"""Matched CleanRL-derived learning loops for speed and return comparisons."""

from __future__ import annotations

import math
import platform
import time
from dataclasses import asdict, dataclass, field
from functools import partial

import gymnasium as gym
import torch
from torch import nn

from gymtosha.benchmarks.adapters import encode, make_backend, synchronize
from gymtosha.benchmarks.algorithms import (
    ActorCritic,
    OffPolicyLearner,
    ReplayBuffer,
    generalized_advantage,
)


@dataclass
class TrainConfig:
    env_id: str = "CartPole-v1"
    algorithm: str = "ppo"
    backend: str = "gymtosha"
    device: str = "cpu"
    num_envs: int = 64
    total_timesteps: int = 65536
    seed: int = 1
    compile_env: bool = False
    num_threads: int = 1
    learning_rate: float | None = None
    gamma: float = 0.99
    rollout_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    buffer_size: int = 100000
    batch_size: int = 256
    learning_starts: int = 1024
    train_every: int = 256
    gradient_steps: int = 1
    log_every: int = 4096
    exploration_fraction: float = 0.5
    exploration_hold_steps: int = 1
    normalize_observations: bool = False
    normalize_rewards: bool = False
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    ppo_minibatch_size: int | None = None
    adam_eps: float = 1e-5
    anneal_learning_rate: bool = True
    anneal_by_optimizer_step: bool = False
    rollout_timeout_seconds: float | None = None
    native_same_step: bool = False

    def validate(self):
        if self.rollout_timeout_seconds is not None and (
            not math.isfinite(self.rollout_timeout_seconds) or self.rollout_timeout_seconds <= 0
        ):
            raise ValueError("rollout timeout must be finite and positive")
        if not math.isfinite(self.adam_eps) or self.adam_eps <= 0:
            raise ValueError("Adam epsilon must be finite and positive")
        if not self.hidden_sizes or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive integers")
        if self.algorithm != "ppo" and (self.normalize_observations or self.normalize_rewards):
            raise ValueError("running normalization is supported for PPO")
        if not 0 < self.exploration_fraction <= 1:
            raise ValueError("exploration_fraction must be in (0, 1]")
        if self.algorithm not in ("ppo", "dqn", "sac", "td3"):
            raise ValueError(f"unknown algorithm {self.algorithm!r}")
        positive = (
            "num_envs",
            "total_timesteps",
            "num_threads",
            "rollout_steps",
            "update_epochs",
            "num_minibatches",
            "buffer_size",
            "batch_size",
            "train_every",
            "gradient_steps",
            "log_every",
            "exploration_hold_steps",
        )
        if any(getattr(self, key) < 1 for key in positive):
            raise ValueError("counts and frequencies must be positive")
        if self.total_timesteps % self.num_envs:
            raise ValueError("total_timesteps must be divisible by num_envs")
        if self.algorithm == "ppo":
            rollout_size = self.num_envs * self.rollout_steps
            divisor = self.num_minibatches
            if self.ppo_minibatch_size is not None:
                divisor = self.ppo_minibatch_size
                if isinstance(divisor, bool) or not isinstance(divisor, int) or divisor < 2:
                    raise ValueError("PPO minibatch size must be an integer of at least two")
            if self.total_timesteps % rollout_size or rollout_size % divisor:
                raise ValueError("PPO budget and minibatches must divide whole rollouts")
            minibatch_size = self.ppo_minibatch_size or rollout_size // self.num_minibatches
            if minibatch_size < 2:
                raise ValueError("PPO minibatches require at least two samples")
        elif self.buffer_size < self.num_envs or self.train_every % self.num_envs:
            raise ValueError("replay must fit num_envs and train_every must divide vector steps")
        if self.algorithm != "ppo" and not 0 <= self.learning_starts < self.total_timesteps:
            raise ValueError("learning_starts must be inside the training budget")


class Recorder:
    def __init__(self, device):
        self.stats = torch.zeros(3, device=device)
        self.checkpoints = []
        self.start = time.perf_counter()
        self.device = device

    def add(self, transition):
        done = transition.done.float()
        self.stats += torch.stack(
            (
                (transition.episode_returns * done).sum(),
                (transition.episode_lengths * done).sum(),
                done.sum(),
            )
        )

    def record(self, timesteps, loss):
        synchronize(self.device)
        sums = self.stats.cpu().tolist()
        self.checkpoints.append(
            {
                "timesteps": timesteps,
                "seconds": time.perf_counter() - self.start,
                "mean_return": sums[0] / sums[2] if sums[2] else None,
                "mean_episode_length": sums[1] / sums[2] if sums[2] else None,
                "episodes": int(sums[2]),
                "loss": float(loss.detach().cpu()),
            }
        )
        self.stats.zero_()


def _ppo_diagnostics(minibatches, clip_coef, learning_rate, metrics):
    entropy, policy_loss, value_loss, log_ratio = zip(*minibatches, strict=True)
    log_ratio = torch.cat(log_ratio)
    ratio_delta = log_ratio.exp() - 1
    means = (
        torch.stack(
            (
                torch.stack(entropy).mean(),
                torch.stack(policy_loss).mean(),
                torch.stack(value_loss).mean(),
                (ratio_delta - log_ratio).mean(),
                (ratio_delta.abs() > clip_coef).float().mean(),
            )
        )
        .cpu()
        .tolist()
    )
    names = ("entropy", "policy_loss", "value_loss", "approx_kl", "clip_fraction")
    return {**metrics, "learning_rate": learning_rate, **dict(zip(names, means, strict=True))}


def _ppo(
    config, env, obs, obs_dim, action_dim, continuous, rate, evaluator=None, *, startup_seconds=0
):
    learner_setup_start = time.perf_counter()
    device = env.device
    agent = ActorCritic(obs_dim, action_dim, continuous, hidden_sizes=config.hidden_sizes).to(
        device
    )
    from gymtosha.benchmarks.normalization import RewardNormalizer, RunningMoments

    observation_normalizer = (
        RunningMoments(obs_dim, device) if config.normalize_observations else None
    )
    reward_normalizer = (
        RewardNormalizer(config.num_envs, device, config.gamma)
        if config.normalize_rewards
        else None
    )
    agent.observation_normalizer = observation_normalizer
    if observation_normalizer is not None:
        observation_normalizer.update(obs)
        obs = observation_normalizer.normalize(obs)
    optimizer = torch.optim.Adam(agent.parameters(), lr=rate, eps=config.adam_eps)
    initial = nn.utils.parameters_to_vector(agent.parameters()).detach().clone()
    size = config.num_envs * config.rollout_steps
    minibatch_size = config.ppo_minibatch_size or size // config.num_minibatches
    shape = (config.rollout_steps, config.num_envs)
    observations = torch.empty((*shape, obs_dim), device=device)
    action_shape = (*shape, action_dim) if continuous else shape
    actions = torch.empty(
        action_shape, dtype=torch.float32 if continuous else torch.long, device=device
    )
    log_probs, values, rewards, next_values = [torch.empty(shape, device=device) for _ in range(4)]
    terminated, dones = [torch.empty(shape, dtype=torch.bool, device=device) for _ in range(2)]
    valid = torch.empty(shape, dtype=torch.bool, device=device)
    masked_resets = getattr(env, "next_step_autoreset", False)
    with torch.no_grad():
        agent.evaluate(obs)
    synchronize(device)
    recorder = Recorder(device)
    learner_setup_seconds = time.perf_counter() - learner_setup_start
    collection_seconds = 0.0
    update_seconds = 0.0
    updates = 0
    iterations = config.total_timesteps // size
    planned_updates = config.total_timesteps // minibatch_size * config.update_epochs
    timesteps = 0
    valid_transitions = torch.zeros((), dtype=torch.long, device=device)
    log_updates = evaluator is not None and evaluator.on_checkpoint is not None
    completed_episodes = 0
    stop = evaluator is not None and evaluator.check(
        agent, 0, config.algorithm, valid_transitions=0, gradient_updates=0
    )
    for iteration in range(iterations):
        if stop:
            break
        if config.anneal_learning_rate and not config.anneal_by_optimizer_step:
            optimizer.param_groups[0]["lr"] = rate * (1 - iteration / iterations)
        collection_start = time.perf_counter()
        initial_seconds = startup_seconds + learner_setup_seconds if iteration == 0 else 0.0
        for step in range(config.rollout_steps):
            observations[step] = obs
            with torch.no_grad():
                action, log_prob, _, value = agent.evaluate(obs)
                # CleanRL's continuous PPO uses ClipAction at the environment
                # boundary, while the likelihood uses the original sample.
                env_action = action.clamp(-1, 1) if continuous else action
                transition = env.step(env_action)
                raw_next_obs = encode(transition.observation, env.observation_space)
                final_obs = encode(transition.final_observation, env.observation_space)
                if observation_normalizer is not None:
                    observation_normalizer.update(raw_next_obs)
                    final_obs = observation_normalizer.normalize(final_obs)
                next_value = agent.critic(final_obs)
            actions[step], log_probs[step], values[step] = action, log_prob, value
            rewards[step] = (
                reward_normalizer.normalize(transition.reward, transition.done)
                if reward_normalizer is not None
                else transition.reward
            )
            next_values[step] = next_value.squeeze(-1)
            terminated[step], dones[step] = transition.terminated, transition.done
            valid[step] = True if transition.valid is None else transition.valid
            obs = (
                observation_normalizer.normalize(raw_next_obs)
                if observation_normalizer is not None
                else raw_next_obs
            )
            recorder.add(transition)
            if config.rollout_timeout_seconds is not None and (
                time.perf_counter() - collection_start + initial_seconds
                >= config.rollout_timeout_seconds
            ):
                raise TimeoutError(
                    "environment setup and rollout collection exceeded the time limit"
                )
        if config.rollout_timeout_seconds is not None or log_updates:
            synchronize(device)
            collected = time.perf_counter() - collection_start
            if (
                config.rollout_timeout_seconds is not None
                and collected + initial_seconds >= config.rollout_timeout_seconds
            ):
                raise TimeoutError(
                    "environment setup and rollout collection exceeded the time limit"
                )
            collection_seconds += collected
        update_start = time.perf_counter()
        advantages = generalized_advantage(
            rewards, values, next_values, terminated, dones, config.gamma, config.gae_lambda
        )
        returns = advantages + values
        flat_obs = observations.reshape(size, obs_dim)
        flat_actions = actions.reshape(size, action_dim) if continuous else actions.reshape(size)
        flat_log_probs, flat_values = log_probs.reshape(size), values.reshape(size)
        flat_advantages, flat_returns = advantages.reshape(size), returns.reshape(size)
        eligible = valid.flatten().nonzero().flatten() if masked_resets else None
        valid_transitions += valid.sum()
        loss = torch.zeros((), device=device)
        diagnostics = []
        for _ in range(config.update_epochs):
            indices = (
                eligible[torch.randperm(eligible.numel(), device=device)]
                if masked_resets
                else torch.randperm(size, device=device)
            )
            for index in indices.split(minibatch_size):
                if index.numel() < 2:
                    continue
                _, new_log_prob, entropy, new_value = agent.evaluate(
                    flat_obs[index], flat_actions[index]
                )
                log_ratio = new_log_prob - flat_log_probs[index]
                ratio = log_ratio.exp()
                advantage = flat_advantages[index]
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                raw_policy_loss = -advantage * ratio
                clipped_policy_loss = -advantage * ratio.clamp(
                    1 - config.clip_coef, 1 + config.clip_coef
                )
                policy_loss = torch.maximum(raw_policy_loss, clipped_policy_loss).mean()
                clipped_value = flat_values[index] + (new_value - flat_values[index]).clamp(
                    -config.clip_coef, config.clip_coef
                )
                value_loss = (
                    0.5
                    * torch.maximum(
                        (new_value - flat_returns[index]).square(),
                        (clipped_value - flat_returns[index]).square(),
                    ).mean()
                )
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy_mean
                )
                if log_updates:
                    diagnostics.append(
                        (
                            entropy_mean.detach(),
                            policy_loss.detach(),
                            value_loss.detach(),
                            log_ratio.detach(),
                        )
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                if config.anneal_learning_rate and config.anneal_by_optimizer_step:
                    optimizer.param_groups[0]["lr"] = rate * (1 - updates / planned_updates)
                optimizer.step()
                updates += 1
        timesteps = (iteration + 1) * size
        recorder.record(timesteps, loss)
        update_seconds += time.perf_counter() - update_start
        if evaluator is not None:
            telemetry = None
            if log_updates:
                completed_episodes += recorder.checkpoints[-1]["episodes"]
                telemetry = partial(
                    _ppo_diagnostics,
                    diagnostics,
                    config.clip_coef,
                    optimizer.param_groups[0]["lr"],
                    {
                        "ppo_update": iteration + 1,
                        "completed_episodes": completed_episodes,
                        "learner_setup_seconds": learner_setup_seconds,
                        "collection_seconds": collection_seconds,
                        "update_seconds": update_seconds,
                    },
                )
            stop = evaluator.check(
                agent,
                timesteps,
                config.algorithm,
                force=timesteps == config.total_timesteps,
                telemetry=telemetry,
                valid_transitions=valid_transitions,
                gradient_updates=updates,
            )
    synchronize(device)
    elapsed = time.perf_counter() - recorder.start
    if evaluator is not None:
        elapsed -= evaluator.evaluation_seconds
    delta = float(
        (nn.utils.parameters_to_vector(agent.parameters()).detach() - initial).norm().cpu()
    )
    phases = {
        "collection_seconds": collection_seconds
        if config.rollout_timeout_seconds is not None
        else None,
        "update_seconds": update_seconds if config.rollout_timeout_seconds is not None else None,
        "learner_setup_seconds": learner_setup_seconds,
    }
    return (
        elapsed,
        recorder.checkpoints,
        updates,
        delta,
        timesteps,
        int(valid_transitions.cpu()),
        phases,
    )


def _off_policy(
    config, env, obs, obs_dim, action_dim, continuous, rate, evaluator=None, *, startup_seconds=0
):
    device = env.device
    learner = OffPolicyLearner(config.algorithm, obs_dim, action_dim, device, rate, config.gamma)
    initial = nn.utils.parameters_to_vector(learner.actor.parameters()).detach().clone()
    buffer = ReplayBuffer(
        config.buffer_size,
        obs_dim,
        (action_dim,) if continuous else (),
        torch.float32 if continuous else torch.long,
        device,
    )
    with torch.no_grad():
        learner.act(obs)
    synchronize(device)
    recorder = Recorder(device)
    loss = torch.zeros((), device=device)
    timesteps = 0
    held_random_actions = None
    stop = evaluator is not None and evaluator.check(learner.actor, 0, config.algorithm)
    for next_timesteps in range(config.num_envs, config.total_timesteps + 1, config.num_envs):
        if stop:
            break
        timesteps = next_timesteps
        before_step = timesteps - config.num_envs
        if continuous and before_step < config.learning_starts:
            action = torch.rand(config.num_envs, action_dim, device=device) * 2 - 1
        else:
            epsilon = max(
                0.05,
                1 - 0.95 * before_step / (config.exploration_fraction * config.total_timesteps),
            )
            if config.algorithm == "dqn":
                if before_step % (config.num_envs * config.exploration_hold_steps) == 0:
                    held_random_actions = torch.randint(
                        action_dim, (config.num_envs,), device=device
                    )
                action = learner.act(obs, epsilon, random_actions=held_random_actions)
            else:
                action = learner.act(obs, epsilon)
        transition = env.step(action)
        buffer.add(
            obs,
            action,
            encode(transition.final_observation, env.observation_space),
            transition.reward,
            transition.terminated,
        )
        obs = encode(transition.observation, env.observation_space)
        recorder.add(transition)
        if timesteps >= config.learning_starts and timesteps % config.train_every == 0:
            for _ in range(config.gradient_steps):
                loss = learner.update(buffer.sample(config.batch_size))
        if timesteps % config.log_every == 0 or timesteps == config.total_timesteps:
            recorder.record(timesteps, loss)
        if evaluator is not None:
            stop = evaluator.check(
                learner.actor,
                timesteps,
                config.algorithm,
                force=timesteps == config.total_timesteps,
            )
    if not recorder.checkpoints or recorder.checkpoints[-1]["timesteps"] != timesteps:
        recorder.record(timesteps, loss)
    synchronize(device)
    elapsed = time.perf_counter() - recorder.start
    if evaluator is not None:
        elapsed -= evaluator.evaluation_seconds
    delta = float(
        (nn.utils.parameters_to_vector(learner.actor.parameters()).detach() - initial).norm().cpu()
    )
    return elapsed, recorder.checkpoints, learner.updates, delta, timesteps, timesteps, {}


def train(config: TrainConfig, *, evaluation=None, on_progress=None, on_checkpoint=None) -> dict:
    from gymtosha.benchmarks.evaluation import TargetEvaluator

    config.validate()
    if config.backend == "gymnasium-native" and config.algorithm != "ppo":
        raise ValueError("the native vector backend currently supports PPO only")
    if evaluation is not None:
        evaluation.validate(config)
    if on_checkpoint is not None and (
        config.algorithm != "ppo"
        or evaluation is None
        or not evaluation.training_time_only
        or evaluation.evaluate_at_start
        or evaluation.stop_on_target
        or evaluation.interval != config.num_envs * config.rollout_steps
    ):
        raise ValueError("checkpoint logging requires fixed-length PPO with post-update evaluation")
    torch.set_num_threads(config.num_threads)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    setup_start = time.perf_counter()
    evaluator = (
        TargetEvaluator(
            config.env_id,
            evaluation,
            started=setup_start,
            on_progress=on_progress,
            on_checkpoint=on_checkpoint,
        )
        if evaluation is not None
        else None
    )
    env = make_backend(
        config.env_id,
        config.backend,
        config.num_envs,
        config.device,
        config.compile_env,
        native_same_step=config.native_same_step,
    )
    try:
        continuous = isinstance(env.action_space, gym.spaces.Box)
        if config.algorithm == "dqn" and continuous:
            raise ValueError("DQN requires a discrete action space")
        if config.algorithm in ("sac", "td3") and not continuous:
            raise ValueError("SAC and TD3 require a continuous action space")
        obs = encode(env.reset(config.seed), env.observation_space)
        action_dim = math.prod(env.action_space.shape) if continuous else env.action_space.n
        setup_seconds = time.perf_counter() - setup_start
        warmup_start = time.perf_counter()
        zeros = torch.zeros(
            (config.num_envs, *env.action_space.shape),
            device=env.device,
            dtype=torch.float32 if continuous else torch.long,
        )
        for _ in range(3):
            env.step(zeros)
        synchronize(env.device)
        warmup_seconds = time.perf_counter() - warmup_start
        if on_checkpoint is not None:
            evaluator.setup_metrics = {
                "setup_seconds": setup_seconds,
                "warmup_seconds": warmup_seconds,
            }
        obs = encode(env.reset(config.seed), env.observation_space)
        rate = config.learning_rate or (2.5e-4 if config.algorithm in ("ppo", "dqn") else 3e-4)
        loop = _ppo if config.algorithm == "ppo" else _off_policy
        elapsed, checkpoints, updates, delta, timesteps, valid_transitions, phases = loop(
            config,
            env,
            obs,
            obs.shape[-1],
            action_dim,
            continuous,
            rate,
            evaluator,
            startup_seconds=time.perf_counter() - setup_start,
        )
        completed_episodes = sum(point["episodes"] for point in checkpoints)
        episode_return_sum = sum(
            point["mean_return"] * point["episodes"] for point in checkpoints if point["episodes"]
        )
        result = {
            "config": asdict(config),
            "timesteps": timesteps,
            "seconds": elapsed,
            "steps_per_second": timesteps / elapsed,
            "valid_transitions": valid_transitions,
            "valid_transitions_per_second": valid_transitions / elapsed,
            "completed_episodes": completed_episodes,
            "mean_training_episode_return": episode_return_sum / completed_episodes
            if completed_episodes
            else None,
            "setup_seconds": setup_seconds,
            "warmup_seconds": warmup_seconds,
            "gradient_updates": updates,
            "parameter_delta": delta,
            "checkpoints": checkpoints,
            "physics_equivalent": "LunarLander" not in config.env_id,
            "phase_seconds": phases,
            "metadata": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "torch": torch.__version__,
                "gymnasium": gym.__version__,
                "device": str(env.device),
                "num_threads": torch.get_num_threads(),
                "cuda_device": torch.cuda.get_device_name(env.device)
                if env.device.type == "cuda"
                else None,
            },
        }
        if evaluator is not None:
            result["evaluation"] = evaluator.summary()
            result["wall_seconds"] = time.perf_counter() - setup_start
            if evaluation.training_time_only:
                result["wall_seconds"] -= evaluator.evaluation_seconds
        return result
    finally:
        env.close()
