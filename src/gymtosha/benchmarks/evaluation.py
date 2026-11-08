"""Held-out Gymnasium evaluation and time-to-target confirmation."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch

from gymtosha.benchmarks.adapters import encode, make_backend, synchronize
from gymtosha.benchmarks.algorithms import ActorCritic, SoftActor


@dataclass(frozen=True)
class EvaluationConfig:
    target: float | None = None
    interval: int = 65536
    episodes: int = 100
    confirmations: int = 2
    seed: int = 100000
    stop_on_target: bool = True
    min_episode_success_rate: float = 0.0
    evaluate_at_start: bool = True
    training_time_only: bool = False
    native_cartpole: bool = False

    def resolved_target(self, env_id):
        if self.target is not None:
            if not np.isfinite(self.target):
                raise ValueError("target must be finite")
            return float(self.target)
        gym_id = "LunarLander-v3" if env_id == "LunarLanderContinuous-v3" else env_id
        target = gym.spec(gym_id).reward_threshold
        if target is None:
            raise ValueError(f"provide a target for {env_id}")
        return float(target)

    def validate(self, training):
        if min(self.interval, self.episodes, self.confirmations) < 1 or self.seed < 0:
            raise ValueError("evaluation counts must be positive and seed nonnegative")
        if (
            not np.isfinite(self.min_episode_success_rate)
            or not 0 <= self.min_episode_success_rate <= 1
        ):
            raise ValueError("minimum episode success rate must be between zero and one")
        boundary = training.num_envs
        if training.algorithm == "ppo":
            boundary *= training.rollout_steps
        if self.interval % boundary:
            raise ValueError("evaluation interval must align with complete training batches")
        self.resolved_target(training.env_id)


@torch.no_grad()
def evaluate_policy(policy, env_id, *, episodes=100, seed=100000, native_cartpole=False):
    """Evaluate exactly one seeded episode per row on the common CPU reference."""
    if native_cartpole and env_id != "CartPole-v1":
        raise ValueError("native evaluation requires CartPole-v1")
    env = (
        gym.make_vec(env_id, num_envs=episodes, vectorization_mode="vector_entry_point")
        if native_cartpole
        else make_backend(env_id, "gymnasium-sync", episodes, "cpu")
    )
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            if native_cartpole:
                return _evaluate_native_cartpole(policy, env, episodes, seed)
            observation = env.reset(seed)
            active = torch.ones(episodes, dtype=torch.bool)
            scores = torch.zeros(episodes)
            while active.any():
                actions = policy(encode(observation, env.observation_space))
                if isinstance(env.action_space, gym.spaces.Box):
                    actions = actions.clamp(-1, 1)
                transition = env.step(actions)
                scores += transition.reward * active
                active &= ~transition.done
                observation = transition.observation
            return scores.numpy().copy()
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        env.close()


def _evaluate_native_cartpole(policy, env, episodes, seed):
    env.reset(seed=seed)
    # Match the independent seed+i resets of SyncVectorEnv for the first episode.
    env.state[:] = np.stack(
        [np.random.default_rng(seed + i).uniform(env.low, env.high, 4) for i in range(episodes)],
        axis=1,
    )
    observation = env.state.T.astype(np.float32)
    active = np.ones(episodes, dtype=bool)
    scores = np.zeros(episodes, dtype=np.float32)
    while active.any():
        action = policy(torch.as_tensor(observation)).numpy()
        observation, reward, terminated, truncated, _ = env.step(action)
        scores += reward * active
        active &= ~(terminated | truncated)
    return scores


class TargetEvaluator:
    def __init__(self, env_id, config, *, started=None, on_progress=None, on_checkpoint=None):
        self.env_id = env_id
        self.config = config
        self.target = config.resolved_target(env_id)
        self.started = time.perf_counter() if started is None else started
        self.on_progress = on_progress
        self.on_checkpoint = on_checkpoint
        self.setup_metrics = {}
        self.checkpoints = []
        self.streak = 0
        self.reached = False
        self.first_target = None
        self.next_checkpoint = 0 if config.evaluate_at_start else config.interval
        self.evaluation_seconds = 0.0

    def record(self, timesteps, scores, *, elapsed, **counts):
        score = float(np.mean(scores))
        success_rate = float(np.mean(np.asarray(scores) >= self.target))
        passed = score >= self.target and success_rate >= self.config.min_episode_success_rate
        self.streak = self.streak + 1 if passed else 0
        confirmed = self.streak >= self.config.confirmations
        self.checkpoints.append(
            {
                "timesteps": timesteps,
                "wall_seconds": elapsed,
                "mean_return": score,
                "std_return": float(np.std(scores)),
                "episode_success_rate": success_rate,
                "episode_returns": [float(value) for value in scores],
                "consecutive_passes": self.streak,
                **counts,
            }
        )
        if confirmed and not self.reached:
            self.first_target = self.checkpoints[-1]
            self.reached = True
        self.next_checkpoint = timesteps + self.config.interval
        if self.on_progress is not None:
            self.on_progress(self.summary())
        return self.reached and self.config.stop_on_target

    def check(self, model, timesteps, algorithm, *, force=False, telemetry=None, **counts):
        if timesteps < self.next_checkpoint and not force:
            return False
        counts = {key: int(value) for key, value in counts.items()}
        if self.config.training_time_only:
            synchronize(next(model.parameters()).device)
        # Copy weights to CPU so evaluation has the same device and overhead for
        # both training backends and does not mutate the training model or RNG.
        evaluation_started = time.perf_counter()
        policy = copy.deepcopy(model).cpu().eval()

        def action(observation):
            normalizer = getattr(policy, "observation_normalizer", None)
            if normalizer is not None:
                observation = normalizer.normalize(observation)
            if isinstance(policy, ActorCritic):
                output = policy.actor(observation)
                return output if policy.continuous else output.argmax(-1)
            if isinstance(policy, SoftActor):
                return policy.mean(torch.relu(policy.features(observation))).tanh()
            output = policy(observation)
            return output.argmax(-1) if algorithm == "dqn" else output

        scores = evaluate_policy(
            action,
            self.env_id,
            episodes=self.config.episodes,
            seed=self.config.seed,
            native_cartpole=self.config.native_cartpole,
        )
        evaluated = time.perf_counter()
        elapsed = (
            evaluation_started - self.started - self.evaluation_seconds
            if self.config.training_time_only
            else evaluated - self.started
        )
        self.evaluation_seconds += evaluated - evaluation_started
        try:
            if self.on_checkpoint is not None:
                self.on_checkpoint(
                    {
                        "timesteps": timesteps,
                        "wall_seconds": elapsed,
                        "episode_returns": scores,
                        **counts,
                        **self.setup_metrics,
                        **telemetry(),
                    }
                )
                self.next_checkpoint = timesteps + self.config.interval
                return False
            return self.record(timesteps, scores, elapsed=elapsed, **counts)
        finally:
            if self.config.training_time_only:
                self.evaluation_seconds += time.perf_counter() - evaluated

    def summary(self):
        if self.on_checkpoint is not None:
            return {
                "environment": "gymnasium",
                "evaluation_seconds": self.evaluation_seconds,
                "env_id": self.env_id,
                "config": asdict(self.config),
            }
        final = self.checkpoints[-1] if self.checkpoints else None
        return {
            "environment": "gymnasium",
            "evaluation_seconds": self.evaluation_seconds,
            "env_id": self.env_id,
            "config": asdict(self.config),
            "target": self.target,
            "status": "reached" if self.reached else "budget_exhausted",
            "time_to_target_seconds": self.first_target["wall_seconds"] if self.reached else None,
            "transitions_to_target": self.first_target["timesteps"] if self.reached else None,
            "valid_transitions_to_target": (
                self.first_target.get("valid_transitions") if self.reached else None
            ),
            "gradient_updates_to_target": (
                self.first_target.get("gradient_updates") if self.reached else None
            ),
            "observed_wall_seconds": final["wall_seconds"] if final else 0.0,
            "checkpoints": list(self.checkpoints),
        }
