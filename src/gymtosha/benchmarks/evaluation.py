"""Held-out Gymnasium evaluation and time-to-target confirmation."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch

from gymtosha.benchmarks.adapters import encode, make_backend
from gymtosha.benchmarks.algorithms import ActorCritic, SoftActor


@dataclass(frozen=True)
class EvaluationConfig:
    target: float | None = None
    interval: int = 65536
    episodes: int = 100
    confirmations: int = 2
    seed: int = 100000

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
        boundary = training.num_envs
        if training.algorithm == "ppo":
            boundary *= training.rollout_steps
        if self.interval % boundary:
            raise ValueError("evaluation interval must align with complete training batches")
        self.resolved_target(training.env_id)


@torch.no_grad()
def evaluate_policy(policy, env_id, *, episodes=100, seed=100000):
    """Evaluate exactly one seeded episode per row on the common CPU reference."""
    env = make_backend(env_id, "gymnasium-sync", episodes, "cpu")
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
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


class TargetEvaluator:
    def __init__(self, env_id, config, *, started=None, on_progress=None):
        self.env_id = env_id
        self.config = config
        self.target = config.resolved_target(env_id)
        self.started = time.perf_counter() if started is None else started
        self.on_progress = on_progress
        self.checkpoints = []
        self.streak = 0
        self.reached = False
        self.next_checkpoint = 0
        self.evaluation_seconds = 0.0

    def record(self, timesteps, scores, *, elapsed):
        score = float(np.mean(scores))
        self.streak = self.streak + 1 if score >= self.target else 0
        self.reached = self.streak >= self.config.confirmations
        self.checkpoints.append(
            {
                "timesteps": timesteps,
                "wall_seconds": elapsed,
                "mean_return": score,
                "std_return": float(np.std(scores)),
                "episode_returns": [float(value) for value in scores],
                "consecutive_passes": self.streak,
            }
        )
        self.next_checkpoint = timesteps + self.config.interval
        if self.on_progress is not None:
            self.on_progress(self.summary())
        return self.reached

    def check(self, model, timesteps, algorithm, *, force=False):
        if timesteps < self.next_checkpoint and not force:
            return False
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
            action, self.env_id, episodes=self.config.episodes, seed=self.config.seed
        )
        self.evaluation_seconds += time.perf_counter() - evaluation_started
        return self.record(timesteps, scores, elapsed=time.perf_counter() - self.started)

    def summary(self):
        final = self.checkpoints[-1] if self.checkpoints else None
        return {
            "environment": "gymnasium",
            "evaluation_seconds": self.evaluation_seconds,
            "env_id": self.env_id,
            "config": asdict(self.config),
            "target": self.target,
            "status": "reached" if self.reached else "budget_exhausted",
            "time_to_target_seconds": final["wall_seconds"] if self.reached else None,
            "transitions_to_target": final["timesteps"] if self.reached else None,
            "observed_wall_seconds": final["wall_seconds"] if final else 0.0,
            "checkpoints": list(self.checkpoints),
        }
