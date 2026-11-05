"""Batched discrete and continuous MountainCar dynamics."""

import numpy as np
import torch
from gymnasium import spaces

from gymtosha.base import BatchedEnv


class MountainCar(BatchedEnv):
    def __init__(self, *, continuous=False, max_episode_steps=None, goal_velocity=0.0, **kwargs):
        self.continuous = continuous
        self.goal_velocity = goal_velocity
        self.max_speed = 0.07
        self.goal_position = 0.45 if continuous else 0.5
        self.single_observation_space = spaces.Box(
            np.array([-1.2, -self.max_speed], dtype=np.float32),
            np.array([0.6, self.max_speed], dtype=np.float32),
        )
        self.single_action_space = (
            spaces.Box(-1.0, 1.0, (1,), dtype=np.float32) if continuous else spaces.Discrete(3)
        )
        if max_episode_steps is None:
            max_episode_steps = 999 if continuous else 200
        super().__init__(max_episode_steps=max_episode_steps, **kwargs)

    def _initial_state(self, mask):
        position = self._uniform(1, mask).squeeze(-1) * 0.2 - 0.6
        return torch.stack((position, torch.zeros_like(position)), dim=-1)

    def _transition(self, actions):
        position, velocity = self.state.unbind(-1)
        force = actions[:, 0].clamp(-1, 1) if self.continuous else actions.float() - 1
        power = 0.0015 if self.continuous else 0.001
        velocity = (velocity + force * power - 0.0025 * (3 * position).cos()).clamp(
            -self.max_speed, self.max_speed
        )
        position = (position + velocity).clamp(-1.2, 0.6)
        velocity = torch.where((position <= -1.2) & (velocity < 0), 0.0, velocity)
        terminated = (position >= self.goal_position) & (velocity >= self.goal_velocity)
        reward = (
            terminated.float() * 100 - 0.1 * actions[:, 0].square()
            if self.continuous
            else -torch.ones_like(position)
        )
        return torch.stack((position, velocity), dim=-1), reward, terminated
