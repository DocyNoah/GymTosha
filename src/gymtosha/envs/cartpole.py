"""Batched CartPole equations matching Gymnasium's Euler integration."""

import math

import numpy as np
import torch
from gymnasium import spaces

from gymtosha.base import BatchedEnv


class CartPole(BatchedEnv):
    def __init__(self, *, max_episode_steps=500, **kwargs):
        high = np.array([4.8, np.inf, 24 * math.pi / 180, np.inf], dtype=np.float32)
        self.single_observation_space = spaces.Box(-high, high)
        self.single_action_space = spaces.Discrete(2)
        super().__init__(max_episode_steps=max_episode_steps, **kwargs)

    def _initial_state(self, mask):
        return self._uniform(4, mask) * 0.1 - 0.05

    def _transition(self, actions):
        x, velocity, angle, angular_velocity = self.state.unbind(-1)
        force = actions.to(torch.float32) * 20.0 - 10.0
        sine, cosine = angle.sin(), angle.cos()
        intermediate = (force + 0.05 * angular_velocity.square() * sine) / 1.1
        angular_acceleration = (9.8 * sine - cosine * intermediate) / (
            0.5 * (4.0 / 3.0 - 0.1 * cosine.square() / 1.1)
        )
        acceleration = intermediate - 0.05 * angular_acceleration * cosine / 1.1
        state = torch.stack(
            (
                x + 0.02 * velocity,
                velocity + 0.02 * acceleration,
                angle + 0.02 * angular_velocity,
                angular_velocity + 0.02 * angular_acceleration,
            ),
            dim=-1,
        )
        terminated = (state[:, 0].abs() > 2.4) | (state[:, 2].abs() > 12 * math.pi / 180)
        return state, torch.ones_like(x), terminated
