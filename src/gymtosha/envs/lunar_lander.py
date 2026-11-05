"""Approximate LunarLander with batched rigid-body and two-foot contact dynamics.

The action/observation conventions and shaping follow Gymnasium. This solver
uses rigid feet and a projected impulse response rather than Box2D's articulated
legs, revolute joints and iterative polygon contact solver. Trajectories differ.
"""

import math

import numpy as np
import torch
from gymnasium import spaces

from gymtosha.base import BatchedEnv


class LunarLander(BatchedEnv):
    physics_backend = "approximate-tensor-rigid-body"

    def __init__(
        self,
        *,
        continuous=False,
        gravity=-10.0,
        max_episode_steps=1000,
        num_envs=1,
        device="cpu",
        **kwargs,
    ):
        if not -12 < gravity < 0:
            raise ValueError("gravity must be between -12 and 0")
        self.continuous = continuous
        self.gravity = gravity
        self.single_action_space = (
            spaces.Box(-1.0, 1.0, (2,), dtype=np.float32) if continuous else spaces.Discrete(4)
        )
        high = np.array([2.5, 2.5, 10, 10, 2 * math.pi, 10, 1, 1], dtype=np.float32)
        low = -high
        low[-2:] = 0
        self.single_observation_space = spaces.Box(low, high)
        self.terrain = torch.zeros(num_envs, 11, device=device)
        self._sleep_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._feet_x = torch.tensor([-0.67, 0.67], device=device)
        self._hull = (
            torch.tensor(
                [[-14, 17], [-17, 0], [-17, -10], [17, -10], [17, 0], [14, 17]],
                device=device,
                dtype=torch.float32,
            )
            / 30
        )
        super().__init__(
            num_envs=num_envs, device=device, max_episode_steps=max_episode_steps, **kwargs
        )

    def _initial_state(self, mask):
        noise = self._uniform(14, mask)
        heights = noise[:, :12] * (20 / 3)
        pad = torch.arange(12, device=self.device)
        heights = torch.where((pad >= 3) & (pad <= 7), 10 / 3, heights)
        left = torch.cat((heights[:, -1:], heights[:, :10]), dim=-1)
        terrain = (left + heights[:, :11] + heights[:, 1:12]) / 3
        self.terrain = torch.where(mask[:, None], terrain, self.terrain)
        vx = (noise[:, 12] * 2 - 1) * (20 / 4.816)
        vy = (noise[:, 13] * 2 - 1) * (20 / 4.816) + self.gravity * 0.02
        zero = torch.zeros_like(vx)
        return torch.stack(
            (10 + vx * 0.02, 40 / 3 + vy * 0.02, vx, vy, zero, zero, zero, zero), dim=-1
        )

    def _reset_extra(self, mask):
        self._sleep_steps = torch.where(mask, 0, self._sleep_steps)

    def _observation(self, state):
        x, y, vx, vy, angle, omega, left, right = state.unbind(-1)
        return torch.stack(
            (
                (x - 10) / 10,
                (y - (10 / 3 + 0.6)) / (20 / 3),
                vx * 0.2,
                vy * (2 / 15),
                angle,
                omega * 0.4,
                left,
                right,
            ),
            dim=-1,
        )

    def _ground_height(self, x):
        segment = (x / 2).floor().long().clamp(0, 9)
        fraction = (x / 2 - segment).clamp(0, 1)
        low = self.terrain.gather(1, segment)
        high = self.terrain.gather(1, segment + 1)
        return low + fraction * (high - low)

    @staticmethod
    def _shaping(observation):
        distance = torch.linalg.vector_norm(observation[:, :2], dim=-1)
        speed = torch.linalg.vector_norm(observation[:, 2:4], dim=-1)
        return -100 * (distance + speed + observation[:, 4].abs()) + 10 * observation[:, 6:].sum(-1)

    def _transition(self, actions):
        x, y, vx, vy, angle, omega, _, _ = self.state.unbind(-1)
        if self.continuous:
            action = actions.clamp(-1, 1)
            main = torch.where(action[:, 0] > 0, (action[:, 0] + 1) * 0.5, 0.0)
            side = torch.where(action[:, 1].abs() > 0.5, action[:, 1].abs(), 0.0)
            direction = action[:, 1].sign()
        else:
            main = (actions == 2).float()
            side = ((actions == 1) | (actions == 3)).float()
            direction = (actions - 2).float()
        dispersion = (self._uniform(2) * 2 - 1) / 30
        sine, cosine = angle.sin(), angle.cos()
        main_x = sine * (4 / 30 + 2 * dispersion[:, 0]) - cosine * dispersion[:, 1]
        main_y = -cosine * (4 / 30 + 2 * dispersion[:, 0]) - sine * dispersion[:, 1]
        side_x = sine * dispersion[:, 0] - cosine * (3 * dispersion[:, 1] + direction * 0.4)
        side_y = -cosine * dispersion[:, 0] - sine * (3 * dispersion[:, 1] + direction * 0.4)
        impulse_x = -13 * main * main_x - 0.6 * side * side_x
        impulse_y = -13 * main * main_y - 0.6 * side * side_y
        # Main engine's impulse is radial. Side engine has an offset torque arm.
        side_arm_x = side_x - sine * 17 / 30
        side_arm_y = side_y + cosine * 14 / 30
        torque = -0.6 * side * (side_arm_x * side_y - side_arm_y * side_x)
        vx = vx + impulse_x / 4.816
        vy = vy + impulse_y / 4.816 + self.gravity * 0.02
        omega = omega + torque / 0.83
        x, y = x + vx * 0.02, y + vy * 0.02
        angle = angle + omega * 0.02
        sine, cosine = angle.sin()[:, None], angle.cos()[:, None]
        # Detect body collision before correcting penetrations at the feet.
        hull_x = x[:, None] + cosine * self._hull[:, 0] - sine * self._hull[:, 1]
        hull_y = y[:, None] + sine * self._hull[:, 0] + cosine * self._hull[:, 1]
        body_contact = (hull_y <= self._ground_height(hull_x)).any(-1)
        arm_x = cosine * self._feet_x + sine * 0.9
        arm_y = sine * self._feet_x - cosine * 0.9
        foot_x, foot_y = x[:, None] + arm_x, y[:, None] + arm_y
        depth = self._ground_height(foot_x) - foot_y
        contacts = depth >= -0.002
        count = contacts.sum(-1).clamp_min(1)
        foot_velocity = vy[:, None] + omega[:, None] * arm_x
        impulse = (
            torch.where(
                contacts, (-foot_velocity).clamp_min(0) / (1 / 4.816 + arm_x.square() / 0.83), 0.0
            )
            / count[:, None]
        )
        vy = vy + impulse.sum(-1) / 4.816
        omega = omega + (impulse * arm_x).sum(-1) / 0.83
        touch = contacts.any(-1)
        both = contacts.all(-1)
        y = y + depth.clamp_min(0).amax(-1)
        vx = torch.where(touch, vx * 0.8, vx)
        vy = torch.where(both, vy.clamp_min(0) * 0.5, vy)
        omega = torch.where(touch, omega * 0.7, omega)
        quiet = both & (vx.abs() < 0.08) & (vy.abs() < 0.08) & (omega.abs() < 0.08)
        self._sleep_steps = torch.where(quiet, self._sleep_steps + 1, 0)
        landed = self._sleep_steps >= 25
        crashed = body_contact | (x <= 0) | (x >= 20)
        state = torch.stack((x, y, vx, vy, angle, omega, contacts[:, 0], contacts[:, 1]), dim=-1)
        reward = self._shaping(self._observation(state)) - self._shaping(
            self._observation(self.state)
        )
        reward = reward - 0.30 * main - 0.03 * side
        reward = torch.where(landed & ~crashed, 100.0, reward)
        reward = torch.where(crashed, -100.0, reward)
        return state, reward, landed | crashed
