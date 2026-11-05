"""Device-native batched environment lifecycle and independent random streams."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import torch
from gymnasium.vector.utils import batch_space
from torch import Tensor


class BatchedEnv(ABC):
    """Tensor API with optional same-step autoreset and a fixed-shape info dict.

    Inputs must contain valid actions in the single action space. Value checks
    are intentionally outside the simulation loop to avoid host synchronization.
    Returned tensors are snapshots, not mutable views into the next step.
    """

    continuous = False
    metadata = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}

    def __init__(
        self,
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        *,
        max_episode_steps: int,
        autoreset: bool = True,
    ):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not isinstance(max_episode_steps, int) or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.max_episode_steps = max_episode_steps
        self.autoreset = autoreset
        self.metadata = dict(self.metadata)
        if not autoreset:
            self.metadata["autoreset_mode"] = gym.vector.AutoresetMode.DISABLED
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.elapsed_steps = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.episode_returns = torch.zeros(num_envs, device=self.device)
        self._rng = torch.ones(num_envs, device=self.device, dtype=torch.long)
        self._step_fn: Callable = self._step
        self.reset(seed=0)

    def _uniform(self, count: int, mask: Tensor | None = None) -> Tensor:
        # Park-Miller streams are device-local. Multiplication fits signed int64.
        # No torch.Generator argument or global RNG mutation inside compiled graphs.
        state = self._rng
        samples = []
        for _ in range(count):
            state = (state * 48271) % 2147483647
            samples.append(state.to(torch.float32) / 2147483647)
        self._rng = state if mask is None else torch.where(mask, state, self._rng)
        return torch.stack(samples, dim=-1)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        mask = None if options is None else options.get("reset_mask")
        if mask is None:
            mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        if mask.shape != (self.num_envs,) or mask.dtype != torch.bool or mask.device != self.device:
            raise ValueError("reset_mask must be a boolean tensor on the environment device")
        if seed is not None:
            # Scramble adjacent indices before using them as independent streams.
            seeds = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            seeds = ((seeds + int(seed) % 2147483646) * 104729 + 12345) % 2147483646 + 1
            self._rng = torch.where(mask, seeds, self._rng)
        fresh = self._initial_state(mask)
        if hasattr(self, "state"):
            expanded = mask if fresh.ndim == 1 else mask[:, None]
            self.state = torch.where(expanded, fresh, self.state)
        else:
            self.state = fresh
        self.elapsed_steps = torch.where(mask, 0, self.elapsed_steps)
        self.episode_returns = torch.where(mask, 0.0, self.episode_returns)
        self._reset_extra(mask)
        return self._observation(self.state).clone(), {}

    def step(self, actions: Tensor):
        if not isinstance(actions, Tensor):
            raise TypeError("actions must be a torch.Tensor")
        shape = (self.num_envs, *self.single_action_space.shape)
        if actions.shape != shape or actions.device != self.device:
            raise ValueError(f"actions must have shape {shape} on {self.device}")
        if self.continuous:
            if not actions.is_floating_point():
                raise TypeError("continuous actions must be floating-point tensors")
        elif actions.dtype not in (torch.int32, torch.int64):
            raise TypeError("discrete actions must have integer dtype")
        with torch.no_grad():
            return self._step_fn(actions)

    def _step(self, actions: Tensor):
        state, reward, terminated = self._transition(actions)
        elapsed = self.elapsed_steps + 1
        returns = self.episode_returns + reward
        truncated = elapsed >= self.max_episode_steps
        done = terminated | truncated
        final_obs = self._observation(state)
        info = {
            "final_observation": final_obs,
            "_final_observation": done,
            "episode_return": returns,
            "episode_length": elapsed,
        }
        self.state = state
        self.elapsed_steps = elapsed
        self.episode_returns = returns
        if self.autoreset:
            fresh = self._initial_state(done)
            self.state = torch.where(done if state.ndim == 1 else done[:, None], fresh, state)
            self.elapsed_steps = torch.where(done, 0, elapsed)
            self.episode_returns = torch.where(done, 0.0, returns)
            self._reset_extra(done)
        return self._observation(self.state), reward, terminated, truncated, info

    def compile(self, **kwargs: Any) -> BatchedEnv:
        """Compile simulation and masked autoreset together, with graph breaks forbidden."""
        kwargs.setdefault("fullgraph", True)
        self._step_fn = torch.compile(self._step, **kwargs)
        return self

    def _observation(self, state: Tensor) -> Tensor:
        return state

    def _reset_extra(self, mask: Tensor) -> None:
        return None

    @abstractmethod
    def _initial_state(self, mask: Tensor) -> Tensor: ...

    @abstractmethod
    def _transition(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]: ...

    def close(self) -> None:
        return None
