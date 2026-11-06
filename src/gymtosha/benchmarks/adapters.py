"""A common learner boundary for native tensors and Gymnasium vector envs."""

from dataclasses import dataclass
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

import gymtosha


@dataclass
class Transition:
    observation: Tensor
    final_observation: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    episode_returns: Tensor
    episode_lengths: Tensor
    valid: Tensor | None = None

    @property
    def done(self):
        return self.terminated | self.truncated


class TensorBackend:
    def __init__(self, env_id, num_envs, device, compile_env=False, **kwargs):
        self.env = gymtosha.make(env_id, num_envs=num_envs, device=device, **kwargs)
        if compile_env:
            self.env.compile()
        self.observation_space = self.env.single_observation_space
        self.action_space = self.env.single_action_space
        self.device = self.env.device
        self.num_envs = num_envs

    def reset(self, seed):
        return self.env.reset(seed=seed)[0]

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return Transition(
            obs,
            info["final_observation"],
            reward,
            terminated,
            truncated,
            info["episode_return"],
            info["episode_length"],
        )

    def close(self):
        self.env.close()


class GymnasiumBackend:
    def __init__(
        self,
        env_id,
        num_envs,
        device,
        *,
        asynchronous=False,
        native=False,
        native_same_step=False,
        **kwargs,
    ):
        if env_id == "LunarLanderContinuous-v3":
            env_id = "LunarLander-v3"
            kwargs["continuous"] = True
        vector_type = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
        self.env = (
            gym.make_vec(
                env_id, num_envs=num_envs, vectorization_mode="vector_entry_point", **kwargs
            )
            if native
            else vector_type(
                [partial(gym.make, env_id, **kwargs) for _ in range(num_envs)],
                autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
            )
        )
        self.observation_space = self.env.single_observation_space
        self.action_space = self.env.single_action_space
        self.device = torch.empty(0, device=device).device
        self.num_envs = num_envs
        self.returns = torch.zeros(num_envs, device=self.device)
        self.lengths = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.native_same_step = native and native_same_step
        if self.native_same_step:
            from gymnasium.envs.classic_control.cartpole import CartPoleVectorEnv

            if not isinstance(self.env, CartPoleVectorEnv):
                raise ValueError("native same-step reset is implemented for CartPole only")
        self.next_step_autoreset = native and not self.native_same_step
        self.pending_reset = torch.zeros(num_envs, device=self.device, dtype=torch.bool)

    def reset(self, seed):
        self.returns.zero_()
        self.lengths.zero_()
        self.pending_reset.zero_()
        obs, _ = self.env.reset(seed=seed)
        return torch.as_tensor(obs, device=self.device).clone()

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions.detach().cpu().numpy())
        # Object-array terminal observations exist only for ended Gymnasium rows.
        final_obs = obs.copy()
        if "final_obs" in info:
            for index in np.flatnonzero(info["_final_obs"]):
                final_obs[index] = info["final_obs"][index]
        reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
        terminated = torch.as_tensor(terminated, device=self.device)
        truncated = torch.as_tensor(truncated, device=self.device)
        valid = ~self.pending_reset if self.next_step_autoreset else None
        returns = self.returns + reward
        lengths = self.lengths + (valid.long() if valid is not None else 1)
        done = terminated | truncated
        if self.native_same_step and self.env.prev_done.any():
            # Use the native reset distribution before the next physics step.
            mask = self.env.prev_done
            self.env.state[:, mask] = self.env.np_random.uniform(
                low=self.env.low, high=self.env.high, size=(4, mask.sum())
            )
            self.env.steps[mask] = 0
            obs[mask] = self.env.state[:, mask].T
            mask.fill(False)
        if self.next_step_autoreset:
            self.pending_reset = done
        self.returns = torch.where(done, 0.0, returns)
        self.lengths = torch.where(done, 0, lengths)
        return Transition(
            torch.as_tensor(obs, device=self.device).clone(),
            torch.as_tensor(final_obs, device=self.device),
            reward,
            terminated,
            truncated,
            returns,
            lengths,
            valid,
        )

    def close(self):
        self.env.close()


def make_backend(
    env_id, backend, num_envs, device, compile_env=False, *, native_same_step=False, **kwargs
):
    if backend == "gymtosha":
        return TensorBackend(env_id, num_envs, device, compile_env, **kwargs)
    if backend in ("gymnasium-sync", "gymnasium-async", "gymnasium-native"):
        if compile_env:
            raise ValueError("compile_env applies only to gymtosha")
        return GymnasiumBackend(
            env_id,
            num_envs,
            device,
            asynchronous=backend == "gymnasium-async",
            native=backend == "gymnasium-native",
            native_same_step=native_same_step,
            **kwargs,
        )
    raise ValueError(f"unknown backend {backend!r}")


def encode(observation, space):
    if isinstance(space, gym.spaces.Discrete):
        return torch.nn.functional.one_hot(observation.long(), space.n).float()
    return observation.float().flatten(start_dim=1)


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
