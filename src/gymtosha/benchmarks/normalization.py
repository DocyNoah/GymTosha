"""Device-local running moments for CleanRL-style PPO preprocessing."""

import torch
from torch import nn


class RunningMoments(nn.Module):
    def __init__(self, dimensions, device):
        super().__init__()
        shape = (dimensions,) if dimensions else ()
        self.register_buffer("mean", torch.zeros(shape, device=device))
        self.register_buffer("variance", torch.ones(shape, device=device))
        self.register_buffer("count", torch.tensor(1e-4, device=device))

    @torch.no_grad()
    def update(self, values):
        size = values.shape[0]
        delta = values.mean(dim=0) - self.mean
        total = self.count + size
        variance_sum = self.variance * self.count + values.var(dim=0, unbiased=False) * size
        variance_sum += delta.square() * self.count * size / total
        self.mean.copy_(self.mean + delta * size / total)
        self.variance.copy_(variance_sum / total)
        self.count.copy_(total)

    def normalize(self, values):
        return ((values - self.mean) / (self.variance + 1e-8).sqrt()).clamp(-10, 10)


class RewardNormalizer:
    def __init__(self, num_envs, device, gamma):
        self.moments = RunningMoments(0, device)
        self.discounted_returns = torch.zeros(num_envs, device=device)
        self.gamma = gamma

    @torch.no_grad()
    def normalize(self, rewards, done):
        self.discounted_returns = self.discounted_returns * self.gamma + rewards
        self.moments.update(self.discounted_returns)
        result = (rewards / (self.moments.variance + 1e-8).sqrt()).clamp(-10, 10)
        self.discounted_returns = torch.where(done, 0.0, self.discounted_returns)
        return result
