"""CleanRL-derived DQN, PPO, SAC and TD3 learning components.

Adapted from vwxyzjn/cleanrl at fe8d8a03c41a7ef5b523e2e354bd01c363e786bb.
See THIRD_PARTY_NOTICES.md for sources, changes and the MIT license.
"""

import copy
import math

import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


def mlp(sizes, activation=nn.ReLU, orthogonal=False, output_gain=1.0):
    layers = []
    for index, (inputs, outputs) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        layer = nn.Linear(inputs, outputs)
        if orthogonal:
            gain = output_gain if index == len(sizes) - 2 else math.sqrt(2)
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)
        layers.append(layer)
        if index < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, continuous):
        super().__init__()
        self.continuous = continuous
        self.actor = mlp([obs_dim, 64, 64, action_dim], nn.Tanh, True, 0.01)
        self.critic = mlp([obs_dim, 64, 64, 1], nn.Tanh, True)
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def evaluate(self, obs, action=None):
        output = self.actor(obs)
        distribution = (
            Normal(output, self.log_std.exp().expand_as(output))
            if self.continuous
            else Categorical(logits=output)
        )
        if action is None:
            action = distribution.sample()
        log_prob, entropy = distribution.log_prob(action), distribution.entropy()
        if self.continuous:
            log_prob, entropy = log_prob.sum(-1), entropy.sum(-1)
        return action, log_prob, entropy, self.critic(obs).squeeze(-1)


class SoftActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.features = mlp([obs_dim, 256, 256])
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def sample(self, obs):
        features = F.relu(self.features(obs))
        mean = self.mean(features)
        log_std = -5 + 3.5 * (self.log_std(features).tanh() + 1)
        distribution = Normal(mean, log_std.exp())
        raw = distribution.rsample()
        action = raw.tanh()
        log_prob = (distribution.log_prob(raw) - (1 - action.square() + 1e-6).log()).sum(-1)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = mlp([obs_dim + action_dim, 256, 256, 1])

    def forward(self, obs, actions):
        return self.network(torch.cat((obs, actions), dim=-1)).squeeze(-1)


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = mlp([obs_dim, 256, 256, action_dim])

    def forward(self, obs):
        return self.network(obs).tanh()


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_shape, action_dtype, device):
        self.capacity, self.position, self.size = capacity, 0, 0
        self.device = device
        self.observations = torch.empty(capacity, obs_dim, device=device)
        self.next_observations = torch.empty_like(self.observations)
        self.actions = torch.empty((capacity, *action_shape), device=device, dtype=action_dtype)
        self.rewards = torch.empty(capacity, device=device)
        self.terminated = torch.empty(capacity, device=device)

    def add(self, obs, actions, next_obs, rewards, terminated):
        count = len(obs)
        if count > self.capacity:
            raise ValueError("replay capacity must be at least num_envs")
        indices = (torch.arange(count, device=self.device) + self.position) % self.capacity
        self.observations[indices] = obs
        self.actions[indices] = actions
        self.next_observations[indices] = next_obs
        self.rewards[indices] = rewards
        self.terminated[indices] = terminated.float()
        self.position = (self.position + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, count):
        indices = torch.randint(self.size, (count,), device=self.device)
        return (
            self.observations[indices],
            self.actions[indices],
            self.next_observations[indices],
            self.rewards[indices],
            self.terminated[indices],
        )


def generalized_advantage(rewards, values, next_values, terminated, done, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_values[step] * (~terminated[step]) - values[step]
        tail = delta + gamma * gae_lambda * (~done[step]) * tail
        advantages[step] = tail
    return advantages


@torch.no_grad()
def soft_update(source, target, tau):
    for source_param, target_param in zip(source.parameters(), target.parameters(), strict=True):
        target_param.lerp_(source_param, tau)


class OffPolicyLearner:
    def __init__(self, algorithm, obs_dim, action_dim, device, learning_rate, gamma):
        self.algorithm, self.gamma = algorithm, gamma
        self.updates = 0
        self.device = device
        if algorithm == "dqn":
            self.actor = mlp([obs_dim, 120, 84, action_dim]).to(device)
            self.target_actor = copy.deepcopy(self.actor)
            self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        else:
            self.actor = (
                SoftActor(obs_dim, action_dim)
                if algorithm == "sac"
                else DeterministicActor(obs_dim, action_dim)
            ).to(device)
            self.target_actor = copy.deepcopy(self.actor)
            self.critic1 = Critic(obs_dim, action_dim).to(device)
            self.critic2 = Critic(obs_dim, action_dim).to(device)
            self.target1, self.target2 = copy.deepcopy(self.critic1), copy.deepcopy(self.critic2)
            self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
            self.q_optimizer = torch.optim.Adam(
                list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=learning_rate
            )
            if algorithm == "sac":
                self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
                self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
                self.target_entropy = -action_dim

    @torch.no_grad()
    def act(self, obs, epsilon=0.0, *, random_actions=None):
        if self.algorithm == "dqn":
            greedy = self.actor(obs).argmax(-1)
            if random_actions is None:
                random_actions = torch.randint(
                    self.actor[-1].out_features, (len(obs),), device=self.device
                )
            return torch.where(
                torch.rand(len(obs), device=self.device) < epsilon, random_actions, greedy
            )
        if self.algorithm == "sac":
            return self.actor.sample(obs)[0]
        return (
            self.actor(obs)
            + 0.1 * torch.randn(len(obs), self.actor.network[-1].out_features, device=self.device)
        ).clamp(-1, 1)

    def update(self, batch):
        obs, actions, next_obs, rewards, terminated = batch
        self.updates += 1
        if self.algorithm == "dqn":
            with torch.no_grad():
                targets = rewards + self.gamma * (1 - terminated) * self.target_actor(
                    next_obs
                ).amax(-1)
            loss = F.mse_loss(self.actor(obs).gather(1, actions[:, None]).squeeze(-1), targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if self.updates % 100 == 0:
                soft_update(self.actor, self.target_actor, 1.0)
            return loss.detach()
        with torch.no_grad():
            if self.algorithm == "sac":
                next_actions, log_prob = self.actor.sample(next_obs)
                entropy_bonus = self.log_alpha.exp() * log_prob
            else:
                noise = (torch.randn_like(actions) * 0.2).clamp(-0.5, 0.5)
                next_actions = (self.target_actor(next_obs) + noise).clamp(-1, 1)
                entropy_bonus = 0.0
            next_value = (
                torch.minimum(
                    self.target1(next_obs, next_actions), self.target2(next_obs, next_actions)
                )
                - entropy_bonus
            )
            targets = rewards + self.gamma * (1 - terminated) * next_value
        loss = F.mse_loss(self.critic1(obs, actions), targets) + F.mse_loss(
            self.critic2(obs, actions), targets
        )
        self.q_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.q_optimizer.step()
        if self.updates % 2 == 0:
            # CleanRL SAC performs two actor updates per delayed policy update.
            policy_updates = 2 if self.algorithm == "sac" else 1
            for _ in range(policy_updates):
                if self.algorithm == "sac":
                    chosen, log_prob = self.actor.sample(obs)
                    actor_loss = (
                        self.log_alpha.exp().detach() * log_prob
                        - torch.minimum(self.critic1(obs, chosen), self.critic2(obs, chosen))
                    ).mean()
                else:
                    actor_loss = -self.critic1(obs, self.actor(obs)).mean()
                self.optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.optimizer.step()
                if self.algorithm == "sac":
                    alpha_loss = -(
                        self.log_alpha.exp() * (log_prob.detach() + self.target_entropy)
                    ).mean()
                    self.alpha_optimizer.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
            soft_update(self.actor, self.target_actor, 0.005)
        if self.algorithm == "sac" or self.updates % 2 == 0:
            soft_update(self.critic1, self.target1, 0.005)
            soft_update(self.critic2, self.target2, 0.005)
        return loss.detach()
