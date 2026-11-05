import gymnasium as gym
import numpy as np
import pytest
import torch

import gymtosha


@pytest.mark.parametrize("env_id", ["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0"])
def test_dynamics_match_gymnasium(env_id):
    reference = gym.make(env_id)
    env = gymtosha.make(env_id, num_envs=8, autoreset=False)
    obs, _ = env.reset(seed=43)
    rng = np.random.default_rng(25)
    for i in range(8):
        reference.reset(seed=i)
        reference.unwrapped.state = obs[i].numpy().astype(np.float64)
        action = np.array([0.7], dtype=np.float32) if env.continuous else 1
        expected_obs, expected_reward, expected_term, _, _ = reference.step(action)
        if i == 0:
            actions = torch.full((8, 1), 0.7) if env.continuous else torch.ones(8, dtype=torch.long)
            actual = env.step(actions)
        np.testing.assert_allclose(actual[0][i].numpy(), expected_obs, rtol=2e-5, atol=2e-6)
        assert float(actual[1][i]) == pytest.approx(expected_reward, abs=2e-6)
        assert bool(actual[2][i]) == expected_term
    # Boundary conditions, with state injected identically in both implementations.
    for _ in range(20):
        state = rng.uniform(-0.04, 0.04, env.state.shape[1]).astype(np.float32)
        if "MountainCar" in env_id:
            state[:] = [-1.2, -0.02]
        env.state[:] = torch.tensor(state)
        reference.reset()
        reference.unwrapped.state = state.astype(np.float64)
        action = np.array([-1.0], dtype=np.float32) if env.continuous else 0
        expected = reference.step(action)
        actions = -torch.ones(8, 1) if env.continuous else torch.zeros(8, dtype=torch.long)
        actual = env.step(actions)
        np.testing.assert_allclose(actual[0][0].numpy(), expected[0], rtol=2e-5, atol=2e-6)
        assert float(actual[1][0]) == pytest.approx(expected[1], abs=2e-6)
        assert bool(actual[2][0]) == expected[2]
    reference.close()


def test_frozen_lake_all_deterministic_transitions():
    reference = gym.make("FrozenLake-v1", is_slippery=False).unwrapped
    env = gymtosha.make("FrozenLake-v1", num_envs=64, is_slippery=False, autoreset=False)
    states = torch.arange(16).repeat_interleave(4)
    actions = torch.arange(4).repeat(16)
    env.state = states.clone()
    obs, reward, terminated, _, _ = env.step(actions)
    for i, (state, action) in enumerate(zip(states.tolist(), actions.tolist(), strict=True)):
        _, next_state, expected_reward, done = reference.P[state][action][0]
        assert obs[i] == next_state
        assert reward[i] == expected_reward
        assert terminated[i] == done


def test_frozen_lake_slip_probabilities():
    env = gymtosha.make("FrozenLake-v1", num_envs=30000, autoreset=False)
    env.reset(seed=1)
    env.state.fill_(10)
    obs, _, _, _, _ = env.step(torch.zeros(30000, dtype=torch.long))
    for state in [6, 9, 14]:
        assert abs(float((obs == state).float().mean()) - 1 / 3) < 0.02
