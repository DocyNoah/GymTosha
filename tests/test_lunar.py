import pytest
import torch

import gymtosha


@pytest.mark.parametrize("continuous", [False, True])
def test_lunar_shapes_seeds_and_long_rollout(continuous):
    env_id = "LunarLanderContinuous-v3" if continuous else "LunarLander-v3"
    env = gymtosha.make(env_id, num_envs=32, max_episode_steps=200)
    obs, _ = env.reset(seed=9)
    assert obs.shape == (32, 8)
    assert env.single_action_space.shape == ((2,) if continuous else ())
    assert env.physics_backend == "approximate-tensor-rigid-body"
    terrain = env.terrain.clone()
    expected, _ = env.reset(seed=9)
    torch.testing.assert_close(obs, expected)
    torch.testing.assert_close(terrain, env.terrain)
    ended = torch.zeros(32, dtype=torch.bool)
    for _ in range(250):
        action = torch.zeros(32, 2) if continuous else torch.zeros(32, dtype=torch.long)
        obs, reward, term, trunc, info = env.step(action)
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        ended |= term | trunc
    assert ended.all()


def test_lunar_thrust_and_touchdown():
    env = gymtosha.make("LunarLander-v3", num_envs=2, autoreset=False)
    env.reset(seed=1)
    env.state[1] = env.state[0]
    env.terrain[1] = env.terrain[0]
    env.step(torch.tensor([0, 2]))
    assert env.state[1, 3] > env.state[0, 3]
    env.state.zero_()
    env.state[:, 0] = 10
    env.state[:, 1] = 3.33333333 + 0.9
    for _ in range(150):
        obs, reward, term, _, _ = env.step(torch.zeros(2, dtype=torch.long))
        if term.all():
            break
    assert term.all()
    assert (reward == 100).all()
    assert (obs[:, 6:] == 1).all()


def test_lunar_body_contact_is_crash():
    env = gymtosha.make("LunarLander-v3", num_envs=2, autoreset=False)
    env.state.zero_()
    env.state[:, 0] = 10
    env.state[:, 1] = 3.3
    _, reward, term, _, _ = env.step(torch.zeros(2, dtype=torch.long))
    assert term.all() and (reward == -100).all()
