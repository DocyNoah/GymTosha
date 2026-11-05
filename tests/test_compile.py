import pytest
import torch

import gymtosha


@pytest.mark.compile
@pytest.mark.parametrize(
    "env_id",
    [
        "CartPole-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "FrozenLake-v1",
        "LunarLander-v3",
        "LunarLanderContinuous-v3",
    ],
)
def test_fullgraph_eager_equivalence(env_id):
    eager = gymtosha.make(env_id, num_envs=8, max_episode_steps=4)
    compiled = gymtosha.make(env_id, num_envs=8, max_episode_steps=4).compile(backend="aot_eager")
    eager.reset(seed=7)
    compiled.reset(seed=7)
    for _ in range(10):
        action = torch.zeros(
            (8, *eager.single_action_space.shape),
            dtype=torch.float32 if eager.continuous else torch.long,
        )
        expected = eager.step(action)
        actual = compiled.step(action)
        for left, right in zip(expected[:4], actual[:4], strict=True):
            torch.testing.assert_close(left, right)
        for key in expected[4]:
            torch.testing.assert_close(expected[4][key], actual[4][key])


@pytest.mark.accelerator
@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("env_id", ["CartPole-v1", "FrozenLake-v1", "LunarLander-v3"])
def test_accelerator(device, env_id):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} unavailable")
    env = gymtosha.make(env_id, num_envs=64, device=device)
    obs, _ = env.reset(seed=20)
    for _ in range(20):
        obs, reward, _, _, _ = env.step(torch.zeros(64, device=device, dtype=torch.long))
        assert obs.device.type == device and reward.device.type == device
        assert torch.isfinite(obs).all()
