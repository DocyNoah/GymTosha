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


@pytest.mark.accelerator
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
@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_inductor_including_autoreset(env_id, device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} unavailable")
    eager = gymtosha.make(env_id, num_envs=8, device=device, max_episode_steps=3)
    compiled = gymtosha.make(env_id, num_envs=8, device=device, max_episode_steps=3).compile()
    eager.reset(seed=14)
    compiled.reset(seed=14)
    for _ in range(5):
        actions = torch.zeros(
            (8, *eager.single_action_space.shape),
            device=device,
            dtype=torch.float32 if eager.continuous else torch.long,
        )
        expected, actual = eager.step(actions), compiled.step(actions)
        for left, right in zip(expected[:4], actual[:4], strict=True):
            torch.testing.assert_close(left, right, rtol=1e-4, atol=2e-5)
        for key in expected[4]:
            torch.testing.assert_close(expected[4][key], actual[4][key], rtol=1e-4, atol=2e-5)


@pytest.mark.accelerator
@pytest.mark.compile
def test_mps_compile_mode():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    env = gymtosha.make("CartPole-v1", num_envs=4, device="mps").compile(mode="default")
    env.step(torch.zeros(4, dtype=torch.long, device="mps"))


@pytest.mark.compile
def test_many_environment_compilations_do_not_share_cache_limit():
    torch.compiler.reset()
    for env_id in [
        "CartPole-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "FrozenLake-v1",
        "LunarLander-v3",
    ]:
        for size in [1, 4, 16]:
            env = gymtosha.make(env_id, num_envs=size).compile(backend="eager")
            action = torch.zeros(
                (size, *env.single_action_space.shape),
                dtype=torch.float32 if env.continuous else torch.long,
            )
            env.step(action)
    torch.compiler.reset()
