import pytest
import torch

import gymtosha

IDS = ["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", "FrozenLake-v1"]


@pytest.mark.parametrize("env_id", IDS)
def test_batch_seed_reset_and_step(env_id):
    env = gymtosha.make(env_id, num_envs=17)
    obs, _ = env.reset(seed=123)
    other = gymtosha.make(env_id, num_envs=17)
    expected, _ = other.reset(seed=123)
    torch.testing.assert_close(obs, expected)
    actions = torch.zeros(
        (17, 1) if env.continuous else (17,), dtype=torch.float32 if env.continuous else torch.long
    )
    result = env.step(actions)
    assert len(result) == 5
    assert result[0].shape == obs.shape
    assert all(t.shape == (17,) for t in result[1:4])
    assert result[2].dtype == result[3].dtype == torch.bool
    assert torch.isfinite(result[0]).all()
    assert all(isinstance(value, torch.Tensor) for value in result[4].values())
    torch.testing.assert_close(result[0], other.step(actions)[0])
    env.close()


@pytest.mark.parametrize("env_id", IDS)
def test_masked_reset_preserves_active_rows(env_id):
    env = gymtosha.make(env_id, num_envs=4)
    env.reset(seed=2)
    env.elapsed_steps += 10
    before = env.state.clone()
    mask = torch.tensor([True, False, True, False])
    env.reset(options={"reset_mask": mask})
    torch.testing.assert_close(env.state[~mask], before[~mask])
    assert torch.equal(env.elapsed_steps, torch.tensor([0, 10, 0, 10]))


@pytest.mark.parametrize("env_id", IDS)
def test_same_step_autoreset_preserves_final_observation(env_id):
    env = gymtosha.make(env_id, num_envs=3, max_episode_steps=1)
    env.reset(seed=5)
    action = torch.zeros(
        (3, 1) if env.continuous else (3,), dtype=torch.float32 if env.continuous else torch.long
    )
    obs, _, _, trunc, info = env.step(action)
    assert trunc.all()
    assert info["_final_observation"].all()
    assert torch.equal(info["episode_length"], torch.ones(3, dtype=torch.long))
    assert env.elapsed_steps.sum() == 0
    assert obs.shape == info["final_observation"].shape


def test_local_rng_does_not_modify_torch_rng():
    torch.manual_seed(44)
    expected = torch.rand(3)
    torch.manual_seed(44)
    env = gymtosha.make("CartPole-v1", num_envs=5)
    env.reset(seed=987)
    for _ in range(5):
        env.step(torch.zeros(5, dtype=torch.long))
    torch.testing.assert_close(torch.rand(3), expected)


def test_validation_and_registry():
    with pytest.raises(ValueError):
        gymtosha.make("typo")
    with pytest.raises(ValueError):
        gymtosha.make("CartPole-v1", num_envs=0)
    env = gymtosha.make("CartPole-v1", num_envs=2)
    with pytest.raises(ValueError):
        env.step(torch.zeros(3, dtype=torch.long))
    with pytest.raises(TypeError):
        env.step(torch.zeros(2))


def test_continuous_float64_action_preserves_observation_dtype():
    env = gymtosha.make("MountainCarContinuous-v0", num_envs=2)
    observation, reward, _, _, _ = env.step(torch.zeros(2, 1, dtype=torch.float64))
    assert observation.dtype == reward.dtype == torch.float32
