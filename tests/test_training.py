import pytest
import torch

from gymtosha.benchmarks.adapters import make_backend
from gymtosha.benchmarks.algorithms import ReplayBuffer, generalized_advantage
from gymtosha.benchmarks.train import TrainConfig, train


def test_truncation_bootstrap_and_episode_boundary():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[3.0], [4.0]])
    next_values = torch.tensor([[10.0], [20.0]])
    terminated = torch.tensor([[False], [True]])
    done = torch.tensor([[True], [True]])
    advantages = generalized_advantage(rewards, values, next_values, terminated, done, 0.9, 0.95)
    torch.testing.assert_close(advantages, torch.tensor([[7.0], [-2.0]]))


@pytest.mark.parametrize("backend", ["gymtosha", "gymnasium-sync"])
def test_adapter_returns_terminal_observation(backend):
    env = make_backend("CartPole-v1", backend, 2, "cpu", max_episode_steps=1)
    env.reset(123)
    result = env.step(torch.ones(2, dtype=torch.long))
    assert result.truncated.all()
    assert result.final_observation.shape == result.observation.shape == (2, 4)
    assert not torch.equal(result.final_observation, result.observation)
    assert (result.episode_lengths == 1).all()
    env.close()


def test_replay_wrap_and_snapshot():
    buffer = ReplayBuffer(5, 2, (), torch.long, "cpu")
    obs = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    buffer.add(obs, torch.ones(3, dtype=torch.long), obs + 10, torch.ones(3), torch.zeros(3))
    obs.zero_()
    assert buffer.observations[0, 1] == 1
    buffer.add(
        torch.ones(3, 2),
        torch.zeros(3, dtype=torch.long),
        torch.ones(3, 2),
        torch.zeros(3),
        torch.ones(3),
    )
    assert buffer.size == 5 and buffer.position == 1
    assert buffer.sample(4)[0].shape == (4, 2)


@pytest.mark.parametrize(
    "algorithm,env_id",
    [
        ("ppo", "CartPole-v1"),
        ("ppo", "MountainCarContinuous-v0"),
        ("dqn", "FrozenLake-v1"),
        ("sac", "MountainCarContinuous-v0"),
        ("td3", "MountainCarContinuous-v0"),
    ],
)
@pytest.mark.parametrize("backend", ["gymtosha", "gymnasium-sync"])
def test_learning_updates_parameters(algorithm, env_id, backend):
    result = train(
        TrainConfig(
            algorithm=algorithm,
            env_id=env_id,
            backend=backend,
            num_envs=4,
            total_timesteps=256,
            rollout_steps=16,
            update_epochs=2,
            num_minibatches=2,
            learning_starts=32,
            train_every=16,
            batch_size=32,
            buffer_size=512,
            log_every=64,
        )
    )
    assert result["timesteps"] == 256
    assert result["gradient_updates"] > 0
    assert result["parameter_delta"] > 0
    assert result["seconds"] > 0
    assert result["checkpoints"][-1]["timesteps"] == 256
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in result["checkpoints"])


def test_invalid_training_combination():
    with pytest.raises(ValueError, match="discrete"):
        train(TrainConfig(algorithm="dqn", env_id="MountainCarContinuous-v0"))


def test_ppo_sends_bounded_actions_to_environment(monkeypatch):
    from gymtosha.benchmarks.adapters import TensorBackend

    original = TensorBackend.step

    def checked_step(self, actions):
        assert (actions.abs() <= 1).all(), "out-of-bounds continuous action"
        return original(self, actions)

    monkeypatch.setattr(TensorBackend, "step", checked_step)
    train(
        TrainConfig(
            env_id="MountainCarContinuous-v0",
            algorithm="ppo",
            num_envs=4,
            total_timesteps=128,
            rollout_steps=16,
            update_epochs=1,
            num_minibatches=2,
        )
    )
