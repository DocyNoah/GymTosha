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


def test_dqn_can_use_temporally_held_random_actions():
    from gymtosha.benchmarks.algorithms import OffPolicyLearner

    learner = OffPolicyLearner("dqn", 2, 3, "cpu", 0.00025, 0.99)
    observation = torch.zeros(8, 2)
    random_actions = torch.full((8,), 2, dtype=torch.long)
    assert torch.equal(learner.act(observation, 1.0, random_actions=random_actions), random_actions)
    torch.testing.assert_close(
        learner.act(observation, 0.0, random_actions=random_actions),
        learner.actor(observation).argmax(-1),
    )


def test_configurable_ppo_hidden_layers_and_minibatches():
    from gymtosha.benchmarks.algorithms import ActorCritic

    agent = ActorCritic(4, 2, False, hidden_sizes=[256, 256, 256])
    layers = [module for module in agent.actor if isinstance(module, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [
        (4, 256),
        (256, 256),
        (256, 256),
        (256, 2),
    ]
    result = train(
        TrainConfig(
            num_envs=4,
            total_timesteps=64,
            rollout_steps=16,
            update_epochs=1,
            hidden_sizes=[256, 256, 256],
            ppo_minibatch_size=16,
        )
    )
    assert result["gradient_updates"] == 4
    assert result["parameter_delta"] > 0
    assert result["config"]["hidden_sizes"] == [256, 256, 256]


@pytest.mark.parametrize("size", [0, 1, True, 2.5, 16.0, 65])
def test_invalid_ppo_minibatch_size(size):
    with pytest.raises(ValueError, match="minibatch"):
        TrainConfig(
            num_envs=4, rollout_steps=16, total_timesteps=64, ppo_minibatch_size=size
        ).validate()


def test_native_gymnasium_marks_next_step_resets_as_invalid_samples():
    env = make_backend("CartPole-v1", "gymnasium-native", 4, "cpu", max_episode_steps=1)
    env.reset(11)
    first = env.step(torch.zeros(4, dtype=torch.long))
    assert first.valid.all() and first.truncated.all()
    second = env.step(torch.zeros(4, dtype=torch.long))
    assert not second.valid.any()
    assert not second.done.any()
    assert (second.episode_lengths == 0).all()
    env.reset(11)
    assert env.step(torch.zeros(4, dtype=torch.long)).valid.all()
    env.close()


def test_native_ppo_excludes_autoreset_slots_from_updates(monkeypatch):
    import importlib

    module = importlib.import_module("gymtosha.benchmarks.train")
    original = module.make_backend

    def make_short_env(*args, **kwargs):
        return original(*args, **kwargs, max_episode_steps=1)

    monkeypatch.setattr(module, "make_backend", make_short_env)
    result = train(
        TrainConfig(
            backend="gymnasium-native",
            num_envs=4,
            total_timesteps=64,
            rollout_steps=4,
            update_epochs=1,
            ppo_minibatch_size=4,
        )
    )
    assert result["gradient_updates"] == 8
    assert result["valid_transitions"] == 32
    assert result["completed_episodes"] == 32
    assert result["mean_training_episode_return"] == 1.0
    assert result["parameter_delta"] > 0
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in result["checkpoints"])


def test_native_vector_rejects_unmasked_off_policy_learning():
    with pytest.raises(ValueError, match="PPO only"):
        train(TrainConfig(backend="gymnasium-native", algorithm="dqn"))


def test_ppo_uses_configured_adam_epsilon_and_constant_learning_rate(monkeypatch):
    original = torch.optim.Adam
    epsilons, rates = [], []

    def tracked_adam(parameters, **kwargs):
        epsilons.append(kwargs["eps"])
        optimizer = original(parameters, **kwargs)
        step = optimizer.step

        def tracked_step(*args, **step_kwargs):
            rates.append(optimizer.param_groups[0]["lr"])
            return step(*args, **step_kwargs)

        optimizer.step = tracked_step
        return optimizer

    monkeypatch.setattr(torch.optim, "Adam", tracked_adam)
    train(
        TrainConfig(
            num_envs=4,
            rollout_steps=4,
            total_timesteps=32,
            update_epochs=1,
            ppo_minibatch_size=4,
            learning_rate=0.001,
            adam_eps=1e-8,
            anneal_learning_rate=False,
        )
    )
    assert epsilons == [1e-8]
    assert len(rates) == 8 and all(rate == 0.001 for rate in rates)
