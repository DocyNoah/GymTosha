import gymnasium as gym
import numpy as np
import pytest
import torch

from gymtosha.benchmarks.evaluation import EvaluationConfig, TargetEvaluator, evaluate_policy
from gymtosha.benchmarks.train import TrainConfig, train


def test_evaluation_uses_held_out_episodes_and_preserves_rng():
    torch.manual_seed(19)
    before = torch.random.get_rng_state().clone()
    scores = evaluate_policy(
        lambda obs: torch.zeros(len(obs), dtype=torch.long),
        "CartPole-v1",
        episodes=8,
        seed=123,
    )
    np.testing.assert_allclose(
        scores,
        evaluate_policy(
            lambda obs: torch.zeros(len(obs), dtype=torch.long),
            "CartPole-v1",
            episodes=8,
            seed=123,
        ),
    )
    assert len(scores) == 8 and min(scores) >= 1
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("angular_velocity_weight", [0.0, 0.19])
def test_numpy_cartpole_evaluation_matches_sync_gymnasium_episodes(angular_velocity_weight):
    def actions(obs):
        return (obs[:, 2] + angular_velocity_weight * obs[:, 3] > 0).long()

    kwargs = {"episodes": 100, "seed": 200000}
    expected = evaluate_policy(actions, "CartPole-v1", **kwargs)
    actual = evaluate_policy(actions, "CartPole-v1", **kwargs, native_cartpole=True)
    np.testing.assert_array_equal(actual, expected)


def test_target_needs_consecutive_confirmations():
    evaluation = TargetEvaluator("CartPole-v1", EvaluationConfig(target=10, confirmations=2))
    assert not evaluation.record(0, [9, 11], elapsed=1.0)
    assert not evaluation.record(10, [0, 0], elapsed=2.0)
    assert not evaluation.record(20, [10, 12], elapsed=3.0)
    assert evaluation.record(30, [11, 12], elapsed=4.0)
    result = evaluation.summary()
    assert result["status"] == "reached"
    assert result["time_to_target_seconds"] == 4.0
    assert result["transitions_to_target"] == 30


def test_missed_target_is_censored():
    evaluation = TargetEvaluator("FrozenLake-v1", EvaluationConfig())
    evaluation.record(100, [0, 0], elapsed=2.0)
    result = evaluation.summary()
    assert result["status"] == "budget_exhausted"
    assert result["time_to_target_seconds"] is None
    assert result["transitions_to_target"] is None


@pytest.mark.parametrize("backend", ["gymtosha", "gymnasium-sync"])
def test_target_training_stops_and_reports_actual_transitions(backend):
    config = TrainConfig(
        backend=backend,
        num_envs=4,
        total_timesteps=256,
        rollout_steps=16,
        update_epochs=1,
        num_minibatches=2,
    )
    evaluation = EvaluationConfig(target=-1, interval=64, episodes=4, confirmations=2)
    result = train(config, evaluation=evaluation)
    assert result["timesteps"] == 64
    assert result["evaluation"]["status"] == "reached"
    assert result["evaluation"]["time_to_target_seconds"] > result["seconds"]
    assert [r["timesteps"] for r in result["evaluation"]["checkpoints"]] == [0, 64]
    assert result["evaluation"]["environment"] == "gymnasium"


def test_evaluation_config_validates_interval_before_training():
    with pytest.raises(ValueError, match="interval"):
        train(TrainConfig(), evaluation=EvaluationConfig(interval=3))


def test_gymnasium_thresholds():
    for name in ["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0", "LunarLander-v3"]:
        assert EvaluationConfig().resolved_target(name) == gym.spec(name).reward_threshold
    assert EvaluationConfig().resolved_target("FrozenLake-v1") == 0.7


def test_fixed_length_evaluation_latches_first_target_without_stopping():
    config = EvaluationConfig(target=10, confirmations=2, stop_on_target=False)
    evaluator = TargetEvaluator("CartPole-v1", config)
    assert not evaluator.record(64, [12, 12], elapsed=1)
    assert not evaluator.record(128, [12, 12], elapsed=2)
    assert not evaluator.record(192, [0, 0], elapsed=3)
    summary = evaluator.summary()
    assert summary["status"] == "reached"
    assert summary["time_to_target_seconds"] == 2
    assert summary["transitions_to_target"] == 128
    assert len(summary["checkpoints"]) == 3


def test_target_measurement_can_run_full_training_budget():
    result = train(
        TrainConfig(num_envs=4, total_timesteps=128, rollout_steps=16, update_epochs=1),
        evaluation=EvaluationConfig(target=-1, interval=64, episodes=4, stop_on_target=False),
    )
    assert result["timesteps"] == 128
    assert result["evaluation"]["transitions_to_target"] == 64


def test_post_update_target_excludes_evaluation_and_latches_first_mean(monkeypatch):
    from gymtosha.benchmarks import evaluation as module

    clock = [100.0]
    evaluations = iter([[500] * 50 + [450] * 50, [450] * 100])

    def evaluate(*args, **kwargs):
        clock[0] += 20
        return next(evaluations)

    def progress(summary):
        clock[0] += 2

    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(module, "evaluate_policy", evaluate)
    config = EvaluationConfig(
        target=475,
        interval=64,
        confirmations=1,
        stop_on_target=False,
        evaluate_at_start=False,
        training_time_only=True,
    )
    evaluator = TargetEvaluator("CartPole-v1", config, on_progress=progress)
    model = torch.nn.Linear(4, 2)
    assert not evaluator.check(model, 0, "ppo")
    clock[0] += 3
    assert not evaluator.check(model, 64, "ppo", valid_transitions=64, gradient_updates=4)
    clock[0] += 5
    assert not evaluator.check(model, 128, "ppo")
    result = evaluator.summary()
    assert result["time_to_target_seconds"] == 3
    assert result["valid_transitions_to_target"] == 64
    assert result["observed_wall_seconds"] == 8
    assert result["evaluation_seconds"] == 44
    assert [p["timesteps"] for p in result["checkpoints"]] == [64, 128]


def test_stable_target_requires_five_mean_and_episode_rate_confirmations():
    config = EvaluationConfig(
        target=475, confirmations=5, stop_on_target=False, min_episode_success_rate=0.95
    )
    evaluator = TargetEvaluator("CartPole-v1", config)
    passing = [500] * 95 + [0] * 5
    unreliable = [500] * 94 + [474] * 6
    for step in range(1, 5):
        assert not evaluator.record(step, passing, elapsed=step)
    assert evaluator.summary()["status"] == "budget_exhausted"
    evaluator.record(5, unreliable, elapsed=5)
    assert evaluator.streak == 0
    for step in range(6, 11):
        evaluator.record(step, passing, elapsed=step)
    summary = evaluator.summary()
    assert summary["status"] == "reached"
    assert summary["transitions_to_target"] == 10
    assert summary["time_to_target_seconds"] == 10
    assert summary["checkpoints"][-1]["episode_success_rate"] == 0.95
    evaluator.record(11, unreliable, elapsed=11)
    assert evaluator.summary()["transitions_to_target"] == 10
