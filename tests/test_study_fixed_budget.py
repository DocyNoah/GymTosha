import pytest
import torch

from gymtosha.benchmarks.evaluation import EvaluationConfig
from gymtosha.benchmarks.study import reference_config
from gymtosha.benchmarks.train import TrainConfig, train


def test_budget_configs_match_across_horizons_and_environment_counts():
    from gymtosha.benchmarks.study_fixed_budget import Budget

    plan = Budget((32, 1024, 16384), (16, 32, 512), 134_217_728, 512)
    base = reference_config(32)
    configs = [
        plan.configure(base, 1024, 16),
        plan.configure(base, 1024, 512),
        plan.configure(base, 32, 16),
        plan.configure(base, 16384, 512),
    ]
    assert [c.total_timesteps // (c.num_envs * c.rollout_steps) for c in configs] == [
        8192,
        256,
        262144,
        16,
    ]
    assert {c.total_timesteps for c in configs} == {134_217_728}
    assert {c.ppo_minibatch_size for c in configs} == {512}
    assert {c.update_epochs for c in configs} == {4}
    assert plan.evaluation_interval == 8_388_608


def test_budget_rejects_truncated_rollouts_or_insufficient_final_updates():
    from gymtosha.benchmarks.study_fixed_budget import Budget

    with pytest.raises(ValueError, match="divisible"):
        Budget((32, 16384), (16, 512), 134_217_727, 512)
    with pytest.raises(ValueError, match="16"):
        Budget((32, 16384), (16, 512), 67_108_864, 512)
    with pytest.raises(ValueError, match="minibatch"):
        Budget((32, 16384), (16, 512), 134_217_728, 1024)


def test_horizon_uses_returns_and_transitions_with_shared_workers():
    from gymtosha.benchmarks.study_fixed_budget import select_horizon

    def trial(horizon, reached, reward, steps):
        return {
            "value": horizon,
            "summary": {
                "target_reached": reached,
                "mean_final_eval_return": reward,
                "std_final_eval_return": 0.0,
                "mean_target_transitions": steps,
                "mean_target_seconds": None,
            },
        }

    candidates = [
        trial(32, 10, 498, 40_000),
        trial(64, 10, 500, 80_000),
        trial(128, 10, 500, 60_000),
        trial(512, 9, 500, None),
    ]
    assert select_horizon(candidates) == (128, "stable_final_score")


def test_native_exact_budget_uses_all_slots_after_cartpole_resets():
    config = TrainConfig(
        backend="gymnasium-native",
        native_same_step=True,
        num_envs=4,
        rollout_steps=32,
        total_timesteps=512,
        num_minibatches=4,
        update_epochs=2,
        hidden_sizes=[8],
    )
    result = train(
        config, evaluation=EvaluationConfig(interval=128, episodes=2, stop_on_target=False)
    )
    assert result["completed_episodes"] > 0
    assert result["valid_transitions"] == 512
    assert result["gradient_updates"] == 32
    assert [p["valid_transitions"] for p in result["evaluation"]["checkpoints"]] == [
        0,
        128,
        256,
        384,
        512,
    ]


def test_optimizer_schedule_is_identical_at_different_horizons(monkeypatch):
    rates = []
    original_step = torch.optim.Adam.step

    def record_rate(optimizer, *args, **kwargs):
        rates.append(optimizer.param_groups[0]["lr"])
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", record_rate)
    for horizon in (8, 16):
        rates.clear()
        train(
            TrainConfig(
                num_envs=4,
                rollout_steps=horizon,
                total_timesteps=64,
                ppo_minibatch_size=8,
                update_epochs=2,
                hidden_sizes=[8],
                learning_rate=0.003,
                anneal_by_optimizer_step=True,
            )
        )
        assert rates == pytest.approx([0.003 * (1 - i / 16) for i in range(16)])
