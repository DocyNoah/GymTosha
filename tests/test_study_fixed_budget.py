import json
import sys
from pathlib import Path

import pytest
import torch

from gymtosha.benchmarks.evaluation import EvaluationConfig
from gymtosha.benchmarks.study import reference_config, study_evaluation
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
    with pytest.raises(ValueError, match="8"):
        Budget((32, 16384), (16, 512), 33_554_432, 512)
    with pytest.raises(ValueError, match="minibatch"):
        Budget((32, 16384), (16, 512), 134_217_728, 1024)


def test_cartpole_tuning_budget_uses_only_the_fixed_tuning_count():
    from gymtosha.benchmarks.study_fixed_budget import HORIZONS, Budget, final_budget

    tuning = Budget((1024,), HORIZONS, 4_194_304, 512)
    assert tuning.evaluation_interval == 524_288
    assert tuning.configure(reference_config(32), 1024, 512).total_timesteps == 4_194_304
    assert final_budget(tuning, 16).total_timesteps == 4_194_304
    assert final_budget(tuning, 64).total_timesteps == 8_388_608
    assert final_budget(tuning, 512).total_timesteps == 67_108_864
    final = final_budget(tuning, 64)
    assert final.horizons == (64,)
    assert final.evaluation_interval == 1_048_576
    assert final.configure(reference_config(32), 16384, 64).total_timesteps == 8_388_608


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


def test_fixed_horizon_starts_comparison_without_tuning(tmp_path, monkeypatch):
    from gymtosha.benchmarks import study_fixed_budget as study

    baseline = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/results/cartpole-study/scaling/selected.json"
    )
    output = tmp_path / "comparison"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "study_fixed_budget",
            "--baseline",
            str(baseline),
            "--output",
            str(output),
            "--devices",
            "cpu",
            "--fixed-horizon",
            "64",
        ],
    )

    def unexpected_tuning(*args):
        pytest.fail("fixed horizon should skip tuning")

    def verify_selected(selected, directory, devices, *, cpu_limit):
        assert directory == output
        assert devices == ["cpu"]
        assert cpu_limit == 1024
        assert {row["config"]["rollout_steps"] for row in selected.values()} == {64}
        for count, row in selected.items():
            assert row["evaluation_interval"] == int(count) * 64
            assert row["evaluate_each_update"]
            evaluation = study_evaluation(TrainConfig(**row["config"]), each_update=True)
            assert evaluation.interval == row["evaluation_interval"]
            assert evaluation.confirmations == 1
            assert evaluation.min_episode_success_rate == 0
            assert evaluation.episodes == 100
            assert evaluation.training_time_only and not evaluation.evaluate_at_start
            assert not evaluation.stop_on_target
        return []

    def compare(args):
        plan = json.loads((args.output / "final_plan.json").read_text())
        assert plan["horizon"] == 64
        assert plan["total_timesteps"] == 8_388_608
        assert plan["evaluation_intervals"] == {str(n): n * 64 for n in study.ENV_COUNTS}
        assert plan["rollouts"] == {str(n): 8_388_608 // (n * 64) for n in study.ENV_COUNTS}

    monkeypatch.setattr(study.signal, "signal", lambda *args: None)
    monkeypatch.setattr(study, "tune", unexpected_tuning)
    monkeypatch.setattr(study, "verify_selected", verify_selected)
    monkeypatch.setattr(study, "compare", compare)
    study.main()

    assert json.loads((output / "protocol.json").read_text())["fixed_horizon"] == 64


def test_capacity_cpu_limit_does_not_limit_accelerators(tmp_path, monkeypatch):
    from gymtosha.benchmarks import study_grouped as study

    def probe(config, directory):
        assert config.device != "cpu" or config.num_envs <= 1024
        return {"status": "passed"}

    monkeypatch.setattr(study, "probe", probe)
    selected = {str(n): {"config": {}} for n in (32, 1024, 4096)}
    capacities = study.verify_selected(selected, tmp_path, ("cpu", "cuda", "mps"), cpu_limit=1024)
    for row in capacities:
        if row["backend"] == "gymnasium-async":
            assert row["verified_counts"] == [32]
        elif row["device"] == "cpu":
            assert row["verified_counts"] == [32, 1024]
        else:
            assert row["verified_counts"] == [32, 1024, 4096]


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


@pytest.mark.parametrize("schedule_timesteps, scheduled_updates", [(None, 16), (256, 64)])
def test_optimizer_schedule_is_identical_at_different_horizons(
    monkeypatch, schedule_timesteps, scheduled_updates
):
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
                lr_schedule_timesteps=schedule_timesteps,
            )
        )
        assert rates == pytest.approx([0.003 * (1 - i / scheduled_updates) for i in range(16)])
