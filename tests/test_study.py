from dataclasses import asdict

import pytest

from gymtosha.benchmarks.study import (
    SEEDS,
    candidate_values,
    change_parameter,
    reference_config,
    study_evaluation,
    summarize_trials,
)


def test_study_scales_transitions_and_minibatches_with_environment_count():
    first = reference_config(32)
    larger = reference_config(64, previous=first)
    assert first.hidden_sizes == larger.hidden_sizes == [256, 256, 256]
    assert larger.total_timesteps == first.total_timesteps * 2
    assert larger.ppo_minibatch_size == first.ppo_minibatch_size * 2
    assert first.total_timesteps // (32 * first.rollout_steps) == 64
    assert (
        first.total_timesteps // first.ppo_minibatch_size
        == larger.total_timesteps // larger.ppo_minibatch_size
    )


def test_study_keeps_update_count_when_rollout_is_changed():
    first = reference_config(32)
    changed = change_parameter(first, "rollout_steps", 256)
    assert changed.total_timesteps == 32 * 256 * 64
    assert changed.ppo_minibatch_size == first.ppo_minibatch_size
    changed.validate()


def test_study_preserves_stability_protocol_without_early_stopping():
    config = reference_config(32)
    evaluation = study_evaluation(config)
    assert evaluation.episodes == 100
    assert evaluation.confirmations == 5
    assert evaluation.min_episode_success_rate == 0.95
    assert not evaluation.stop_on_target
    evaluation.validate(config)


def test_larger_batches_search_incumbent_and_neighbors_first():
    config = reference_config(64)
    assert candidate_values("learning_rate", config, local=True) == [0.0001, 0.0003, 0.001]
    assert candidate_values("learning_rate", config, local=False) == [
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
    ]


def trials(count=None):
    config = reference_config(32)
    return [
        {
            "status": "completed",
            "seed": seed,
            "result": {
                "config": {**asdict(config), "seed": seed},
                "mean_training_episode_return": float(seed),
                "seconds": 2.0,
                "wall_seconds": 3.0,
                "evaluation": {
                    "status": "budget_exhausted",
                    "time_to_target_seconds": None,
                    "checkpoints": [{"mean_return": float(seed)} for _ in range(5)],
                },
            },
        }
        for seed in SEEDS[:count]
    ]


def test_summary_requires_ten_seeds_and_reports_sample_standard_deviation():
    result = summarize_trials(trials())
    assert result["complete"] and result["successful_trials"] == 10
    assert result["mean_training_return"] == 5.5
    assert result["std_training_return"] == pytest.approx(3.0276503541)
    assert not summarize_trials(trials(9))["complete"]
    assert summarize_trials(trials(9))["mean_training_return"] is None


def test_summary_does_not_drop_a_failed_trial():
    records = trials()
    records[-1] = {"status": "failed", "seed": 10, "error": "out of memory"}
    result = summarize_trials(records)
    assert result["complete"]
    assert result["successful_trials"] == 9
    assert result["mean_training_return"] is None


def test_selection_uses_last_five_evaluation_checkpoints():
    from gymtosha.benchmarks.study import selection_key

    first = trials()
    second = trials()
    for lower, higher in zip(first, second, strict=True):
        lower["result"]["mean_training_episode_return"] = 490
        lower["result"]["evaluation"]["checkpoints"] = [
            {"mean_return": score} for score in [500, 500, 425, 425, 425, 425, 425]
        ]
        higher["result"]["mean_training_episode_return"] = 300
        higher["result"]["evaluation"]["checkpoints"] = [
            {"mean_return": score} for score in [5, 5, 480, 490, 500, 490, 490]
        ]
    first, second = summarize_trials(first), summarize_trials(second)
    assert first["mean_final_eval_return"] == 425
    assert second["mean_final_eval_return"] == 490
    assert selection_key(second) < selection_key(first)


def test_summary_rejects_duplicate_seeds():
    records = trials()
    records[-1] = records[0]
    with pytest.raises(ValueError, match="duplicate"):
        summarize_trials(records)


def test_summary_rejects_mixed_training_configs():
    records = trials()
    records[-1]["result"]["config"]["gamma"] = 0.95
    with pytest.raises(ValueError, match="config"):
        summarize_trials(records)


def test_worker_runs_fixed_budget_and_trial_cache_is_reusable(tmp_path, monkeypatch):
    from dataclasses import replace

    from gymtosha.benchmarks.evaluation import EvaluationConfig
    from gymtosha.benchmarks.study import run_trial

    config = replace(
        reference_config(32),
        num_envs=4,
        device="cpu",
        compile_env=False,
        total_timesteps=256,
        rollout_steps=16,
        update_epochs=1,
        ppo_minibatch_size=16,
        hidden_sizes=[8, 8],
    )
    evaluation = EvaluationConfig(target=-1, interval=64, episodes=4, stop_on_target=False)
    first = run_trial(config, evaluation, tmp_path)
    assert first["status"] == "completed"
    assert first["result"]["timesteps"] == 256
    assert first["result"]["mean_training_episode_return"] > 0

    def no_process(*args, **kwargs):
        raise AssertionError("a cached trial should not launch a new process")

    monkeypatch.setattr("subprocess.Popen", no_process)
    assert run_trial(config, evaluation, tmp_path) == first


def test_parameter_dashboard_contains_real_ten_seed_statistics(tmp_path):
    import json

    from gymtosha.benchmarks.study_plot import plot_parameter, render_dashboard

    phase = {
        "num_envs": 32,
        "parameter": "learning_rate",
        "selected_value": 0.001,
        "candidates": [
            {
                "value": 0.001,
                "summary": summarize_trials(trials()),
                "trial_ids": [str(seed) for seed in SEEDS],
                "seed_scores": list(SEEDS),
            }
        ],
    }
    directory = tmp_path / "search" / "32"
    directory.mkdir(parents=True)
    (directory / "learning_rate.json").write_text(json.dumps(phase))
    trial_directory = tmp_path / "trials"
    trial_directory.mkdir()
    for record in trials():
        record["result"]["checkpoints"] = [
            {"timesteps": step, "episodes": 4, "mean_return": 100 + step} for step in range(1, 9)
        ]
        record["result"]["evaluation"]["checkpoints"] = [
            {"timesteps": 0, "mean_return": 20},
            {"timesteps": 4, "mean_return": 150},
            {"timesteps": 8, "mean_return": 400},
        ]
        (trial_directory / f"{record['seed']}.json").write_text(json.dumps(record))
    plot_parameter(phase, directory / "learning_rate.png")
    render_dashboard(tmp_path)
    assert (directory / "learning_rate.png").stat().st_size > 1000
    assert (directory / "learning_rate-learning.png").stat().st_size > 1000
    text = (tmp_path / "index.html").read_text()
    assert "__DATA__" not in text
    assert "5.5" in text


def test_failed_worker_exit_keeps_a_trial_id_for_resume(tmp_path, monkeypatch):
    from gymtosha.benchmarks.study import run_trial

    class ExitedWorker:
        pid = 999999999
        returncode = 1

        def wait(self):
            return 1

    from gymtosha.benchmarks.study import runtime_fingerprint

    runtime_fingerprint()
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: ExitedWorker())
    config = reference_config(32)
    record = run_trial(config, study_evaluation(config), tmp_path)
    assert record["status"] == "failed"
    assert record["trial_id"]
    assert run_trial(config, study_evaluation(config), tmp_path) == record


def test_tuning_cannot_reuse_selected_settings_from_another_device(tmp_path):
    import json
    from argparse import Namespace

    from gymtosha.benchmarks.study import runtime_fingerprint, source_fingerprint, tune

    config = reference_config(32)
    (tmp_path / "selected.json").write_text(
        json.dumps(
            {
                "32": {
                    "source": source_fingerprint(),
                    "runtime": runtime_fingerprint(),
                    "config": asdict(config),
                }
            }
        )
    )
    with pytest.raises(ValueError, match="device|runtime"):
        tune(Namespace(output=tmp_path, min_envs=32, max_envs=32, device="mps"))


def test_training_fingerprint_includes_cartpole_environment_source(monkeypatch):
    from pathlib import Path

    from gymtosha.benchmarks.study import source_fingerprint

    before = source_fingerprint()
    original = Path.read_bytes

    def changed_source(path):
        content = original(path)
        return content + b"\n# changed dynamics\n" if path.name == "cartpole.py" else content

    monkeypatch.setattr(Path, "read_bytes", changed_source)
    assert source_fingerprint() != before


def test_comparison_records_unsupported_capacity_without_stopping(tmp_path):
    import json
    from argparse import Namespace

    from gymtosha.benchmarks.study import compare

    presets = tmp_path / "presets.json"
    presets.write_text("{}")
    capacities = tmp_path / "capacity.json"
    capacities.write_text(
        json.dumps(
            [
                {
                    "backend": "gymnasium-async",
                    "device": "cuda",
                    "compile_env": False,
                    "maximum_verified": None,
                }
            ]
        )
    )
    compare(
        Namespace(presets=presets, capacity=capacities, devices=["cuda"], output=tmp_path / "run")
    )
    assert (tmp_path / "run/unsupported/gymnasium-async-cuda-0.json").exists()


def test_best_backend_uses_fastest_stably_solved_size():
    from gymtosha.benchmarks.study_plot import best_backend_conditions

    def condition(n, score, target_time):
        return {
            "config": {"backend": "gymtosha", "device": "cuda", "compile_env": True, "num_envs": n},
            "summary": {
                "complete": True,
                "successful_trials": 20,
                "mean_final_eval_return": score,
                "std_final_eval_return": 1.0,
                "mean_wall_seconds": 100.0,
                "mean_target_seconds": target_time,
                "std_target_seconds": 1.0 if target_time else None,
            },
        }

    result = best_backend_conditions(
        [condition(32, 450, 20), condition(64, 470, 15), condition(128, 490, None)]
    )
    assert result[0]["config"]["num_envs"] == 64
    assert result[0]["selection_basis"] == "stable_target_time"


def test_study_lock_rejects_another_controller_and_releases(tmp_path):
    from gymtosha.benchmarks.study import study_lock

    with study_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with study_lock(tmp_path):
                pass
    with study_lock(tmp_path):
        pass


def test_controller_refuses_hot_changed_training_source(tmp_path, monkeypatch):
    from gymtosha.benchmarks.study import run_trial

    monkeypatch.setattr("gymtosha.benchmarks.study.source_fingerprint", lambda: "changed")
    config = reference_config(32)
    with pytest.raises(RuntimeError, match="restart"):
        run_trial(config, study_evaluation(config), tmp_path)


def test_shared_search_reports_rewards_but_does_not_compare_wall_times():
    result = summarize_trials(trials(), timing_comparable=False)
    assert result["mean_training_return"] == 5.5
    assert result["mean_wall_seconds"] is None
    assert result["mean_target_seconds"] is None
    assert not result["timing_comparable"]


@pytest.mark.parametrize("host_gib,expected_peak", [(100, 2), (5, 1)])
def test_parallel_search_respects_memory_budget_and_ten_seeds(
    tmp_path, monkeypatch, host_gib, expected_peak
):
    import threading
    import time

    from gymtosha.benchmarks.study import collect_candidates

    state = {"active": 0, "peak": 0}
    lock = threading.Lock()

    def fake_trial(config, evaluation, output, **kwargs):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.01)
        with lock:
            state["active"] -= 1
        return {
            "status": "completed",
            "seed": config.seed,
            "trial_id": str(config.seed),
            "result": {
                "config": asdict(config),
                "mean_training_episode_return": float(config.seed),
                "wall_seconds": 1.0,
                "evaluation": {"time_to_target_seconds": None},
            },
        }

    monkeypatch.setattr(
        "gymtosha.benchmarks.capacity._available_memory", lambda: host_gib * 1024**3
    )
    monkeypatch.setattr("gymtosha.benchmarks.study.run_trial", fake_trial)
    monkeypatch.setattr("gymtosha.benchmarks.study.estimate_trial_bytes", lambda config: 1)
    results = collect_candidates(
        reference_config(32),
        "learning_rate",
        [0.0003, 0.001],
        tmp_path,
        parallelism=4,
        memory_budget=2,
    )
    assert state["peak"] == expected_peak
    assert all(row["summary"]["attempted_seeds"] == 10 for row in results)
    assert all(row["summary"]["mean_training_return"] == 5.5 for row in results)


def test_search_uses_ten_seeds_and_final_comparison_uses_twenty():
    from gymtosha.benchmarks.study import COMPARISON_SEEDS

    assert SEEDS == tuple(range(1, 11))
    assert COMPARISON_SEEDS == tuple(range(1001, 1021))


@pytest.mark.parametrize(
    "prior_seeds,prior_metric",
    [(range(1, 21), "final_five_evaluation_checkpoints"), (SEEDS, "whole_training_mean")],
)
def test_search_recomputes_phase_when_protocol_changes(
    tmp_path, monkeypatch, prior_seeds, prior_metric
):
    import json

    from gymtosha.benchmarks.study import runtime_fingerprint, source_fingerprint, tune_count

    config = reference_config(32)
    path = tmp_path / "search/32/learning_rate.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "starting_config": asdict(config),
                "source": source_fingerprint(),
                "runtime": runtime_fingerprint(),
                "selected_config": asdict(config),
                "seeds": list(prior_seeds),
                "score_metric": prior_metric,
            }
        )
    )
    calls = []

    def candidates(*args, **kwargs):
        calls.append(True)
        return [
            {
                "value": 0.001,
                "summary": summarize_trials(trials()),
                "trial_ids": [],
                "seed_scores": list(SEEDS),
            }
        ]

    monkeypatch.setattr("gymtosha.benchmarks.study.PARAMETERS", {"learning_rate": [0.001]})
    monkeypatch.setattr("gymtosha.benchmarks.study.collect_candidates", candidates)
    monkeypatch.setattr("gymtosha.benchmarks.study_plot.plot_parameter", lambda *args: None)
    monkeypatch.setattr("gymtosha.benchmarks.study_plot.render_dashboard", lambda *args: None)
    tune_count(config, tmp_path)
    assert calls == [True]
    result = json.loads(path.read_text())
    assert result["seeds"] == list(SEEDS)
    assert result["score_metric"] == "final_five_evaluation_checkpoints"


def test_tuning_rejects_earlier_selection_with_old_seed_protocol(tmp_path):
    import json
    from argparse import Namespace

    from gymtosha.benchmarks.study import source_fingerprint, tune

    config = reference_config(32)
    (tmp_path / "selected.json").write_text(
        json.dumps(
            {
                "32": {
                    "source": source_fingerprint(),
                    "config": asdict(config),
                    "seeds": list(range(1, 21)),
                },
            }
        )
    )
    with pytest.raises(ValueError, match="earlier selected study protocol"):
        tune(Namespace(min_envs=64, max_envs=64, device="cuda", output=tmp_path))


def test_comparison_exports_seed_metrics_and_evaluation_curves(tmp_path, monkeypatch):
    import csv
    import json
    from argparse import Namespace

    from gymtosha.benchmarks.study import COMPARISON_SEEDS, SCORE_METRIC, compare

    presets = tmp_path / "selected.json"
    presets.write_text(
        json.dumps(
            {
                "32": {
                    "config": asdict(reference_config(32)),
                    "score_metric": SCORE_METRIC,
                    "seeds": list(SEEDS),
                }
            }
        )
    )
    capacity = tmp_path / "capacity.json"
    capacity.write_text(
        json.dumps(
            [
                {
                    "backend": "gymtosha",
                    "device": "cuda",
                    "compile_env": False,
                    "maximum_verified": 32,
                }
            ]
        )
    )

    def fake_trial(config, evaluation, output):
        return {
            "status": "completed",
            "seed": config.seed,
            "trial_id": str(config.seed),
            "result": {
                "config": asdict(config),
                "mean_training_episode_return": 400,
                "wall_seconds": 60,
                "evaluation": {
                    "time_to_target_seconds": 50,
                    "transitions_to_target": 81920,
                    "checkpoints": [
                        {"timesteps": 0, "wall_seconds": 1, "mean_return": 20},
                        {"timesteps": 20480, "wall_seconds": 13, "mean_return": 250},
                        {"timesteps": 40960, "wall_seconds": 25, "mean_return": 350},
                        {"timesteps": 61440, "wall_seconds": 38, "mean_return": 450},
                        {"timesteps": 81920, "wall_seconds": 50, "mean_return": 490},
                    ],
                },
            },
        }

    monkeypatch.setattr("gymtosha.benchmarks.study.run_trial", fake_trial)
    output = tmp_path / "run"
    compare(Namespace(presets=presets, capacity=capacity, devices=["cuda"], output=output))
    with (output / "comparison-seeds.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["seed"]) for row in rows] == list(COMPARISON_SEEDS)
    assert {int(row["transitions_to_target"]) for row in rows} == {81920}
    assert {float(row["time_to_target_seconds"]) for row in rows} == {50.0}
    assert (output / "best-learning-curves.png").stat().st_size > 1000
