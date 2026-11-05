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


def trials(count=20):
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
                "evaluation": {"status": "budget_exhausted", "time_to_target_seconds": None},
            },
        }
        for seed in SEEDS[:count]
    ]


def test_summary_requires_twenty_seeds_and_reports_sample_standard_deviation():
    result = summarize_trials(trials())
    assert result["complete"] and result["successful_trials"] == 20
    assert result["mean_training_return"] == 10.5
    assert result["std_training_return"] == pytest.approx(5.9160797831)
    assert not summarize_trials(trials(19))["complete"]
    assert summarize_trials(trials(19))["mean_training_return"] is None


def test_summary_does_not_drop_a_failed_trial():
    records = trials()
    records[-1] = {"status": "failed", "seed": 20, "error": "out of memory"}
    result = summarize_trials(records)
    assert result["complete"]
    assert result["successful_trials"] == 19
    assert result["mean_training_return"] is None


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


def test_parameter_dashboard_contains_real_twenty_seed_statistics(tmp_path):
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
                "seed_rewards": list(range(1, 21)),
            }
        ],
    }
    directory = tmp_path / "search" / "32"
    directory.mkdir(parents=True)
    (directory / "learning_rate.json").write_text(json.dumps(phase))
    plot_parameter(phase, directory / "learning_rate.png")
    render_dashboard(tmp_path)
    assert (directory / "learning_rate.png").stat().st_size > 1000
    text = (tmp_path / "index.html").read_text()
    assert "__DATA__" not in text
    assert "10.5" in text
