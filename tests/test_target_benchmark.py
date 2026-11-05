from dataclasses import asdict

import pytest

from gymtosha.benchmarks.cli import main
from gymtosha.benchmarks.evaluation import EvaluationConfig
from gymtosha.benchmarks.target import target_preset, target_summary, validate_target_comparisons


def record(backend, reached, seed=1):
    config = target_preset("CartPole-v1", backend=backend, device="cpu", seed=seed)
    return {
        "config": asdict(config),
        "timesteps": 64,
        "wall_seconds": 3.0,
        "evaluation": {
            "config": asdict(EvaluationConfig()),
            "target": 475,
            "status": "reached" if reached else "budget_exhausted",
            "time_to_target_seconds": 2.0 if reached else None,
            "transitions_to_target": 64 if reached else None,
            "checkpoints": [{"mean_return": 475, "wall_seconds": 3, "timesteps": 64}],
        },
    }


def test_incomplete_comparison_does_not_claim_speedup():
    records = [record("gymtosha", True), record("gymnasium-sync", False)]
    summary = target_summary(records)
    assert summary[0]["speedup"] is None
    assert summary[0]["gymnasium_reached"] == 0
    assert summary[0]["gymtosha_reached"] == 1


def test_target_comparisons_validate_evaluation_protocol():
    first, second = record("gymtosha", True), record("gymnasium-sync", True)
    second["evaluation"]["config"]["episodes"] = 7
    with pytest.raises(ValueError, match="incompatible"):
        validate_target_comparisons([first, second])


def test_target_cli_creates_real_evaluation_results(tmp_path):
    main(
        [
            "time-to-target",
            "--env-ids",
            "CartPole-v1",
            "--seeds",
            "1",
            "--device",
            "cpu",
            "--num-envs",
            "4",
            "--total-timesteps",
            "128",
            "--rollout-steps",
            "16",
            "--eval-interval",
            "64",
            "--eval-episodes",
            "4",
            "--target",
            "-1",
            "--output",
            str(tmp_path),
        ]
    )
    import json

    records = json.loads((tmp_path / "time_to_target.json").read_text())
    assert len(records) == 2
    assert all(r["evaluation"]["status"] == "reached" for r in records)
    assert all(r["timesteps"] == 64 for r in records)
    assert (tmp_path / "time_to_target.png").exists()
    assert (tmp_path / "evaluation_curves.png").exists()
    assert (tmp_path / "time_to_target.csv").exists()


def test_resume_rejects_changed_protocol_without_overwriting(tmp_path):
    import json

    args = [
        "time-to-target",
        "--env-ids",
        "CartPole-v1",
        "--seeds",
        "1",
        "--device",
        "cpu",
        "--num-envs",
        "4",
        "--total-timesteps",
        "128",
        "--rollout-steps",
        "16",
        "--eval-interval",
        "64",
        "--eval-episodes",
        "4",
        "--target",
        "-1",
        "--output",
        str(tmp_path),
    ]
    main(args)
    before = json.loads((tmp_path / "time_to_target.json").read_text())
    changed = args.copy()
    changed[changed.index("--target") + 1] = "-2"
    with pytest.raises(ValueError, match="resume"):
        main([*changed, "--resume"])
    assert json.loads((tmp_path / "time_to_target.json").read_text()) == before


def test_same_backend_cannot_mix_compilation_modes():
    first, second = record("gymtosha", True, 1), record("gymtosha", True, 2)
    second["config"]["compile_env"] = False
    with pytest.raises(ValueError, match="compilation"):
        validate_target_comparisons([first, second])


def test_new_seed_resume_validates_protocol_before_writing(tmp_path):
    import json

    args = [
        "time-to-target",
        "--env-ids",
        "CartPole-v1",
        "--seeds",
        "1",
        "--device",
        "cpu",
        "--num-envs",
        "4",
        "--total-timesteps",
        "128",
        "--rollout-steps",
        "16",
        "--eval-interval",
        "64",
        "--eval-episodes",
        "4",
        "--target",
        "-1",
        "--output",
        str(tmp_path),
    ]
    main(args)
    before = json.loads((tmp_path / "time_to_target.json").read_text())
    changed = args.copy()
    changed[changed.index("--target") + 1] = "-2"
    changed[changed.index("--seeds") + 1] = "2"
    with pytest.raises(ValueError, match="incompatible"):
        main([*changed, "--resume"])
    assert json.loads((tmp_path / "time_to_target.json").read_text()) == before


def test_target_cli_uses_selected_hyperparameters(tmp_path):
    import json
    from dataclasses import asdict, replace

    from gymtosha.benchmarks.target import target_preset

    config = replace(
        target_preset("CartPole-v1"),
        num_envs=4,
        total_timesteps=128,
        rollout_steps=16,
        update_epochs=1,
        learning_rate=0.0007,
    )
    presets = tmp_path / "selected.json"
    presets.write_text(json.dumps({"CartPole-v1": {"config": asdict(config)}}))
    main(
        [
            "time-to-target",
            "--env-ids",
            "CartPole-v1",
            "--device",
            "cpu",
            "--seeds",
            "11",
            "--presets",
            str(presets),
            "--target",
            "-1",
            "--eval-interval",
            "64",
            "--eval-episodes",
            "4",
            "--output",
            str(tmp_path / "run"),
        ]
    )
    records = json.loads((tmp_path / "run/time_to_target.json").read_text())
    assert all(r["config"]["learning_rate"] == 0.0007 for r in records)
    assert all(r["config"]["seed"] == 11 for r in records)


def test_alternative_algorithm_gets_its_own_valid_preset(tmp_path):
    import json

    main(
        [
            "time-to-target",
            "--env-ids",
            "MountainCarContinuous-v0",
            "--algorithms",
            "sac",
            "--device",
            "cpu",
            "--seeds",
            "11",
            "--total-timesteps",
            "16384",
            "--num-envs",
            "4",
            "--eval-interval",
            "64",
            "--eval-episodes",
            "2",
            "--target",
            "-1000",
            "--output",
            str(tmp_path),
        ]
    )
    records = json.loads((tmp_path / "time_to_target.json").read_text())
    assert all(not r["config"]["normalize_observations"] for r in records)


def test_zero_evaluation_episodes_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="counts"):
        main(
            [
                "time-to-target",
                "--env-ids",
                "CartPole-v1",
                "--device",
                "cpu",
                "--seeds",
                "11",
                "--num-envs",
                "4",
                "--rollout-steps",
                "16",
                "--total-timesteps",
                "128",
                "--eval-interval",
                "64",
                "--eval-episodes",
                "0",
                "--target",
                "-1",
                "--output",
                str(tmp_path),
            ]
        )


def test_target_plot_preserves_flat_csv_and_does_not_create_empty_results(tmp_path):
    import csv

    source = tmp_path / "source"
    output = tmp_path / "plot"
    main(
        [
            "time-to-target",
            "--env-ids",
            "CartPole-v1",
            "--seeds",
            "1",
            "--device",
            "cpu",
            "--num-envs",
            "4",
            "--total-timesteps",
            "128",
            "--rollout-steps",
            "16",
            "--eval-interval",
            "64",
            "--eval-episodes",
            "4",
            "--target",
            "-1",
            "--output",
            str(source),
        ]
    )
    main(["plot", "--inputs", str(source), "--output", str(output)])
    with (output / "time_to_target.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows and "status" in rows[0] and "evaluation" not in rows[0]
    assert (output / "target_summary.json").exists()
    assert not (output / "training.json").exists()
    assert not (output / "throughput.json").exists()


def test_independent_tuning_allows_backend_configs_but_preserves_evaluation():
    first, second = record("gymtosha", True), record("gymnasium-sync", True)
    first["config"]["num_envs"] = 4096
    first["evaluation"]["config"]["interval"] = 262144
    for row in (first, second):
        row["comparison_mode"] = "independently-tuned"
    assert target_summary([first, second])[0]["speedup"] == 1
    second["evaluation"]["config"]["episodes"] = 7
    with pytest.raises(ValueError, match="incompatible"):
        target_summary([first, second])


def test_independent_tuning_rejects_changed_settings_within_backend():
    first, second = record("gymtosha", True, 1), record("gymtosha", True, 2)
    for row in (first, second):
        row["comparison_mode"] = "independently-tuned"
    second["config"]["num_envs"] = 8192
    with pytest.raises(ValueError, match="incompatible"):
        validate_target_comparisons([first, second])


def test_target_cli_loads_backend_specific_presets(tmp_path):
    import json
    from dataclasses import replace

    configs = {
        backend: {
            "config": asdict(
                replace(
                    target_preset("CartPole-v1", backend=backend),
                    num_envs=count,
                    total_timesteps=128,
                    rollout_steps=8,
                    update_epochs=1,
                )
            ),
            "evaluation_interval": count * 8,
        }
        for backend, count in (("gymnasium-sync", 4), ("gymtosha", 8))
    }
    presets = tmp_path / "selected.json"
    presets.write_text(json.dumps({"CartPole-v1": {"backends": configs}}))
    main(
        [
            "time-to-target",
            "--env-ids",
            "CartPole-v1",
            "--device",
            "cpu",
            "--seeds",
            "11",
            "--presets",
            str(presets),
            "--comparison-mode",
            "independently-tuned",
            "--target",
            "-1",
            "--eval-episodes",
            "4",
            "--output",
            str(tmp_path / "run"),
        ]
    )
    rows = json.loads((tmp_path / "run/time_to_target.json").read_text())
    assert [(r["config"]["backend"], r["config"]["num_envs"]) for r in rows] == [
        ("gymnasium-sync", 4),
        ("gymtosha", 8),
    ]
    assert [r["evaluation"]["config"]["interval"] for r in rows] == [32, 64]
    assert all(r["comparison_mode"] == "independently-tuned" for r in rows)


def test_zero_evaluation_interval_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="counts"):
        main(
            [
                "time-to-target",
                "--env-ids",
                "CartPole-v1",
                "--backends",
                "gymtosha",
                "--device",
                "cpu",
                "--num-envs",
                "4",
                "--rollout-steps",
                "16",
                "--total-timesteps",
                "128",
                "--eval-interval",
                "0",
                "--output",
                str(tmp_path),
            ]
        )


def test_flat_target_rows_preserve_comparison_protocol():
    from gymtosha.benchmarks.target import target_rows

    row = record("gymtosha", True)
    row["comparison_mode"] = "independently-tuned"
    result = target_rows([row])[0]
    assert result["comparison_mode"] == "independently-tuned"
    assert result["evaluation_interval"] == 65536


def test_comparisons_accept_legacy_default_network_settings():
    first, second = record("gymtosha", True), record("gymnasium-sync", True)
    first["config"].pop("hidden_sizes")
    first["config"].pop("ppo_minibatch_size")
    validate_target_comparisons([first, second])
