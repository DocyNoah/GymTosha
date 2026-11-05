import json

from gymtosha.benchmarks.cli import main


def test_benchmark_cli_writes_measurements_and_plots(tmp_path):
    main(
        [
            "train",
            "--env-ids",
            "CartPole-v1",
            "--algorithms",
            "ppo",
            "--backends",
            "gymtosha",
            "gymnasium-sync",
            "--num-envs",
            "4",
            "--total-timesteps",
            "128",
            "--rollout-steps",
            "16",
            "--seeds",
            "1",
            "--output",
            str(tmp_path),
        ]
    )
    records = json.loads((tmp_path / "training.json").read_text())
    assert len(records) == 2
    assert records[0]["gradient_updates"] == records[1]["gradient_updates"]
    assert (tmp_path / "training_speed.png").stat().st_size > 1000
    assert (tmp_path / "learning_curves.png").stat().st_size > 1000
    assert (tmp_path / "training.csv").exists()


def test_throughput_cli(tmp_path):
    main(
        [
            "throughput",
            "--env-ids",
            "FrozenLake-v1",
            "--backends",
            "gymtosha",
            "--batch-sizes",
            "4",
            "--steps",
            "10",
            "--repeats",
            "2",
            "--output",
            str(tmp_path),
        ]
    )
    records = json.loads((tmp_path / "throughput.json").read_text())
    assert len(records) == 2
    assert all(record["steps_per_second"] > 0 for record in records)
    assert (tmp_path / "environment_throughput.png").exists()


def test_plots_reject_mismatched_training_settings(tmp_path):
    from dataclasses import asdict

    import pytest

    from gymtosha.benchmarks.plot import plot_training
    from gymtosha.benchmarks.train import TrainConfig

    first = {"config": asdict(TrainConfig()), "steps_per_second": 100, "checkpoints": []}
    second = {
        "config": {**first["config"], "num_envs": 128},
        "steps_per_second": 200,
        "checkpoints": [],
    }
    with pytest.raises(ValueError, match="incompatible"):
        plot_training([first, second], tmp_path)


def test_default_training_uses_one_representative_algorithm(tmp_path):
    main(
        [
            "train",
            "--env-ids",
            "CartPole-v1",
            "--num-envs",
            "4",
            "--total-timesteps",
            "128",
            "--rollout-steps",
            "16",
            "--seeds",
            "1",
            "--output",
            str(tmp_path),
        ]
    )
    records = json.loads((tmp_path / "training.json").read_text())
    assert len(records) == 2
    assert {r["config"]["algorithm"] for r in records} == {"ppo"}
