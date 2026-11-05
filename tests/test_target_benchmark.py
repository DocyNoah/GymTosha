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
