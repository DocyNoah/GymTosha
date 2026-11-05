import runpy
from pathlib import Path

from gymtosha.benchmarks.evaluation import EvaluationConfig

MODULE = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/tune_targets.py"))


def test_candidates_are_valid_and_one_algorithm_per_environment():
    for env in MODULE["ENVIRONMENTS"]:
        candidates = MODULE["candidates"](env, "cpu")
        assert len(candidates) == 4
        assert len({c.algorithm for c in candidates}) == 1
        for config in candidates:
            config.validate()
            EvaluationConfig().validate(config)


def test_tuning_prefers_confirmed_success_over_failure():
    def trial(status, seconds, score):
        return {
            "result": {
                "evaluation": {
                    "status": status,
                    "time_to_target_seconds": seconds,
                    "checkpoints": [{"mean_return": score}],
                }
            }
        }

    assert MODULE["rank"]([trial("reached", 10, 500)]) < MODULE["rank"](
        [trial("budget_exhausted", None, 470)]
    )
    assert MODULE["rank"]([trial("reached", 5, 500)]) < MODULE["rank"]([trial("reached", 10, 500)])


def test_resume_keeps_finalists_and_other_environment_selections(tmp_path, monkeypatch):
    import json
    import sys
    from dataclasses import asdict, replace

    configs = MODULE["candidates"]("CartPole-v1", "cpu")
    trials = []
    for index, seconds in enumerate([10, 20, 30, 40]):
        trials.append(
            {
                "env_id": "CartPole-v1",
                "candidate": index,
                "seed": 101,
                "result": {
                    "config": asdict(configs[index]),
                    "evaluation": {
                        "config": asdict(EvaluationConfig(seed=900000)),
                        "status": "reached",
                        "time_to_target_seconds": seconds,
                        "checkpoints": [{"mean_return": 500}],
                    },
                },
            }
        )
    for index, seconds in [(0, 100), (1, 20)]:
        trials.append(
            {
                "env_id": "CartPole-v1",
                "candidate": index,
                "seed": 202,
                "result": {
                    "config": asdict(replace(configs[index], seed=202)),
                    "evaluation": {
                        "config": asdict(EvaluationConfig(seed=900000)),
                        "status": "reached",
                        "time_to_target_seconds": seconds,
                        "checkpoints": [{"mean_return": 500}],
                    },
                },
            }
        )
    (tmp_path / "search.json").write_text(json.dumps(trials))
    (tmp_path / "selected.json").write_text(json.dumps({"MountainCar-v0": {"preserved": True}}))

    def no_training(*args, **kwargs):
        raise AssertionError("a complete search must not run new trials on resume")

    monkeypatch.setitem(MODULE["main"].__globals__, "train", no_training)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune",
            "--device",
            "cpu",
            "--env-ids",
            "CartPole-v1",
            "--output",
            str(tmp_path),
            "--resume",
        ],
    )
    MODULE["main"]()
    selected = json.loads((tmp_path / "selected.json").read_text())
    assert selected["MountainCar-v0"] == {"preserved": True}
    assert selected["CartPole-v1"]["candidate"] == 1
