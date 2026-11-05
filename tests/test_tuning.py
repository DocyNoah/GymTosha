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
