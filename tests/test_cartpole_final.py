import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gymtosha.benchmarks.study import study_evaluation
from gymtosha.benchmarks.study_grouped import load_baseline
from gymtosha.benchmarks.train import TrainConfig

BASELINE = (
    Path(__file__).resolve().parents[1] / "benchmarks/results/cartpole-study/scaling/selected.json"
)
RUNTIME = {"node": "test-host", "platform": "test-platform", "torch": "2.15", "gymnasium": "1.4"}


def legacy_trial():
    config = TrainConfig(
        num_envs=32,
        rollout_steps=64,
        update_epochs=4,
        num_minibatches=4,
        ppo_minibatch_size=512,
        total_timesteps=4096,
        seed=1001,
    )
    evaluation = asdict(study_evaluation(config, comparison=True, each_update=True))
    record = {
        "request": {"config": asdict(config), "evaluation": evaluation, "runtime": RUNTIME},
        "status": "completed",
        "seed": 1001,
        "timing_scope": "isolated",
        "result": {
            "config": asdict(config),
            "timesteps": 4096,
            "valid_transitions": 4096,
            "gradient_updates": 32,
            "wall_seconds": 2.1,
            "setup_seconds": 0.1,
            "warmup_seconds": 0.2,
            "phase_seconds": {
                "learner_setup_seconds": 0.3,
                "collection_seconds": 0.7,
                "update_seconds": 0.9,
            },
            "completed_episodes": 7,
            "checkpoints": [{"timesteps": 2048, "episodes": 3}, {"timesteps": 4096, "episodes": 4}],
            "evaluation": {
                "config": evaluation,
                "env_id": "CartPole-v1",
                "transitions_to_target": 7,
                "checkpoints": [
                    {
                        "timesteps": 2048,
                        "valid_transitions": 2048,
                        "gradient_updates": 16,
                        "wall_seconds": 1.0,
                        "episode_returns": [460.0] * 100,
                        "mean_return": 500.0,
                    },
                    {
                        "timesteps": 4096,
                        "valid_transitions": 4096,
                        "gradient_updates": 32,
                        "wall_seconds": 2.0,
                        "episode_returns": [500.0] * 100,
                    },
                ],
            },
        },
    }
    return config, record


def test_legacy_uses_raw_evaluation_and_marks_unrecorded_diagnostics():
    from gymtosha.benchmarks.cartpole_final import legacy_points

    config, record = legacy_trial()
    points, missing = legacy_points(record, config, RUNTIME)
    assert points[0] == {
        "timesteps": 2048,
        "valid_transitions": 2048,
        "gradient_updates": 16,
        "wall_seconds": 1.0,
        "episode_returns": [460.0] * 100,
        "ppo_update": 1,
        "completed_episodes": 3,
        "setup_seconds": 0.1,
        "warmup_seconds": 0.2,
        "learner_setup_seconds": 0.3,
    }
    assert points[1]["completed_episodes"] == 7
    assert points[1]["ppo_update"] == 2
    assert points[1]["collection_seconds"] == 0.7
    assert points[1]["update_seconds"] == 0.9
    assert set(missing) == {
        "collection_seconds",
        "update_seconds",
        "learning_rate",
        "entropy",
        "policy_loss",
        "value_loss",
        "approx_kl",
        "clip_fraction",
    }


@pytest.mark.parametrize("violation", ["runtime", "scope", "protocol", "coverage", "telemetry"])
def test_legacy_excludes_incompatible_or_incomplete_trials(violation):
    from gymtosha.benchmarks.cartpole_final import legacy_points

    config, record = legacy_trial()
    match violation:
        case "runtime":
            record["request"]["runtime"] = {**RUNTIME, "node": "another-host"}
        case "scope":
            record["timing_scope"] = "shared_search"
        case "protocol":
            record["request"]["evaluation"] = {"interval": 4096}
        case "coverage":
            record["result"]["evaluation"]["checkpoints"].pop(0)
        case "telemetry":
            record["result"]["evaluation"]["checkpoints"][0]["episode_returns"][0] = float("nan")
    assert legacy_points(record, config, RUNTIME) is None


def test_capacity_matrix_reuses_only_exact_host_probes_and_includes_failures(tmp_path, monkeypatch):
    from gymtosha.benchmarks import cartpole_final as final
    from gymtosha.benchmarks import study_grouped

    output = tmp_path / "final"
    legacy = tmp_path / "legacy"
    capacity = legacy / "capacity-selected"
    capacity.mkdir(parents=True)
    (legacy / "protocol.json").write_text(json.dumps({"runtime": RUNTIME}))
    base, _ = load_baseline(BASELINE)
    selected = final.final_presets(base)
    assert len(selected) == 7
    assert {row["config"]["total_timesteps"] for row in selected.values()} == {8_388_608}
    config = TrainConfig(**selected["32"]["config"])
    assert (config.update_epochs, config.ppo_minibatch_size, config.rollout_steps) == (4, 512, 64)
    matching = replace(config, backend="gymnasium-native", device="cpu", compile_env=False)
    saved = {
        "config": asdict(replace(matching, total_timesteps=2048)),
        "status": "passed",
        "observed_seconds": 1.0,
        "limit_seconds": 60,
    }
    (capacity / "gymnasium-native-cpu-0-32.json").write_text(json.dumps(saved))
    saved["config"].update(backend="gymnasium-sync", rollout_steps=256)
    (capacity / "gymnasium-sync-cpu-0-32.json").write_text(json.dumps(saved))
    saved["config"].update(backend="gymnasium-async", rollout_steps=64)
    saved["observed_seconds"] = 60.5
    (capacity / "gymnasium-async-cpu-0-32.json").write_text(json.dumps(saved))

    def isolate(args, count):
        assert (args.backend, count) != ("gymnasium-native", 32)
        assert count <= 1024
        return {
            "status": "failed",
            "error_type": "test_capacity",
            "observed_seconds": 1.0,
            "limit_seconds": 60,
        }

    monkeypatch.setattr(study_grouped, "_isolated_probe", isolate)
    matrix = final.verify_matrix(selected, output, ["cpu"], legacy, RUNTIME)
    assert matrix["seeds"] == [1001, 1002, 1003, 1004, 1005]
    assert matrix["budget"] == 8_388_608 and matrix["horizon"] == 64
    rows = {row["id"]: row for row in matrix["conditions"]}
    assert len(rows) == 17
    assert rows["gymnasium-native-cpu-0-32"]["status"] == "passed"
    assert rows["gymnasium-native-cpu-0-32"]["capacity_source"] == "legacy_json"
    assert rows["gymnasium-sync-cpu-0-32"]["status"] == "failed"
    assert rows["gymnasium-sync-cpu-0-32"]["capacity_source"] == "live"
    assert rows["gymnasium-async-cpu-0-32"]["status"] == "failed"
    assert final.verify_matrix(selected, output, ["cpu"], legacy, RUNTIME) == matrix


def test_controller_stops_on_worker_failure_and_reuses_completed_seed(tmp_path, monkeypatch):
    from gymtosha.benchmarks import cartpole_final as final
    from gymtosha.benchmarks import study_grouped

    calls = []
    probe_calls = []
    condition = "gymnasium-async-cpu-0-32"

    def isolate(args, count):
        probe_calls.append((args.backend, count))
        return {
            "status": "passed" if args.backend == "gymnasium-async" else "failed",
            "observed_seconds": 1.0,
            "limit_seconds": 60,
        }

    def worker(config, runtime, version, path, root, name):
        assert (root / "matrix.json").is_file()
        assert name == condition
        calls.append(config.seed)
        if calls == [1001, 1002]:
            raise RuntimeError("test worker failure")
        path.mkdir(parents=True)
        final._write(
            {
                **final._run_metadata(config, runtime, source="live", code_version=version),
                "status": "completed",
            },
            path / "metadata.json",
        )
        (path / "events.out.tfevents.test").write_bytes(b"test stream")

    monkeypatch.setattr(study_grouped, "_isolated_probe", isolate)
    monkeypatch.setattr(final, "runtime_fingerprint", lambda: RUNTIME)
    monkeypatch.setattr(final, "_run_worker", worker)
    with pytest.raises(RuntimeError, match="test worker failure"):
        final.run_final(BASELINE, tmp_path, ["cpu"], None)
    assert calls == [1001, 1002]
    final.run_final(BASELINE, tmp_path, ["cpu"], None)
    assert calls == [1001, 1002, 1002, 1003, 1004, 1005]
    assert len(probe_calls) == 17


def test_execution_order_finishes_accelerator_variants_before_cpu():
    from gymtosha.benchmarks import cartpole_final as final

    priority = [
        ("gymtosha", "cuda", True),
        ("gymtosha", "cuda", False),
        ("gymtosha", "mps", True),
        ("gymtosha", "mps", False),
        ("gymtosha", "cpu", True),
        ("gymtosha", "cpu", False),
        ("gymnasium-native", "cpu", False),
        ("gymnasium-sync", "cpu", False),
        ("gymnasium-async", "cpu", False),
    ]
    rows = [
        {"backend": backend, "device": device, "compile_env": compiled, "num_envs": count}
        for backend, device, compiled in reversed(priority)
        for count in (256, 32)
    ]
    order = sorted(rows, key=final._execution_key)
    assert [
        (row["backend"], row["device"], row["compile_env"], row["num_envs"]) for row in order
    ] == [(*variant, count) for variant in priority for count in (32, 256)]
