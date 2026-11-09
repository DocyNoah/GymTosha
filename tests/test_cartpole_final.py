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


def test_final_worker_uses_the_declared_replacement_seed(tmp_path, monkeypatch):
    from gymtosha.benchmarks import cartpole_final as final

    root = tmp_path / "final"
    root.mkdir()
    condition = "gymtosha-cuda-1-256"
    (root / "seed-replacements.json").write_text(json.dumps({condition: {"1006": 1021}}))
    monkeypatch.setattr(final, "runtime_fingerprint", lambda: RUNTIME)
    monkeypatch.setattr(
        final,
        "verify_matrix",
        lambda *args, seeds, **kwargs: {
            "seeds": seeds,
            "conditions": [
                {
                    "id": condition,
                    "backend": "gymtosha",
                    "device": "cuda",
                    "compile_env": True,
                    "num_envs": 256,
                    "status": "passed",
                }
            ],
        },
    )
    workers = []
    monkeypatch.setattr(
        final,
        "_run_worker",
        lambda config, runtime, version, path, *args: workers.append(
            (config.seed, path.relative_to(root).as_posix())
        ),
    )
    final.run_final(BASELINE, root, ["cuda"], None, seed_start=1006)
    assert workers[0] == (1021, f"replacements/{condition}/seed-1021")
    assert [seed for seed, _ in workers] == [1021, 1007, 1008, 1009, 1010]


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


def test_legacy_crops_raw_returns_without_using_full_run_phase_totals():
    from gymtosha.benchmarks.cartpole_final import legacy_points

    config, record = legacy_trial()
    for saved in (record["request"]["config"], record["result"]["config"]):
        saved.pop("lr_schedule_timesteps", None)
    short = replace(config, total_timesteps=2048, lr_schedule_timesteps=4096)
    points, missing = legacy_points(record, short, RUNTIME)
    assert len(points) == 1
    assert points[0]["episode_returns"] == [460.0] * 100
    assert points[0]["completed_episodes"] == 3
    assert {"collection_seconds", "update_seconds"} <= set(missing)
    assert {"collection_seconds", "update_seconds"}.isdisjoint(points[0])


@pytest.mark.parametrize(
    "violation", [None, "coverage", "config", "counter", "returns", "diagnostics", "lr"]
)
def test_running_tensorboard_stream_requires_a_valid_prefix(tmp_path, violation):
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
    from tensorboard.util.tensor_util import make_ndarray

    from gymtosha.benchmarks import cartpole_final as final
    from gymtosha.benchmarks.cartpole_events import EventRun

    base, _ = load_baseline(BASELINE)
    config = TrainConfig(**final.final_presets(base)["32"]["config"])
    short = replace(config, seed=1001, total_timesteps=4096)
    original = replace(short, total_timesteps=8_388_608, lr_schedule_timesteps=None)
    metadata = final._run_metadata(original, RUNTIME, source="live", code_version="original")
    del metadata["config"]["lr_schedule_timesteps"]
    if violation == "config":
        metadata["config"]["gamma"] = 0.5
    old = tmp_path / "old" / "events" / "condition" / "seed-1001"
    output = tmp_path / "new" / "events" / "condition" / "seed-1001"
    with EventRun(old, metadata) as run:
        for step in (1,) if violation == "coverage" else (1, 2, 3):
            point = {
                "ppo_update": step,
                "valid_transitions": step * 2048,
                "gradient_updates": step * 16,
                "episode_returns": [400 + step] * 100,
                "wall_seconds": step * 1.25,
                "completed_episodes": step * 2,
                "setup_seconds": 0.1,
                "warmup_seconds": 0.2,
                "learner_setup_seconds": 0.3,
                "collection_seconds": step * 0.4,
                "update_seconds": step * 0.6,
                "learning_rate": 0.003 * (1 - (step * 16 - 1) / 65_536),
                "entropy": 0.3,
                "policy_loss": -0.1,
                "value_loss": 0.4,
                "approx_kl": 0.02,
                "clip_fraction": 0.01,
            }
            if step == 2:
                match violation:
                    case "counter":
                        point["gradient_updates"] = 31
                    case "returns":
                        point["episode_returns"][0] = 501
                    case "diagnostics":
                        del point["policy_loss"]
                    case "lr":
                        point["learning_rate"] = 0.001
            run.record(point)
        run.writer.flush()
        path = old / "metadata.json"
        stream = next(old.glob("events.out.tfevents.*"))
        archived = stream.read_bytes(), path.read_bytes()
        assert json.loads(archived[1])["status"] == "running"
        assert final._migrate_prefix(old, short, RUNTIME, output) is (violation is None)
        assert (stream.read_bytes(), path.read_bytes()) == archived

    if violation is not None:
        assert not output.exists()
        return

    migrated = json.loads((output / "metadata.json").read_text())
    assert migrated["status"] == "completed"
    assert migrated["source"] == "live"
    assert migrated["config"]["total_timesteps"] == 4096
    assert migrated["config"]["lr_schedule_timesteps"] == 8_388_608
    assert migrated["reused_prefix"] == {
        "path": str(old.resolve()),
        "budget": 8_388_608,
        "code_version": "original",
    }
    events = LegacyEventFileLoader(str(next(output.glob("events.out.tfevents.*")))).Load()
    returns = {
        event.step: make_ndarray(value.tensor).tolist()
        for event in events
        for value in event.summary.value
        if value.tag == "eval/episode_returns"
    }
    assert returns == {1: [401.0] * 100, 2: [402.0] * 100}


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
    assert set(selected) == {"32", "256", "512", "1024"}
    assert {row["config"]["total_timesteps"] for row in selected.values()} == {2_097_152}
    assert {row["config"]["lr_schedule_timesteps"] for row in selected.values()} == {8_388_608}
    config = TrainConfig(**selected["32"]["config"])
    assert (config.update_epochs, config.ppo_minibatch_size, config.rollout_steps) == (4, 512, 64)
    matching = replace(config, backend="gymnasium-native", device="cpu", compile_env=False)
    saved = {
        "config": asdict(replace(matching, total_timesteps=2048, lr_schedule_timesteps=None)),
        "status": "passed",
        "observed_seconds": 1.0,
        "limit_seconds": 60,
    }
    del saved["config"]["lr_schedule_timesteps"]
    prefix = tmp_path / "prefix"
    (prefix / "capacity-selected").mkdir(parents=True)
    (prefix / "protocol.json").write_text(json.dumps({"runtime": RUNTIME}))
    (prefix / "capacity-selected/gymnasium-native-cpu-0-32.json").write_text(json.dumps(saved))
    saved["status"] = "failed"
    (capacity / "gymnasium-native-cpu-0-32.json").write_text(json.dumps(saved))
    saved["status"] = "passed"
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
    matrix = final.verify_matrix(selected, output, ["cpu"], legacy, RUNTIME, prefix_from=prefix)
    assert matrix["seeds"] == [1001, 1002, 1003, 1004, 1005]
    assert matrix["budget"] == 2_097_152 and matrix["horizon"] == 64
    rows = {row["id"]: row for row in matrix["conditions"]}
    assert len(rows) == 17
    assert rows["gymnasium-native-cpu-0-32"]["status"] == "passed"
    assert rows["gymnasium-native-cpu-0-32"]["capacity_source"] == "prefix_tensorboard"
    assert rows["gymnasium-sync-cpu-0-32"]["status"] == "failed"
    assert rows["gymnasium-sync-cpu-0-32"]["capacity_source"] == "live"
    assert rows["gymnasium-async-cpu-0-32"]["status"] == "failed"
    assert final.verify_matrix(selected, output, ["cpu"], legacy, RUNTIME) == matrix


@pytest.mark.parametrize("first_seed", [1001, 1006])
def test_controller_stops_on_worker_failure_and_reuses_completed_seed(
    tmp_path, monkeypatch, first_seed
):
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
        if calls == [first_seed, first_seed + 1]:
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
        final.run_final(BASELINE, tmp_path, ["cpu"], None, seed_start=first_seed)
    assert calls == [first_seed, first_seed + 1]
    final.run_final(BASELINE, tmp_path, ["cpu"], None, seed_start=first_seed)
    assert calls == [first_seed, first_seed + 1, *range(first_seed + 1, first_seed + 5)]
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


def test_existing_matrix_can_narrow_without_retesting_capacity(tmp_path, monkeypatch):
    from gymtosha.benchmarks import cartpole_final as final
    from gymtosha.benchmarks import study_grouped

    probes = []

    def isolate(args, count):
        probes.append(count)
        return {"status": "passed", "observed_seconds": 1.0, "limit_seconds": 60}

    monkeypatch.setattr(study_grouped, "_isolated_probe", isolate)
    base, _ = load_baseline(BASELINE)
    original_budget = final.Budget((32, 256, 512, 1024, 4096), (64,), final.BUDGET, 512)
    original = final.expected_presets(base, original_budget, 64, each_update=True)
    old_matrix = final.verify_matrix(original, tmp_path, ["mps"], None, RUNTIME)
    assert len(probes) == 10
    selected = final.final_presets(base)
    matrix = final.verify_matrix(selected, tmp_path, ["mps"], None, RUNTIME)
    assert len(probes) == 10
    assert len(matrix["conditions"]) == 8
    assert max(row["num_envs"] for row in matrix["conditions"]) == 1024
    assert json.loads((tmp_path / "matrix-before-env-cap.json").read_text()) == old_matrix
    assert (tmp_path / "capacity-selected/gymtosha-mps-1-4096.json").exists()
