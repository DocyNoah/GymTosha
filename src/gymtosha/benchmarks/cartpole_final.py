"""Run the five-seed, fixed-horizon CartPole comparison on one host."""

import argparse
import hashlib
import json
import signal
import subprocess
import sys
from dataclasses import asdict, replace
from math import isfinite
from pathlib import Path

from gymtosha.benchmarks.capacity import terminate_probe
from gymtosha.benchmarks.cli import _write
from gymtosha.benchmarks.study import (
    runtime_fingerprint,
    source_fingerprint,
    study_evaluation,
    study_lock,
)
from gymtosha.benchmarks.study_fixed_budget import (
    CPU_LIMIT,
    Budget,
    expected_presets,
)
from gymtosha.benchmarks.study_grouped import conditions, load_baseline, verify_selected
from gymtosha.benchmarks.train import TrainConfig, train

HORIZON = 64
BUDGET = 8_388_608
ENV_COUNTS = (32, 256, 512, 1024)
CAPACITY_SECONDS = 60
SEEDS = (1001, 1002, 1003, 1004, 1005)
EXECUTION_ORDER = (
    ("gymtosha", "cuda", True),
    ("gymtosha", "cuda", False),
    ("gymtosha", "mps", True),
    ("gymtosha", "mps", False),
    ("gymtosha", "cpu", True),
    ("gymtosha", "cpu", False),
    ("gymnasium-native", "cpu", False),
    ("gymnasium-sync", "cpu", False),
    ("gymnasium-async", "cpu", False),
)
LEGACY_MISSING = (
    "collection_seconds",
    "update_seconds",
    "learning_rate",
    "entropy",
    "policy_loss",
    "value_loss",
    "approx_kl",
    "clip_fraction",
)


def final_presets(base):
    if base.update_epochs != 4 or base.num_threads != 1:
        raise ValueError("the final CartPole baseline requires four PPO epochs and one thread")
    final = Budget(ENV_COUNTS, (HORIZON,), BUDGET, 512)
    return expected_presets(base, final, HORIZON, each_update=True)


def _read_record(path):
    try:
        record = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        return {}
    return record if isinstance(record, dict) else {}


def _condition_config(preset, backend, device, compile_env):
    return replace(
        TrainConfig(**preset["config"]), backend=backend, device=device, compile_env=compile_env
    )


def _probe_configs(selected, devices):
    for backend, device, compile_env in conditions(devices):
        for value, preset in selected.items():
            count = int(value)
            if backend == "gymnasium-async" and count != 32:
                continue
            if device == "cpu" and count > CPU_LIMIT:
                continue
            condition = f"{backend}-{device}-{int(compile_env)}-{count}"
            config = _condition_config(preset, backend, device, compile_env)
            yield condition, replace(config, total_timesteps=count * config.rollout_steps)


def _valid_probe(record, config, runtime):
    elapsed = record.get("observed_seconds")
    status = record.get("status")
    return (
        record.get("config") == asdict(config)
        and record.get("runtime", runtime) == runtime
        and record.get("limit_seconds") == CAPACITY_SECONDS
        and _time(elapsed)
        and (status == "failed" or (status == "passed" and elapsed < CAPACITY_SECONDS))
    )


def verify_matrix(selected, output, devices, legacy, runtime):
    capacity = output / "capacity-selected"
    matrix_path = output / "matrix.json"
    published = matrix_path.exists()
    configs = list(_probe_configs(selected, devices))
    if not published:
        previous = _read_record(legacy / "protocol.json") if legacy else {}
        for name, config in configs:
            target = capacity / f"{name}.json"
            if target.exists():
                if _valid_probe(_read_record(target), config, runtime):
                    continue
                target.replace(target.with_suffix(".unverified.json"))
            if previous.get("runtime") == runtime:
                path = legacy / "capacity-selected" / target.name
                saved = _read_record(path)
                if _valid_probe(saved, config, runtime):
                    _write({**saved, "legacy_probe": str(path.resolve())}, target)
        verify_selected(selected, output, devices, cpu_limit=CPU_LIMIT)

    rows = []
    for name, config in configs:
        path = capacity / f"{name}.json"
        probe = _read_record(path)
        if not _valid_probe(probe, config, runtime):
            raise RuntimeError(f"missing capacity evidence for {name}")
        rows.append(
            {
                "id": name,
                "backend": config.backend,
                "device": config.device,
                "compile_env": config.compile_env,
                "num_envs": config.num_envs,
                "host": runtime["node"],
                "status": probe["status"],
                "capacity_probe": str(path.relative_to(output)),
                "capacity_source": "legacy_json" if "legacy_probe" in probe else "live",
                **{key: probe[key] for key in ("error_type", "error") if key in probe},
            }
        )
    matrix = {"conditions": rows, "seeds": list(SEEDS), "budget": BUDGET, "horizon": HORIZON}
    if published:
        previous = _read_record(matrix_path)
        if matrix == previous:
            return matrix
        narrowed = {
            **previous,
            "conditions": [
                row for row in previous.get("conditions", ()) if row["num_envs"] <= ENV_COUNTS[-1]
            ],
        }
        if narrowed != matrix:
            raise RuntimeError("final CartPole matrix differs from its capacity evidence")
        archive = output / "matrix-before-env-cap.json"
        if archive.exists() and _read_record(archive) != previous:
            raise RuntimeError("the archived CartPole matrix differs from the previous scope")
        _write(previous, archive)
    _write(matrix, matrix_path)
    return matrix


def _time(value):
    return type(value) in (int, float) and isfinite(value) and value >= 0


def _count(value, expected):
    return type(value) is int and value == expected


def _counters(point, timesteps, updates):
    return (
        _count(point.get("timesteps"), timesteps)
        and _count(point.get("valid_transitions"), timesteps)
        and _count(point.get("gradient_updates"), updates)
    )


def _training_episode_counts(result, rollout, num_points):
    points = result.get("checkpoints")
    if not isinstance(points, list) or len(points) != num_points:
        return []
    totals = []
    total = 0
    for index, point in enumerate(points, start=1):
        if not isinstance(point, dict) or not _count(point.get("timesteps"), index * rollout):
            return []
        episodes = point.get("episodes")
        if type(episodes) is not int or not 0 <= episodes <= rollout:
            return []
        total += episodes
        totals.append(total)
    return totals if _count(result.get("completed_episodes"), total) else []


def legacy_points(record, config, runtime):
    """Return validated raw evaluation telemetry from a completed legacy trial."""
    evaluation = study_evaluation(config, comparison=True, each_update=True)
    expected_config, expected_evaluation = asdict(config), asdict(evaluation)
    try:
        request, result = record["request"], record["result"]
        saved_evaluation = result["evaluation"]
        points = saved_evaluation["checkpoints"]
        if (
            record["status"] != "completed"
            or record["timing_scope"] != "isolated"
            or not _count(record["seed"], config.seed)
            or request["runtime"] != runtime
            or request["config"] != expected_config
            or result["config"] != expected_config
            or request["evaluation"] != expected_evaluation
            or saved_evaluation["config"] != expected_evaluation
            or saved_evaluation["env_id"] != config.env_id
        ):
            return None
        result_runtime = result.get("metadata", {})
        if any(key in runtime and runtime[key] != value for key, value in result_runtime.items()):
            return None
        rollout = evaluation.interval
        num_points = config.total_timesteps // rollout
        updates = config.num_minibatches * config.update_epochs
        if (
            not isinstance(points, list)
            or len(points) != num_points
            or not _counters(result, config.total_timesteps, num_points * updates)
            or not _time(result["wall_seconds"])
        ):
            return None
        missing = list(LEGACY_MISSING)
        phases = result.get("phase_seconds")
        if not isinstance(phases, dict):
            phases = {}
        startup = {}
        for name, value in (
            ("setup_seconds", result.get("setup_seconds")),
            ("warmup_seconds", result.get("warmup_seconds")),
            ("learner_setup_seconds", phases.get("learner_setup_seconds")),
        ):
            if _time(value):
                startup[name] = value
            else:
                missing.append(name)
        episodes = _training_episode_counts(result, rollout, num_points)
        if not episodes:
            missing.append("completed_episodes")
        migrated = []
        previous = 0
        for index, point in enumerate(points, start=1):
            elapsed, scores = point["wall_seconds"], point["episode_returns"]
            if (
                not _counters(point, index * rollout, index * updates)
                or not _time(elapsed)
                or not previous < elapsed <= result["wall_seconds"]
                or not isinstance(scores, list)
                or len(scores) != evaluation.episodes
                or not all(
                    _time(score) and 1 <= score <= 500 and score % 1 == 0 for score in scores
                )
            ):
                return None
            previous = elapsed
            migrated.append(
                {
                    **{
                        name: point[name]
                        for name in ("timesteps", "valid_transitions", "gradient_updates")
                    },
                    "wall_seconds": elapsed,
                    "episode_returns": scores,
                    "ppo_update": index,
                    **startup,
                    **({"completed_episodes": episodes[index - 1]} if episodes else {}),
                }
            )
        for name in ("collection_seconds", "update_seconds"):
            if _time(phases.get(name)):
                migrated[-1][name] = phases[name]
    except (AttributeError, KeyError, TypeError, OverflowError):
        return None
    return migrated, missing


def _run_metadata(config, runtime, *, source, code_version):
    return {
        "host": runtime["node"],
        "config": asdict(config),
        "evaluation": asdict(study_evaluation(config, comparison=True, each_update=True)),
        "runtime": runtime,
        "code_version": code_version,
        "source": source,
        "timing_scope": "isolated",
        "missing_fields": [],
    }


def _migrate_legacy(legacy, protocol, config, runtime, output):
    if protocol.get("runtime") != runtime or not protocol.get("source"):
        return False
    source = protocol["source"]
    metadata = _run_metadata(
        config, runtime, source="legacy_json", code_version=protocol.get("git_sha") or source
    )
    request = {
        "config": metadata["config"],
        "evaluation": metadata["evaluation"],
        "source": source,
        "runtime": runtime,
    }
    # The legacy controller hashes the request, including its own training fingerprint.
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
    path = legacy / "trials" / f"{key}.json"
    record = _read_record(path)
    if record.get("request") != request:
        return False
    trial = legacy_points(record, config, runtime)
    if trial is None:
        return False

    from gymtosha.benchmarks.cartpole_events import EventRun

    points, missing = trial
    metadata.update(
        legacy_record=str(path.resolve()), legacy_source_fingerprint=source, missing_fields=missing
    )
    with EventRun(output, metadata) as run:
        for point in points:
            run.record(point)
        run.finish()
    return True


def _completed(path, config, runtime):
    metadata = _read_record(path / "metadata.json")
    if metadata.get("status") != "completed":
        return False
    expected = _run_metadata(config, runtime, source="live", code_version=None)
    keys = ("host", "config", "evaluation", "runtime", "timing_scope")
    if (
        any(metadata.get(key) != expected[key] for key in keys)
        or metadata.get("source") not in ("live", "legacy_json")
        or not metadata.get("code_version")
        or not any(path.glob("events.out.tfevents.*"))
    ):
        raise RuntimeError(f"invalid completed CartPole event run at {path}")
    return True


def _worker(request_path, output):
    from gymtosha.benchmarks.cartpole_events import EventRun

    metadata = _read_record(request_path)
    config = TrainConfig(**metadata["config"])
    evaluation = study_evaluation(config, comparison=True, each_update=True)
    expected = _run_metadata(
        config, runtime_fingerprint(), source="live", code_version=source_fingerprint()
    )
    if metadata != expected:
        raise ValueError("training source or runtime changed while the controller was running")

    num_points = config.total_timesteps // evaluation.interval
    updates_per_rollout = config.num_minibatches * config.update_epochs
    with EventRun(output, metadata) as run:
        checkpoint = 0

        def record(point):
            nonlocal checkpoint
            checkpoint += 1
            if not _counters(
                point, checkpoint * evaluation.interval, checkpoint * updates_per_rollout
            ):
                raise RuntimeError(f"incomplete training telemetry at PPO update {checkpoint}")
            run.record(point)

        result = train(config, evaluation=evaluation, on_checkpoint=record)
        if checkpoint != num_points or not _counters(
            result, config.total_timesteps, num_points * updates_per_rollout
        ):
            raise RuntimeError("CartPole worker did not reach the full transition budget")
        run.finish()


def _run_worker(config, runtime, version, output, root, condition):
    metadata = _run_metadata(config, runtime, source="live", code_version=version)
    request = root / "requests" / condition / f"seed-{config.seed}.json"
    _write(metadata, request)
    log_path = root / "worker-logs" / condition / f"seed-{config.seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gymtosha.benchmarks.cartpole_final",
                "--worker-request",
                str(request),
                "--worker-output",
                str(output),
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            returncode = process.wait()
        finally:
            if process.poll() is None:
                terminate_probe(process)
    if returncode != 0 or not _completed(output, config, runtime):
        message = f"CartPole worker exited with code {returncode}: {log_path}"
        metadata_path = output / "metadata.json"
        if _read_record(metadata_path).get("status") not in ("failed", "interrupted"):
            _write({**metadata, "status": "failed", "error": message}, metadata_path)
        raise RuntimeError(message)


def _execution_key(row):
    variant = (row["backend"], row["device"], row["compile_env"])
    return EXECUTION_ORDER.index(variant), row["num_envs"]


def run_final(baseline, output, devices, legacy):
    if legacy is not None and (not legacy.is_dir() or legacy.resolve() == output.resolve()):
        raise ValueError("the legacy directory must exist and differ from the final output")
    with study_lock(output):
        base, _ = load_baseline(baseline)
        selected = final_presets(base)
        runtime, version = runtime_fingerprint(), source_fingerprint()
        protocol = {
            "design": "cartpole_tensorboard_t64",
            "runtime": runtime,
            "baseline": asdict(base),
            "devices": devices,
            "seeds": list(SEEDS),
            "horizon": HORIZON,
            "budget": BUDGET,
        }
        protocol_path = output / "protocol.json"
        if protocol_path.exists():
            if _read_record(protocol_path) != protocol:
                raise ValueError("use a new output directory for a different final protocol")
        else:
            if any(
                (output / name).exists() for name in ("matrix.json", "capacity-selected", "events")
            ):
                raise ValueError("final output already has artifacts without a recorded protocol")
            _write(protocol, protocol_path)
        matrix = verify_matrix(selected, output, devices, legacy, runtime)
        previous = _read_record(legacy / "protocol.json") if legacy else {}
        for row in sorted(matrix["conditions"], key=_execution_key):
            if row["status"] != "passed":
                continue
            name = row["id"]
            config = _condition_config(
                selected[str(row["num_envs"])], row["backend"], row["device"], row["compile_env"]
            )
            for seed in matrix["seeds"]:
                trial = replace(config, seed=seed)
                path = output / "events" / name / f"seed-{seed}"
                if _completed(path, trial, runtime):
                    continue
                if _migrate_legacy(legacy, previous, trial, runtime, path):
                    print(f"migrated {name} seed={seed}", flush=True)
                    continue
                print(f"training {name} seed={seed}", flush=True)
                _run_worker(trial, runtime, version, path, output, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    if args.worker_request is not None and args.worker_output is not None:
        _worker(args.worker_request, args.worker_output)
    elif args.baseline is None or args.output is None or not args.devices:
        parser.error("--baseline, --output and --devices are required")
    elif len(set(args.devices)) != len(args.devices):
        parser.error("--devices must not contain duplicates")
    else:
        run_final(args.baseline, args.output, args.devices, args.legacy)


if __name__ == "__main__":
    main()
