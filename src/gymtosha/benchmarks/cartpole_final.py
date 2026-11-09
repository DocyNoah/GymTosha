"""Run the five-seed, fixed-horizon CartPole comparison on one host."""

import argparse
import hashlib
import json
import signal
import subprocess
import sys
from dataclasses import asdict, replace
from math import isclose, isfinite
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
from gymtosha.benchmarks.study_grouped import conditions, load_baseline, probe
from gymtosha.benchmarks.train import TrainConfig, train

HORIZON = 64
BUDGET = 2_097_152
LR_SCHEDULE_TIMESTEPS = 8_388_608
ENV_COUNTS = (32, 256, 512, 1024)
CAPACITY_SECONDS = 60
SEED_BLOCKS = {start: tuple(range(start, start + 5)) for start in range(1001, 1021, 5)}
SEEDS = SEED_BLOCKS[1001]
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
    base = replace(base, lr_schedule_timesteps=LR_SCHEDULE_TIMESTEPS)
    return expected_presets(base, final, HORIZON, each_update=True)


def _read_record(path):
    try:
        record = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        return {}
    return record if isinstance(record, dict) else {}


def read_seed_replacements(root, seeds):
    path = root / "seed-replacements.json"
    try:
        replacements = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(replacements, dict):
        raise ValueError(f"{path}: expected a JSON object")
    declared = {str(seed) for seed in seeds}
    for condition, substituted in replacements.items():
        if (
            not isinstance(substituted, dict)
            or not substituted.keys() <= declared
            or any(type(seed) is not int or seed < 0 for seed in substituted.values())
            or len(set(substituted.values()).union(seeds)) != len(seeds) + len(substituted)
        ):
            raise ValueError(f"{path}: invalid replacements for {condition}")
    return replacements


def seed_locations(root, condition, seeds, replacements):
    for seed in seeds:
        actual = replacements.get(str(seed), seed)
        directory = "events" if seed == actual else "replacements"
        yield seed, actual, root / directory / condition / f"seed-{actual}"


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
            yield (
                condition,
                replace(
                    config, total_timesteps=count * config.rollout_steps, lr_schedule_timesteps=None
                ),
            )


def _same_config(saved, config):
    return isinstance(saved, dict) and {"lr_schedule_timesteps": None, **saved} == asdict(config)


def _valid_probe(record, config, runtime):
    elapsed = record.get("observed_seconds")
    status = record.get("status")
    return (
        _same_config(record.get("config"), config)
        and record.get("runtime", runtime) == runtime
        and record.get("limit_seconds") == CAPACITY_SECONDS
        and _time(elapsed)
        and (status == "failed" or (status == "passed" and elapsed < CAPACITY_SECONDS))
    )


def verify_matrix(selected, output, devices, legacy, runtime, *, prefix_from=None, seeds=SEEDS):
    capacity = output / "capacity-selected"
    matrix_path = output / "matrix.json"
    published = matrix_path.exists()
    configs = list(_probe_configs(selected, devices))
    if not published:
        sources = [
            (root, marker)
            for root, marker in ((prefix_from, "prefix_probe"), (legacy, "legacy_probe"))
            if root is not None and _read_record(root / "protocol.json").get("runtime") == runtime
        ]
        for name, config in configs:
            target = capacity / f"{name}.json"
            if target.exists():
                if _valid_probe(_read_record(target), config, runtime):
                    continue
                target.replace(target.with_suffix(".unverified.json"))
            for root, marker in sources:
                path = root / "capacity-selected" / target.name
                saved = _read_record(path)
                if _valid_probe(saved, config, runtime):
                    _write({**saved, marker: str(path.resolve())}, target)
                    break
            else:
                probe(config, capacity)

    rows = []
    for name, config in configs:
        path = capacity / f"{name}.json"
        saved = _read_record(path)
        if not _valid_probe(saved, config, runtime):
            raise RuntimeError(f"missing capacity evidence for {name}")
        rows.append(
            {
                "id": name,
                "backend": config.backend,
                "device": config.device,
                "compile_env": config.compile_env,
                "num_envs": config.num_envs,
                "host": runtime["node"],
                "status": saved["status"],
                "capacity_probe": str(path.relative_to(output)),
                "capacity_source": (
                    "prefix_tensorboard"
                    if "prefix_probe" in saved
                    else "legacy_json"
                    if "legacy_probe" in saved
                    else "live"
                ),
                **{key: saved[key] for key in ("error_type", "error") if key in saved},
            }
        )
    matrix = {"conditions": rows, "seeds": list(seeds), "budget": BUDGET, "horizon": HORIZON}
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


def _scores_valid(scores, episodes):
    return (
        isinstance(scores, list)
        and len(scores) == episodes
        and all(_time(score) and 1 <= score <= 500 and score % 1 == 0 for score in scores)
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


def _full_budget_configs(config):
    """Match both the implicit and explicit forms of the full-budget schedule."""
    budget = config.lr_schedule_timesteps
    if budget is not None and budget > config.total_timesteps:
        yield replace(config, total_timesteps=budget, lr_schedule_timesteps=None)
        yield replace(config, total_timesteps=budget)


def legacy_points(record, config, runtime):
    """Return validated raw evaluation telemetry from a completed legacy trial."""
    evaluation = study_evaluation(config, comparison=True, each_update=True)
    expected_evaluation = asdict(evaluation)
    try:
        request, result = record["request"], record["result"]
        saved_evaluation = result["evaluation"]
        points = saved_evaluation["checkpoints"]
        saved_config = request["config"]
        original = next(
            (
                item
                for item in (config, *_full_budget_configs(config))
                if _same_config(saved_config, item)
            ),
            None,
        )
        if (
            record["status"] != "completed"
            or record["timing_scope"] != "isolated"
            or not _count(record["seed"], config.seed)
            or request["runtime"] != runtime
            or original is None
            or not _same_config(result["config"], original)
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
        original_points = original.total_timesteps // rollout
        updates = config.num_minibatches * config.update_epochs
        if (
            not num_points
            or config.total_timesteps % rollout
            or not isinstance(points, list)
            or len(points) != original_points
            or not _counters(result, original.total_timesteps, original_points * updates)
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
        episodes = _training_episode_counts(result, rollout, original_points)
        if not episodes:
            missing.append("completed_episodes")
        migrated = []
        previous = 0
        for index, point in enumerate(points[:num_points], start=1):
            elapsed, scores = point["wall_seconds"], point["episode_returns"]
            if (
                not _counters(point, index * rollout, index * updates)
                or not _time(elapsed)
                or not previous < elapsed <= result["wall_seconds"]
                or not _scores_valid(scores, evaluation.episodes)
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
        if original_points == num_points:
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
    for original in (config, *_full_budget_configs(config)):
        saved = asdict(original)
        configs = [saved]
        if original.lr_schedule_timesteps is None:
            configs.append(
                {key: value for key, value in saved.items() if key != "lr_schedule_timesteps"}
            )
        for saved_config in configs:
            request = {
                "config": saved_config,
                "evaluation": metadata["evaluation"],
                "source": source,
                "runtime": runtime,
            }
            # The old controller hashed the complete request with its training fingerprint.
            key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
            path = legacy / "trials" / f"{key}.json"
            record = _read_record(path)
            if record.get("request") != request:
                continue
            trial = legacy_points(record, config, runtime)
            if trial is None:
                continue
            points, missing = trial
            metadata.update(
                legacy_record=str(path.resolve()),
                legacy_source_fingerprint=source,
                missing_fields=missing,
            )
            if original.total_timesteps != config.total_timesteps:
                metadata["reused_prefix"] = _prefix_provenance(path, original, metadata)
            _write_seed(output, metadata, points)
            return True
    return False


def _prefix_provenance(path, config, metadata):
    return {
        "path": str(path.resolve()),
        "budget": config.total_timesteps,
        "code_version": metadata["code_version"],
    }


def _write_seed(output, metadata, points):
    from gymtosha.benchmarks.cartpole_events import EventRun

    with EventRun(output, metadata) as run:
        for point in points:
            run.record(point)
        run.finish()


def _read_event_prefix(path, metadata, config, optional):
    from google.protobuf.message import DecodeError
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
    from tensorboard.util.tensor_util import make_ndarray

    from gymtosha.benchmarks.cartpole_events import METRIC_TAGS

    paths = list(path.glob("events.out.tfevents.*"))
    if len(paths) != 1:
        return None
    rollout = metadata["evaluation"]["interval"]
    count = config.total_timesteps // rollout
    if not count or config.total_timesteps % rollout:
        return None
    updates = config.num_minibatches * config.update_epochs
    planned_updates = (
        config.lr_schedule_timesteps // config.ppo_minibatch_size
    ) * config.update_epochs
    names = {tag: name for name, tag in METRIC_TAGS.items()}
    names["eval/episode_returns"] = "episode_returns"
    required = set(names.values()) - optional
    startup = ("setup_seconds", "warmup_seconds", "learner_setup_seconds")
    cumulative = ("wall_seconds", "collection_seconds", "update_seconds")
    expected_metadata = {key: val for key, val in metadata.items() if key != "error"}
    expected_metadata["status"] = "running"
    points, missing = [], set()
    previous = {key: 0 for key in (*cumulative, "completed_episodes")}
    episode_step = 0
    saw_metadata = False
    try:
        for event in LegacyEventFileLoader(str(paths[0])).Load():
            if event.step > count:
                break
            point = {}
            for value in event.summary.value:
                if value.tag == "run/metadata":
                    if (
                        saw_metadata
                        or json.loads(make_ndarray(value.tensor).item()) != expected_metadata
                    ):
                        return None
                    saw_metadata = True
                if value.tag not in names:
                    continue
                name = names[value.tag]
                if name in point:
                    return None
                array = make_ndarray(value.tensor)
                if array.dtype.kind not in "iuf":
                    return None
                if name == "episode_returns":
                    if array.shape != (metadata["evaluation"]["episodes"],):
                        return None
                    point[name] = array.tolist()
                elif array.shape == () and isfinite(array.item()):
                    point[name] = array.item()
                else:
                    return None
            if not point:
                continue
            step = len(points) + 1
            point["timesteps"] = step * rollout
            if (
                not saw_metadata
                or required.difference(point)
                or event.step != step
                or not _count(point["ppo_update"], step)
                or not _counters(point, step * rollout, step * updates)
                or not _scores_valid(point["episode_returns"], metadata["evaluation"]["episodes"])
                or point["wall_seconds"] <= previous["wall_seconds"]
                or any(point[key] < previous[key] for key in cumulative if key in point)
                or any(point[key] < 0 for key in startup if key in point)
            ):
                return None
            if "completed_episodes" in point:
                episodes = point["completed_episodes"]
                if (
                    type(episodes) is not int
                    or not 0
                    <= episodes - previous["completed_episodes"]
                    <= (step - episode_step) * rollout
                ):
                    return None
                previous["completed_episodes"] = episodes
                episode_step = step
            if "learning_rate" in point:
                rate = config.learning_rate or 2.5e-4
                if config.anneal_learning_rate:
                    rate *= 1 - (step * updates - 1) / planned_updates
                if not isclose(point["learning_rate"], rate, rel_tol=1e-6):
                    return None
            previous.update({key: point[key] for key in cumulative if key in point})
            missing.update(optional.difference(point))
            points.append(point)
    except (DecodeError, OSError, TypeError, ValueError, OverflowError):
        return None
    return (points, missing) if len(points) == count else None


def _migrate_prefix(path, config, runtime, output):
    from gymtosha.benchmarks.cartpole_events import METRIC_TAGS

    metadata = _read_record(path / "metadata.json")
    original = next(
        (
            item
            for item in _full_budget_configs(config)
            if _same_config(metadata.get("config"), item)
        ),
        None,
    )
    source, code_version = metadata.get("source"), metadata.get("code_version")
    expected = _run_metadata(config, runtime, source=source, code_version=code_version)
    missing = metadata.get("missing_fields")
    core = {"valid_transitions", "gradient_updates", "ppo_update", "wall_seconds"}
    if (
        original is None
        or metadata.get("status") not in ("running", "completed", "interrupted", "failed")
        or source not in ("live", "legacy_json")
        or not isinstance(code_version, str)
        or not code_version
        or any(
            metadata.get(key) != expected[key]
            for key in ("host", "runtime", "evaluation", "timing_scope")
        )
        or not isinstance(missing, list)
        or any(
            not isinstance(field, str) or field not in METRIC_TAGS or field in core
            for field in missing
        )
        or (source == "live" and missing)
    ):
        return False
    prefix = _read_event_prefix(path, metadata, config, set(missing))
    if prefix is None:
        return False
    points, gaps = prefix
    migrated = {key: val for key, val in metadata.items() if key not in ("status", "error")}
    migrated.update(
        config=expected["config"],
        missing_fields=[field for field in missing if field in gaps],
        reused_prefix=_prefix_provenance(path, original, metadata),
    )
    _write_seed(output, migrated, points)
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


def run_final(baseline, output, devices, legacy, *, prefix_from=None, seed_start=1001):
    seeds = SEED_BLOCKS[seed_start]
    for source in (prefix_from, legacy):
        if source is not None and (
            not source.is_dir() or output.resolve().is_relative_to(source.resolve())
        ):
            raise ValueError(
                "the source directory must exist and be separate from the final output"
            )
    with study_lock(output):
        base, _ = load_baseline(baseline)
        selected = final_presets(base)
        runtime, version = runtime_fingerprint(), source_fingerprint()
        protocol = {
            "design": "cartpole_tensorboard_t64",
            "runtime": runtime,
            "baseline": asdict(base),
            "devices": devices,
            "seeds": list(seeds),
            "horizon": HORIZON,
            "budget": BUDGET,
            "lr_schedule_timesteps": LR_SCHEDULE_TIMESTEPS,
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
        matrix = verify_matrix(
            selected, output, devices, legacy, runtime, prefix_from=prefix_from, seeds=seeds
        )
        replacements = read_seed_replacements(output, seeds)
        eligible = {row["id"] for row in matrix["conditions"] if row["status"] == "passed"}
        if replacements.keys() - eligible:
            raise ValueError(f"{output}: replacements require an eligible condition")
        previous = _read_record(legacy / "protocol.json") if legacy else {}
        pending = []
        for row in sorted(matrix["conditions"], key=_execution_key):
            if row["status"] != "passed":
                continue
            name = row["id"]
            config = _condition_config(
                selected[str(row["num_envs"])], row["backend"], row["device"], row["compile_env"]
            )
            for _, seed, path in seed_locations(output, name, seeds, replacements.get(name, {})):
                trial = replace(config, seed=seed)
                if _completed(path, trial, runtime):
                    continue
                if prefix_from is not None and _migrate_prefix(
                    prefix_from / "events" / name / f"seed-{seed}", trial, runtime, path
                ):
                    print(f"reused event prefix {name} seed={seed}", flush=True)
                    continue
                if _migrate_legacy(legacy, previous, trial, runtime, path):
                    print(f"migrated {name} seed={seed}", flush=True)
                    continue
                pending.append((name, trial, path))
        for name, trial, path in pending:
            print(f"training {name} seed={trial.seed}", flush=True)
            _run_worker(trial, runtime, version, path, output, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--legacy", type=Path, help="prior same-host CartPole JSON study")
    parser.add_argument("--prefix-from", type=Path, help="prior same-host 8M TensorBoard study")
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--seed-start", type=int, choices=SEED_BLOCKS, default=1001)
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
        run_final(
            args.baseline,
            args.output,
            args.devices,
            args.legacy,
            prefix_from=args.prefix_from,
            seed_start=args.seed_start,
        )


if __name__ == "__main__":
    main()
