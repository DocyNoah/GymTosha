"""CartPole coordinate searches and fixed-length, twenty-seed comparisons."""

from contextlib import contextmanager
from dataclasses import asdict, replace
from statistics import mean, stdev
from threading import Event, Lock

from gymtosha.benchmarks.evaluation import EvaluationConfig
from gymtosha.benchmarks.train import TrainConfig

SEEDS = tuple(range(1, 21))
COMPARISON_SEEDS = tuple(range(1001, 1021))
ROLLOUT_UPDATES = 64
_ACTIVE_WORKERS = {}
_WORKERS_LOCK = Lock()
_CANCELLED = Event()
PARAMETERS = {
    "learning_rate": [0.0001, 0.0003, 0.001, 0.003, 0.01],
    "gamma": [0.95, 0.98, 0.99, 0.995, 0.998],
    "rollout_steps": [16, 32, 64, 128, 256, 512],
    "ppo_minibatch_size": [],
    "update_epochs": [2, 4, 8, 10],
    "gae_lambda": [0.90, 0.95, 0.97, 0.99],
    "clip_coef": [0.10, 0.20, 0.30],
    "entropy_coef": [0.0, 0.001, 0.01, 0.03],
    "value_coef": [0.25, 0.5, 1.0],
    "max_grad_norm": [0.5, 1.0, 2.0],
    "adam_eps": [1e-8, 1e-5],
    "anneal_learning_rate": [False, True],
}


def reference_config(num_envs, *, previous=None):
    config = previous or TrainConfig(
        env_id="CartPole-v1",
        algorithm="ppo",
        backend="gymtosha",
        device="cuda",
        num_envs=32,
        compile_env=True,
        hidden_sizes=[256, 256, 256],
        rollout_steps=128,
        ppo_minibatch_size=256,
        num_minibatches=16,
        update_epochs=4,
        learning_rate=0.0003,
        rollout_timeout_seconds=60.0,
    )
    if num_envs < 32 or num_envs & (num_envs - 1):
        raise ValueError("study environment counts must be powers of two starting at 32")
    minibatch = config.ppo_minibatch_size * num_envs // config.num_envs
    result = replace(
        config,
        num_envs=num_envs,
        ppo_minibatch_size=minibatch,
        total_timesteps=num_envs * config.rollout_steps * ROLLOUT_UPDATES,
    )
    result.validate()
    return result


def change_parameter(config, parameter, value):
    result = replace(config, **{parameter: value})
    result = replace(
        result, total_timesteps=result.num_envs * result.rollout_steps * ROLLOUT_UPDATES
    )
    result.validate()
    return result


def candidate_values(parameter, config, *, local):
    values = PARAMETERS[parameter]
    if parameter == "ppo_minibatch_size":
        size = 32
        values = []
        while size <= config.num_envs * config.rollout_steps:
            values.append(size)
            size *= 2
    valid = []
    for value in values:
        try:
            change_parameter(config, parameter, value)
        except ValueError:
            continue
        valid.append(value)
    incumbent = getattr(config, parameter)
    if local and incumbent in valid:
        index = valid.index(incumbent)
        valid = valid[max(0, index - 1) : index + 2]
    return valid


def study_evaluation(config, *, comparison=False):
    return EvaluationConfig(
        target=475.0,
        interval=config.num_envs * config.rollout_steps * 4,
        episodes=100,
        confirmations=5,
        seed=200000 if comparison else 100000,
        stop_on_target=False,
        min_episode_success_rate=0.95,
    )


def summarize_trials(records, *, seeds=SEEDS, timing_comparable=True):
    actual = [record["seed"] for record in records]
    if len(set(actual)) != len(actual):
        raise ValueError("duplicate study seeds")
    if not set(actual).issubset(seeds):
        raise ValueError("unexpected study seed")
    complete = set(actual) == set(seeds)
    good = [record["result"] for record in records if record["status"] == "completed"]
    configs = [{k: v for k, v in row["config"].items() if k != "seed"} for row in good]
    if configs and any(config != configs[0] for config in configs[1:]):
        raise ValueError("mixed study training configurations")
    returns = [row["mean_training_episode_return"] for row in good]
    usable = complete and len(good) == len(seeds) and all(value is not None for value in returns)
    times = [row["wall_seconds"] for row in good]
    target_times = [row["evaluation"]["time_to_target_seconds"] for row in good]
    reached = sum(value is not None for value in target_times)
    return {
        "complete": complete,
        "attempted_seeds": len(actual),
        "successful_trials": len(good),
        "mean_training_return": mean(returns) if usable else None,
        "std_training_return": stdev(returns) if usable else None,
        "timing_comparable": timing_comparable,
        "mean_wall_seconds": mean(times) if usable and timing_comparable else None,
        "std_wall_seconds": stdev(times) if usable and timing_comparable else None,
        "target_reached": reached,
        "mean_target_seconds": mean(target_times)
        if usable and timing_comparable and reached == len(seeds)
        else None,
        "std_target_seconds": stdev(target_times)
        if usable and timing_comparable and reached == len(seeds)
        else None,
    }


def selection_key(summary):
    if summary["mean_training_return"] is None:
        return (1, 0, 0)
    return (
        0,
        -summary["mean_training_return"],
        summary["std_training_return"],
    )


def source_fingerprint():
    import hashlib
    from pathlib import Path

    digest = hashlib.sha256()
    for name in (
        "train.py",
        "algorithms.py",
        "adapters.py",
        "evaluation.py",
        "normalization.py",
        "../base.py",
        "../registry.py",
        "../__init__.py",
        "../envs/cartpole.py",
    ):
        digest.update(name.encode())
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def runtime_fingerprint():
    import platform

    import gymnasium
    import torch

    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
    }


def run_trial(config, evaluation, output, *, timing_scope="isolated"):
    import hashlib
    import json
    import subprocess
    import sys
    from concurrent.futures import CancelledError
    from pathlib import Path

    from gymtosha.benchmarks.capacity import terminate_probe
    from gymtosha.benchmarks.cli import _write

    if _CANCELLED.is_set():
        raise CancelledError()
    if source_fingerprint() != _LOADED_SOURCE:
        raise RuntimeError("training source changed; restart the study controller")
    request = {
        "config": asdict(config),
        "evaluation": asdict(evaluation),
        "source": source_fingerprint(),
        "runtime": runtime_fingerprint(),
    }
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:24]
    directory = Path(output) / "trials"
    path = directory / f"{key}.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record["request"] != request:
            raise ValueError("cached trial request changed")
        record.setdefault("trial_id", key)
        return record
    request_path = directory / f"{key}.request.json"
    _write(request, request_path)
    with path.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gymtosha.benchmarks.study",
                "worker",
                "--request",
                str(request_path),
                "--result",
                str(path),
                "--timing-scope",
                timing_scope,
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        with _WORKERS_LOCK:
            _ACTIVE_WORKERS[process.pid] = process
        try:
            if _CANCELLED.is_set():
                terminate_probe(process)
                raise CancelledError()
            process.wait()
        except BaseException:
            if process.poll() is None:
                terminate_probe(process)
            raise
        finally:
            with _WORKERS_LOCK:
                _ACTIVE_WORKERS.pop(process.pid, None)
    if _CANCELLED.is_set():
        raise CancelledError()
    if not path.exists():
        _write(
            {
                "request": request,
                "status": "failed",
                "seed": config.seed,
                "trial_id": key,
                "timing_scope": timing_scope,
                "error": f"worker exited with code {process.returncode}",
            },
            path,
        )
    return json.loads(path.read_text())


def run_worker(request_path, result_path, *, timing_scope="isolated"):
    import json
    import math
    from pathlib import Path

    from gymtosha.benchmarks.cli import _write
    from gymtosha.benchmarks.train import train

    request = json.loads(Path(request_path).read_text())
    config = TrainConfig(**request["config"])
    record = {
        "request": request,
        "seed": config.seed,
        "trial_id": Path(result_path).stem,
        "timing_scope": timing_scope,
    }
    if request["source"] != source_fingerprint() or request["runtime"] != runtime_fingerprint():
        raise ValueError("training source or runtime changed while the study was running")

    def progress(value):
        point = value["checkpoints"][-1]
        _write(
            {
                "config": asdict(config),
                "checkpoint": {k: v for k, v in point.items() if k != "episode_returns"},
            },
            Path(result_path).with_suffix(".progress.json"),
        )

    try:
        result = train(
            config, evaluation=EvaluationConfig(**request["evaluation"]), on_progress=progress
        )
        score = result["mean_training_episode_return"]
        if score is None or not math.isfinite(score):
            raise RuntimeError("training produced no finite completed-episode reward")
        record.update(status="completed", result=result)
        _write(record, Path(result_path))
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        _write(record, Path(result_path))


def estimate_trial_bytes(config):
    rollout = config.num_envs * config.rollout_steps
    minibatch = config.ppo_minibatch_size or rollout // config.num_minibatches
    return 512 * 1024**2 + 64 * rollout + 16384 * minibatch


def search_memory_budget(device):
    import torch

    if device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        return int(free * 0.8)
    if device == "mps":
        from gymtosha.benchmarks.capacity import _available_memory

        available = _available_memory()
        recommended = torch.mps.recommended_max_memory()
        return int(min(recommended, available or recommended) * 0.7)
    return 8 * 1024**3


def cancel_active_trials():
    from gymtosha.benchmarks.capacity import terminate_probe

    _CANCELLED.set()
    with _WORKERS_LOCK:
        processes = list(_ACTIVE_WORKERS.values())
    for process in processes:
        if process.poll() is None:
            terminate_probe(process)


def collect_candidates(config, parameter, values, output, *, parallelism=16, memory_budget=None):
    import random
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from pathlib import Path

    from gymtosha.benchmarks.capacity import _available_memory
    from gymtosha.benchmarks.cli import _write

    if parallelism < 1:
        raise ValueError("parallelism must be positive")
    _CANCELLED.clear()
    budget = search_memory_budget(config.device) if memory_budget is None else memory_budget
    host_available = _available_memory()
    # A compiled worker and its compiler process measured about 2.2 GiB on the server.
    host_limit = max(1, int(host_available * 0.7) // (2560 * 1024**2)) if host_available else 1
    worker_limit = min(parallelism, host_limit)
    pending = []
    for seed in SEEDS:
        order = list(values)
        random.Random(f"{config.num_envs}:{parameter}:{seed}").shuffle(order)
        for value in order:
            candidate = replace(change_parameter(config, parameter, value), seed=seed)
            pending.append((seed, value, candidate, estimate_trial_bytes(candidate)))
    records = {str(value): [] for value in values}
    running = {}
    reserved = 0
    completed = 0
    total = len(pending)
    executor = ThreadPoolExecutor(max_workers=worker_limit)
    try:
        while pending or running:
            while pending and len(running) < worker_limit:
                available_index = next(
                    (i for i, job in enumerate(pending) if job[3] + reserved <= budget), None
                )
                if available_index is None:
                    if running:
                        break
                    available_index = 0
                seed, value, candidate, reservation = pending.pop(available_index)
                future = executor.submit(
                    run_trial,
                    candidate,
                    study_evaluation(candidate),
                    output,
                    timing_scope="shared_search",
                )
                running[future] = (seed, value, reservation)
                reserved += reservation
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                seed, value, reservation = running.pop(future)
                reserved -= reservation
                trial = future.result()
                records[str(value)].append(trial)
                completed += 1
                _write(
                    {
                        "num_envs": config.num_envs,
                        "parameter": parameter,
                        "value": value,
                        "seed": seed,
                        "completed_seeds": len(records[str(value)]),
                        "completed_jobs": completed,
                        "total_jobs": total,
                        "running_trials": len(running),
                        "parallelism_limit": worker_limit,
                        "requested_parallelism": parallelism,
                        "host_available_bytes": host_available,
                        "device_budget_bytes": budget,
                    },
                    Path(output) / "progress.json",
                )
                print(
                    f"tune N={config.num_envs} {parameter}={value} seed={seed} "
                    f"{trial['status']} jobs={completed}/{total}",
                    flush=True,
                )
    except BaseException:
        cancel_active_trials()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    for trials in records.values():
        trials.sort(key=lambda record: record["seed"])
    return [
        {
            "value": value,
            "summary": summarize_trials(records[str(value)], timing_comparable=False),
            "trial_ids": [record["trial_id"] for record in records[str(value)]],
            "seed_rewards": [
                record["result"]["mean_training_episode_return"]
                if record["status"] == "completed"
                else None
                for record in records[str(value)]
            ],
        }
        for value in values
    ]


def tune_count(config, output, *, parallelism=16):
    import json
    from pathlib import Path

    from gymtosha.benchmarks.cli import _write
    from gymtosha.benchmarks.study_plot import plot_parameter, render_dashboard

    directory = Path(output) / "search" / str(config.num_envs)
    for parameter in PARAMETERS:
        phase_path = directory / f"{parameter}.json"
        if phase_path.exists():
            phase = json.loads(phase_path.read_text())
            if (
                phase["starting_config"] != asdict(config)
                or phase["source"] != source_fingerprint()
                or phase.get("runtime") != runtime_fingerprint()
            ):
                raise ValueError("saved parameter search starts from a different configuration")
            config = TrainConfig(**phase["selected_config"])
            continue
        starting_config = asdict(config)
        full_grid = candidate_values(parameter, config, local=False)
        values = candidate_values(parameter, config, local=config.num_envs > 32)
        while True:
            candidates = collect_candidates(
                config, parameter, values, output, parallelism=parallelism
            )
            winner = min(candidates, key=lambda candidate: selection_key(candidate["summary"]))
            if winner["summary"]["mean_training_return"] is None:
                raise RuntimeError(
                    f"no valid twenty-seed candidate for {parameter}, N={config.num_envs}"
                )
            # Expand a local bracket while its best candidate lies on an open edge.
            position = full_grid.index(winner["value"])
            expanded = set(values)
            if winner["value"] == min(values) and position > 0:
                expanded.add(full_grid[position - 1])
            if winner["value"] == max(values) and position + 1 < len(full_grid):
                expanded.add(full_grid[position + 1])
            if expanded == set(values) or config.num_envs == 32:
                break
            values = sorted(expanded)
        config = change_parameter(config, parameter, winner["value"])
        phase = {
            "source": source_fingerprint(),
            "runtime": runtime_fingerprint(),
            "num_envs": config.num_envs,
            "parameter": parameter,
            "starting_config": starting_config,
            "candidates": candidates,
            "selected_value": winner["value"],
            "selected_config": asdict(config),
        }
        _write(phase, phase_path)
        plot_parameter(phase, directory / f"{parameter}.png")
        render_dashboard(output)
        print(f"SELECTED N={config.num_envs} {parameter}={winner['value']}", flush=True)
    return config


def tune(args):
    import json
    from pathlib import Path

    from gymtosha.benchmarks.cli import _write
    from gymtosha.benchmarks.study_plot import render_dashboard

    path = Path(args.output) / "selected.json"
    selected = json.loads(path.read_text()) if path.exists() else {}
    previous = None
    for count in sorted(int(value) for value in selected if int(value) < args.min_envs):
        previous = TrainConfig(**selected[str(count)]["config"])
    count = args.min_envs
    while count <= args.max_envs:
        base = reference_config(count, previous=previous)
        base = replace(base, device=args.device, compile_env=True)
        if str(count) in selected:
            saved = selected[str(count)]
            if saved["source"] != source_fingerprint():
                raise ValueError("selected training source changed")
            if (
                saved["config"]["device"] != args.device
                or saved.get("runtime") != runtime_fingerprint()
            ):
                raise ValueError("selected device or runtime changed")
            previous = TrainConfig(**saved["config"])
        else:
            previous = tune_count(
                base, args.output, parallelism=getattr(args, "parallel_trials", 16)
            )
            selected[str(count)] = {
                "source": source_fingerprint(),
                "runtime": runtime_fingerprint(),
                "config": asdict(previous),
            }
            _write(selected, path)
        render_dashboard(args.output)
        count *= 2


def compare(args):
    import json
    from pathlib import Path

    from gymtosha.benchmarks.cli import _write
    from gymtosha.benchmarks.study_plot import render_dashboard

    selected = json.loads(Path(args.presets).read_text())
    capacity = json.loads(Path(args.capacity).read_text())
    directory = Path(args.output)
    for condition in capacity:
        if condition["device"] not in args.devices:
            continue
        if condition["maximum_verified"] is None:
            name = f"{condition['backend']}-{condition['device']}-{int(condition['compile_env'])}"
            _write(
                {"condition": condition, "status": "unsupported"},
                directory / "unsupported" / f"{name}.json",
            )
            continue
        for count in sorted(map(int, selected)):
            maximum = condition["maximum_verified"]
            if condition["backend"] == "gymnasium-async":
                maximum = min(maximum, 32)
            if count > maximum:
                continue
            config = TrainConfig(**selected[str(count)]["config"])
            config = replace(
                config,
                backend=condition["backend"],
                device=condition["device"],
                compile_env=condition["compile_env"],
            )
            records = []
            for seed in COMPARISON_SEEDS:
                records.append(
                    run_trial(
                        replace(config, seed=seed),
                        study_evaluation(config, comparison=True),
                        directory,
                    )
                )
                _write(
                    {"condition": condition, "num_envs": count, "seed": seed},
                    directory / "progress.json",
                )
                print(
                    f"compare {config.backend}/{config.device}/{config.compile_env} "
                    f"N={count} seed={seed}",
                    flush=True,
                )
            name = f"{config.backend}-{config.device}-{int(config.compile_env)}-{count}"
            _write(
                {
                    "name": name,
                    "config": asdict(config),
                    "summary": summarize_trials(records, seeds=COMPARISON_SEEDS),
                    "seeds": list(COMPARISON_SEEDS),
                    "trial_ids": [record["trial_id"] for record in records],
                },
                directory / "comparison" / f"{name}.json",
            )
            render_dashboard(directory)


def main():
    import argparse
    import signal
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("tune")
    search.add_argument("--min-envs", type=int, default=32)
    search.add_argument("--max-envs", type=int, required=True)
    search.add_argument("--device", choices=["cuda", "mps"], default="cuda")
    search.add_argument("--output", type=Path, required=True)
    search.add_argument("--parallel-trials", type=int, choices=range(1, 33), default=16)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--presets", type=Path, required=True)
    comparison.add_argument("--capacity", type=Path, required=True)
    comparison.add_argument("--devices", nargs="+", choices=["cpu", "cuda", "mps"], required=True)
    comparison.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--request", type=Path, required=True)
    worker.add_argument("--result", type=Path, required=True)
    worker.add_argument("--timing-scope", choices=["isolated", "shared_search"], default="isolated")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    if args.command == "worker":
        run_worker(args.request, args.result, timing_scope=args.timing_scope)
    else:
        with study_lock(args.output):
            if args.command == "tune":
                tune(args)
            else:
                compare(args)


@contextmanager
def study_lock(output):
    import fcntl
    from pathlib import Path

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "a study controller is already running in this output directory"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


_LOADED_SOURCE = source_fingerprint()


if __name__ == "__main__":
    main()
