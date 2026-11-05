"""Isolated CartPole PPO capacity checks with doubling and halving."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def search_capacity(probe, *, asynchronous=False):
    count = 32 if asynchronous else 4096
    trials = []
    first_failed = None
    maximum = None
    increasing = None
    while count >= 1:
        result = probe(count)
        trials.append(result)
        passed = result["status"] == "passed"
        if increasing is None:
            increasing = passed
        if passed:
            maximum = count
            if asynchronous or not increasing:
                break
            count *= 2
        else:
            first_failed = count
            if increasing:
                break
            count //= 2
    return {
        "maximum_verified": maximum,
        "first_failed": first_failed,
        "capped": asynchronous and maximum == 32,
        "trials": trials,
    }


def _write(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _probe(args):
    from dataclasses import asdict

    import torch

    from gymtosha.benchmarks.train import TrainConfig, train

    started = time.perf_counter()
    config = TrainConfig(
        env_id="CartPole-v1",
        algorithm="ppo",
        backend=args.backend,
        device=args.device,
        compile_env=args.compile,
        num_envs=args.num_envs,
        total_timesteps=args.num_envs * 128,
        rollout_steps=128,
        update_epochs=1,
        num_minibatches=4,
        hidden_sizes=[256, 256, 256],
    )
    result = {"config": asdict(config), "num_envs": args.num_envs, "status": "failed"}
    try:
        training = train(config)
        losses = torch.tensor([point["loss"] for point in training["checkpoints"]])
        if training["parameter_delta"] <= 0 or not torch.isfinite(losses).all():
            raise RuntimeError("PPO update did not produce finite changed parameters")
        result.update(status="passed", training=training)
    except Exception as error:
        result.update(error_type=type(error).__name__, error=str(error))
    result["wall_seconds"] = time.perf_counter() - started
    _write(result, args.output)


def _available_memory():
    if Path("/proc/meminfo").exists():
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.split()[0]) * 1024
        return values["MemAvailable"]
    if sys.platform == "darwin":
        output = subprocess.check_output(["vm_stat"], text=True)
        page_size = int(output.split("page size of ")[1].split()[0])
        values = {}
        for line in output.splitlines()[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().strip("."))
        return page_size * sum(
            values.get(key, 0) for key in ("Pages free", "Pages inactive", "Pages speculative")
        )
    return None


def _isolated_probe(args, count):
    path = args.output / f"{args.backend}-{args.device}-{int(args.compile)}-{count}.json"
    command = [
        sys.executable,
        "-m",
        "gymtosha.benchmarks.capacity",
        "--probe",
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--num-envs",
        str(count),
        "--output",
        str(path),
    ]
    if args.compile:
        command.append("--compile")
    started = time.perf_counter()
    reason = None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with path.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        try:
            while process.poll() is None:
                if time.perf_counter() - started > args.timeout:
                    reason = "timeout"
                    break
                available = _available_memory()
                if available is not None and available < 1024**3:
                    reason = "host_memory_pressure"
                    break
                time.sleep(1)
        finally:
            if process.poll() is None:
                terminate_probe(process)
    if path.exists():
        result = json.loads(path.read_text())
        if reason is not None:
            result["supervisor_stop_reason"] = reason
            _write(result, path)
    else:
        result = {
            "num_envs": count,
            "status": "failed",
            "error_type": reason or "process_exit",
            "exit_code": process.returncode,
            "wall_seconds": time.perf_counter() - started,
        }
        _write(result, path)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("training", "config")}), flush=True
    )
    return result


def terminate_probe(process):
    try:
        rows = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True).splitlines()
        parents = {int(pid): int(parent) for pid, parent in (row.split() for row in rows)}
    except (OSError, subprocess.SubprocessError, ValueError):
        parents = {}
    descendants = set()
    pending = {process.pid}
    while pending:
        children = {pid for pid, parent in parents.items() if parent in pending}
        descendants.update(children)
        pending = children
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (PermissionError, ProcessLookupError):
        for pid in descendants:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (PermissionError, ProcessLookupError):
            if process.poll() is None:
                process.kill()
        process.wait()
    for pid in descendants:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["gymtosha", "gymnasium-sync", "gymnasium-native", "gymnasium-async"],
        required=True,
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--num-envs", type=int)
    args = parser.parse_args()
    if args.backend == "gymnasium-async" and args.num_envs is not None and args.num_envs > 32:
        parser.error("AsyncVectorEnv is capped at 32 environments")
    if args.probe:
        _probe(args)
        return
    name = f"{args.backend}-{args.device}-{int(args.compile)}"
    summary = search_capacity(
        lambda count: _isolated_probe(args, count), asynchronous=args.backend == "gymnasium-async"
    )
    summary.update(
        backend=args.backend,
        device=args.device,
        compile_env=args.compile,
        timeout_seconds=args.timeout,
        scope="128-step PPO rollout plus one epoch of four minibatch updates",
    )
    _write(summary, args.output / f"{name}-summary.json")


if __name__ == "__main__":
    main()
