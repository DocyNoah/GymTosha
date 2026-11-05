"""Reproducible benchmark command line with raw, resumable measurement files."""

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import gymnasium as gym
import torch

from gymtosha.benchmarks.adapters import make_backend, synchronize
from gymtosha.benchmarks.plot import plot_throughput, plot_time_to_target, plot_training
from gymtosha.benchmarks.train import TrainConfig, train

ENVIRONMENTS = [
    "CartPole-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "LunarLander-v3",
    "FrozenLake-v1",
]


def _write(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _csv(records, path):
    rows = []
    for record in records:
        row = dict(record.get("config", {}))
        row.update(
            {
                key: value
                for key, value in record.items()
                if key not in ("config", "checkpoints", "metadata")
            }
        )
        rows.append(row)
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fields)
            writer.writeheader()
            writer.writerows(rows)


def measure_throughput(env_id, backend, device, num_envs, steps, repeat, compile_env=False):
    torch.set_num_threads(1)
    setup_start = time.perf_counter()
    env = make_backend(env_id, backend, num_envs, device, compile_env)
    try:
        env.reset(repeat + 1)
        continuous = isinstance(env.action_space, gym.spaces.Box)
        action = torch.zeros(
            (num_envs, *env.action_space.shape),
            device=env.device,
            dtype=torch.float32 if continuous else torch.long,
        )
        setup_seconds = time.perf_counter() - setup_start
        warmup_start = time.perf_counter()
        for _ in range(10):
            env.step(action)
        synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_start
        env.reset(repeat + 1)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(steps):
            env.step(action)
        synchronize(device)
        elapsed = time.perf_counter() - start
        return {
            "env_id": env_id,
            "backend": backend,
            "device": device,
            "compile_env": compile_env,
            "num_envs": num_envs,
            "vector_steps": steps,
            "repeat": repeat,
            "seconds": elapsed,
            "steps_per_second": steps * num_envs / elapsed,
            "setup_seconds": setup_seconds,
            "warmup_seconds": warmup_seconds,
            "physics_equivalent": "LunarLander" not in env_id,
            "metadata": {
                "platform": platform.platform(),
                "torch": torch.__version__,
                "gymnasium": gym.__version__,
                "num_threads": torch.get_num_threads(),
            },
        }
    finally:
        env.close()


def _compatible(algorithm, env_id):
    continuous = "Continuous" in env_id
    return (
        algorithm == "ppo"
        or (algorithm == "dqn" and not continuous)
        or (algorithm in ("sac", "td3") and continuous)
    )


def _common(parser):
    parser.add_argument("--env-ids", nargs="+", default=ENVIRONMENTS)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["gymnasium-sync", "gymtosha"],
        choices=["gymtosha", "gymnasium-sync", "gymnasium-async"],
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--compile", action="store_true", dest="compile_env")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--append", action="store_true")


def _time_to_target(args):
    from gymtosha.benchmarks.evaluation import EvaluationConfig
    from gymtosha.benchmarks.target import (
        target_preset,
        target_rows,
        target_summary,
        validate_target_comparisons,
    )

    path = args.output / "time_to_target.json"
    records = json.loads(path.read_text()) if args.resume and path.exists() else []
    selected = json.loads(args.presets.read_text()) if args.presets else {}
    base_evaluation = EvaluationConfig(
        target=args.target,
        interval=args.eval_interval if args.eval_interval is not None else 65536,
        episodes=args.eval_episodes if args.eval_episodes is not None else 100,
        confirmations=args.confirmations,
        seed=args.eval_seed,
    )
    for env_id in args.env_ids:
        evaluation = replace(
            base_evaluation,
            episodes=(
                args.eval_episodes
                if args.eval_episodes is not None
                else (1000 if env_id.startswith("FrozenLake") else 100)
            ),
        )
        if args.presets and env_id not in selected:
            raise ValueError(f"no selected hyperparameters for {env_id}")
        selection = selected.get(env_id, {})
        backend_presets = selection.get("backends", {})
        first_selection = next(iter(backend_presets.values())) if backend_presets else selection
        preset = TrainConfig(**first_selection["config"]) if args.presets else target_preset(env_id)
        algorithms = args.algorithms or [preset.algorithm]
        for algorithm in algorithms:
            if not _compatible(algorithm, env_id):
                continue
            for seed in args.seeds:
                for backend in args.backends:
                    if args.presets:
                        backend_selection = backend_presets.get(backend, selection)
                        if "config" not in backend_selection:
                            raise ValueError(f"no selected hyperparameters for {env_id}/{backend}")
                        preset = TrainConfig(**backend_selection["config"])
                        if algorithm != preset.algorithm:
                            raise ValueError(
                                "selected preset uses a different representative algorithm"
                            )
                        config = replace(
                            preset,
                            env_id=env_id,
                            backend=backend,
                            device=args.device,
                            seed=seed,
                            compile_env=backend == "gymtosha",
                        )
                        evaluation = replace(
                            evaluation,
                            interval=(
                                args.eval_interval
                                if args.eval_interval is not None
                                else backend_selection.get("evaluation_interval", 65536)
                            ),
                        )
                    else:
                        config = target_preset(
                            env_id,
                            backend=backend,
                            device=args.device,
                            seed=seed,
                            algorithm=algorithm,
                        )
                    overrides = {
                        name: getattr(args, name)
                        for name in ("total_timesteps", "num_envs", "rollout_steps")
                        if getattr(args, name) is not None
                    }
                    config = replace(config, **overrides)
                    if args.device == "cpu" and not args.compile_env:
                        config = replace(config, compile_env=False)
                    saved = [
                        r
                        for r in records
                        if all(
                            r["config"][field] == getattr(config, field)
                            for field in ("env_id", "algorithm", "backend", "device", "seed")
                        )
                    ]
                    if saved:
                        if (
                            saved[0]["config"] != asdict(config)
                            or saved[0]["evaluation"]["config"] != asdict(evaluation)
                            or saved[0].get("comparison_mode", "matched") != args.comparison_mode
                        ):
                            raise ValueError("resume settings differ from the existing trial")
                        continue
                    validate_target_comparisons(
                        [
                            *records,
                            {
                                "config": asdict(config),
                                "evaluation": {"config": asdict(evaluation)},
                                "comparison_mode": args.comparison_mode,
                            },
                        ]
                    )
                    progress_path = (
                        args.output / f"{env_id}__{algorithm}__{backend}__{seed}.progress.json"
                    )

                    def progress(value, config=config, progress_path=progress_path):
                        _write(
                            {"state": "running", "config": asdict(config), "evaluation": value},
                            progress_path,
                        )
                        point = value["checkpoints"][-1]
                        print(
                            f"{config.env_id} {config.algorithm} {config.backend} "
                            f"seed={config.seed} "
                            f"step={point['timesteps']} score={point['mean_return']:.3f} "
                            f"wall={point['wall_seconds']:.1f}s",
                            flush=True,
                        )

                    result = train(config, evaluation=evaluation, on_progress=progress)
                    result["comparison_mode"] = args.comparison_mode
                    records.append(result)
                    _write(records, path)
                    summary = target_summary(records)
                    _write(summary, args.output / "target_summary.json")
                    _csv(summary, args.output / "target_summary.csv")
                    _csv(target_rows(records), args.output / "time_to_target.csv")
                    progress_path.unlink(missing_ok=True)
                    plot_time_to_target(records, args.output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    _common(training)
    training.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        choices=["ppo", "dqn", "sac", "td3"],
    )
    defaults = asdict(TrainConfig())
    for name in (
        "num_envs",
        "total_timesteps",
        "rollout_steps",
        "update_epochs",
        "num_minibatches",
        "batch_size",
        "buffer_size",
        "learning_starts",
        "train_every",
        "gradient_steps",
        "log_every",
        "num_threads",
    ):
        training.add_argument("--" + name.replace("_", "-"), type=int, default=defaults[name])
    training.add_argument("--learning-rate", type=float, default=None)
    training.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    throughput = commands.add_parser("throughput")
    _common(throughput)
    throughput.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64, 1024])
    throughput.add_argument("--steps", type=int, default=200)
    throughput.add_argument("--repeats", type=int, default=3)
    target = commands.add_parser("time-to-target")
    _common(target)
    target.add_argument("--algorithms", nargs="+", choices=["ppo", "dqn", "sac", "td3"])
    target.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    target.add_argument("--total-timesteps", type=int)
    target.add_argument("--num-envs", type=int)
    target.add_argument("--rollout-steps", type=int)
    target.add_argument("--eval-interval", type=int)
    target.add_argument("--eval-episodes", type=int)
    target.add_argument("--eval-seed", type=int, default=100000)
    target.add_argument("--confirmations", type=int, default=2)
    target.add_argument("--target", type=float)
    target.add_argument("--resume", action="store_true")
    target.add_argument("--presets", type=Path)
    target.add_argument(
        "--comparison-mode", choices=["matched", "independently-tuned"], default="matched"
    )
    plots = commands.add_parser("plot")
    plots.add_argument("--inputs", type=Path, nargs="+", required=True)
    plots.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "time-to-target":
        _time_to_target(args)
        return
    if args.command == "plot":
        for name, plotter in [
            ("training", plot_training),
            ("throughput", plot_throughput),
            ("time_to_target", plot_time_to_target),
        ]:
            records = []
            for folder in args.inputs:
                path = folder / f"{name}.json"
                if path.exists():
                    records.extend(json.loads(path.read_text()))
            if not records:
                continue
            _write(records, args.output / f"{name}.json")
            if name == "time_to_target":
                from gymtosha.benchmarks.target import target_rows, target_summary

                summary = target_summary(records)
                _write(summary, args.output / "target_summary.json")
                _csv(summary, args.output / "target_summary.csv")
                _csv(target_rows(records), args.output / f"{name}.csv")
            else:
                _csv(records, args.output / f"{name}.csv")
            plotter(records, args.output)
        return
    name = "training" if args.command == "train" else "throughput"
    path = args.output / f"{name}.json"
    records = json.loads(path.read_text()) if args.append and path.exists() else []
    if args.command == "train":
        from gymtosha.benchmarks.target import REPRESENTATIVE_ALGORITHMS

        for env_id in args.env_ids:
            algorithms = args.algorithms or [REPRESENTATIVE_ALGORITHMS.get(env_id, "ppo")]
            for algorithm in algorithms:
                if not _compatible(algorithm, env_id):
                    continue
                for seed in args.seeds:
                    for backend in args.backends:
                        fields = {key: getattr(args, key) for key in defaults if hasattr(args, key)}
                        fields.update(
                            env_id=env_id,
                            algorithm=algorithm,
                            backend=backend,
                            seed=seed,
                            compile_env=args.compile_env and backend == "gymtosha",
                        )
                        result = train(TrainConfig(**fields))
                        records.append(result)
                        _write(records, path)
                        print(
                            f"{env_id} {algorithm} {backend} {args.device} seed={seed} "
                            f"{result['steps_per_second']:.0f} transitions/s",
                            flush=True,
                        )
        _csv(records, args.output / "training.csv")
        plot_training(records, args.output)
    else:
        if args.steps < 1 or args.repeats < 1:
            parser.error("steps and repeats must be positive")
        for env_id in args.env_ids:
            for count in args.batch_sizes:
                for repeat in range(args.repeats):
                    for backend in args.backends:
                        result = measure_throughput(
                            env_id,
                            backend,
                            args.device,
                            count,
                            args.steps,
                            repeat,
                            args.compile_env and backend == "gymtosha",
                        )
                        records.append(result)
                        _write(records, path)
                        print(
                            f"{env_id} n={count} {backend} {args.device} "
                            f"{result['steps_per_second']:.0f} transitions/s",
                            flush=True,
                        )
        _csv(records, args.output / "throughput.csv")
        plot_throughput(records, args.output)


if __name__ == "__main__":
    main()
