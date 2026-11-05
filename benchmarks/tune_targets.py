"""Screen four configurations per task, then confirm the two best on a second seed."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median

from gymtosha.benchmarks.cli import ENVIRONMENTS, _write
from gymtosha.benchmarks.evaluation import EvaluationConfig
from gymtosha.benchmarks.target import target_preset
from gymtosha.benchmarks.train import train


def candidates(env_id, device, backend="gymnasium-sync"):
    if backend == "gymtosha":
        return native_candidates(env_id, device)
    base = target_preset(env_id, backend="gymnasium-sync", device=device, seed=101)
    variants = {
        "CartPole-v1": [
            {},
            {"num_envs": 8, "rollout_steps": 128, "update_epochs": 4},
            {"num_envs": 32, "rollout_steps": 64},
            {"update_epochs": 4, "learning_rate": 1e-3},
        ],
        "MountainCar-v0": [
            {},
            {"exploration_hold_steps": 8, "learning_rate": 1e-3},
            {"exploration_hold_steps": 32},
            {"learning_rate": 1e-3, "gamma": 0.995},
        ],
        "MountainCarContinuous-v0": [
            {},
            {"num_envs": 8, "rollout_steps": 256},
            {"num_envs": 16, "rollout_steps": 128, "update_epochs": 4, "learning_rate": 1e-3},
            {"num_envs": 16, "rollout_steps": 128, "entropy_coef": 0.01},
        ],
        "LunarLander-v3": [
            {},
            {"num_envs": 8, "rollout_steps": 256},
            {
                "num_envs": 16,
                "rollout_steps": 128,
                "normalize_observations": True,
                "normalize_rewards": True,
            },
            {"num_envs": 16, "rollout_steps": 128, "update_epochs": 4, "learning_rate": 1e-3},
        ],
        "FrozenLake-v1": [
            {},
            {"num_envs": 16, "train_every": 16},
            {"num_envs": 16, "train_every": 16, "learning_rate": 3e-4},
            {"num_envs": 32, "train_every": 32, "gamma": 0.995},
        ],
    }
    if env_id == "CartPole-v1":
        base = replace(base, total_timesteps=524288)
    return [replace(base, **variant) for variant in variants[env_id]]


def native_candidates(env_id, device):
    base = replace(
        target_preset(env_id, device=device, seed=101),
        num_envs=4096,
        total_timesteps=8388608,
        rollout_steps=64,
        update_epochs=4,
        num_minibatches=16,
        learning_rate=3e-4,
        buffer_size=1048576,
        batch_size=1024,
        learning_starts=65536,
        train_every=4096,
        gradient_steps=8,
        log_every=262144,
    )
    variants = {
        "CartPole-v1": [{}, {"learning_rate": 1e-3}, {"update_epochs": 8}, {"num_minibatches": 32}],
        "MountainCar-v0": [
            {},
            {"learning_rate": 1e-3},
            {"gradient_steps": 16},
            {"exploration_hold_steps": 32, "learning_rate": 1e-3},
        ],
        "MountainCarContinuous-v0": [
            {},
            {"learning_rate": 1e-3},
            {"update_epochs": 8},
            {"rollout_steps": 32, "learning_rate": 1e-3},
        ],
        "LunarLander-v3": [
            {},
            {"learning_rate": 1e-3},
            {"update_epochs": 8},
            {"rollout_steps": 64, "learning_rate": 1e-3},
        ],
        "FrozenLake-v1": [
            {},
            {"learning_rate": 1e-3},
            {"gradient_steps": 16},
            {"batch_size": 2048, "learning_rate": 1e-3},
        ],
    }
    if env_id == "CartPole-v1":
        base = replace(base, total_timesteps=4194304, rollout_steps=16, num_minibatches=8)
    if env_id == "MountainCar-v0":
        base = replace(base, total_timesteps=16777216, gamma=0.995)
    if env_id == "LunarLander-v3":
        base = replace(
            base,
            total_timesteps=16777216,
            rollout_steps=128,
            normalize_observations=True,
            normalize_rewards=True,
        )
    if env_id == "FrozenLake-v1":
        base = replace(base, total_timesteps=4194304)
    return [replace(base, **variant) for variant in variants[env_id]]


def evaluation_config(config):
    interval = 65536
    if config.backend == "gymtosha":
        interval = {
            "CartPole-v1": 65536,
            "MountainCar-v0": 262144,
            "MountainCarContinuous-v0": 262144,
            "LunarLander-v3": 524288,
            "FrozenLake-v1": 65536,
        }[config.env_id]
    return EvaluationConfig(
        seed=900000,
        interval=interval,
        episodes=1000 if config.env_id.startswith("FrozenLake") else 100,
    )


def rank(trials):
    hits = [
        t["result"]["evaluation"]["time_to_target_seconds"]
        for t in trials
        if t["result"]["evaluation"]["status"] == "reached"
    ]
    score = median(t["result"]["evaluation"]["checkpoints"][-1]["mean_return"] for t in trials)
    return (len(trials) - len(hits), median(hits) if len(hits) == len(trials) else -score)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend", choices=["gymnasium-sync", "gymtosha"], default="gymnasium-sync"
    )
    parser.add_argument("--env-ids", nargs="+", default=ENVIRONMENTS)
    parser.add_argument("--output", type=Path, default=Path("outputs/tuning"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    path = args.output / "search.json"
    trials = json.loads(path.read_text()) if args.resume and path.exists() else []
    selected_path = args.output / "selected.json"
    selected = (
        json.loads(selected_path.read_text()) if args.resume and selected_path.exists() else {}
    )
    for env_id in args.env_ids:
        configs = candidates(env_id, args.device, args.backend)
        evaluation = evaluation_config(configs[0])
        for seed in (101, 202):
            ids = list(range(len(configs)))
            if seed == 202:
                ids = sorted(
                    ids,
                    key=lambda i: rank(
                        [
                            t
                            for t in trials
                            if t["env_id"] == env_id and t["candidate"] == i and t["seed"] == 101
                        ]
                    ),
                )[:2]
            for index in ids:
                config = replace(configs[index], seed=seed)
                old = [
                    t
                    for t in trials
                    if t["env_id"] == env_id and t["candidate"] == index and t["seed"] == seed
                ]
                if old:
                    if old[0]["result"]["config"] != asdict(config) or old[0]["result"][
                        "evaluation"
                    ]["config"] != asdict(evaluation):
                        raise ValueError("resume tuning configuration changed")
                    continue

                def progress(value, env_id=env_id, index=index, seed=seed):
                    point = value["checkpoints"][-1]
                    _write(
                        {"env_id": env_id, "candidate": index, "seed": seed, "evaluation": value},
                        args.output / "progress.json",
                    )
                    print(
                        f"{env_id} candidate={index} seed={seed} "
                        f"step={point['timesteps']} score={point['mean_return']:.3f} "
                        f"wall={point['wall_seconds']:.1f}s",
                        flush=True,
                    )

                result = train(config, evaluation=evaluation, on_progress=progress)
                trials.append(
                    {"env_id": env_id, "candidate": index, "seed": seed, "result": result}
                )
                _write(trials, path)
        confirmed = [
            i
            for i in range(len(configs))
            if len([t for t in trials if t["env_id"] == env_id and t["candidate"] == i]) == 2
        ]
        winner = min(
            confirmed,
            key=lambda i: rank(
                [t for t in trials if t["env_id"] == env_id and t["candidate"] == i]
            ),
        )
        winner_trials = [t for t in trials if t["env_id"] == env_id and t["candidate"] == winner]
        selected[env_id] = {
            "candidate": winner,
            "config": asdict(configs[winner]),
            "evaluation_interval": evaluation.interval,
            "confirmed_target": all(
                t["result"]["evaluation"]["status"] == "reached" for t in winner_trials
            ),
            "selection_rank": rank(winner_trials),
        }
        _write(selected, args.output / "selected.json")
        print("SELECTED", env_id, winner, selected[env_id]["confirmed_target"], flush=True)


if __name__ == "__main__":
    main()
