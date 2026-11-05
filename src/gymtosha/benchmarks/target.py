"""Prespecified target benchmarks and matched-seed summaries."""

from collections import defaultdict
from dataclasses import replace
from statistics import median

from gymtosha.benchmarks.train import TrainConfig


def target_preset(env_id, *, backend="gymtosha", device="cuda", seed=1, algorithm=None):
    algorithms = {
        "CartPole-v1": "ppo",
        "MountainCar-v0": "dqn",
        "MountainCarContinuous-v0": "ppo",
        "LunarLander-v3": "ppo",
        "FrozenLake-v1": "dqn",
    }
    if env_id not in algorithms:
        raise ValueError(f"no target preset for {env_id}")
    config = TrainConfig(
        env_id=env_id,
        algorithm=algorithm or algorithms[env_id],
        backend=backend,
        device=device,
        seed=seed,
        compile_env=backend == "gymtosha",
        num_envs=32,
        total_timesteps=2097152,
        rollout_steps=64,
        update_epochs=10,
        num_minibatches=4,
        learning_rate=3e-4,
        learning_starts=10000,
        train_every=32,
        log_every=65536,
        exploration_fraction=0.25,
    )
    if env_id == "CartPole-v1":
        config = replace(
            config, num_envs=16, rollout_steps=128, total_timesteps=1048576, train_every=16
        )
    if env_id == "MountainCarContinuous-v0":
        config = replace(config, entropy_coef=0.0)
    if env_id == "LunarLander-v3":
        config = replace(config, rollout_steps=128)
    if env_id == "FrozenLake-v1":
        config = replace(
            config, num_envs=64, train_every=64, learning_rate=1e-3, total_timesteps=1048576
        )
    return config


def validate_target_comparisons(records):
    settings = {}
    seen = set()
    for row in records:
        config = row["config"]
        key = (config["env_id"], config["algorithm"], config["device"])
        identity = (*key, config["backend"], config["seed"])
        if identity in seen:
            raise ValueError(f"duplicate target trial {identity}")
        seen.add(identity)
        protocol = {k: v for k, v in config.items() if k not in ("backend", "compile_env", "seed")}
        protocol["evaluation"] = row["evaluation"]["config"]
        if key in settings and settings[key] != protocol:
            raise ValueError(f"incompatible target comparison for {key}")
        settings[key] = protocol


def target_summary(records):
    validate_target_comparisons(records)
    groups = defaultdict(lambda: defaultdict(dict))
    for row in records:
        config = row["config"]
        groups[(config["env_id"], config["algorithm"], config["device"])][config["backend"]][
            config["seed"]
        ] = row
    result = []
    for (env_id, algorithm, device), backends in groups.items():
        baseline, native = backends.get("gymnasium-sync", {}), backends.get("gymtosha", {})
        seeds = set(baseline) | set(native)
        baseline_hits = [
            r["evaluation"]["time_to_target_seconds"]
            for r in baseline.values()
            if r["evaluation"]["status"] == "reached"
        ]
        native_hits = [
            r["evaluation"]["time_to_target_seconds"]
            for r in native.values()
            if r["evaluation"]["status"] == "reached"
        ]
        complete = bool(seeds) and len(baseline_hits) == len(native_hits) == len(seeds)
        paired_ratios = (
            [
                baseline[s]["evaluation"]["time_to_target_seconds"]
                / native[s]["evaluation"]["time_to_target_seconds"]
                for s in sorted(seeds)
            ]
            if complete
            else []
        )
        result.append(
            {
                "env_id": env_id,
                "algorithm": algorithm,
                "device": device,
                "seeds": len(seeds),
                "gymnasium_reached": len(baseline_hits),
                "gymtosha_reached": len(native_hits),
                "gymnasium_median_seconds": median(baseline_hits)
                if len(baseline_hits) == len(seeds)
                else None,
                "gymtosha_median_seconds": median(native_hits)
                if len(native_hits) == len(seeds)
                else None,
                "speedup": median(paired_ratios) if complete else None,
            }
        )
    return result
