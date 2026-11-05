"""CartPole PPO scaling at measured environment counts and predicted presets."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from gymtosha.benchmarks.cli import _write
from gymtosha.benchmarks.study import (
    PARAMETERS,
    ROLLOUT_UPDATES,
    SCORE_METRIC,
    SEEDS,
    collect_configurations,
    reference_config,
    run_trial,
    runtime_fingerprint,
    selection_key,
    source_fingerprint,
    study_evaluation,
    tune_count,
)
from gymtosha.benchmarks.train import TrainConfig

SCALING_ANCHORS = (32, 256, 1024, 4096)
SCALING_DESIGN = "joint_rollout_optimizer_v1"
DEVICE_PROBE_SEED = 9001
FIXED_PARAMETERS = {"gae_lambda": 0.99}


def minibatches(config):
    return config.num_envs * config.rollout_steps // config.ppo_minibatch_size


def scaling_config(base, rollout, batches, epochs, rate):
    samples = base.num_envs * rollout
    batches = max(1, min(batches, samples // 32))
    config = replace(
        base,
        rollout_steps=rollout,
        ppo_minibatch_size=samples // batches,
        update_epochs=epochs,
        learning_rate=rate,
        total_timesteps=samples * ROLLOUT_UPDATES,
    )
    config.validate()
    return config


def scaling_candidates(count, previous):
    base = reference_config(count, previous=previous)
    batches = minibatches(previous)
    center = 2 ** round(math.log2(previous.rollout_steps * math.sqrt(previous.num_envs / count)))
    candidates = {"reference": base}
    for horizon, factor in (("short", 0.5), ("mid", 1), ("long", 2)):
        rollout = max(16, min(512, int(center * factor)))
        for regime, groups, epochs, rate in (
            (
                "more",
                batches * 2,
                max(1, previous.update_epochs // 2),
                previous.learning_rate / math.sqrt(2),
            ),
            ("base", batches, previous.update_epochs, previous.learning_rate),
            ("fewer", batches // 2, previous.update_epochs, previous.learning_rate * math.sqrt(2)),
        ):
            config = scaling_config(base, rollout, groups, epochs, min(0.01, max(0.0001, rate)))
            if config not in candidates.values():
                candidates[f"{horizon}/{regime}"] = config
    return candidates


def protocol_metadata():
    return {
        "source": source_fingerprint(),
        "runtime": runtime_fingerprint(),
        "seeds": list(SEEDS),
        "score_metric": SCORE_METRIC,
        "scaling_design": SCALING_DESIGN,
    }


def scaled_presets(anchors, maximum):
    counts = sorted(anchors)
    metadata = protocol_metadata()
    presets = {}
    for exponent in range(5, int(math.log2(maximum)) + 1):
        count = 2**exponent
        lower = max(anchor for anchor in counts if anchor <= count)
        upper = next((anchor for anchor in counts if anchor >= count), lower)
        base = reference_config(count, previous=anchors[lower])
        if lower == count:
            config, basis = anchors[lower], "anchor_search"
        elif lower == upper:
            config, basis = base, "last_anchor_scaling"
        else:
            first, second = anchors[lower], anchors[upper]
            fraction = math.log2(count / lower) / math.log2(upper / lower)
            rollout = 2 ** round(
                math.log2(first.rollout_steps) * (1 - fraction)
                + math.log2(second.rollout_steps) * fraction
            )
            batches = 2 ** round(
                math.log2(minibatches(first)) * (1 - fraction)
                + math.log2(minibatches(second)) * fraction
            )
            rate = first.learning_rate ** (1 - fraction) * second.learning_rate**fraction
            epochs = round(first.update_epochs * (1 - fraction) + second.update_epochs * fraction)
            config, basis = (
                scaling_config(base, rollout, batches, epochs, rate),
                "log_interpolation",
            )
        presets[str(count)] = {
            **metadata,
            "selection_basis": basis,
            "anchor_counts": [lower] if lower == upper else [lower, upper],
            "config": asdict(config),
        }
    return presets


def choose_search_device(config, probe_config, accelerator, output):
    path = Path(output) / "device-probes" / f"{config.num_envs}.json"
    metadata = {**protocol_metadata(), "accelerator": accelerator}
    if path.exists():
        saved = json.loads(path.read_text())
        if any(saved.get(key) != value for key, value in metadata.items()):
            raise ValueError("saved device probe uses a different study protocol")
        selected = saved["selected"]
        return replace(config, device=selected["device"], compile_env=selected["compile_env"])

    trials = []
    for device in ("cpu", accelerator):
        for compiled in (False, True):
            probe = replace(
                probe_config, device=device, compile_env=compiled, seed=DEVICE_PROBE_SEED
            )
            record = run_trial(probe, study_evaluation(probe), output)
            row = {
                "device": device,
                "compile_env": compiled,
                "status": record["status"],
                "trial_id": record["trial_id"],
                "wall_seconds": record.get("result", {}).get("wall_seconds"),
                "error": record.get("error"),
            }
            trials.append(row)
            print(
                f"DEVICE N={config.num_envs} {device}/{compiled} {row['wall_seconds']} s",
                flush=True,
            )
    completed = [row for row in trials if row["status"] == "completed"]
    if not completed:
        raise RuntimeError(f"no device finished the N={config.num_envs} search probe")
    selected = min(completed, key=lambda row: row["wall_seconds"])
    _write(
        {
            **metadata,
            "config": asdict(probe_config),
            "seed": DEVICE_PROBE_SEED,
            "trials": trials,
            "selected": selected,
        },
        path,
    )
    return replace(config, device=selected["device"], compile_env=selected["compile_env"])


def tune_joint_count(config, previous, output, *, parallelism):
    from gymtosha.benchmarks.study_plot import plot_parameter, render_dashboard

    path = Path(output) / "search" / str(config.num_envs) / "joint_scaling.json"
    metadata = protocol_metadata()
    if path.exists():
        phase = json.loads(path.read_text())
        if phase["starting_config"] != asdict(config) or any(
            phase.get(key) != value for key, value in metadata.items()
        ):
            raise ValueError("saved joint search starts from a different configuration or protocol")
        return TrainConfig(**phase["selected_config"])
    configs = {
        name: replace(candidate, device=config.device, compile_env=config.compile_env)
        for name, candidate in scaling_candidates(config.num_envs, previous).items()
    }
    candidates = collect_configurations(configs, "joint_scaling", output, parallelism=parallelism)
    for candidate in candidates:
        chosen = configs[candidate["value"]]
        candidate["scaling_config"] = {
            "rollout_steps": chosen.rollout_steps,
            "ppo_minibatch_size": chosen.ppo_minibatch_size,
            "minibatches_per_epoch": minibatches(chosen),
            "update_epochs": chosen.update_epochs,
            "learning_rate": chosen.learning_rate,
        }
    winner = min(
        candidates,
        key=lambda candidate: (
            selection_key(candidate["summary"]),
            -candidate["summary"]["target_reached"],
            candidate["scaling_config"]["rollout_steps"],
            candidate["scaling_config"]["minibatches_per_epoch"]
            * candidate["scaling_config"]["update_epochs"],
        ),
    )
    if winner["summary"]["mean_final_eval_return"] is None:
        raise RuntimeError(f"no valid joint scaling candidate for N={config.num_envs}")
    selected = configs[winner["value"]]
    phase = {
        **metadata,
        "num_envs": config.num_envs,
        "parameter": "joint_scaling",
        "starting_config": asdict(config),
        "candidates": candidates,
        "selected_value": winner["value"],
        "selected_config": asdict(selected),
    }
    _write(phase, path)
    plot_parameter(phase, path.with_suffix(".png"))
    render_dashboard(output)
    return selected


def tune_anchors(args):
    from gymtosha.benchmarks.study_plot import render_dashboard

    maximum = args.max_envs
    if args.min_envs != 32 or maximum < 4096 or maximum & (maximum - 1):
        raise ValueError("anchor tuning starts at 32 and requires a power-of-two maximum >= 4096")
    path = Path(args.output) / "anchors.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    metadata = protocol_metadata()
    anchors = {}
    previous = None
    for count in SCALING_ANCHORS:
        if str(count) in saved:
            item = saved[str(count)]
            if any(item.get(key) != value for key, value in metadata.items()):
                raise ValueError("saved anchor uses a different study protocol")
            config = TrainConfig(**item["config"])
        else:
            base = reference_config(count, previous=previous)
            probe_config = base
            if previous is None:
                for parameter in PARAMETERS:
                    phase = Path(args.output) / "search/32" / f"{parameter}.json"
                    if not phase.exists():
                        break
                    probe_config = TrainConfig(**json.loads(phase.read_text())["selected_config"])
            base = choose_search_device(base, probe_config, args.device, args.output)
            if previous is None:
                config = tune_count(
                    base,
                    args.output,
                    parallelism=args.parallel_trials,
                    reuse_other_devices=True,
                    fixed_parameters=FIXED_PARAMETERS,
                )
            else:
                config = tune_joint_count(
                    base, previous, args.output, parallelism=args.parallel_trials
                )
            saved[str(count)] = {**metadata, "config": asdict(config)}
            _write(saved, path)
        anchors[count] = config
        previous = config
    _write(scaled_presets(anchors, maximum), Path(args.output) / "selected.json")
    render_dashboard(args.output)
