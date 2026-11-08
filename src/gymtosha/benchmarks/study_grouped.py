"""Run the grouped CartPole PPO study on a single host."""

import argparse
import json
import signal
from dataclasses import asdict, replace
from pathlib import Path

from gymtosha.benchmarks.capacity import _isolated_probe
from gymtosha.benchmarks.cli import _write
from gymtosha.benchmarks.study import (
    ROLLOUT_UPDATES,
    SCORE_METRIC,
    SEEDS,
    collect_configurations,
    compare,
    run_trial,
    runtime_fingerprint,
    source_fingerprint,
    study_evaluation,
    study_lock,
    summarize_trials,
)
from gymtosha.benchmarks.study_plot import plot_parameter, render_dashboard
from gymtosha.benchmarks.train import TrainConfig

GROUPS = ((32,), (256, 512, 1024), (4096, 8192, 16384))
DESIGN = "grouped_fixed_batch_v1"
TARGET = 475
SCORE_TOLERANCE = 5


def group_config(base, count, batch, minibatch):
    if min(count, batch, minibatch) < 1 or batch % count or batch % minibatch:
        raise ValueError("group batch must be divisible by the environment count and minibatch")
    config = replace(
        base,
        num_envs=count,
        rollout_steps=batch // count,
        num_minibatches=batch // minibatch,
        ppo_minibatch_size=minibatch,
        total_timesteps=ROLLOUT_UPDATES * batch,
    )
    config.validate()
    return config


def qualified_candidates(phases, *, target):
    """Return candidates that solve at both endpoints and stay close to their best scores."""
    scores = [
        {
            row["value"]: row["summary"]["mean_final_eval_return"]
            for row in phase
            if row["summary"]["target_reached"] == len(SEEDS)
            and row["summary"]["mean_final_eval_return"] is not None
            and row["summary"]["mean_final_eval_return"] >= target
        }
        for phase in phases
    ]
    common = set.intersection(*(set(endpoint) for endpoint in scores))
    if not common:
        return []
    floors = [max(endpoint[name] for name in common) - SCORE_TOLERANCE for endpoint in scores]
    return sorted(
        (name for name in common if all(s[name] >= f for s, f in zip(scores, floors, strict=True))),
        key=lambda name: (-sum(endpoint[name] for endpoint in scores), name),
    )


def load_baseline(path):
    item = json.loads(path.read_text())["32"]
    config = TrainConfig(**item["config"])
    if (
        item["seeds"] != list(SEEDS)
        or item["score_metric"] != SCORE_METRIC
        or config.env_id != "CartPole-v1"
        or config.algorithm != "ppo"
        or config.num_envs != 32
        or config.gae_lambda != 0.99
        or config.hidden_sizes != [256, 256, 256]
        or config.total_timesteps != 32 * config.rollout_steps * ROLLOUT_UPDATES
    ):
        raise ValueError("reference must be the completed 10-seed CartPole 32-environment preset")
    config.validate()
    return config, item


def conditions(devices):
    if "cpu" in devices:
        for backend in ("gymnasium-async", "gymnasium-sync", "gymnasium-native"):
            yield backend, "cpu", False
    for device in devices:
        for compile_env in (False, True):
            yield "gymtosha", device, compile_env


def probe(config, directory, *, screening=False):
    name = f"{config.backend}-{config.device}-{int(config.compile_env)}-{config.num_envs}"
    path = directory / f"{name}.json"
    if screening:
        config = replace(
            config,
            rollout_steps=128,
            update_epochs=1,
            num_minibatches=4,
            ppo_minibatch_size=None,
        )
    config = replace(config, total_timesteps=config.num_envs * config.rollout_steps)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("config") == asdict(config):
            return saved
    request = directory / "requests" / f"{name}.json"
    _write(asdict(config), request)
    args = argparse.Namespace(
        backend=config.backend,
        device=config.device,
        compile=config.compile_env,
        output=directory,
        timeout=60,
        training_config=request,
    )
    result = _isolated_probe(args, config.num_envs)
    result["config"] = asdict(config)
    _write(result, path)
    return result


def screen(base, output, devices):
    directory = output / "capacity-screening"
    for backend, device, compiled in conditions(devices):
        counts = (32,) if backend == "gymnasium-async" else sum(GROUPS, ())
        for count in counts:
            probe(
                replace(base, backend=backend, device=device, compile_env=compiled, num_envs=count),
                directory,
                screening=True,
            )


def search_device(config, output, devices):
    probes = [
        (device, compiled, probe(replace(config, device=device, compile_env=compiled), output))
        for backend, device, compiled in conditions(devices)
        if backend == "gymtosha"
    ]
    passed = [(d, c, row) for d, c, row in probes if row["status"] == "passed"]
    if not passed:
        return None
    device, compiled, _ = min(passed, key=lambda item: item[2]["wall_seconds"])
    return device, compiled


def candidate_settings(group, *, fallback):
    # T at the largest endpoint is the shortest horizon in a matched-batch group.
    horizons = (32, 64, 128) if group[-1] == 1024 else (16, 32, 64)
    for horizon in horizons:
        batch = group[-1] * horizon
        for batches in (64, 128):
            label = f"TmaxN={horizon},K={batches}"
            yield (
                label,
                {
                    count: (count * horizon, count * horizon // batches)
                    if fallback
                    else (batch, batch // batches)
                    for count in group
                },
            )


def timed_contender(configs, output):
    summaries = [
        summarize_trials(
            [
                run_trial(replace(config, seed=seed), study_evaluation(config), output)
                for seed in SEEDS
            ]
        )
        for config in configs
    ]
    times = [row["mean_target_seconds"] for row in summaries]
    return sum(times) / len(times) if all(value is not None for value in times) else None


def tune_group(base, group, args, *, fallback=False):
    rule = "fixed_horizon" if fallback else "fixed_batch"
    output = args.output
    group_name = f"{group[0]}-{group[-1]}"
    path = output / "groups" / f"{group_name}-{rule}.json"
    if path.exists():
        return json.loads(path.read_text())["presets"]
    settings = dict(candidate_settings(group, fallback=fallback))
    endpoints = (group[0], group[-1])
    by_count, phases, modes = {}, {}, {}
    parameter = f"group_{rule}"
    for count in endpoints:
        configs = {
            name: group_config(base, count, *sizes[count]) for name, sizes in settings.items()
        }
        representative = list(configs.values())[len(configs) // 2]
        mode = search_device(
            representative, output / "search-device" / rule / str(count), args.search_devices
        )
        if mode is None:
            _write({"presets": {}, "failed_count": count, "reason": "device_probe"}, path)
            return {}
        device, compiled = modes[count] = mode
        configs = {
            name: replace(config, device=device, compile_env=compiled)
            for name, config in configs.items()
        }
        by_count[count] = configs
        phases[count] = collect_configurations(
            configs, parameter, output, parallelism=args.parallel_trials
        )
    finalists = qualified_candidates(list(phases.values()), target=TARGET)
    timing = {}
    if len(finalists) > 1:
        for name in finalists:
            timing[name] = timed_contender(
                [by_count[count][name] for count in endpoints], output / "isolated-tuning"
            )
        finalists.sort(key=lambda name: float("inf") if timing[name] is None else timing[name])
    selected = None
    midpoint = group[1]
    for name in finalists:
        config = group_config(base, midpoint, *settings[name][midpoint])
        mode = search_device(
            config, output / "search-device" / rule / name / str(midpoint), args.search_devices
        )
        if mode is None:
            continue
        device, compiled = mode
        config = replace(config, device=device, compile_env=compiled)
        checked = collect_configurations(
            {name: config}, parameter, output, parallelism=args.parallel_trials
        )[0]
        _write(checked, output / "midpoints" / rule / f"{midpoint}-{name}.json")
        if qualified_candidates([[checked]], target=TARGET):
            modes[midpoint] = mode
            selected = name
            break
    for count, candidates in phases.items():
        for row in candidates:
            row["scaling_config"] = asdict(by_count[count][row["value"]])
        phase_path = output / "search" / str(count) / f"{parameter}.json"
        phase = {
            "parameter": parameter,
            "num_envs": count,
            "candidates": candidates,
            "selected_value": selected,
            "starting_config": asdict(next(iter(by_count[count].values()))),
        }
        _write(phase, phase_path)
        plot_parameter(phase, phase_path.with_suffix(".png"))
    presets = {}
    if selected is not None:
        for count in group:
            device, compiled = modes[count]
            config = replace(
                group_config(base, count, *settings[selected][count]),
                device=device,
                compile_env=compiled,
            )
            presets[str(count)] = preset(config, group, rule)
    _write({"presets": presets, "selected": selected, "isolated_times": timing}, path)
    render_dashboard(output)
    return presets


def preset(config, group, rule):
    return {
        "config": asdict(config),
        "group": list(group),
        "batch_rule": rule,
        "scaling_design": DESIGN,
        "score_metric": SCORE_METRIC,
        "seeds": list(SEEDS),
        "selection_basis": "anchor_search"
        if config.num_envs in (group[0], group[-1])
        else "group_midpoint_validation",
    }


def verify_selected(selected, output, devices, *, cpu_limit=None):
    capacity = []
    for backend, device, compiled in conditions(devices):
        verified = []
        for value, row in selected.items():
            count = int(value)
            if backend == "gymnasium-async" and count != 32:
                continue
            if device == "cpu" and cpu_limit is not None and count > cpu_limit:
                continue
            config = replace(
                TrainConfig(**row["config"]), backend=backend, device=device, compile_env=compiled
            )
            if probe(config, output / "capacity-selected")["status"] == "passed":
                verified.append(count)
        capacity.append(
            {
                "backend": backend,
                "device": device,
                "compile_env": compiled,
                "maximum_verified": max(verified, default=None),
                "verified_counts": verified,
            }
        )
    return capacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--parallel-trials", type=int, choices=range(1, 33), default=8)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    args.search_devices = [device for device in args.devices if device in ("cpu", "cuda")]
    args.presets = args.output / "selected.json"
    args.capacity = args.output / "capacity-selected" / f"summary-{'-'.join(args.devices)}.json"
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    with study_lock(args.output):
        base, original = load_baseline(args.baseline)
        if args.compare_only:
            selected = json.loads(args.presets.read_text())
            if any(row.get("scaling_design") != DESIGN for row in selected.values()):
                raise ValueError("compare-only requires grouped presets")
        else:
            metadata = {
                "design": DESIGN,
                "source": source_fingerprint(),
                "runtime": runtime_fingerprint(),
                "devices": args.devices,
                "baseline": original,
                "min_target_reached": len(SEEDS),
                "min_final_eval_return": TARGET,
                "score_tolerance": SCORE_TOLERANCE,
            }
            protocol = args.output / "protocol.json"
            if protocol.exists() and json.loads(protocol.read_text()) != metadata:
                raise ValueError("use a new output directory for a different grouped protocol")
            _write(metadata, protocol)
            screen(base, args.output, args.devices)
            selected = {"32": preset(base, GROUPS[0], "fixed_batch")}
            for group in GROUPS[1:]:
                matches = tune_group(base, group, args)
                if not matches:
                    matches = tune_group(base, group, args, fallback=True)
                selected.update(matches)
                _write(selected, args.presets)
        _write(verify_selected(selected, args.output, args.devices), args.capacity)
        compare(args)


if __name__ == "__main__":
    main()
