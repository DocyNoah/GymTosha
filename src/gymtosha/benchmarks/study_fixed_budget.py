"""Retune CartPole's rollout horizon and compare backends at one experience budget."""

import argparse
import json
import signal
from dataclasses import asdict, dataclass, replace
from math import lcm
from pathlib import Path

from gymtosha.benchmarks.cli import _write
from gymtosha.benchmarks.study import (
    SCORE_METRIC,
    SEEDS,
    collect_configurations,
    compare,
    runtime_fingerprint,
    source_fingerprint,
    study_lock,
)
from gymtosha.benchmarks.study_grouped import load_baseline, verify_selected
from gymtosha.benchmarks.study_plot import plot_parameter, render_dashboard

DESIGN = "fixed_experience_budget_v1"
ENV_COUNTS = (32, 256, 512, 1024, 4096, 8192, 16384)
HORIZONS = (16, 32, 64, 128, 256, 512)
MIN_ROLLOUTS = 16
TARGET = 475


@dataclass(frozen=True)
class Budget:
    counts: tuple[int, ...]
    horizons: tuple[int, ...]
    total_timesteps: int
    minibatch: int

    def __post_init__(self):
        if (
            not self.counts
            or not self.horizons
            or min(*self.counts, *self.horizons, self.total_timesteps) < 1
            or self.minibatch < 2
        ):
            raise ValueError("budget dimensions must be positive and minibatch at least two")
        if self.total_timesteps % self.evaluation_interval:
            raise ValueError("budget must be divisible by the common evaluation interval")
        if self.total_timesteps // self.evaluation_interval < MIN_ROLLOUTS:
            raise ValueError(f"budget must allow at least {MIN_ROLLOUTS} common rollout intervals")
        if any(n * t % self.minibatch for n in self.counts for t in self.horizons):
            raise ValueError("minibatch size must divide all rollout batches")

    @property
    def evaluation_interval(self):
        return lcm(*(n * t for n in self.counts for t in self.horizons))

    def configure(self, base, count, horizon):
        if count not in self.counts or horizon not in self.horizons:
            raise ValueError("configuration lies outside the declared counts or horizons")
        config = replace(
            base,
            num_envs=count,
            rollout_steps=horizon,
            num_minibatches=count * horizon // self.minibatch,
            ppo_minibatch_size=self.minibatch,
            total_timesteps=self.total_timesteps,
            native_same_step=True,
            anneal_by_optimizer_step=True,
        )
        config.validate()
        return config


def select_horizon(candidates):
    completed = [c for c in candidates if c["summary"]["mean_final_eval_return"] is not None]
    if not completed:
        raise RuntimeError("no horizon completed the fixed-budget tuning seeds")
    stable = [
        c
        for c in completed
        if c["summary"]["target_reached"] == len(SEEDS)
        and c["summary"]["mean_final_eval_return"] >= TARGET
    ]
    winner = min(
        stable or completed,
        key=lambda c: (
            -c["summary"]["mean_final_eval_return"],
            -c["summary"]["target_reached"],
            c["summary"]["std_final_eval_return"],
            c["summary"]["mean_target_transitions"] or float("inf"),
            c["value"],
        ),
    )
    return winner["value"], "stable_final_score" if stable else "final_score_no_stable_horizon"


def expected_presets(base, budget, horizon):
    return {
        str(count): {
            "config": asdict(budget.configure(base, count, horizon)),
            "batch_rule": DESIGN,
            "scaling_design": DESIGN,
            "selection_basis": "common_horizon",
            "evaluation_interval": budget.evaluation_interval,
            "score_metric": SCORE_METRIC,
            "seeds": list(SEEDS),
        }
        for count in budget.counts
    }


def tune(base, budget, args):
    path = args.output / "search" / str(args.tune_envs) / "rollout_steps.json"
    if path.exists():
        return json.loads(path.read_text())["selected_value"]
    configs = {
        horizon: budget.configure(
            replace(base, device=args.tune_device, compile_env=False), args.tune_envs, horizon
        )
        for horizon in budget.horizons
    }
    candidates = collect_configurations(
        configs,
        "rollout_steps",
        args.output,
        parallelism=args.parallel_trials,
        evaluation_interval=budget.evaluation_interval,
    )
    horizon, basis = select_horizon(candidates)
    phase = {
        "num_envs": args.tune_envs,
        "parameter": "rollout_steps",
        "starting_config": asdict(next(iter(configs.values()))),
        "candidates": candidates,
        "selected_value": horizon,
        "selection_basis": basis,
    }
    _write(phase, path)
    plot_parameter(phase, path.with_suffix(".png"))
    render_dashboard(args.output)
    return horizon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--tune-device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--tune-envs", type=int, choices=ENV_COUNTS, default=1024)
    parser.add_argument("--total-timesteps", type=int, default=134_217_728)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--parallel-trials", type=int, choices=range(1, 33), default=16)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    budget = Budget(ENV_COUNTS, HORIZONS, args.total_timesteps, args.minibatch_size)
    args.presets = args.output / "selected.json"
    args.capacity = args.output / "capacity-selected" / f"summary-{'-'.join(args.devices)}.json"
    with study_lock(args.output):
        base, _ = load_baseline(args.baseline)
        protocol = args.output / "protocol.json"
        metadata = {
            "design": DESIGN,
            "source": source_fingerprint(),
            "runtime": runtime_fingerprint(),
            "devices": args.devices,
            "tune_envs": args.tune_envs,
            "tune_device": args.tune_device,
            "compare_only": args.compare_only,
            "counts": list(budget.counts),
            "horizons": list(budget.horizons),
            "total_timesteps": budget.total_timesteps,
            "minibatch": budget.minibatch,
            "evaluation_interval": budget.evaluation_interval,
            "baseline": asdict(base),
        }
        if protocol.exists() and json.loads(protocol.read_text()) != metadata:
            raise ValueError("use a new output directory for a different fixed-budget protocol")
        if args.compare_only:
            selected = json.loads(args.presets.read_text())
            horizon = selected[str(budget.counts[0])]["config"]["rollout_steps"]
            if selected != expected_presets(base, budget, horizon):
                raise ValueError("presets do not match the fixed-budget protocol")
            _write(metadata, protocol)
        else:
            _write(metadata, protocol)
            horizon = tune(base, budget, args)
            selected = expected_presets(base, budget, horizon)
            _write(selected, args.presets)
        _write(verify_selected(selected, args.output, args.devices), args.capacity)
        compare(args)


if __name__ == "__main__":
    main()
