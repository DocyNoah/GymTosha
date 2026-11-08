"""Compare CartPole backends at matched budgets with a tuned or fixed rollout horizon."""

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

DESIGN = "fixed_experience_budget_v2"
ENV_COUNTS = (32, 256, 512, 1024, 4096, 8192, 16384)
HORIZONS = (16, 32, 64, 128, 256, 512)
MIN_ROLLOUTS = 8
TUNING_TIMESTEPS = 4_194_304
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


def final_budget(tuning, horizon):
    if horizon not in tuning.horizons:
        raise ValueError("final horizon must be a registered tuning candidate")
    interval = lcm(*(n * horizon for n in ENV_COUNTS))
    intervals = max(MIN_ROLLOUTS, (tuning.total_timesteps + interval - 1) // interval)
    return Budget(ENV_COUNTS, (horizon,), intervals * interval, tuning.minibatch)


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
    parser.add_argument("--tune-timesteps", type=int, default=TUNING_TIMESTEPS)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--parallel-trials", type=int, choices=range(1, 33), default=16)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--compare-only", action="store_true")
    selection.add_argument(
        "--fixed-horizon",
        type=int,
        choices=HORIZONS,
        help="skip tuning and use this rollout length",
    )
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    tuning = Budget((args.tune_envs,), HORIZONS, args.tune_timesteps, args.minibatch_size)
    final_horizons = (args.fixed_horizon,) if args.fixed_horizon is not None else tuning.horizons
    if any(n * t % tuning.minibatch for n in ENV_COUNTS for t in final_horizons):
        parser.error("minibatch must divide every possible final rollout batch")
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
            "counts": list(ENV_COUNTS),
            "horizons": list(tuning.horizons),
            "tune_timesteps": tuning.total_timesteps,
            "minibatch": tuning.minibatch,
            "tune_evaluation_interval": tuning.evaluation_interval,
            "baseline": asdict(base),
        }
        if args.fixed_horizon is not None:
            metadata["fixed_horizon"] = args.fixed_horizon
        if protocol.exists() and json.loads(protocol.read_text()) != metadata:
            raise ValueError("use a new output directory for a different fixed-budget protocol")
        if args.compare_only:
            selected = json.loads(args.presets.read_text())
            horizon = selected[str(ENV_COUNTS[0])]["config"]["rollout_steps"]
            final = final_budget(tuning, horizon)
            if selected != expected_presets(base, final, horizon):
                raise ValueError("presets do not match the fixed-budget protocol")
            _write(metadata, protocol)
        else:
            _write(metadata, protocol)
            horizon = (
                args.fixed_horizon if args.fixed_horizon is not None else tune(base, tuning, args)
            )
            final = final_budget(tuning, horizon)
            selected = expected_presets(base, final, horizon)
            _write(selected, args.presets)
        _write(
            {
                "horizon": horizon,
                "total_timesteps": final.total_timesteps,
                "evaluation_interval": final.evaluation_interval,
                "minibatch": final.minibatch,
                "rollouts": {n: final.total_timesteps // (n * horizon) for n in final.counts},
            },
            args.output / "final_plan.json",
        )
        _write(verify_selected(selected, args.output, args.devices), args.capacity)
        compare(args)


if __name__ == "__main__":
    main()
