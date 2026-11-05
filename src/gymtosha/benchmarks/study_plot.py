"""Plots and an offline interactive index of the CartPole study."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gymtosha.benchmarks.plot import _pyplot


def plot_parameter(phase, path):
    plt = _pyplot()
    candidates = phase["candidates"]
    positions = np.arange(len(candidates))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for index, candidate in enumerate(candidates):
        values = [value for value in candidate.get("seed_rewards", []) if value is not None]
        axes[0].scatter(np.full(len(values), index), values, s=13, alpha=0.35, color="#327d63")
    for axis, metric, deviation, title in (
        (axes[0], "mean_training_return", "std_training_return", "Mean training episode return"),
        (axes[1], "mean_wall_seconds", "std_wall_seconds", "Full training wall time (s)"),
    ):
        values = [candidate["summary"][metric] for candidate in candidates]
        errors = [candidate["summary"][deviation] for candidate in candidates]
        axis.errorbar(
            positions,
            [np.nan if v is None else v for v in values],
            yerr=[0 if v is None else v for v in errors],
            fmt="o-",
            capsize=4,
        )
        axis.set_title(title)
    axes[2].bar(
        positions,
        [candidate["summary"]["target_reached"] for candidate in candidates],
        color="#327d63",
    )
    axes[2].set_ylim(0, 20)
    axes[2].set_title("Stable target reached / 20 seeds")
    for axis in axes:
        axis.set_xticks(
            positions, [str(candidate["value"]) for candidate in candidates], rotation=30
        )
        axis.set_xlabel(phase["parameter"])
    fig.suptitle(f"CartPole PPO · N={phase['num_envs']:,} · mean ± sample SD · 20 seeds")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparisons(comparisons, path):
    if not comparisons:
        return
    plt = _pyplot()
    groups = defaultdict(list)
    for record in comparisons:
        config = record["config"]
        label = f"{config['backend']} / {config['device']}" + (
            " compiled" if config["compile_env"] else ""
        )
        groups[label].append(record)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics = [
        ("mean_training_return", "std_training_return", "Training episode return"),
        ("mean_wall_seconds", "std_wall_seconds", "Fixed-length wall time (s)"),
        ("mean_target_seconds", "std_target_seconds", "Stable target time (s), 20/20 reached"),
    ]
    for label, records in groups.items():
        records.sort(key=lambda record: record["config"]["num_envs"])
        counts = [record["config"]["num_envs"] for record in records]
        for axis, (metric, deviation, title) in zip(axes, metrics, strict=True):
            values = [record["summary"][metric] for record in records]
            errors = [record["summary"][deviation] for record in records]
            axis.errorbar(
                counts,
                [np.nan if v is None else v for v in values],
                yerr=[0 if v is None else v for v in errors],
                marker="o",
                capsize=3,
                label=label,
            )
            axis.set_xscale("log", base=2)
            axis.set_xlabel("Environments")
            axis.set_title(title)
    axes[0].legend(fontsize=7)
    fig.suptitle("CartPole PPO · mean ± sample SD across 20 independent seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_heatmaps(phases, output):
    if len({phase["num_envs"] for phase in phases}) < 2:
        return
    plt = _pyplot()
    grouped = defaultdict(list)
    for phase in phases:
        grouped[phase["parameter"]].append(phase)
    for parameter, records in grouped.items():
        counts = sorted({row["num_envs"] for row in records})
        values = sorted(
            {
                candidate["value"] / phase["num_envs"]
                if parameter == "ppo_minibatch_size"
                else candidate["value"]
                for phase in records
                for candidate in phase["candidates"]
            }
        )
        grid = np.full((len(values), len(counts)), np.nan)
        for phase in records:
            for candidate in phase["candidates"]:
                value = (
                    candidate["value"] / phase["num_envs"]
                    if parameter == "ppo_minibatch_size"
                    else candidate["value"]
                )
                result = candidate["summary"]["mean_training_return"]
                if result is not None:
                    grid[values.index(value), counts.index(phase["num_envs"])] = result
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(counts)), max(3, 0.45 * len(values))))
        plotted = ax.imshow(
            np.ma.masked_invalid(grid), aspect="auto", cmap="viridis", vmin=0, vmax=500
        )
        ax.set_xticks(range(len(counts)), [f"{count:,}" for count in counts], rotation=35)
        ax.set_yticks(range(len(values)), [str(value) for value in values])
        ax.set_xlabel("Environments")
        ax.set_ylabel(parameter + (" / num_envs" if parameter == "ppo_minibatch_size" else ""))
        ax.set_title("Mean training episode return · 20 seeds per measured cell")
        fig.colorbar(plotted, ax=ax)
        fig.tight_layout()
        fig.savefig(Path(output) / f"heatmap-{parameter}.png", dpi=150)
        plt.close(fig)


def render_dashboard(output):
    root = Path(output)
    phases = [json.loads(path.read_text()) for path in sorted(root.glob("search/*/*.json"))]
    comparisons = [json.loads(path.read_text()) for path in sorted(root.glob("comparison/*.json"))]
    plot_heatmaps(phases, root)
    plot_comparisons(comparisons, root / "comparison.png")
    data = json.dumps({"phases": phases, "comparisons": comparisons}).replace("</", "<\\/")
    page = Path(__file__).with_name("study.html").read_text().replace("__DATA__", data)
    (root / "index.html").write_text(page)
    manifest = {
        "completed_parameter_searches": len(phases),
        "completed_comparisons": len(comparisons),
        "counts": sorted({phase["num_envs"] for phase in phases}),
    }
    (root / "dashboard.json").write_text(json.dumps(manifest, indent=2) + "\n")
