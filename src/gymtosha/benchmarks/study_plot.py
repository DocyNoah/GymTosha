"""Plots and an offline interactive index of the CartPole study."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gymtosha.benchmarks.plot import _pyplot


def plot_parameter(phase, path):
    plt = _pyplot()
    candidates = phase["candidates"]
    seed_count = candidates[0]["summary"].get(
        "expected_seeds", candidates[0]["summary"]["attempted_seeds"]
    )
    positions = np.arange(len(candidates))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for index, candidate in enumerate(candidates):
        values = [value for value in candidate.get("seed_rewards", []) if value is not None]
        axes[0].scatter(np.full(len(values), index), values, s=13, alpha=0.35, color="#327d63")
    for axis, metric, deviation, title in (
        (axes[0], "mean_training_return", "std_training_return", "Mean training episode return"),
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
    axes[1].bar(
        positions,
        [candidate["summary"]["target_reached"] for candidate in candidates],
        color="#327d63",
    )
    axes[1].set_ylim(0, seed_count)
    axes[1].set_title(f"Stable target reached / {seed_count} seeds")
    for axis in axes:
        axis.set_xticks(
            positions, [str(candidate["value"]) for candidate in candidates], rotation=30
        )
        axis.set_xlabel(phase["parameter"])
    fig.suptitle(f"CartPole PPO · N={phase['num_envs']:,} · mean ± sample SD · {seed_count} seeds")
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
        seed_counts = sorted(
            {
                candidate["summary"].get("expected_seeds", candidate["summary"]["attempted_seeds"])
                for phase in records
                for candidate in phase["candidates"]
            }
        )
        counts_label = "/".join(map(str, seed_counts))
        ax.set_title(f"Mean training episode return · {counts_label} seeds per measured cell")
        fig.colorbar(plotted, ax=ax)
        fig.tight_layout()
        fig.savefig(Path(output) / f"heatmap-{parameter}.png", dpi=150)
        plt.close(fig)


def render_dashboard(output, *, figures=True):
    root = Path(output)
    phases = [json.loads(path.read_text()) for path in sorted(root.glob("search/*/*.json"))]
    comparisons = [json.loads(path.read_text()) for path in sorted(root.glob("comparison/*.json"))]
    best = best_backend_conditions(comparisons)
    if figures:
        plot_heatmaps(phases, root)
        plot_comparisons(comparisons, root / "comparison.png")
        plot_best(best, root)
        plot_best_learning_curves(best, root / "best-learning-curves.png")
    (root / "best-backends.json").write_text(json.dumps(best, indent=2) + "\n")
    selected_path = root / "selected.json"
    if figures and selected_path.exists():
        plot_selected_trends(json.loads(selected_path.read_text()), root)
    export_tables(phases, comparisons, root)
    view_phases = [
        {
            "num_envs": phase["num_envs"],
            "parameter": phase["parameter"],
            "selected_value": phase["selected_value"],
            "candidates": [
                {"value": candidate["value"], "summary": candidate["summary"]}
                for candidate in phase["candidates"]
            ],
        }
        for phase in phases
    ]
    progress_path = root / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
    data = json.dumps(
        {
            "phases": view_phases,
            "has_comparisons": bool(comparisons),
            "progress": progress,
            "has_trends": selected_path.exists(),
            "has_best": bool(best),
            "has_curves": (root / "best-learning-curves.png").exists(),
            "tables": sorted(path.name for path in root.glob("*.csv")),
        }
    ).replace("</", "<\\/")
    page = Path(__file__).with_name("study.html").read_text().replace("__DATA__", data)
    (root / "index.html").write_text(page)
    manifest = {
        "completed_parameter_searches": len(phases),
        "completed_comparisons": len(comparisons),
        "counts": sorted({phase["num_envs"] for phase in phases}),
    }
    (root / "dashboard.json").write_text(json.dumps(manifest, indent=2) + "\n")


def best_backend_conditions(comparisons):
    groups = defaultdict(list)
    for record in comparisons:
        summary = record["summary"]
        if not summary["complete"] or summary["mean_training_return"] is None:
            continue
        config = record["config"]
        groups[(config["backend"], config["device"], config["compile_env"])].append(record)
    selected = []
    for records in groups.values():
        solved = [
            record for record in records if record["summary"]["mean_target_seconds"] is not None
        ]
        if solved:
            best = min(
                solved,
                key=lambda record: (
                    record["summary"]["mean_target_seconds"],
                    record["summary"]["std_target_seconds"],
                ),
            )
            basis = "stable_target_time"
        else:
            best = min(
                records,
                key=lambda record: (
                    -record["summary"]["mean_training_return"],
                    record["summary"]["std_training_return"],
                    record["summary"]["mean_wall_seconds"],
                ),
            )
            basis = "training_reward_only_no_fully_stable_size"
        selected.append({**best, "selection_basis": basis})
    return selected


def plot_best(best, output):
    if not best:
        return
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    labels = []
    for row in best:
        config = row["config"]
        compiled = " compiled" if config["compile_env"] else ""
        labels.append(
            f"{config['backend']}\n{config['device']}{compiled}\nN={config['num_envs']:,}"
        )
    for axis, metric, deviation, title in (
        (axes[0], "mean_training_return", "std_training_return", "Training episode return"),
        (axes[1], "mean_wall_seconds", "std_wall_seconds", "Full fixed-length wall time (s)"),
        (
            axes[2],
            "mean_target_seconds",
            "std_target_seconds",
            "Stable target time (s), 20/20 reached",
        ),
    ):
        values = [row["summary"][metric] for row in best]
        errors = [row["summary"][deviation] for row in best]
        axis.errorbar(
            range(len(best)),
            [np.nan if v is None else v for v in values],
            yerr=[0 if v is None else v for v in errors],
            fmt="o",
            capsize=5,
        )
        axis.set_xticks(range(len(best)), labels, rotation=25, ha="right", fontsize=7)
        axis.set_title(title)
    fig.suptitle("Best environment count per backend · mean ± sample SD · 20 seeds")
    fig.tight_layout()
    fig.savefig(Path(output) / "best-backends.png", dpi=160)
    plt.close(fig)


def plot_selected_trends(selected, output):
    if not selected:
        return
    from gymtosha.benchmarks.study import PARAMETERS

    plt = _pyplot()
    counts = sorted(map(int, selected))
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    for axis, parameter in zip(axes.flat, PARAMETERS, strict=True):
        values = [selected[str(count)]["config"][parameter] for count in counts]
        axis.plot(counts, values, marker="o")
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Environments")
        axis.set_title(parameter)
        if min(values) > 0 and max(values) / min(values) > 10:
            axis.set_yscale("log")
    fig.suptitle("Selected hyperparameters across environment counts")
    fig.tight_layout()
    fig.savefig(Path(output) / "selected-trends.png", dpi=150)
    plt.close(fig)


def plot_best_learning_curves(best, path):
    series = [row for row in best if row.get("seed_results")]
    if not series:
        return
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for row in series:
        curves = [seed["evaluation_curve"] for seed in row["seed_results"]]
        returns = np.array([[point["mean_return"] for point in curve] for curve in curves])
        elapsed = np.array([[point["wall_seconds"] for point in curve] for curve in curves])
        timesteps = [point["timesteps"] for point in curves[0]]
        mean = returns.mean(axis=0)
        deviation = returns.std(axis=0, ddof=1)
        config = row["config"]
        compiled = " compiled" if config["compile_env"] else ""
        label = f"{config['backend']} / {config['device']}{compiled} / N={config['num_envs']:,}"
        for axis, x in zip(axes, (timesteps, elapsed.mean(axis=0)), strict=True):
            line = axis.plot(x, mean, label=label)[0]
            axis.fill_between(
                x, mean - deviation, mean + deviation, alpha=0.12, color=line.get_color()
            )
    for axis in axes:
        axis.axhline(475, color="#666", linestyle="--", linewidth=1)
        axis.set_ylabel("Evaluation episode return")
    axes[0].set_xlabel("Sampling slots")
    axes[1].set_xlabel("Mean wall time at each checkpoint (s)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Best environment count per backend · mean ± sample SD · 20 seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export_tables(phases, comparisons, root):
    import csv

    searches = [
        {
            "num_envs": phase["num_envs"],
            "parameter": phase["parameter"],
            "value": candidate["value"],
            "selected": candidate["value"] == phase["selected_value"],
            **candidate["summary"],
        }
        for phase in phases
        for candidate in phase["candidates"]
    ]
    results = [{**row["config"], **row["summary"]} for row in comparisons]
    search_seeds = [
        {
            "num_envs": phase["num_envs"],
            "parameter": phase["parameter"],
            "value": candidate["value"],
            "seed": seed,
            "trial_id": trial_id,
            "mean_training_episode_return": reward,
        }
        for phase in phases
        for candidate in phase["candidates"]
        for seed, trial_id, reward in zip(
            phase.get("seeds", range(1, candidate["summary"]["attempted_seeds"] + 1)),
            candidate.get("trial_ids", []),
            candidate.get("seed_rewards", []),
            strict="trial_ids" in candidate,
        )
    ]
    comparison_seeds = [
        {
            "condition": row["name"],
            **{key: value for key, value in seed.items() if key != "evaluation_curve"},
        }
        for row in comparisons
        for seed in row.get("seed_results", [])
    ]
    for name, rows in [
        ("search-summary.csv", searches),
        ("search-seeds.csv", search_seeds),
        ("comparison-summary.csv", results),
        ("comparison-seeds.csv", comparison_seeds),
    ]:
        if not rows:
            continue
        with (Path(root) / name).open("w", newline="") as stream:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
