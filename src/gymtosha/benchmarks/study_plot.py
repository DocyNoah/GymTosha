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
        values = [value for value in candidate.get("seed_scores", []) if value is not None]
        axes[0].scatter(np.full(len(values), index), values, s=13, alpha=0.35, color="#327d63")
    for axis, metric, deviation, title in (
        (
            axes[0],
            "mean_final_eval_return",
            "std_final_eval_return",
            "Mean final-five evaluation return",
        ),
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
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    metrics = [
        ("mean_final_eval_return", "std_final_eval_return", "Final-five evaluation return"),
        ("mean_wall_seconds", "std_wall_seconds", "Fixed-length wall time (s)"),
        ("mean_target_seconds", "std_target_seconds", "Time to target (s), 20/20 reached"),
        (
            "mean_target_transitions",
            "std_target_transitions",
            "Transitions to target, 20/20 reached",
        ),
    ]
    for label, records in groups.items():
        records.sort(key=lambda record: record["config"]["num_envs"])
        counts = [record["config"]["num_envs"] for record in records]
        for axis, (metric, deviation, title) in zip(axes, metrics, strict=True):
            values = [record["summary"].get(metric) for record in records]
            errors = [record["summary"].get(deviation) for record in records]
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
                result = candidate["summary"]["mean_final_eval_return"]
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
        ax.set_title(f"Mean final-five evaluation return · {counts_label} seeds per measured cell")
        fig.colorbar(plotted, ax=ax)
        fig.tight_layout()
        fig.savefig(Path(output) / f"heatmap-{parameter}.png", dpi=150)
        plt.close(fig)


def plot_search_learning_curves(phase, trial_dir, path):
    series = []
    for candidate in phase["candidates"]:
        summary = candidate["summary"]
        paths = [trial_dir / f"{trial_id}.json" for trial_id in candidate.get("trial_ids", [])]
        if (
            summary["mean_final_eval_return"] is not None
            and len(paths) == summary["attempted_seeds"]
            and all(trial.exists() for trial in paths)
        ):
            series.append((candidate["value"], paths))
    if not series:
        return

    plt = _pyplot()
    fig, (training, evaluation) = plt.subplots(2, 1, sharex=True, figsize=(10, 7.2))
    for value, paths in series:
        windows, assessments = [], []
        for trial in paths:
            result = json.loads(trial.read_text())["result"]
            checkpoints = result["checkpoints"]
            points = []
            for start in range(0, len(checkpoints), 4):
                group = checkpoints[start : start + 4]
                episodes = sum(point["episodes"] for point in group)
                reward = sum(
                    point["episodes"] * point["mean_return"] for point in group if point["episodes"]
                )
                points.append((group[-1]["timesteps"], reward / episodes if episodes else np.nan))
            windows.append(points)
            assessments.append(
                [
                    (point.get("valid_transitions", point["timesteps"]), point["mean_return"])
                    for point in result["evaluation"]["checkpoints"]
                ]
            )
        selected = value == phase["selected_value"]
        label = f"{value} · selected" if selected else str(value)
        color = None
        for axis, points in ((training, windows), (evaluation, assessments)):
            points = np.array(points)
            x, scores = points[0, :, 0], points[:, :, 1]
            mean, deviation = scores.mean(axis=0), scores.std(axis=0, ddof=1)
            line = axis.plot(
                x,
                mean,
                label=label,
                color=color,
                marker="o",
                markersize=2.5,
                linewidth=2.5 if selected else 1.3,
            )[0]
            color = line.get_color()
            axis.fill_between(
                x, mean - deviation, mean + deviation, color=color, alpha=0.13 if selected else 0.06
            )
    training.set_title("Training return · completed episodes per four updates", loc="left")
    evaluation.set_title("Evaluation return · 100 Gymnasium episodes per checkpoint", loc="left")
    evaluation.axhline(475, linestyle="--", linewidth=1, color="#777")
    for axis in (training, evaluation):
        axis.set_ylabel("Episode return")
        axis.set_ylim(0, 535)
        axis.set_xscale("symlog", linthresh=8192)
        axis.set_xlim(left=0)
    training.legend(title=phase["parameter"], ncols=3, loc="upper left", fontsize=8)
    training.set_xlabel("Sampling slots (symlog scale)")
    evaluation.set_xlabel("Valid transitions (symlog scale)")
    fig.suptitle(
        f"CartPole PPO · {phase['num_envs']:,} environments\n"
        f"Mean ± sample SD across {len(series[0][1])} training seeds per candidate"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render_dashboard(output, *, figures=True):
    root = Path(output)
    phase_paths = sorted(root.glob("search/*/*.json"))
    phases = [json.loads(path.read_text()) for path in phase_paths]
    comparisons = [json.loads(path.read_text()) for path in sorted(root.glob("comparison/*.json"))]
    best = best_backend_conditions(comparisons)
    if figures:
        plot_heatmaps(phases, root)
        plot_comparisons(comparisons, root / "comparison.png")
        plot_best(best, root)
        plot_best_learning_curves(best, root / "best-learning-curves.png")
        if any("batch_rule" in row for row in comparisons):
            plot_comparison_heatmaps(comparisons, root / "comparison-heatmaps.png")
    (root / "best-backends.json").write_text(json.dumps(best, indent=2) + "\n")
    selected_path = root / "selected.json"
    if figures and selected_path.exists():
        plot_selected_trends(json.loads(selected_path.read_text()), root)
    export_tables(phases, comparisons, root, include_checkpoints=figures)
    learning_curves = {}
    for phase, phase_path in zip(phases, phase_paths, strict=True):
        curve = phase_path.with_name(f"{phase_path.stem}-learning.png")
        current = curve.exists() and curve.stat().st_mtime_ns >= phase_path.stat().st_mtime_ns
        if figures and not current:
            plot_search_learning_curves(phase, root / "trials", curve)
            current = curve.exists() and curve.stat().st_mtime_ns >= phase_path.stat().st_mtime_ns
        if current:
            learning_curves[(phase["num_envs"], phase["parameter"])] = curve.relative_to(
                root
            ).as_posix()
    view_phases = [
        {
            "num_envs": phase["num_envs"],
            "parameter": phase["parameter"],
            "selected_value": phase["selected_value"],
            "reused_original_search": "resume_from_config" in phase,
            "selection_override": phase.get("selection_override"),
            "device": phase.get("starting_config", {}).get("device"),
            "learning_curve": learning_curves.get((phase["num_envs"], phase["parameter"])),
            "candidates": [
                {
                    "value": candidate["value"],
                    "summary": candidate["summary"],
                    "scaling_config": candidate.get("scaling_config"),
                }
                for candidate in phase["candidates"]
            ],
        }
        for phase in phases
    ]
    progress_path = root / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    data = json.dumps(
        {
            "fixed_experience_budget": protocol.get("design", "").startswith(
                "fixed_experience_budget_v"
            ),
            "evaluate_each_update": protocol.get("evaluate_each_update", False),
            "phases": view_phases,
            "has_comparisons": bool(comparisons),
            "progress": progress,
            "has_trends": selected_path.exists(),
            "has_best": bool(best),
            "has_curves": (root / "best-learning-curves.png").exists(),
            "has_comparison_heatmaps": (root / "comparison-heatmaps.png").exists(),
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
        if not summary["complete"] or summary["mean_final_eval_return"] is None:
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
                    -record["summary"]["mean_final_eval_return"],
                    record["summary"]["std_final_eval_return"],
                    record["summary"]["mean_wall_seconds"],
                ),
            )
            basis = "final_evaluation_no_fully_stable_size"
        selected.append({**best, "selection_basis": basis})
    return selected


def plot_best(best, output):
    if not best:
        return
    plt = _pyplot()
    fig, axes = plt.subplots(1, 4, figsize=(19, 6))
    labels = []
    for row in best:
        config = row["config"]
        compiled = " compiled" if config["compile_env"] else ""
        labels.append(
            f"{config['backend']}\n{config['device']}{compiled}\nN={config['num_envs']:,}"
        )
    for axis, metric, deviation, title in (
        (
            axes[0],
            "mean_final_eval_return",
            "std_final_eval_return",
            "Final-five evaluation return",
        ),
        (axes[1], "mean_wall_seconds", "std_wall_seconds", "Full fixed-length wall time (s)"),
        (
            axes[2],
            "mean_target_seconds",
            "std_target_seconds",
            "Time to target (s), 20/20 reached",
        ),
        (
            axes[3],
            "mean_target_transitions",
            "std_target_transitions",
            "Transitions to target, 20/20 reached",
        ),
    ):
        values = [row["summary"].get(metric) for row in best]
        errors = [row["summary"].get(deviation) for row in best]
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
    searched = [
        selected[str(count)].get("selection_basis", "anchor_search") == "anchor_search"
        for count in counts
    ]
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    for axis, parameter in zip(axes.flat, PARAMETERS, strict=True):
        values = [selected[str(count)]["config"][parameter] for count in counts]
        axis.plot(counts, values, color="#7a8b85", linewidth=1)
        for measured, label, marker in ((True, "Searched", "o"), (False, "Scaled", "x")):
            points = [
                (count, value)
                for count, value, anchor in zip(counts, values, searched, strict=True)
                if anchor == measured
            ]
            if points:
                x, y = zip(*points, strict=True)
                axis.scatter(x, y, label=label, marker=marker, s=21)
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Environments")
        axis.set_title(parameter)
        if min(values) > 0 and max(values) / min(values) > 10:
            axis.set_yscale("log")
    fig.suptitle("Selected hyperparameters across environment counts")
    axes.flat[0].legend(fontsize=7)
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
        transitions = np.array(
            [
                [point.get("valid_transitions", point["timesteps"]) for point in curve]
                for curve in curves
            ]
        )
        mean = returns.mean(axis=0)
        deviation = returns.std(axis=0, ddof=1)
        config = row["config"]
        compiled = " compiled" if config["compile_env"] else ""
        label = f"{config['backend']} / {config['device']}{compiled} / N={config['num_envs']:,}"
        for axis, x in zip(axes, (transitions.mean(axis=0), elapsed.mean(axis=0)), strict=True):
            line = axis.plot(x, mean, label=label)[0]
            axis.fill_between(
                x, mean - deviation, mean + deviation, alpha=0.12, color=line.get_color()
            )
    for axis in axes:
        axis.axhline(475, color="#666", linestyle="--", linewidth=1)
        axis.set_ylabel("Evaluation episode return")
    axes[0].set_xlabel("Mean valid transitions at each checkpoint")
    axes[1].set_xlabel("Mean wall time at each checkpoint (s)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Best environment count per backend · mean ± sample SD · 20 seeds")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_comparison_heatmaps(comparisons, path):
    plt = _pyplot()
    labels = sorted(
        {(c["backend"], c["device"], c["compile_env"]) for c in (r["config"] for r in comparisons)}
    )
    counts = sorted({row["config"]["num_envs"] for row in comparisons})
    reach = np.full((len(labels), len(counts)), np.nan)
    times = np.full_like(reach, np.nan)
    for row in comparisons:
        config, stats = row["config"], row["summary"]
        key = (config["backend"], config["device"], config["compile_env"])
        position = labels.index(key), counts.index(config["num_envs"])
        reach[position] = stats["target_reached"] / stats["expected_seeds"]
        if stats["mean_target_seconds"] is not None:
            times[position] = stats["mean_target_seconds"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis, grid, title, upper in zip(
        axes,
        (reach, times),
        ("Target reach rate", "Mean time to target (s)"),
        (1, None),
        strict=True,
    ):
        rendered = axis.imshow(np.ma.masked_invalid(grid), aspect="auto", vmin=0, vmax=upper)
        axis.set_xticks(range(len(counts)), [f"{count:,}" for count in counts], rotation=30)
        axis.set_yticks(
            range(len(labels)),
            [f"{b} / {d}" + (" compiled" if c else "") for b, d, c in labels],
        )
        axis.set_title(title)
        axis.set_xlabel("Environments")
        fig.colorbar(rendered, ax=axis)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def final_checkpoint_rows(root, seeds):
    from gymtosha.benchmarks.study import FINAL_EVAL_CHECKPOINTS

    for seed in seeds:
        path = root / "trials" / f"{seed['trial_id']}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if record["status"] != "completed":
            continue
        points = record["result"]["evaluation"]["checkpoints"]
        tail = points[-FINAL_EVAL_CHECKPOINTS:]
        if len(tail) != FINAL_EVAL_CHECKPOINTS:
            raise ValueError(f"trial {seed['trial_id']} has fewer than five evaluation checkpoints")
        scope = {
            key: seed[key]
            for key in ("condition", "num_envs", "parameter", "value", "seed", "trial_id")
            if key in seed
        }
        for index, point in enumerate(tail, start=len(points) - len(tail) + 1):
            scores = np.asarray(point["episode_returns"], dtype=np.float32)
            if len(scores) != 100:
                raise ValueError(
                    f"trial {seed['trial_id']} checkpoint {index} must have 100 episodes"
                )
            yield {
                **scope,
                "checkpoint": index,
                "sampling_slots": point["timesteps"],
                "valid_transitions": point.get("valid_transitions"),
                "gradient_updates": point.get("gradient_updates"),
                "episodes": len(scores),
                "mean_return": float(scores.mean()),
                "episode_population_std": float(scores.std()),
            }


def export_tables(phases, comparisons, root, *, include_checkpoints=False):
    import csv

    searches = [
        {
            "num_envs": phase["num_envs"],
            "parameter": phase["parameter"],
            "value": candidate["value"],
            "selected": candidate["value"] == phase["selected_value"],
            **candidate.get("scaling_config", {}),
            **candidate["summary"],
        }
        for phase in phases
        for candidate in phase["candidates"]
    ]
    results = [
        {
            **row["config"],
            **row["summary"],
            **{key: row[key] for key in ("group", "batch_rule") if key in row},
        }
        for row in comparisons
    ]
    search_seeds = [
        {
            "num_envs": phase["num_envs"],
            "parameter": phase["parameter"],
            "value": candidate["value"],
            "seed": seed,
            "trial_id": trial_id,
            "final_eval_return": reward,
        }
        for phase in phases
        for candidate in phase["candidates"]
        for seed, trial_id, reward in zip(
            phase.get("seeds", range(1, candidate["summary"]["attempted_seeds"] + 1)),
            candidate.get("trial_ids", []),
            candidate.get("seed_scores", []),
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
    tables = [
        ("search-summary.csv", searches),
        ("search-seeds.csv", search_seeds),
        ("comparison-summary.csv", results),
        ("comparison-seeds.csv", comparison_seeds),
    ]
    device_rows = []
    for path in sorted((root / "device-probes").glob("*.json")):
        if path.stem.isdigit():
            probe = json.loads(path.read_text())
            config = probe["config"]
            updates = config["total_timesteps"] // (config["num_envs"] * config["rollout_steps"])
            device_rows.extend(
                {
                    "num_envs": int(path.stem),
                    "rollout_updates": updates,
                    **row,
                    "selected": row["trial_id"] == probe["selected"]["trial_id"],
                }
                for row in probe["trials"]
            )
    tables.append(("device-probes.csv", device_rows))
    presets = root / "selected.json"
    if presets.exists():
        tables.append(
            (
                "selected-presets.csv",
                [
                    {
                        **row["config"],
                        "selection_basis": row.get("selection_basis", "searched"),
                        "anchor_counts": row.get("anchor_counts", [int(count)]),
                        **{key: row[key] for key in ("group", "batch_rule") if key in row},
                    }
                    for count, row in sorted(
                        json.loads(presets.read_text()).items(), key=lambda item: int(item[0])
                    )
                ],
            )
        )
    if include_checkpoints:
        rows = list(final_checkpoint_rows(root, search_seeds + comparison_seeds))
        tables.append(("final-evaluation-checkpoints.csv", rows))
    for name, rows in tables:
        if not rows:
            continue
        with (Path(root) / name).open("w", newline="") as stream:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
