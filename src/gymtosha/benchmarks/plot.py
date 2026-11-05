"""Publication-style plots made only from recorded benchmark observations."""

from collections import defaultdict
from pathlib import Path

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    return plt


def label(config):
    compiled = " compiled" if config.get("compile_env") else ""
    return f"{config['backend']} · {config['device']}{compiled}"


def plot_training(records, output):
    if not records:
        return
    conditions = {}
    for record in records:
        config = record["config"]
        key = (config["env_id"], config["algorithm"])
        settings = {
            name: value
            for name, value in config.items()
            if name not in {"backend", "device", "compile_env", "seed"}
        }
        if key in conditions and conditions[key] != settings:
            raise ValueError(f"incompatible training settings for {key}")
        conditions[key] = settings
    plt = _pyplot()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for record in records:
        config = record["config"]
        groups[(config["env_id"], config["algorithm"])].append(record)
    env_order = list(dict.fromkeys(record["config"]["env_id"] for record in records))
    algo_order = ["ppo", "dqn", "sac", "td3"]
    groups = dict(
        sorted(
            groups.items(),
            key=lambda item: (env_order.index(item[0][0]), algo_order.index(item[0][1])),
        )
    )
    labels = list(dict.fromkeys(label(record["config"]) for record in records))
    colors = dict(zip(labels, plt.get_cmap("tab10").colors, strict=False))
    fig, ax = plt.subplots(figsize=(max(9, len(groups) * 1.2), 5.2))
    width = 0.8 / len(labels)
    for offset, name in enumerate(labels):
        means, errors = [], []
        for rows in groups.values():
            rates = [row["steps_per_second"] for row in rows if label(row["config"]) == name]
            means.append(np.median(rates) if rates else np.nan)
            errors.append(np.std(rates) if len(rates) > 1 else 0)
        positions = np.arange(len(groups)) - 0.4 + width / 2 + offset * width
        ax.bar(positions, means, width, yerr=errors, label=name, color=colors[name], capsize=2)
    ax.set_xticks(
        np.arange(len(groups)),
        [f"{env}\n{algo.upper()}" for env, algo in groups],
        rotation=35,
        ha="right",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Training transitions / second (log scale)")
    ax.set_title("End-to-end training speed · identical learner and budget per comparison")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.subplots_adjust(bottom=0.34, right=0.82)
    fig.text(
        0.01,
        0.015,
        "Bars = median; whiskers = seed standard deviation. "
        "Warm-up excluded and recorded separately. LunarLander uses approximate tensor physics.",
        fontsize=8,
    )
    for extension in ("png", "svg"):
        fig.savefig(output / f"training_speed.{extension}")
    plt.close(fig)

    columns = min(3, len(groups))
    rows = (len(groups) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.6 * rows), squeeze=False)
    for ax, ((env_id, algorithm), runs) in zip(axes.flat, groups.items(), strict=False):
        shown = set()
        for run in runs:
            name = label(run["config"])
            points = [row for row in run["checkpoints"] if row["mean_return"] is not None]
            if not points:
                continue
            ax.plot(
                [row["seconds"] for row in points],
                [row["mean_return"] for row in points],
                color=colors[name],
                alpha=0.65,
                linewidth=1.4,
                marker=".",
                markersize=4,
                label=name if name not in shown else None,
            )
            shown.add(name)
        suffix = " (approximate)" if "LunarLander" in env_id else ""
        ax.set_title(f"{env_id} · {algorithm.upper()}{suffix}", fontsize=10)
        ax.set_xlabel("Training wall time (s)")
        ax.set_ylabel("Mean completed-episode return")
        if not shown:
            ax.text(0.5, 0.5, "No completed episodes", ha="center", transform=ax.transAxes)
    for ax in list(axes.flat)[len(groups) :]:
        ax.set_visible(False)
    handles = [plt.Line2D([0], [0], color=colors[name], label=name) for name in labels]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=min(len(labels), 3),
        fontsize=8,
    )
    fig.suptitle("Training returns versus wall time · individual seeds", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for extension in ("png", "svg"):
        fig.savefig(output / f"learning_curves.{extension}")
    plt.close(fig)


def plot_throughput(records, output):
    if not records:
        return
    plt = _pyplot()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    environments = list(dict.fromkeys(row["env_id"] for row in records))
    labels = list(dict.fromkeys(label(row) for row in records))
    colors = dict(zip(labels, plt.get_cmap("tab10").colors, strict=False))
    columns = min(3, len(environments))
    fig, axes = plt.subplots(
        (len(environments) + columns - 1) // columns,
        columns,
        figsize=(5 * columns, 7 if len(environments) > 3 else 4),
        squeeze=False,
    )
    for ax, env_id in zip(axes.flat, environments, strict=False):
        for name in labels:
            by_size = defaultdict(list)
            for row in records:
                if row["env_id"] == env_id and label(row) == name:
                    by_size[row["num_envs"]].append(row["steps_per_second"])
            sizes = sorted(by_size)
            means = [np.median(by_size[size]) for size in sizes]
            ax.plot(sizes, means, marker="o", color=colors[name], label=name)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Environments in batch")
        ax.set_ylabel("Transitions / second")
        ax.set_title(env_id + (" (approximate)" if "LunarLander" in env_id else ""))
    for ax in list(axes.flat)[len(environments) :]:
        ax.set_visible(False)
    handles = [plt.Line2D([0], [0], color=colors[name], label=name) for name in labels]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=min(len(labels), 3),
        fontsize=8,
    )
    fig.suptitle("Environment + tensor-adapter throughput · synchronized device timing", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for extension in ("png", "svg"):
        fig.savefig(output / f"environment_throughput.{extension}")
    plt.close(fig)
