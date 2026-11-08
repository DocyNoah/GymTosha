"""Generate the final CartPole T=64 comparison from raw TensorBoard evaluation events."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from statistics import mean, stdev

import numpy as np

HORIZON = 64
BUDGET = 8_388_608
TARGET = 475
INITIAL_SEEDS = frozenset(range(1001, 1006))
RETURNS = "eval/episode_returns"
TRANSITIONS = "train/valid_transitions"
TRAINING_SECONDS = "time/training_seconds"
OPTIMIZER_CALLS = "train/optimizer_calls"
UPDATE = "train/ppo_update"
REQUIRED_TAGS = {RETURNS, TRANSITIONS, TRAINING_SECONDS, OPTIMIZER_CALLS, UPDATE}
FIXED_CONFIG = {
    "env_id": "CartPole-v1",
    "algorithm": "ppo",
    "rollout_steps": HORIZON,
    "total_timesteps": BUDGET,
    "ppo_minibatch_size": 512,
    "update_epochs": 4,
    "gae_lambda": 0.99,
    "hidden_sizes": [256, 256, 256],
    "num_threads": 1,
    "anneal_by_optimizer_step": True,
    "anneal_learning_rate": True,
    "native_same_step": True,
    "normalize_observations": False,
    "normalize_rewards": False,
}
FIXED_EVALUATION = {
    "target": TARGET,
    "episodes": 100,
    "confirmations": 1,
    "seed": 200000,
    "stop_on_target": False,
    "min_episode_success_rate": 0.0,
    "evaluate_at_start": False,
    "training_time_only": True,
    "native_cartpole": True,
}

CONDITION_FIELDS = (
    "host",
    "backend",
    "device",
    "compile_env",
    "num_envs",
    "evaluation_spacing_transitions",
)
SEED_FIELDS = (
    "seed",
    "status",
    "source",
    "code_version",
    "missing_fields",
    "issue",
    "observed_checkpoints",
    "expected_checkpoints",
    "target_update",
    "target_mean_return",
    "target_transitions",
    "target_training_seconds",
    "target_optimizer_calls",
    "last_mean_return",
    "last_transitions",
    "last_training_seconds",
    "last_optimizer_calls",
)
SUMMARY_FIELDS = (
    "expected_seeds",
    "completed_seeds",
    "coverage",
    "incomplete_seeds",
    "missing_seeds",
    "invalid_seeds",
    "reached_seeds",
    "censored_seeds",
    "reach_rate",
    "sources",
    "mean_target_transitions",
    "sd_target_transitions",
    "mean_target_training_seconds",
    "sd_target_training_seconds",
    "mean_target_optimizer_calls",
    "sd_target_optimizer_calls",
    "mean_last_return",
    "sd_last_return",
)


@dataclass(frozen=True)
class Condition:
    host: str
    backend: str
    device: str
    compile_env: bool
    num_envs: int

    @property
    def folder(self):
        return f"{self.backend}-{self.device}-{int(self.compile_env)}-{self.num_envs}"

    @property
    def spacing(self):
        return self.num_envs * HORIZON

    @property
    def checkpoints(self):
        return BUDGET // self.spacing

    @property
    def label(self):
        mode = "compile" if self.compile_env else "eager"
        return f"{self.backend} / {self.device} / {mode}"

    def columns(self):
        return dict(
            zip(
                CONDITION_FIELDS,
                (
                    self.host,
                    self.backend,
                    self.device,
                    self.compile_env,
                    self.num_envs,
                    self.spacing,
                ),
                strict=True,
            )
        )


@dataclass(frozen=True)
class Checkpoint:
    update: int
    mean_return: float
    transitions: int
    training_seconds: float
    optimizer_calls: int


@dataclass(frozen=True)
class SeedResult:
    seed: int
    status: str
    points: tuple[Checkpoint, ...] = ()
    source: str = ""
    code_version: str = ""
    missing_fields: tuple[str, ...] = ()
    issue: str = ""

    @property
    def first_target(self):
        return next((point for point in self.points if point.mean_return >= TARGET), None)

    def columns(self, condition):
        hit = self.first_target
        last = self.points[-1] if self.points else None
        return {
            **condition.columns(),
            "seed": self.seed,
            "status": self.status,
            "source": self.source,
            "code_version": self.code_version,
            "missing_fields": ",".join(self.missing_fields),
            "issue": self.issue,
            "observed_checkpoints": len(self.points),
            "expected_checkpoints": condition.checkpoints,
            "target_update": hit.update if hit else None,
            "target_mean_return": hit.mean_return if hit else None,
            "target_transitions": hit.transitions if hit else None,
            "target_training_seconds": hit.training_seconds if hit else None,
            "target_optimizer_calls": hit.optimizer_calls if hit else None,
            "last_mean_return": last.mean_return if last else None,
            "last_transitions": last.transitions if last else None,
            "last_training_seconds": last.training_seconds if last else None,
            "last_optimizer_calls": last.optimizer_calls if last else None,
        }


def _read_json(path):
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return result


def _read_checkpoints(folder, condition, *, complete, optional):
    from google.protobuf.message import DecodeError
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
    from tensorboard.util.tensor_util import make_ndarray

    from gymtosha.benchmarks.cartpole_events import METRIC_TAGS

    all_tags = {RETURNS, *METRIC_TAGS.values()}
    expected_tags = all_tags - optional
    # Read all events and preserve duplicate steps instead of using TensorBoard's reservoir.
    records = defaultdict(dict)
    for path in sorted(folder.glob("events.out.tfevents.*")):
        try:
            for event in LegacyEventFileLoader(str(path)).Load():
                for value in event.summary.value:
                    if value.tag not in all_tags:
                        continue
                    values = records[event.step]
                    if value.tag in values:
                        raise ValueError(f"duplicate {value.tag} at PPO update {event.step}")
                    array = make_ndarray(value.tensor)
                    if value.tag == RETURNS:
                        if (
                            array.shape != (100,)
                            or array.dtype.kind not in "iuf"
                            or not np.isfinite(array).all()
                        ):
                            raise ValueError(
                                f"PPO update {event.step} must have 100 finite returns"
                            )
                        values[value.tag] = float(array.mean(dtype=np.float64))
                    else:
                        if (
                            array.shape != ()
                            or array.dtype.kind not in "iuf"
                            or not np.isfinite(array)
                        ):
                            raise ValueError(
                                f"{value.tag} at PPO update {event.step} must be a finite scalar"
                            )
                        values[value.tag] = array.item()
        except (DecodeError, OSError, TypeError) as error:
            raise ValueError(f"invalid TensorBoard event in {path.name}: {error}") from error

    count = len(records)
    if count > condition.checkpoints or (complete and count != condition.checkpoints):
        raise ValueError(f"found {count} checkpoints, expected {condition.checkpoints}")

    points = []
    previous_time = 0.0
    observed_gaps = set()
    for expected_step, step in enumerate(sorted(records), start=1):
        values = records[step]
        if missing := expected_tags.difference(values):
            raise ValueError(f"PPO update {step} missing {', '.join(sorted(missing))}")
        observed_gaps.update(optional.difference(values))
        if step != expected_step or values[UPDATE] != step:
            raise ValueError(f"PPO update {step} does not match expected update {expected_step}")
        transitions = step * condition.spacing
        if values[TRANSITIONS] != transitions:
            raise ValueError(f"PPO update {step} expected {transitions} valid transitions")
        optimizer_calls = transitions * 4 // 512
        if values[OPTIMIZER_CALLS] != optimizer_calls:
            raise ValueError(f"PPO update {step} expected {optimizer_calls} optimizer calls")
        elapsed = float(values[TRAINING_SECONDS])
        if not math.isfinite(elapsed) or elapsed < previous_time:
            raise ValueError(f"PPO update {step} training seconds must be finite and monotonic")
        points.append(Checkpoint(step, values[RETURNS], transitions, elapsed, optimizer_calls))
        previous_time = elapsed
    if complete and optional != observed_gaps:
        raise ValueError("legacy missing fields must describe at least one absent checkpoint")
    return tuple(points)


def _read_seed(folder, condition, seed, *, partial):
    from gymtosha.benchmarks.cartpole_events import METRIC_TAGS

    if not folder.exists():
        return SeedResult(seed, "missing")
    metadata_path = folder / "metadata.json"
    if not metadata_path.exists():
        return SeedResult(seed, "incomplete", issue="missing metadata.json")

    provenance = {}
    try:
        metadata = _read_json(metadata_path)
        source, code_version = metadata["source"], metadata["code_version"]
        missing_fields = metadata["missing_fields"]
        if (
            source not in ("live", "legacy_json")
            or not isinstance(code_version, str)
            or not code_version
            or not isinstance(missing_fields, list)
            or not all(isinstance(field, str) for field in missing_fields)
            or (source == "live" and missing_fields)
        ):
            raise ValueError("invalid event source, code version or missing fields")
        provenance = {
            "source": source,
            "code_version": code_version,
            "missing_fields": tuple(missing_fields),
        }
        if set(missing_fields).difference(METRIC_TAGS):
            raise ValueError("unknown missing fields in metadata")
        optional = {METRIC_TAGS[field] for field in missing_fields}
        if optional & REQUIRED_TAGS:
            raise ValueError("core benchmark fields cannot be missing")
        if metadata["host"] != condition.host:
            raise ValueError("seed host differs from matrix host")
        if metadata.get("timing_scope") != "isolated":
            raise ValueError("seed timing scope must be isolated")
        config = metadata["config"]
        config_values = {
            **FIXED_CONFIG,
            "seed": seed,
            "num_envs": condition.num_envs,
            "num_minibatches": condition.spacing // 512,
            "backend": condition.backend,
            "device": condition.device,
            "compile_env": condition.compile_env,
        }
        if not isinstance(config, dict):
            raise ValueError("seed config must be a JSON object")
        for name, value in config_values.items():
            if config.get(name) != value:
                raise ValueError(f"seed config {name} differs from matrix protocol")
        if metadata.get("evaluation") != {**FIXED_EVALUATION, "interval": condition.spacing}:
            raise ValueError("seed evaluation differs from matrix protocol")
        complete = metadata["status"] == "completed"
        if not complete and not partial:
            return SeedResult(seed, "incomplete", **provenance)
        points = _read_checkpoints(folder, condition, complete=complete, optional=optional)
    except (KeyError, TypeError, ValueError) as error:
        if not partial:
            raise ValueError(f"{folder}: {error}") from error
        return SeedResult(seed, "invalid", issue=str(error), **provenance)

    status = "incomplete"
    if complete:
        status = "reached" if any(point.mean_return >= TARGET for point in points) else "censored"
    return SeedResult(seed, status, points, issue=metadata.get("error", ""), **provenance)


def analyze(roots, *, partial=False):
    """Load the declared comparison matrix and verify every eligible seed."""
    capacity = []
    results = {}
    seen_conditions = set()
    expected_seeds = None
    incomplete = []
    for root in roots:
        manifest = _read_json(root / "matrix.json")
        if manifest.get("horizon") != HORIZON or manifest.get("budget") != BUDGET:
            raise ValueError(f"{root}: expected horizon {HORIZON} and budget {BUDGET}")
        declared = manifest.get("seeds", [])
        if not isinstance(declared, list) or any(type(seed) is not int for seed in declared):
            raise ValueError(f"{root}: seeds must be a list of integers")
        seeds = set(declared)
        if len(seeds) != len(declared) or not INITIAL_SEEDS <= seeds:
            raise ValueError(f"{root}: duplicate seed or missing one of seeds 1001–1005")
        if expected_seeds is not None and expected_seeds != seeds:
            raise ValueError(f"{root}: all matrices must declare the same seeds")
        expected_seeds = seeds
        if not manifest.get("conditions"):
            raise ValueError(f"{root}: matrix has no capacity outcomes")
        for entry in manifest["conditions"]:
            try:
                condition = Condition(**{name: entry[name] for name in CONDITION_FIELDS[:-1]})
                status = entry["status"]
            except (KeyError, TypeError) as error:
                raise ValueError(f"{root}: invalid capacity condition {entry!r}") from error
            if (
                not condition.host
                or type(condition.num_envs) is not int
                or condition.num_envs < 1
                or not isinstance(condition.compile_env, bool)
                or status not in ("passed", "failed")
                or condition.spacing % 512
                or BUDGET % condition.spacing
                or not re.fullmatch(r"[a-z0-9_-]+", condition.folder)
            ):
                raise ValueError(f"{root}: invalid capacity condition {entry!r}")
            if condition in seen_conditions:
                raise ValueError(
                    f"{root}: duplicate condition {condition.host} / {condition.folder}"
                )
            seen_conditions.add(condition)
            capacity.append({**entry, **condition.columns()})
            if status == "failed":
                continue
            folder = root / "events" / condition.folder
            known_folders = {f"seed-{seed}" for seed in seeds}
            for seed_folder in folder.glob("seed-*"):
                if seed_folder.name not in known_folders:
                    raise ValueError(f"{root}: undeclared seed directory {seed_folder}")
            results[condition] = [
                _read_seed(folder / f"seed-{seed}", condition, seed, partial=partial)
                for seed in sorted(seeds)
            ]
            incomplete.extend(
                f"{condition.host}/{condition.folder}/seed-{result.seed} ({result.status})"
                for result in results[condition]
                if result.status in ("missing", "incomplete")
            )
    if incomplete and not partial:
        examples = ", ".join(incomplete[:8])
        raise ValueError(f"{len(incomplete)} missing or incomplete eligible runs: {examples}")
    return capacity, results


def _summarize(condition, runs):
    completed = [run for run in runs if run.status in ("reached", "censored")]
    hits = [run.first_target for run in completed if run.status == "reached"]
    row = {
        **condition.columns(),
        "expected_seeds": len(runs),
        "completed_seeds": len(completed),
        "coverage": len(completed) / len(runs) if runs else None,
        "reached_seeds": len(hits),
        "reach_rate": len(hits) / len(completed) if completed else None,
        "sources": ",".join(sorted({run.source for run in completed})),
    }
    for status in ("incomplete", "missing", "invalid", "censored"):
        row[f"{status}_seeds"] = sum(run.status == status for run in runs)
    for field in ("transitions", "training_seconds", "optimizer_calls"):
        values = [getattr(point, field) for point in hits]
        row[f"mean_target_{field}"] = mean(values) if values else None
        row[f"sd_target_{field}"] = stdev(values) if len(values) > 1 else None
    final = [run.points[-1].mean_return for run in completed]
    row["mean_last_return"] = mean(final) if final else None
    row["sd_last_return"] = stdev(final) if len(final) > 1 else None
    return row


def _best_rows(rows):
    by_variant = defaultdict(list)
    for condition, row in rows.items():
        if row["coverage"] == 1:
            by_variant[condition.host, condition.label].append((condition, row))

    def rank(candidate):
        _, row = candidate
        if row["reached_seeds"] == 0:
            return (0, -row["mean_last_return"], row["num_envs"])
        return (
            -row["reached_seeds"],
            row["mean_target_training_seconds"],
            row["mean_target_transitions"],
            row["num_envs"],
        )

    return dict(min(candidates, key=rank) for candidates in by_variant.values())


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_rows(results):
    for condition, runs in results.items():
        for run in runs:
            for point in run.points:
                yield {
                    **condition.columns(),
                    "seed": run.seed,
                    "status": run.status,
                    "source": run.source,
                    "ppo_update": point.update,
                    "mean_return": point.mean_return,
                    "valid_transitions": point.transitions,
                    "training_seconds": point.training_seconds,
                    "optimizer_calls": point.optimizer_calls,
                }


def _learning_curve(runs):
    completed = [run for run in runs if run.status in ("reached", "censored")]
    if not completed:
        return None
    rewards = np.array([[point.mean_return for point in run.points] for run in completed])
    times = np.array([[point.training_seconds for point in run.points] for run in completed])
    return {
        "x": np.array([point.transitions for point in completed[0].points]),
        "time": times.mean(axis=0),
        "mean": rewards.mean(axis=0),
        "sd": rewards.std(axis=0, ddof=1) if len(completed) > 1 else None,
    }


def _dashboard_conditions(capacity, results, summaries, curves, best):
    for entry in capacity:
        condition = Condition(**{name: entry[name] for name in CONDITION_FIELDS[:-1]})
        runs = results.get(condition, ())
        summary = summaries.get(condition) or _summarize(condition, runs)
        curve = curves.get(condition)
        if curve is not None:
            curve = {
                key: values.tolist() if values is not None else None
                for key, values in curve.items()
            }
        error = entry.get("error") or entry.get("error_type")
        yield {
            **summary,
            "id": f"{condition.host}/{condition.folder}",
            "status": entry["status"],
            "capacity_source": entry.get("capacity_source"),
            **({"capacity_error": error} if error else {}),
            "best": condition in best,
            "curve": curve,
            "seeds": [run.columns(condition) for run in runs],
        }


def _plot(path, title, conditions, curves, summaries):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import EngFormatter

    figure = Figure(figsize=(12, 5.6), layout="constrained")
    FigureCanvasAgg(figure)
    transitions, wall = figure.subplots(ncols=2, sharey=True)
    for condition in conditions:
        curve = curves[condition]
        if curve is None:
            continue
        summary = summaries[condition]
        label = (
            f"{condition.label}, N={condition.num_envs:,} "
            f"({summary['reached_seeds']}/{summary['completed_seeds']} reached)"
        )
        average = curve["mean"]
        spread = curve["sd"]
        for axis, positions in ((transitions, curve["x"]), (wall, curve["time"])):
            (line,) = axis.plot(positions, average, label=label, linewidth=1.5)
            if spread is not None:
                axis.fill_between(
                    positions, average - spread, average + spread, color=line.get_color(), alpha=0.1
                )

    for axis in (transitions, wall):
        axis.axhline(TARGET, color="#616b78", linewidth=1, linestyle="--")
        axis.grid(axis="y", alpha=0.2)
        axis.set_xlim(left=0)
    transitions.set_xlabel("Valid transitions")
    transitions.xaxis.set_major_formatter(EngFormatter())
    transitions.set_ylabel("Mean evaluation return")
    wall.set_xlabel("Mean training wall time at each PPO update (s)")
    handles, labels = transitions.get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="outside lower center", fontsize=8, ncols=2)
    figure.suptitle(title, fontweight="bold")
    figure.savefig(path, dpi=145)


def _comparison_table(conditions, summaries):
    headings = (
        "Host",
        "Backend",
        "Environments",
        "Evaluation spacing",
        "Complete",
        "Reached",
        "Transitions to target",
        "Training seconds to target",
    )

    def estimate(row, field, decimals):
        average = row[f"mean_target_{field}"]
        deviation = row[f"sd_target_{field}"]
        if average is None:
            return "—"
        center = f"{average:,.{decimals}f}"
        return center if deviation is None else f"{center} ± {deviation:,.{decimals}f}"

    header = "".join(f"<th>{title}</th>" for title in headings)
    body = []
    for condition in conditions:
        row = summaries[condition]
        fields = (
            condition.host,
            condition.label,
            f"{condition.num_envs:,}",
            f"{condition.spacing:,}",
            f"{row['completed_seeds']}/{row['expected_seeds']}",
            f"{row['reached_seeds']}/{row['completed_seeds']}" if row["completed_seeds"] else "—",
            estimate(row, "transitions", 0),
            estimate(row, "training_seconds", 3),
        )
        body.append("<tr>" + "".join(f"<td>{escape(field)}</td>" for field in fields) + "</tr>")
    return (
        f'<div class="scroll"><table><thead><tr>{header}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def write_report(output, capacity, results, *, partial=False):
    output.mkdir(parents=True, exist_ok=True)
    summaries = {condition: _summarize(condition, runs) for condition, runs in results.items()}
    curves = {condition: _learning_curve(runs) for condition, runs in results.items()}
    best = _best_rows(summaries)
    ordered = sorted(results, key=lambda c: (c.host, c.num_envs, c.label))
    capacity_fields = (*CONDITION_FIELDS, "status")
    extra_fields = sorted({key for row in capacity for key in row} - set(capacity_fields))
    _write_csv(output / "capacity.csv", (*capacity_fields, *extra_fields), capacity)
    _write_csv(output / "conditions.csv", (*CONDITION_FIELDS, *SUMMARY_FIELDS), summaries.values())
    _write_csv(
        output / "same-n.csv",
        (*CONDITION_FIELDS, *SUMMARY_FIELDS),
        (summaries[condition] for condition in ordered),
    )
    _write_csv(output / "best-n.csv", (*CONDITION_FIELDS, *SUMMARY_FIELDS), best.values())
    _write_csv(
        output / "seeds.csv",
        (*CONDITION_FIELDS, *SEED_FIELDS),
        (run.columns(condition) for condition, runs in results.items() for run in runs),
    )
    _write_csv(
        output / "checkpoints.csv",
        (
            *CONDITION_FIELDS,
            "seed",
            "status",
            "source",
            "ppo_update",
            "mean_return",
            "valid_transitions",
            "training_seconds",
            "optimizer_calls",
        ),
        _checkpoint_rows(results),
    )

    same_n = defaultdict(list)
    best_n = defaultdict(list)
    for condition in ordered:
        same_n[condition.host, condition.num_envs].append(condition)
        if condition in best:
            best_n[condition.host].append(condition)
    plots = []
    for index, ((host, count), conditions) in enumerate(same_n.items(), start=1):
        if any(summaries[condition]["completed_seeds"] for condition in conditions):
            plots.append((f"same-n-{index:02d}.png", f"{host} | N={count:,}", conditions))
    for index, (host, conditions) in enumerate(best_n.items(), start=1):
        plots.append((f"best-n-{index:02d}.png", f"{host} | Best N by backend", conditions))
    for filename, title, conditions in plots:
        _plot(output / filename, title, conditions, curves, summaries)

    links = "".join(
        f'<li><a href="{name}.csv">{name}.csv</a></li>'
        for name in ("capacity", "conditions", "seeds", "checkpoints", "same-n", "best-n")
    )
    figures = "".join(
        f"<figure><figcaption>{escape(title)}</figcaption>"
        f'<a href="{filename}"><img src="{filename}" alt="{escape(title)}" loading="lazy">'
        "</a></figure>"
        for filename, title, _ in plots
    )
    label = "Partial report" if partial else "Final report"
    same_table = _comparison_table(ordered, summaries)
    best_table = (
        _comparison_table(best, summaries) if best else "<p>Awaiting full seed coverage.</p>"
    )
    (output / "static.html").write_text(
        f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CartPole T=64 | {label}</title>
<style>
body {{font: 16px/1.55 system-ui,sans-serif; max-width: 1100px; margin: 3rem auto; padding: 0 1rem}}
figure {{margin: 2rem 0}} img {{display: block; width: 100%; height: auto}}
.scroll {{overflow-x: auto}}
table {{border-collapse: collapse; width: 100%; font-size: .9rem; white-space: nowrap}}
th, td {{text-align: right; padding: .45rem; border-bottom: 1px solid #ddd}}
th:nth-child(-n+2), td:nth-child(-n+2) {{text-align: left}}
</style>
<h1>CartPole T=64 · {label}</h1>
<p>The first mean of 100 raw evaluation returns at or above 475 is the target checkpoint.
Each condition evaluates after every PPO update, spaced by N × 64 valid transitions.
Each complete seed uses 8,388,608 valid transitions. The recorded training clock excludes
evaluation.</p>
<p>Coverage is complete seeds divided by declared seeds. Reach rate is reached seeds divided
by complete seeds. Complete non-reaching seeds are right-censored at their last checkpoint.
Target means and sample standard deviations use reached seeds. Standard deviations are blank
when fewer than two reached. Learning curves use completed seeds at each update. Shading shows
one sample standard deviation of returns around the mean. Wall-time curves use mean recorded
time at the same update.</p>
<p>Best N is selected per host and backend, device and compile mode among conditions with full
seed coverage. Ranking uses the number of reached seeds, then lower mean training time and
fewer mean transitions among reached seeds. If none reach, higher mean final return ranks
first. A remaining tie favors smaller N.</p>
<p>The source columns identify migrated legacy runs and live TensorBoard runs. Legacy timings
come from the original collector, before the diagnostic instrumentation used in live runs.</p>
<ul>{links}</ul>
<h2>Backends at the same environment count</h2>{same_table}
<h2>Best environment count per backend</h2>{best_table}
<h2>Learning curves</h2>{figures}
</html>""",
        encoding="utf-8",
    )

    payload = {
        "partial": partial,
        "target": TARGET,
        "budget": BUDGET,
        "horizon": HORIZON,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "conditions": list(_dashboard_conditions(capacity, results, summaries, curves, best)),
    }
    # Prevent data from closing the application/json script element.
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("cartpole_dashboard.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(
        template.replace("__DASHBOARD_DATA__", data, 1), encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="host output roots")
    parser.add_argument(
        "--output", type=Path, required=True, help="directory for CSV, PNG and HTML"
    )
    parser.add_argument(
        "--partial", action="store_true", help="report available runs before completion"
    )
    args = parser.parse_args(argv)
    try:
        capacity, results = analyze(args.inputs, partial=args.partial)
    except ValueError as error:
        parser.exit(2, f"error: {error}\n")
    write_report(args.output, capacity, results, partial=args.partial)


if __name__ == "__main__":
    main()
