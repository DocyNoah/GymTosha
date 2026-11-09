import csv
import json
import re
import shutil

import pytest


def write_matrix(root, *, host="RTX 3090", seeds=range(1001, 1006), device="cuda", extra=()):
    root.mkdir()
    (root / "matrix.json").write_text(
        json.dumps(
            {
                "conditions": [
                    {
                        "backend": "gymtosha",
                        "device": device,
                        "compile_env": False,
                        "num_envs": 16384,
                        "host": host,
                        "status": "passed",
                    },
                    {
                        "backend": "gymtosha",
                        "device": device,
                        "compile_env": True,
                        "num_envs": 16384,
                        "host": host,
                        "status": "failed",
                    },
                    *extra,
                ],
                "seeds": list(seeds),
                "horizon": 64,
                "budget": 8388608,
            }
        )
    )


def write_seed(
    root,
    seed,
    *,
    scores=(474, 475, 450, 480, 480, 480, 480, 480),
    mutate=None,
    status="completed",
    missing_fields=(),
    source="live",
    count=16384,
    host="RTX 3090",
    device="cuda",
    duplicate_update=None,
    config_overrides=None,
    error=None,
):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.util.tensor_util import make_tensor_proto

    folder = root / "events" / f"gymtosha-{device}-0-{count}" / f"seed-{seed}"
    folder.mkdir(parents=True)
    writer = EventFileWriter(str(folder))
    try:
        for step, score in enumerate(scores, start=1):
            metrics = {
                "eval/episode_returns": [score] * 100,
                "eval/mean_return": 500,
                "train/valid_transitions": count * 64 * step,
                "time/training_seconds": 2.5 * step,
                "train/optimizer_calls": count * step // 2,
                "train/ppo_update": step,
                "train/completed_episodes": 20 * step,
                "time/setup_seconds": 0.1,
                "time/warmup_seconds": 0.2,
                "time/learner_setup_seconds": 0.3,
                "time/collection_seconds": 0.4 * step,
                "time/update_seconds": 0.5 * step,
                "ppo/learning_rate": 0.001,
                "ppo/entropy": 0.6,
                "ppo/policy_loss": -0.1,
                "ppo/value_loss": 0.7,
                "ppo/approx_kl": 0.05,
                "ppo/clip_fraction": 0.3,
            }
            if mutate:
                mutate(step, metrics)
            values = [
                Summary.Value(tag=key, tensor=make_tensor_proto(value))
                for key, value in metrics.items()
            ]
            event = Event(step=step, summary=Summary(value=values))
            writer.add_event(event)
            if step == duplicate_update:
                writer.add_event(event)
    finally:
        writer.close()
    (folder / "metadata.json").write_text(
        json.dumps(
            {
                "host": host,
                "status": status,
                "config": {
                    "seed": seed,
                    "backend": "gymtosha",
                    "device": device,
                    "compile_env": False,
                    "num_envs": count,
                    "rollout_steps": 64,
                    "total_timesteps": 8_388_608,
                    "ppo_minibatch_size": 512,
                    "num_minibatches": count // 8,
                    "update_epochs": 4,
                    "env_id": "CartPole-v1",
                    "algorithm": "ppo",
                    "gae_lambda": 0.99,
                    "hidden_sizes": [256, 256, 256],
                    "num_threads": 1,
                    "anneal_by_optimizer_step": True,
                    "anneal_learning_rate": True,
                    "native_same_step": True,
                    "normalize_observations": False,
                    "normalize_rewards": False,
                    **(config_overrides or {}),
                },
                "evaluation": {
                    "target": 475.0,
                    "interval": count * 64,
                    "episodes": 100,
                    "confirmations": 1,
                    "seed": 200000,
                    "stop_on_target": False,
                    "min_episode_success_rate": 0.0,
                    "evaluate_at_start": False,
                    "training_time_only": True,
                    "native_cartpole": True,
                },
                "source": source,
                "timing_scope": "isolated",
                "missing_fields": missing_fields,
                "code_version": "test",
                **({"error": error} if error is not None else {}),
            }
        )
    )


def rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def dashboard_data(path):
    page = path.read_text(encoding="utf-8")
    script = re.search(r'<script\b[^>]*\bid="dashboard-data"[^>]*>(.*?)</script>', page, re.S)
    assert script is not None
    return json.loads(script[1])


def test_report_uses_first_raw_mean_and_preserves_censoring(tmp_path):
    from gymtosha.benchmarks.cartpole_report import main

    def slower_training(step, metrics):
        metrics["time/training_seconds"] = 5.0 * step

    root = tmp_path / "input"
    output = tmp_path / "report"
    write_matrix(root)
    write_seed(root, 1001)
    write_seed(root, 1002, scores=(474, 474, 475, 490, 480, 480, 480, 480))
    for seed in (1003, 1004):
        write_seed(root, seed, scores=(474,) * 8)
    write_seed(root, 1005, scores=(474,) * 8, mutate=slower_training)

    main(["--inputs", str(root), "--output", str(output)])

    seed, second, censored, *_ = rows(output / "seeds.csv")
    assert seed["status"] == "reached"
    assert seed["target_update"] == "2"
    assert seed["target_transitions"] == "2097152"
    assert float(seed["target_training_seconds"]) == 5
    assert seed["target_optimizer_calls"] == "16384"
    assert second["target_update"] == "3"
    assert censored["status"] == "censored"
    assert censored["target_transitions"] == ""
    assert censored["last_transitions"] == "8388608"
    assert float(censored["last_mean_return"]) == 474

    (condition,) = rows(output / "conditions.csv")
    assert condition["host"] == "RTX 3090"
    assert condition["evaluation_spacing_transitions"] == "1048576"
    assert condition["completed_seeds"] == "5"
    assert condition["censored_seeds"] == "3"
    assert condition["reached_seeds"] == "2"
    assert float(condition["reach_rate"]) == 0.4
    assert float(condition["mean_target_transitions"]) == 2621440
    assert float(condition["sd_target_transitions"]) == pytest.approx(741455.20019)
    assert float(condition["mean_target_training_seconds"]) == 6.25
    assert float(condition["sd_target_training_seconds"]) == pytest.approx(1.76776695)
    assert len(rows(output / "checkpoints.csv")) == 40
    assert len(rows(output / "capacity.csv")) == 2
    assert len(rows(output / "best-n.csv")) == 1
    assert list(output.glob("same-n-*.png"))
    assert list(output.glob("best-n-*.png"))

    payload = dashboard_data(output / "index.html")
    assert payload["partial"] is False
    assert (payload["target"], payload["budget"], payload["horizon"]) == (475, 8388608, 64)
    assert payload["generated_at"].endswith("+00:00")
    passed, failed = payload["conditions"]
    assert passed["id"] == "RTX 3090/gymtosha-cuda-0-16384"
    assert passed["status"] == "passed"
    assert passed["best"] is True
    assert passed["reached_seeds"] == 2
    assert passed["mean_target_training_seconds"] == 6.25
    assert passed["curve"]["x"][:2] == [1048576, 2097152]
    assert passed["curve"]["time"][:2] == [3.0, 6.0]
    assert passed["curve"]["mean"][:2] == [474, 474.2]
    assert passed["curve"]["sd"][:2] == pytest.approx([0.0, 0.4472135955])
    assert passed["seeds"][2]["status"] == "censored"
    assert passed["seeds"][2]["target_transitions"] is None
    assert failed["status"] == "failed"
    assert failed["reach_rate"] is None
    assert failed["curve"] is None
    assert failed["seeds"] == []
    assert failed["best"] is False
    static = (output / "static.html").read_text(encoding="utf-8")
    assert 'href="same-n-01.png"' in static
    assert 'href="seeds.csv"' in static


def test_partial_mode_distinguishes_unfinished_missing_and_censored_seeds(tmp_path, capsys):
    from gymtosha.benchmarks.cartpole_report import main

    root = tmp_path / "input"
    output = tmp_path / "report"
    write_matrix(root)
    write_seed(root, 1001, scores=(474,) * 8)
    write_seed(root, 1002, scores=(474, 475), status="running")

    with pytest.raises(SystemExit) as error:
        main(["--inputs", str(root), "--output", str(output)])
    assert error.value.code == 2
    assert "missing or incomplete eligible runs" in capsys.readouterr().err
    assert not output.exists()

    main(["--inputs", str(root), "--output", str(output), "--partial"])

    assert [row["status"] for row in rows(output / "seeds.csv")] == [
        "censored",
        "incomplete",
        "missing",
        "missing",
        "missing",
    ]
    (condition,) = rows(output / "conditions.csv")
    assert condition["completed_seeds"] == "1"
    assert condition["expected_seeds"] == "5"
    assert condition["reached_seeds"] == "0"
    assert rows(output / "best-n.csv") == []

    payload = dashboard_data(output / "index.html")
    assert payload["partial"] is True
    passed, _ = payload["conditions"]
    assert passed["coverage"] == 0.2
    assert passed["curve"]["mean"] == [474.0] * 8
    assert passed["curve"]["sd"] is None
    assert [seed["status"] for seed in passed["seeds"]] == [
        "censored",
        "incomplete",
        "missing",
        "missing",
        "missing",
    ]


def test_dashboard_preserves_capacity_errors_without_closing_data_script(tmp_path):
    from gymtosha.benchmarks.cartpole_report import main

    root = tmp_path / "input"
    output = tmp_path / "report"
    error = '</script><img src="invalid">'
    write_matrix(
        root,
        extra=[
            {
                "host": "Mac",
                "backend": "gymtosha",
                "device": "mps",
                "compile_env": True,
                "num_envs": 8192,
                "status": "failed",
                "capacity_source": "legacy_json",
                "error": error,
            }
        ],
    )

    main(["--inputs", str(root), "--output", str(output), "--partial"])

    page = output / "index.html"
    assert error not in page.read_text(encoding="utf-8")
    passed, _, failed = dashboard_data(page)["conditions"]
    assert passed["expected_seeds"] == 5
    assert passed["completed_seeds"] == 0
    assert passed["coverage"] == 0
    assert passed["curve"] is None
    assert failed["capacity_source"] == "legacy_json"
    assert failed["capacity_error"] == error
    assert failed["expected_seeds"] == 0
    assert failed["coverage"] is None


def test_live_dashboard_publishes_terminal_status_and_provisional_statistics(tmp_path):
    from gymtosha.benchmarks import cartpole_report as report

    root = tmp_path / "input"
    output = tmp_path / "report"
    write_matrix(root)
    write_seed(root, 1001)
    write_seed(
        root, 1002, scores=(480,), status="failed", error="worker exited", duplicate_update=1
    )
    write_seed(root, 1003, scores=(480,), status="interrupted")
    write_seed(root, 1004, scores=(480,), status="running")

    capacity, results = report.analyze([root], partial=True)
    assert [seed.status for seed in next(iter(results.values()))] == [
        "reached",
        "failed",
        "interrupted",
        "incomplete",
        "missing",
    ]
    with pytest.raises(ValueError, match=r"seed-1002 \(failed\).*seed-1003 \(interrupted\)"):
        report.analyze([root])

    report.write_dashboard(output, capacity, results, partial=True)

    first_revision = json.loads((output / "revision.json").read_text())
    payload = json.loads((output / "dashboard-data.json").read_text())
    assert {"revision": payload["revision"]} == first_revision
    assert dashboard_data(output / "index.html") == payload
    passed, failed = payload["conditions"]
    assert passed["expected_seeds"] == 5
    assert passed["finished_seeds"] == 3
    assert passed["completed_seeds"] == 1
    assert passed["failed_seeds"] == 1
    assert passed["interrupted_seeds"] == 1
    assert passed["statistics_final"] is False
    assert passed["mean_target_training_seconds"] == 5
    assert passed["seeds"][1]["issue"].startswith("worker exited\n")
    assert "duplicate" in passed["seeds"][1]["issue"]
    assert passed["curve"]["mean"][:2] == [474, 475]
    assert failed["statistics_final"] is False
    assert rows(output / "seeds.csv")[1]["status"] == "failed"
    assert rows(output / "conditions.csv")[0]["finished_seeds"] == "3"
    assert rows(output / "same-n.csv")[0]["completed_seeds"] == "1"
    assert rows(output / "best-n.csv") == []
    assert not list(output.glob("*.png"))
    assert not (output / "checkpoints.csv").exists()

    for seed in (1002, 1003, 1004):
        shutil.rmtree(root / "events" / "gymtosha-cuda-0-16384" / f"seed-{seed}")
    for seed in (1002, 1003, 1004, 1005):
        write_seed(root, seed, scores=(474,) * 8)
    capacity, results = report.analyze([root], partial=True)
    report.write_dashboard(output, capacity, results, partial=True)

    payload = json.loads((output / "dashboard-data.json").read_text())
    assert payload["revision"] != first_revision["revision"]
    assert payload["conditions"][0]["statistics_final"] is True
    assert payload["conditions"][0]["completed_seeds"] == 5
    assert rows(output / "best-n.csv")[0]["completed_seeds"] == "5"


def test_live_analysis_reuses_only_unchanged_validated_seeds(tmp_path):
    from gymtosha.benchmarks.cartpole_report import analyze

    root = tmp_path / "input"
    write_matrix(root)
    for seed in range(1001, 1005):
        write_seed(root, seed)
    write_seed(root, 1005, status="running")
    cache = {}

    _, first_results = analyze([root], partial=True, cache=cache)
    first = next(iter(first_results.values()))
    _, same_results = analyze([root], partial=True, cache=cache)
    same = next(iter(same_results.values()))
    assert same[0] is first[0]
    assert same[4] is not first[4]

    seed_folder = root / "events" / "gymtosha-cuda-0-16384" / "seed-1005"
    metadata_path = seed_folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata_path.write_text(json.dumps({**metadata, "status": "completed"}))
    _, changed_results = analyze([root], partial=True, cache=cache)
    changed = next(iter(changed_results.values()))
    assert changed[4].status == "reached"
    _, strict_results = analyze([root], cache=cache)
    strict = next(iter(strict_results.values()))
    assert strict[0] == first[0]
    assert strict[0] is not first[0]

    events = next(seed_folder.glob("events.out.tfevents.*"))
    shutil.copyfile(events, seed_folder / "events.out.tfevents.duplicate")
    _, invalid_results = analyze([root], partial=True, cache=cache)
    invalid = next(iter(invalid_results.values()))
    assert invalid[4].status == "invalid"
    assert invalid[4].terminal is True
    assert "duplicate" in invalid[4].issue


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("eval/episode_returns", [475] * 99, "100 finite"),
        ("eval/episode_returns", [475] * 99 + [float("nan")], "100 finite"),
        ("train/valid_transitions", 12, "transitions"),
        ("time/training_seconds", 1, "monotonic"),
        ("train/optimizer_calls", 12, "optimizer"),
        ("train/ppo_update", 12, "PPO update"),
        ("ppo/approx_kl", float("nan"), "ppo/approx_kl.*finite"),
    ],
)
def test_invalid_raw_checkpoint_rejects_final_report(tmp_path, field, value, error):
    from gymtosha.benchmarks.cartpole_report import analyze

    def mutate(step, metrics):
        if step == 2:
            metrics[field] = value

    root = tmp_path / "input"
    write_matrix(root)
    write_seed(root, 1001, mutate=mutate)

    with pytest.raises(ValueError, match=error):
        analyze([root])


def test_extra_seed_directory_is_rejected(tmp_path):
    from gymtosha.benchmarks.cartpole_report import analyze

    root = tmp_path / "input"
    write_matrix(root)
    write_seed(root, 1006)

    with pytest.raises(ValueError, match="undeclared seed"):
        analyze([root])


@pytest.mark.parametrize("duplicate", [False, True])
def test_completed_seed_rejects_short_stream_or_duplicate_step(tmp_path, duplicate):
    from gymtosha.benchmarks.cartpole_report import analyze

    root = tmp_path / "input"
    write_matrix(root)
    if duplicate:
        write_seed(root, 1001, duplicate_update=2)
        error = "duplicate .* PPO update 2"
    else:
        write_seed(root, 1001, scores=(474, 475))
        error = "found 2 checkpoints, expected 8"

    with pytest.raises(ValueError, match=error):
        analyze([root])


def test_best_n_prioritizes_reach_rate_within_each_host(tmp_path):
    from gymtosha.benchmarks.cartpole_report import main

    rtx, mac = tmp_path / "rtx", tmp_path / "mac"
    output = tmp_path / "report"
    write_matrix(
        rtx,
        extra=[
            {
                "host": "RTX 3090",
                "backend": "gymtosha",
                "device": "cuda",
                "compile_env": False,
                "num_envs": 8192,
                "status": "passed",
            }
        ],
    )
    write_matrix(mac, host="Mac", device="mps")
    for seed in range(1001, 1006):
        rtx_early_scores = (475,) * 8 if seed == 1001 else (474,) * 8
        write_seed(rtx, seed, scores=rtx_early_scores)
        write_seed(rtx, seed, count=8192, scores=(474, 474) + (475,) * 14)
        write_seed(mac, seed, scores=(400,) * 8, host="Mac", device="mps")

    main(["--inputs", str(rtx), str(mac), "--output", str(output)])

    best = {row["host"]: row for row in rows(output / "best-n.csv")}
    assert best["RTX 3090"]["num_envs"] == "8192"
    assert best["RTX 3090"]["reached_seeds"] == "5"
    assert best["RTX 3090"]["evaluation_spacing_transitions"] == "524288"
    assert best["Mac"]["num_envs"] == "16384"
    assert best["Mac"]["mean_target_training_seconds"] == ""
    assert len(list(output.glob("best-n-*.png"))) == 2


def test_legacy_seed_preserves_source_and_missing_diagnostics(tmp_path):
    from gymtosha.benchmarks.cartpole_report import main

    def omit_historical_diagnostic(step, metrics):
        if step < 8:
            del metrics["time/collection_seconds"]

    root = tmp_path / "input"
    output = tmp_path / "report"
    write_matrix(root)
    write_seed(
        root,
        1001,
        source="legacy_json",
        missing_fields=("collection_seconds",),
        mutate=omit_historical_diagnostic,
    )
    main(["--inputs", str(root), "--output", str(output), "--partial"])

    seed, *_ = rows(output / "seeds.csv")
    assert seed["status"] == "reached"
    assert seed["source"] == "legacy_json"
    assert seed["missing_fields"] == "collection_seconds"


@pytest.mark.parametrize("source", ["live", "legacy_json"])
def test_undeclared_missing_diagnostic_is_rejected(tmp_path, source):
    from gymtosha.benchmarks.cartpole_report import analyze

    def omit_policy_loss(step, metrics):
        del metrics["ppo/policy_loss"]

    root = tmp_path / "input"
    write_matrix(root)
    write_seed(root, 1001, source=source, mutate=omit_policy_loss)

    with pytest.raises(ValueError, match="missing ppo/policy_loss"):
        analyze([root])


def test_seed_configuration_requires_matched_gae(tmp_path):
    from gymtosha.benchmarks.cartpole_report import analyze

    root = tmp_path / "input"
    write_matrix(root)
    write_seed(root, 1001, config_overrides={"gae_lambda": 0.95})

    with pytest.raises(ValueError, match="config gae_lambda"):
        analyze([root])
