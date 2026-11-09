import csv
import json
import re
import shutil

import pytest


def write_matrix(
    root, *, host="RTX 3090", seeds=range(1001, 1006), device="cuda", extra=(), budget=2_097_152
):
    root.mkdir()
    (root / "matrix.json").write_text(
        json.dumps(
            {
                "conditions": [
                    {
                        "backend": "gymtosha",
                        "device": device,
                        "compile_env": False,
                        "num_envs": 1024,
                        "host": host,
                        "status": "passed",
                    },
                    {
                        "backend": "gymtosha",
                        "device": device,
                        "compile_env": True,
                        "num_envs": 1024,
                        "host": host,
                        "status": "failed",
                    },
                    *extra,
                ],
                "seeds": list(seeds),
                "horizon": 64,
                "budget": budget,
            }
        )
    )


def write_seed(
    root,
    seed,
    *,
    scores=(474, 475, 450) + (480,) * 29,
    mutate=None,
    status="completed",
    missing_fields=(),
    source="live",
    count=1024,
    host="RTX 3090",
    device="cuda",
    duplicate_update=None,
    config_overrides=None,
    error=None,
    reused_prefix=None,
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
            optimizer_calls = count * step // 2
            metrics = {
                "eval/episode_returns": [score] * 100,
                "eval/mean_return": 500,
                "train/valid_transitions": count * 64 * step,
                "time/training_seconds": 2.5 * step,
                "train/optimizer_calls": optimizer_calls,
                "train/ppo_update": step,
                "train/completed_episodes": 20 * step,
                "time/setup_seconds": 0.1,
                "time/warmup_seconds": 0.2,
                "time/learner_setup_seconds": 0.3,
                "time/collection_seconds": 0.4 * step,
                "time/update_seconds": 0.5 * step,
                "ppo/learning_rate": 0.003 * (1 - (optimizer_calls - 1) / 65_536),
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
                    "total_timesteps": 2_097_152,
                    "lr_schedule_timesteps": 8_388_608,
                    "learning_rate": 0.003,
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
                **({"reused_prefix": reused_prefix} if reused_prefix is not None else {}),
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
    write_seed(root, 1002, scores=(474, 474, 475, 490) + (480,) * 28)
    for seed in (1003, 1004):
        write_seed(root, seed, scores=(474,) * 32)
    write_seed(root, 1005, scores=(474,) * 32, mutate=slower_training)

    main(["--inputs", str(root), "--output", str(output)])

    seed, second, censored, *_ = rows(output / "seeds.csv")
    assert seed["status"] == "reached"
    assert seed["target_update"] == "2"
    assert seed["target_transitions"] == "131072"
    assert float(seed["target_training_seconds"]) == 5
    assert seed["target_optimizer_calls"] == "1024"
    assert second["target_update"] == "3"
    assert censored["status"] == "censored"
    assert censored["target_transitions"] == ""
    assert censored["last_transitions"] == "2097152"
    assert censored["last_optimizer_calls"] == "16384"
    assert float(censored["last_mean_return"]) == 474

    (condition,) = rows(output / "conditions.csv")
    assert condition["host"] == "RTX 3090"
    assert condition["evaluation_spacing_transitions"] == "65536"
    assert condition["completed_seeds"] == "5"
    assert condition["censored_seeds"] == "3"
    assert condition["reached_seeds"] == "2"
    assert float(condition["reach_rate"]) == 0.4
    assert float(condition["mean_target_transitions"]) == 163840
    assert float(condition["sd_target_transitions"]) == pytest.approx(46340.95001)
    assert float(condition["mean_target_training_seconds"]) == 6.25
    assert float(condition["sd_target_training_seconds"]) == pytest.approx(1.76776695)
    assert len(rows(output / "checkpoints.csv")) == 160
    assert len(rows(output / "capacity.csv")) == 2
    assert len(rows(output / "best-n.csv")) == 1
    assert list(output.glob("same-n-*.png"))
    assert list(output.glob("best-n-*.png"))

    payload = dashboard_data(output / "index.html")
    assert payload["partial"] is False
    assert (payload["target"], payload["budget"], payload["horizon"]) == (475, 2097152, 64)
    assert payload["lr_schedule_timesteps"] == 8388608
    assert payload["generated_at"].endswith("+00:00")
    passed, failed = payload["conditions"]
    assert passed["id"] == "RTX 3090/gymtosha-cuda-0-1024"
    assert passed["status"] == "passed"
    assert passed["best"] is True
    assert passed["reached_seeds"] == 2
    assert passed["mean_target_training_seconds"] == 6.25
    assert passed["curve"]["x"][:2] == [65536, 131072]
    assert passed["curve"]["x"][-1] == 2097152
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
    write_seed(root, 1001, scores=(474,) * 32)
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
    assert passed["curve"]["mean"] == [474.0] * 32
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
                "num_envs": 512,
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
        shutil.rmtree(root / "events" / "gymtosha-cuda-0-1024" / f"seed-{seed}")
    for seed in (1002, 1003, 1004, 1005):
        write_seed(root, seed, scores=(474,) * 32)
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

    seed_folder = root / "events" / "gymtosha-cuda-0-1024" / "seed-1005"
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
        ("ppo/learning_rate", 0.00281268310546875, "learning rate.*optimizer"),
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


def test_old_budget_matrix_is_rejected(tmp_path):
    from gymtosha.benchmarks.cartpole_report import analyze

    root = tmp_path / "input"
    write_matrix(root, budget=8_388_608)

    with pytest.raises(ValueError, match="budget 2097152"):
        analyze([root])


@pytest.mark.parametrize("seeds", [range(1001, 1007), [1001, 1002, 1003, 1004, 1004]])
def test_matrix_requires_exact_protocol_seeds(tmp_path, seeds):
    from gymtosha.benchmarks.cartpole_report import analyze

    root = tmp_path / "input"
    write_matrix(root, seeds=seeds)

    with pytest.raises(ValueError, match="exactly five seeds"):
        analyze([root], partial=True)


def test_report_accepts_next_seed_block_but_rejects_mixing_blocks(tmp_path):
    from gymtosha.benchmarks.cartpole_report import analyze

    first, second = tmp_path / "first", tmp_path / "second"
    write_matrix(first)
    write_matrix(second, host="Mac", device="mps", seeds=range(1006, 1011))
    _, results = analyze([second], partial=True)
    assert [run.seed for runs in results.values() for run in runs] == list(range(1006, 1011))
    with pytest.raises(ValueError, match="same seed block"):
        analyze([first, second], partial=True)


def test_follow_up_extends_only_matching_host_conditions(tmp_path):
    from gymtosha.benchmarks.cartpole_report import analyze, write_dashboard

    rtx, mac, extra = (tmp_path / name for name in ("rtx", "mac", "extra"))
    write_matrix(rtx)
    write_matrix(mac, host="Mac", device="mps")
    write_matrix(extra, host="Mac", device="mps", seeds=range(1006, 1011))
    for folder, seeds in ((mac, range(1001, 1006)), (extra, range(1006, 1011))):
        for seed in seeds:
            write_seed(folder, seed, host="Mac", device="mps")

    capacity, results = analyze([rtx, mac], partial=True, follow_up=[extra])
    out = tmp_path / "report"
    write_dashboard(out, capacity, results, partial=True)
    dashboard = json.loads((out / "dashboard-data.json").read_text())
    passed = {row["host"]: row for row in dashboard["conditions"] if row["status"] == "passed"}
    assert passed["RTX 3090"]["expected_seeds"] == 5
    assert passed["Mac"]["expected_seeds"] == passed["Mac"]["finished_seeds"] == 10
    assert passed["Mac"]["completed_seeds"] == passed["Mac"]["reached_seeds"] == 10
    assert [run["seed"] for run in passed["Mac"]["seeds"]] == list(range(1001, 1011))
    assert passed["Mac"]["statistics_final"]
    with pytest.raises(ValueError, match="duplicate follow-up seeds"):
        analyze([rtx, mac], partial=True, follow_up=[mac])


def test_matrix_rejects_capacity_condition_above_protocol_cap(tmp_path):
    from gymtosha.benchmarks.cartpole_report import analyze

    root = tmp_path / "input"
    write_matrix(
        root,
        extra=[
            {
                "host": "RTX 3090",
                "backend": "gymtosha",
                "device": "cuda",
                "compile_env": False,
                "num_envs": 2048,
                "status": "passed",
            }
        ],
    )

    with pytest.raises(ValueError, match="invalid capacity condition"):
        analyze([root], partial=True)


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
        error = "found 2 checkpoints, expected 32"

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
                "num_envs": 512,
                "status": "passed",
            }
        ],
    )
    write_matrix(mac, host="Mac", device="mps")
    for seed in range(1001, 1006):
        rtx_early_scores = (475,) * 32 if seed == 1001 else (474,) * 32
        write_seed(rtx, seed, scores=rtx_early_scores)
        write_seed(rtx, seed, count=512, scores=(474, 474) + (475,) * 62)
        write_seed(mac, seed, scores=(400,) * 32, host="Mac", device="mps")

    main(["--inputs", str(rtx), str(mac), "--output", str(output)])

    best = {row["host"]: row for row in rows(output / "best-n.csv")}
    assert best["RTX 3090"]["num_envs"] == "512"
    assert best["RTX 3090"]["reached_seeds"] == "5"
    assert best["RTX 3090"]["evaluation_spacing_transitions"] == "32768"
    assert best["Mac"]["num_envs"] == "1024"
    assert best["Mac"]["mean_target_training_seconds"] == ""
    assert len(list(output.glob("best-n-*.png"))) == 2


def test_legacy_seed_preserves_source_and_missing_diagnostics(tmp_path):
    from gymtosha.benchmarks.cartpole_report import main

    def omit_historical_diagnostic(step, metrics):
        del metrics["ppo/learning_rate"]
        if step < 32:
            del metrics["time/collection_seconds"]

    root = tmp_path / "input"
    output = tmp_path / "report"
    write_matrix(root)
    write_seed(
        root,
        1001,
        source="legacy_json",
        missing_fields=("collection_seconds", "learning_rate"),
        mutate=omit_historical_diagnostic,
    )
    main(["--inputs", str(root), "--output", str(output), "--partial"])

    seed, *_ = rows(output / "seeds.csv")
    assert seed["status"] == "reached"
    assert seed["source"] == "legacy_json"
    assert seed["missing_fields"] == "collection_seconds,learning_rate"


@pytest.mark.parametrize("source", ["live", "legacy_json"])
def test_reused_seed_preserves_original_source_and_prefix_provenance(tmp_path, source):
    from gymtosha.benchmarks.cartpole_report import main

    root = tmp_path / "input"
    output = tmp_path / "report"
    write_matrix(root)
    provenance = {"budget": 8_388_608, "path": "/original/events/seed-1001", "code_version": "test"}
    write_seed(root, 1001, source=source, reused_prefix=provenance)
    write_seed(root, 1002)

    main(["--inputs", str(root), "--output", str(output), "--partial"])

    reused, fresh, *_ = rows(output / "seeds.csv")
    assert reused["source"] == source
    assert json.loads(reused["reused_prefix"]) == provenance
    assert fresh["reused_prefix"] == ""
    (condition,) = rows(output / "conditions.csv")
    assert condition["reused_prefix_seeds"] == "1"
    passed, _ = dashboard_data(output / "index.html")["conditions"]
    assert passed["reused_prefix_seeds"] == 1
    assert passed["seeds"][0]["reused_prefix"] == provenance
    assert passed["seeds"][1]["reused_prefix"] is None
    assert {(row["seed"], row["reused_prefix"]) for row in rows(output / "checkpoints.csv")} == {
        ("1001", "True"),
        ("1002", "False"),
    }


def test_reused_seed_requires_original_budget_provenance(tmp_path):
    from gymtosha.benchmarks.cartpole_report import analyze

    root = tmp_path / "input"
    write_matrix(root)
    write_seed(
        root,
        1001,
        reused_prefix={"budget": 2_097_152, "path": "/original/seed-1001", "code_version": "test"},
    )

    with pytest.raises(ValueError, match="reused prefix provenance"):
        analyze([root])


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


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("gae_lambda", 0.95),
        ("learning_rate", 0.001),
        ("total_timesteps", 8_388_608),
        ("lr_schedule_timesteps", 2_097_152),
        ("lr_schedule_timesteps", None),
    ],
)
def test_seed_configuration_requires_matched_training_protocol(tmp_path, name, value):
    from gymtosha.benchmarks.cartpole_report import analyze

    root = tmp_path / "input"
    write_matrix(root)
    write_seed(root, 1001, config_overrides={name: value})

    with pytest.raises(ValueError, match=f"config {name}"):
        analyze([root])
