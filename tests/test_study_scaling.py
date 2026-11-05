from argparse import Namespace
from dataclasses import replace

import pytest

from gymtosha.benchmarks.study import ROLLOUT_UPDATES, reference_config


def test_joint_candidates_preserve_reference_and_vary_only_scaling_parameters():
    from gymtosha.benchmarks.study_scaling import scaling_candidates

    previous = reference_config(32)
    baseline = reference_config(256, previous=previous)
    candidates = scaling_candidates(256, previous)
    assert candidates["reference"] == baseline
    assert 4 <= len(candidates) <= 10
    assert any(
        config.rollout_steps != baseline.rollout_steps
        and config.ppo_minibatch_size != baseline.ppo_minibatch_size
        and config.learning_rate != baseline.learning_rate
        for config in candidates.values()
    )
    for config in candidates.values():
        config.validate()
        assert config.num_envs == 256
        assert config.total_timesteps == 256 * config.rollout_steps * ROLLOUT_UPDATES
        assert config.gamma == previous.gamma
        assert config.gae_lambda == previous.gae_lambda
        assert config.clip_coef == previous.clip_coef
        assert config.hidden_sizes == [256, 256, 256]


def test_non_anchor_presets_follow_interpolated_and_last_anchor_rules():
    from gymtosha.benchmarks.study_scaling import SCALING_ANCHORS, scaled_presets

    base = reference_config(32)
    middle = replace(
        reference_config(256),
        rollout_steps=16,
        total_timesteps=256 * 16 * 64,
        ppo_minibatch_size=256,
        learning_rate=0.0024,
    )
    anchors = {
        32: base,
        256: middle,
        1024: reference_config(1024, previous=middle),
        4096: reference_config(4096, previous=middle),
    }
    presets = scaled_presets(anchors, 131072)
    assert sorted(map(int, presets)) == [2**power for power in range(5, 18)]
    assert [
        int(key) for key, row in presets.items() if row["selection_basis"] == "anchor_search"
    ] == list(SCALING_ANCHORS)
    middle_preset = presets["64"]
    assert middle_preset["anchor_counts"] == [32, 256]
    assert middle_preset["config"]["rollout_steps"] == 64
    assert middle_preset["config"]["ppo_minibatch_size"] == 256
    assert middle_preset["config"]["learning_rate"] == pytest.approx(0.0006)
    large_preset = presets["131072"]
    assert large_preset["anchor_counts"] == [4096]
    assert large_preset["config"]["rollout_steps"] == 16
    assert large_preset["config"]["ppo_minibatch_size"] == 256 * 131072 // 256


def test_only_anchor_counts_are_tuned_and_resume_does_not_research(tmp_path, monkeypatch):
    import json

    from gymtosha.benchmarks.study_scaling import tune_anchors

    searched = []

    def tune_base(config, output, *, parallelism, reuse_other_devices, fixed_parameters):
        assert fixed_parameters == {"gae_lambda": 0.99}
        searched.append(config.num_envs)
        return config

    def tune_joint(config, previous, output, *, parallelism):
        searched.append(config.num_envs)
        return config

    monkeypatch.setattr("gymtosha.benchmarks.study_scaling.tune_count", tune_base)
    monkeypatch.setattr(
        "gymtosha.benchmarks.study_scaling.choose_search_device", lambda config, *args: config
    )
    monkeypatch.setattr("gymtosha.benchmarks.study_scaling.tune_joint_count", tune_joint)
    monkeypatch.setattr("gymtosha.benchmarks.study_plot.render_dashboard", lambda *args: None)
    args = Namespace(
        min_envs=32, max_envs=131072, device="cuda", output=tmp_path, parallel_trials=8
    )
    tune_anchors(args)
    tune_anchors(args)
    assert searched == [32, 256, 1024, 4096]
    assert len(json.loads((tmp_path / "selected.json").read_text())) == 13


def test_device_probes_are_isolated_and_choose_separately_for_each_count(tmp_path, monkeypatch):
    from gymtosha.benchmarks.study_scaling import DEVICE_PROBE_SEED, choose_search_device

    calls = []

    def probe(config, evaluation, output):
        calls.append(config)
        faster = "cpu" if config.num_envs == 32 else "cuda"
        seconds = (1 if config.device == faster else 2) + (0 if config.compile_env else 0.1)
        return {
            "status": "completed",
            "trial_id": str(len(calls)),
            "result": {"wall_seconds": seconds},
        }

    monkeypatch.setattr("gymtosha.benchmarks.study_scaling.run_trial", probe)
    for count, expected in ((32, "cpu"), (256, "cuda")):
        base = reference_config(count)
        config = choose_search_device(base, base, "cuda", tmp_path)
        assert config.device == expected and config.compile_env
        assert choose_search_device(base, base, "cuda", tmp_path) == config
    assert len(calls) == 8
    assert {config.seed for config in calls} == {DEVICE_PROBE_SEED}
    assert {
        (config.num_envs, config.total_timesteps // (config.num_envs * config.rollout_steps))
        for config in calls
    } == {(32, 64), (256, 8)}


def test_fixed_gae_value_bypasses_candidate_search(tmp_path, monkeypatch):
    from gymtosha.benchmarks.study import tune_count

    def unexpected_search(*args, **kwargs):
        raise AssertionError("fixed parameters must not be searched")

    monkeypatch.setattr("gymtosha.benchmarks.study.PARAMETERS", {"gae_lambda": [0.9, 0.99]})
    monkeypatch.setattr("gymtosha.benchmarks.study.collect_candidates", unexpected_search)
    assert (
        tune_count(reference_config(32), tmp_path, fixed_parameters={"gae_lambda": 0.99}).gae_lambda
        == 0.99
    )
