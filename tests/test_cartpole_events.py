import json

import numpy as np
import pytest

from gymtosha.benchmarks.study import study_evaluation
from gymtosha.benchmarks.train import TrainConfig, train


def test_live_cartpole_records_raw_events_after_each_update(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.util.tensor_util import make_ndarray

    from gymtosha.benchmarks.cartpole_events import EventRun

    config = TrainConfig(
        num_envs=4,
        rollout_steps=16,
        total_timesteps=128,
        update_epochs=1,
        ppo_minibatch_size=16,
        hidden_sizes=[8],
    )
    with EventRun(tmp_path / "run", {"source": "live"}) as run:
        result = train(
            config,
            evaluation=study_evaluation(config, comparison=True, each_update=True),
            on_checkpoint=run.record,
        )
        run.finish()
    assert "time_to_target_seconds" not in result["evaluation"]
    path = tmp_path / "run"
    assert json.loads((path / "metadata.json").read_text())["status"] == "completed"
    events = EventAccumulator(str(path), size_guidance={"tensors": 0}).Reload()

    def values(tag):
        return [make_ndarray(e.tensor_proto) for e in events.Tensors(tag)]

    assert values("train/valid_transitions") == [64, 128]
    assert values("train/optimizer_calls") == [4, 8]
    assert values("train/ppo_update") == [1, 2]
    assert len(values("eval/episode_returns")) == 2
    assert all(point.shape == (100,) for point in values("eval/episode_returns"))
    assert np.all(np.diff(values("time/training_seconds")) > 0)
    assert values("time/training_seconds")[-1] <= result["wall_seconds"]
    assert np.all(np.diff(values("time/collection_seconds")) > 0)
    assert np.all(np.diff(values("time/update_seconds")) > 0)
    assert all(value >= 0 for value in values("train/completed_episodes"))
    for name in (
        "learning_rate",
        "entropy",
        "policy_loss",
        "value_loss",
        "approx_kl",
        "clip_fraction",
    ):
        assert all(np.isfinite(v) for v in values(f"ppo/{name}"))
