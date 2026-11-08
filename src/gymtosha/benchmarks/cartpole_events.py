"""Write one TensorBoard event stream per CartPole comparison seed."""

import json
import time
from pathlib import Path

import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
from tensorboard.plugins.text import metadata as text_metadata
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.util.tensor_util import make_tensor_proto

from gymtosha.benchmarks.cli import _write

METRIC_TAGS = {
    "valid_transitions": "train/valid_transitions",
    "gradient_updates": "train/optimizer_calls",
    "ppo_update": "train/ppo_update",
    "completed_episodes": "train/completed_episodes",
    "wall_seconds": "time/training_seconds",
    "setup_seconds": "time/setup_seconds",
    "warmup_seconds": "time/warmup_seconds",
    "learner_setup_seconds": "time/learner_setup_seconds",
    "collection_seconds": "time/collection_seconds",
    "update_seconds": "time/update_seconds",
    **{
        name: f"ppo/{name}"
        for name in (
            "learning_rate",
            "entropy",
            "policy_loss",
            "value_loss",
            "approx_kl",
            "clip_fraction",
        )
    },
}


class EventRun:
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.metadata = {**metadata, "status": "running"}
        self.step = 0
        self.writer = None

    def __enter__(self):
        if self.path.exists():
            metadata_path = self.path / "metadata.json"
            if (
                metadata_path.exists()
                and json.loads(metadata_path.read_text())["status"] == "completed"
            ):
                raise ValueError(f"completed event stream already exists at {self.path}")
            archive = self.path.parent.parent.parent / "interrupted" / self.path.parent.name
            archive.mkdir(parents=True, exist_ok=True)
            self.path.rename(archive / f"{self.path.name}-{time.time_ns()}")
        self.path.mkdir(parents=True)
        self.writer = EventFileWriter(str(self.path))
        self._write_metadata()
        return self

    def _event(self, values):
        self.writer.add_event(
            Event(wall_time=time.time(), step=self.step, summary=Summary(value=values))
        )

    def _write_metadata(self):
        _write(self.metadata, self.path / "metadata.json")
        self._event(
            [
                Summary.Value(
                    tag="run/metadata",
                    tensor=make_tensor_proto(json.dumps(self.metadata, sort_keys=True)),
                    metadata=text_metadata.create_summary_metadata("Run metadata", ""),
                )
            ]
        )

    def record(self, point):
        step = point["ppo_update"]
        scores = np.asarray(point["episode_returns"], dtype=np.float32)
        if step != self.step + 1 or scores.shape != (100,) or not np.isfinite(scores).all():
            raise ValueError("each PPO update must have 100 finite evaluation returns")
        self.step = step
        scalars = {
            tag: point[name] for name, tag in METRIC_TAGS.items() if point.get(name) is not None
        }
        scalars.update(
            {"eval/mean_return": float(scores.mean()), "eval/std_return": float(scores.std())}
        )
        values = [
            Summary.Value(
                tag=tag,
                tensor=make_tensor_proto(np.asarray(value)),
                metadata=scalar_metadata.create_summary_metadata("", ""),
            )
            for tag, value in scalars.items()
        ]
        values.append(
            Summary.Value(
                tag="eval/episode_returns",
                tensor=make_tensor_proto(scores),
                metadata=SummaryMetadata(plugin_data={"plugin_name": "gymtosha"}),
            )
        )
        self._event(values)

    def finish(self):
        if self.step == 0:
            raise ValueError("cannot complete an event stream without checkpoints")
        self.metadata["status"] = "completed"

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            self.metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        elif self.metadata["status"] == "running":
            self.metadata["status"] = "interrupted"
        metadata = self.metadata
        # Flush the event metadata and checkpoints before publishing terminal status on disk.
        self._event(
            [
                Summary.Value(
                    tag="run/status",
                    tensor=make_tensor_proto(metadata["status"]),
                    metadata=text_metadata.create_summary_metadata("Run status", ""),
                )
            ]
        )
        self.writer.close()
        _write(metadata, self.path / "metadata.json")
