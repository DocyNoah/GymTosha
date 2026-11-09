import json
from argparse import Namespace

from gymtosha.benchmarks import cartpole_watch


def test_terminal_state_includes_failed_and_completed_seeds(tmp_path):
    def record(seed, status):
        metadata = tmp_path / "events" / "condition" / f"seed-{seed}" / "metadata.json"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps({"status": status}))

    record(1001, "running")
    assert cartpole_watch.terminal_state(tmp_path) == []
    record(1001, "completed")
    record(1002, "failed")
    assert [row[1] for row in cartpole_watch.terminal_state(tmp_path)] == ["completed", "failed"]
    record(1002, "running")
    assert [row[1] for row in cartpole_watch.terminal_state(tmp_path)] == ["completed"]


def test_watcher_publishes_each_terminal_seed_without_waiting_for_a_condition(
    tmp_path, monkeypatch
):
    changes = iter([[], ["first"], ["first"], ["first", "second"]])
    published = []

    def publish(roots, output, cache, *, follow_up):
        assert not follow_up
        published.append(cache)
        return len(published) == 3

    monkeypatch.setattr(cartpole_watch, "terminal_state", lambda root: next(changes))
    monkeypatch.setattr(cartpole_watch.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(cartpole_watch, "_publish", publish)
    args = Namespace(
        inputs=[tmp_path], output=tmp_path / "report", ssh_host=None, interval=2, follow_up=()
    )
    cartpole_watch.watch(args)
    assert len(published) == 3
    assert all(cache is published[0] for cache in published)
