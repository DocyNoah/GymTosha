from gymtosha.benchmarks.capacity import search_capacity


def test_capacity_doubles_until_first_failure():
    seen = []

    def probe(n):
        seen.append(n)
        return {"num_envs": n, "status": "passed" if n <= 16384 else "failed"}

    result = search_capacity(probe)
    assert seen == [4096, 8192, 16384, 32768]
    assert result["maximum_verified"] == 16384
    assert result["first_failed"] == 32768


def test_capacity_halves_from_failed_start():
    seen = []

    def probe(n):
        seen.append(n)
        return {"num_envs": n, "status": "passed" if n <= 1024 else "failed"}

    result = search_capacity(probe)
    assert seen == [4096, 2048, 1024]
    assert result["maximum_verified"] == 1024
    assert result["first_failed"] == 2048


def test_async_capacity_never_exceeds_32():
    seen = []
    result = search_capacity(
        lambda n: seen.append(n) or {"num_envs": n, "status": "passed"}, asynchronous=True
    )
    assert seen == [32]
    assert result["maximum_verified"] == 32
    assert result["capped"]


def test_child_cleanup_falls_back_when_group_signal_is_denied(monkeypatch):
    import subprocess
    import sys

    from gymtosha.benchmarks.capacity import terminate_probe

    def denied(*args):
        raise PermissionError("group signal denied")

    monkeypatch.setattr("gymtosha.benchmarks.capacity.os.killpg", denied)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        terminate_probe(child)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_cleanup_fallback_also_terminates_worker_descendants(monkeypatch):
    import os
    import subprocess
    import sys

    from gymtosha.benchmarks.capacity import terminate_probe

    def denied(*args):
        raise PermissionError("group signal denied")

    monkeypatch.setattr("gymtosha.benchmarks.capacity.os.killpg", denied)
    code = """
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print(child.pid, flush=True)
time.sleep(60)
"""
    parent = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    child_pid = int(parent.stdout.readline())
    try:
        terminate_probe(parent)
        state = subprocess.run(
            ["ps", "-p", str(child_pid), "-o", "stat="], capture_output=True, text=True
        ).stdout.strip()
        assert not state or state.startswith("Z")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        try:
            os.kill(child_pid, 9)
        except ProcessLookupError:
            pass
        parent.stdout.close()


def test_probe_is_terminated_even_when_process_inventory_fails(monkeypatch):
    import subprocess
    import sys

    from gymtosha.benchmarks.capacity import terminate_probe

    def failed_inventory(*args, **kwargs):
        raise OSError("process inventory unavailable")

    monkeypatch.setattr("gymtosha.benchmarks.capacity.subprocess.check_output", failed_inventory)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        terminate_probe(child)
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
