"""Refresh the CartPole dashboard when any local or remote seed ends."""

import argparse
import json
import os
import re
import selectors
import shlex
import signal
import subprocess
import time
from pathlib import Path, PurePosixPath

TERMINAL = {"completed", "failed", "interrupted"}
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10")


def terminal_state(root):
    """Identify individual terminal seeds, including status changes after retry."""
    entries = []
    for path in sorted((root / "events").glob("*/*/metadata.json")):
        try:
            with path.open() as file:
                record = json.load(file)
                revision = os.fstat(file.fileno()).st_mtime_ns
        except FileNotFoundError:
            continue
        status = record["status"]
        if status in TERMINAL:
            entries.append([path.relative_to(root).as_posix(), status, revision])
    return entries


def _snapshot(root):
    matrix = root / "matrix.json"
    try:
        revision = matrix.stat().st_mtime_ns
    except FileNotFoundError:
        revision = None
    return [revision, terminal_state(root)]


def _signal(root, interval):
    last = None
    while True:
        state = _snapshot(root)
        if state != last:
            print(json.dumps(state), flush=True)
            last = state
        time.sleep(interval)


def _sync(host, repo, root, mirror):
    mirror.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--delete-delay",
            "--include=/matrix.json",
            "--include=/protocol.json",
            "--include=/capacity-selected/",
            "--include=/capacity-selected/*.json",
            "--include=/events/***",
            "--exclude=*",
            f"{host}:{PurePosixPath(repo) / root}/",
            f"{mirror}/",
        ],
        check=True,
        timeout=180,
    )


def _publish(roots, output, cache):
    from gymtosha.benchmarks.cartpole_report import analyze, write_dashboard, write_report

    capacity, results = analyze(roots, partial=True, cache=cache)
    seeds = [seed for runs in results.values() for seed in runs]
    complete = sum(seed.status in ("reached", "censored") for seed in seeds)
    if seeds and complete == len(seeds):
        capacity, results = analyze(roots)
        write_report(output, capacity, results)
        print(f"Final dashboard: {complete}/{len(seeds)} seeds verified", flush=True)
        return True
    write_dashboard(output, capacity, results, partial=True)
    failed = sum(seed.status in ("failed", "interrupted", "invalid") for seed in seeds)
    print(f"Updated dashboard: {complete}/{len(seeds)} validated, {failed} with issues", flush=True)
    return False


def watch(args):
    roots = list(args.inputs)
    if args.ssh_host:
        roots.append(args.mirror)
        command = (
            f"cd {shlex.quote(args.remote_repo)} && "
            '"$HOME/.local/bin/uv" run --no-sync python -m gymtosha.benchmarks.cartpole_watch '
            + shlex.join(["--signal-root", args.remote_root, "--interval", str(args.interval)])
        )
    stream = None
    remote = None
    mirrored = None
    last = None
    cache = {}
    reconnect = 0
    pending = b""
    with selectors.DefaultSelector() as selector:
        try:
            while True:
                try:
                    if args.ssh_host and stream is None and time.monotonic() >= reconnect:
                        stream = subprocess.Popen(
                            ["ssh", *SSH_OPTIONS, args.ssh_host, command],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                        )
                        selector.register(stream.stdout, selectors.EVENT_READ)
                    if stream is not None and selector.select(timeout=args.interval):
                        chunk = os.read(stream.stdout.fileno(), 65_536)
                        if chunk:
                            lines = (pending + chunk).split(b"\n")
                            pending = lines.pop()
                            if lines:
                                remote = json.loads(lines[-1])
                        else:
                            selector.unregister(stream.stdout)
                            stream.stdout.close()
                            stream.wait(timeout=5)
                            stream = None
                            pending = b""
                            reconnect = time.monotonic() + 10
                    elif stream is None:
                        time.sleep(args.interval)
                    if args.ssh_host and remote is None:
                        continue
                    state = [_snapshot(root) for root in args.inputs], remote
                    if state == last:
                        continue
                    if args.ssh_host and remote != mirrored:
                        _sync(args.ssh_host, args.remote_repo, args.remote_root, args.mirror)
                        mirrored = remote
                    if _publish(roots, args.output, cache):
                        return
                    last = state
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    print(f"Retrying dashboard refresh: {error}", flush=True)
                    time.sleep(args.interval)
        finally:
            if stream is not None:
                stream.terminate()
                stream.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ssh-host")
    parser.add_argument("--remote-repo", default="source/GymTosha")
    parser.add_argument("--remote-root")
    parser.add_argument("--mirror", type=Path)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--signal-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 0.5 <= args.interval <= 300:
        parser.error("--interval must be between 0.5 and 300 seconds")
    if args.signal_root is not None:
        _signal(args.signal_root, args.interval)
        return
    if args.inputs is None or args.output is None:
        parser.error("--inputs and --output are required")
    if any((args.ssh_host, args.remote_root, args.mirror)) and not all(
        (args.ssh_host, args.remote_root, args.mirror)
    ):
        parser.error("--ssh-host, --remote-root and --mirror must be supplied together")
    if args.ssh_host and (
        not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.@-]*", args.ssh_host)
        or any(
            not re.fullmatch(r"[a-zA-Z0-9_./-]+", path)
            for path in (args.remote_repo, args.remote_root)
        )
    ):
        parser.error("unsupported SSH host or remote path")
    from gymtosha.benchmarks.study import study_lock

    signal.signal(signal.SIGTERM, signal.default_int_handler)
    with study_lock(args.output):
        watch(args)


if __name__ == "__main__":
    main()
