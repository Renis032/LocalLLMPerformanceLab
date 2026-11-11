"""Live progress of a running experiment, for the dashboard.

`llmlab run` writes one JSON event per line to live/latest.jsonl next to the database:
the session starting, each configuration, each run's result, the telemetry file being
written, and the end. The dashboard reads that file, whether the run was started from
the dashboard or from a terminal, so it never talks to the runner directly.

Stopping works the same way: the dashboard creates live/stop, and the runner checks for
it before each run and stops cleanly, shutting the server down.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def live_dir(db_path: str | Path) -> Path:
    return Path(db_path).resolve().parent / "live"


class LiveLog:
    """Writes events for the dashboard. Every write is flushed, so readers see it at once."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "latest.jsonl"
        self.stop_path = self.dir / "stop"
        self.stop_path.unlink(missing_ok=True)  # a stop request from an earlier run doesn't apply
        self._f = open(self.path, "w", encoding="utf-8")
        # LLMLAB_RUN_ID is set by the dashboard, so it knows which run it started.
        self.emit("start", pid=os.getpid(), run_id=os.environ.get("LLMLAB_RUN_ID"))

    def emit(self, event: str, /, **fields) -> None:
        if self._f.closed:
            return
        self._f.write(json.dumps({"type": event, "time": time.time(), **fields}, default=str) + "\n")
        self._f.flush()

    def stop_requested(self) -> bool:
        return self.stop_path.exists()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


class NoLiveLog:
    """Used when nothing is watching: events go nowhere and a stop is never requested."""

    def emit(self, event: str, /, **fields) -> None:
        pass

    def stop_requested(self) -> bool:
        return False

    def close(self) -> None:
        pass


class StopRequested(KeyboardInterrupt):
    """The dashboard asked the run to stop. Handled like Ctrl+C: the session is marked aborted."""


def check_stop(live) -> None:
    if live.stop_requested():
        raise StopRequested()


def read_events(path: str | Path) -> list[dict]:
    """Every complete event in a live log. A line still being written is skipped."""
    events = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return events
    for line in text.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events


def read_samples(path: str | Path, offset: int = 0) -> tuple[list[dict], int]:
    """Telemetry samples written after byte `offset`, and the offset to continue from."""
    rows: list[dict] = []
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return rows, offset
    end = data.rfind(b"\n") + 1  # only whole lines; the rest comes next time
    for line in data[:end].splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows, offset + end
