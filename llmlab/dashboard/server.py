"""The dashboard's web server: the JSON API, the page, and starting and stopping runs.

Standard library only. It listens on 127.0.0.1, answers only requests addressed to
that host (so another website can't reach it through DNS tricks), and starts a run
only for a JSON request from its own page, naming a file in experiments/.

A run is an ordinary `python -m llmlab run` process. The server never measures
anything itself; it reads the run's live log (see live.py) when the page asks.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import psutil

from .. import db
from ..live import live_dir, read_events, read_samples
from . import api

STATIC = Path(__file__).with_name("static")
REPO = Path(__file__).resolve().parents[2]
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}
MAX_BODY = 10_000
DEMO_REFUSAL = ("This is the demo, which shows recorded results, so starting runs is off. "
                "A run needs llama.cpp and a model on your machine: docs/setup.md explains the setup.")


class RunConflict(RuntimeError):
    """A run is already in progress, or there is none to stop."""


class RunManager:
    """Starts `llmlab run` processes and reports their progress from the live log."""

    def __init__(self, db_path: str | Path, experiments_dir: str | Path, command=None):
        self.db_path = Path(db_path)
        self.experiments_dir = Path(experiments_dir).resolve()
        self.dir = live_dir(db_path)
        self.events_path = self.dir / "latest.jsonl"
        self.command = command or (
            lambda path: [sys.executable, "-m", "llmlab", "run", str(path), "--db", str(self.db_path)]
        )
        self.proc: subprocess.Popen | None = None
        self.run_id: str | None = None
        self.lock = threading.Lock()

    def experiment_path(self, name: str) -> Path:
        if not isinstance(name, str) or not re.fullmatch(r"[\w.\-]+\.yaml", name):
            raise LookupError("name an experiment file, such as smollm3-baseline.yaml")
        path = (self.experiments_dir / name).resolve()
        if path.parent != self.experiments_dir or not path.is_file():
            raise LookupError(f"no experiment file named {name}")
        return path

    def state(self, events: list[dict] | None = None) -> dict:
        events = read_events(self.events_path) if events is None else events
        mine = self.proc is not None
        if mine and self.proc.poll() is None and not events:
            return {"state": "starting"}
        if not events:
            if mine and self.proc.poll() is not None:
                # It exited before writing anything, for example over a bad experiment file.
                return {"state": "ended", "exit_code": self.proc.returncode, "output": self._output()}
            return {"state": "idle"}
        start = next((e for e in events if e["type"] == "start"), {})
        end = next((e for e in reversed(events) if e["type"] == "exit"), None)
        pid = start.get("pid")
        out = {"pid": pid, "started_here": mine and start.get("run_id") == self.run_id, "stopping": (self.dir / "stop").exists()}
        if end is not None:
            return {**out, "state": "ended", "exit_code": end.get("code")}
        if pid and psutil.pid_exists(pid):
            return {**out, "state": "running"}
        return {**out, "state": "ended", "exit_code": None, "output": self._output() if mine else None}

    def _output(self) -> str:
        try:
            return (self.dir / "process.log").read_text(encoding="utf-8", errors="replace")[-3000:]
        except OSError:
            return ""

    def start(self, name: str, allow_busy: bool = False) -> dict:
        path = self.experiment_path(name)
        with self.lock:
            if self.state()["state"] in ("starting", "running"):
                raise RunConflict("a run is already in progress")
            self.dir.mkdir(parents=True, exist_ok=True)
            self.events_path.unlink(missing_ok=True)  # so the page never shows the previous run as this one
            (self.dir / "stop").unlink(missing_ok=True)
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.run_id = uuid.uuid4().hex
            with open(self.dir / "process.log", "w", encoding="utf-8") as log:
                self.proc = subprocess.Popen(
                    self.command(path) + (["--allow-busy"] if allow_busy else []), cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, creationflags=flags, env={**os.environ, "LLMLAB_RUN_ID": self.run_id},
                )
        return {"state": "starting", "file": name}

    def stop(self) -> dict:
        if self.state()["state"] not in ("starting", "running"):
            raise RunConflict("nothing is running")
        (self.dir / "stop").write_text("requested from the dashboard\n", encoding="utf-8")
        return {"state": "stopping"}

    def poll(self, since: int = 0, samples_path: str | None = None, samples_offset: int = 0,
             with_samples: bool = True) -> dict:
        """Events after the first `since`, and telemetry samples the page hasn't seen yet."""
        events = read_events(self.events_path)
        current = next((e["path"] for e in reversed(events) if e["type"] == "samples"), None)
        samples: list[dict] = []
        if current and with_samples:
            if current != samples_path:
                samples_offset = 0
            rows, samples_offset = read_samples(current, samples_offset)
            samples = [api.compact_sample(r) for r in rows]
        return {
            **self.state(events),
            "events": events[since:],
            "next": len(events),
            "samples_path": current,
            "samples_offset": samples_offset,
            "samples": samples,
        }


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, db_path: Path, runs: RunManager, demo: bool = False):
        super().__init__(address, Handler)
        self.db_path = Path(db_path)
        self.runs = runs
        self.demo = demo  # recorded results only: runs can't be started
        port = self.server_address[1]
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self.allowed_origins = {f"http://{h}" for h in self.allowed_hosts}

    def read_connection(self) -> sqlite3.Connection | None:
        """A read-only connection, so the dashboard never blocks a run writing its results."""
        if not self.db_path.exists():
            return None
        conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn


class Handler(BaseHTTPRequestHandler):
    server: DashboardServer
    server_version = "llmlab-dashboard"

    def log_message(self, fmt, *args) -> None:
        pass  # quiet: the terminal shows the address and nothing else

    # ------------------------------------------------------------ responses

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj, default=str).encode("utf-8"), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _static(self, name: str) -> None:
        path = STATIC / name
        if not re.fullmatch(r"[\w.\-]+", name) or not path.is_file():
            return self._error(404, "not found")
        self._send(200, path.read_bytes(), CONTENT_TYPES.get(path.suffix, "application/octet-stream"))

    def _host_ok(self) -> bool:
        return self.headers.get("Host", "") in self.server.allowed_hosts

    # ------------------------------------------------------------ GET

    def do_GET(self) -> None:
        if not self._host_ok():
            return self._error(403, "this dashboard only answers requests to 127.0.0.1 or localhost")
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        parts = [p for p in url.path.split("/") if p]
        if not parts:
            return self._static("index.html")
        if parts[0] == "static" and len(parts) == 2:
            return self._static(parts[1])
        if parts[0] != "api":
            return self._error(404, "not found")
        try:
            return self._api_get(parts[1:], q)
        except (ValueError, KeyError) as e:
            return self._error(400, f"bad request: {e}")

    def _api_get(self, parts: list[str], q: dict) -> None:
        runs = self.server.runs
        if parts == ["live"]:
            return self._json(runs.poll(
                since=int(q.get("since", 0)),
                samples_path=q.get("samples_path"),
                samples_offset=int(q.get("samples_offset", 0)),
                with_samples=q.get("samples", "1") != "0",
            ))
        if parts == ["status"]:
            events = read_events(runs.events_path)
            session = next((e for e in events if e["type"] == "session"), {})
            variant = next((e for e in reversed(events) if e["type"] == "variant"), {})
            return self._json({
                **runs.state(events),
                "demo": self.server.demo,
                "experiment": session.get("experiment"),
                "variant": variant.get("label"),
                "index": variant.get("index"),
                "total": variant.get("total"),
            })
        if parts == ["experiments"]:
            return self._json(api.experiments(runs.experiments_dir))

        conn = self.server.read_connection()
        if conn is None:
            empty = {"overview": {"counts": {}, "latest": [], "quality": {}}, "sessions": [], "quality": []}
            return self._json(empty[parts[0]]) if parts[0] in empty else self._error(404, "no database yet")
        try:
            if parts == ["overview"]:
                return self._json(api.overview(conn))
            if parts == ["sessions"]:
                return self._json(api.sessions(conn))
            if parts == ["quality"]:
                return self._json(api.quality_results(conn))
            if parts == ["quality", "compare"]:
                return self._json(api.quality_compare(conn, int(q["a"]), int(q["b"])))
            if len(parts) == 2 and parts[0] == "quality":
                found = api.quality_detail(conn, int(parts[1]))
                return self._json(found) if found else self._error(404, f"no quality run for configuration {parts[1]}")
            if len(parts) == 2 and parts[0] == "sessions":
                found = api.session_detail(conn, int(parts[1]))
                return self._json(found) if found else self._error(404, f"there is no session {parts[1]}")
            if len(parts) == 3 and parts[0] == "configs" and parts[2] == "timeline":
                found = api.timeline(conn, int(parts[1]))
                return self._json(found) if found else self._error(404, f"there is no configuration {parts[1]}")
            if len(parts) == 3 and parts[0] == "configs" and parts[2] == "log":
                log = api.server_log(conn, int(parts[1]))
                if log is None:
                    return self._error(404, "no server log for this configuration")
                return self._send(200, log.encode("utf-8"), "text/plain; charset=utf-8")
        finally:
            conn.close()
        return self._error(404, "not found")

    # ------------------------------------------------------------ POST

    def do_POST(self) -> None:
        if not self._host_ok():
            return self._error(403, "this dashboard only answers requests to 127.0.0.1 or localhost")
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.server.allowed_origins:
            return self._error(403, "requests from other sites are refused")
        if self.server.demo:
            return self._error(403, DEMO_REFUSAL)
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            return self._error(415, "send JSON")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._error(413, "request too large")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._error(400, "the body is not valid JSON")
        path = urlparse(self.path).path
        runs = self.server.runs
        try:
            if path == "/api/runs":
                return self._json(runs.start(body.get("file"), allow_busy=body.get("allow_busy") is True), 202)
            if path == "/api/runs/stop":
                return self._json(runs.stop(), 202)
        except LookupError as e:
            return self._error(404, str(e))
        except RunConflict as e:
            return self._error(409, str(e))
        return self._error(404, "not found")


def make_server(db_path: str | Path, host: str = "127.0.0.1", port: int = 8765,
                experiments_dir: str | Path | None = None, command=None, demo: bool = False) -> DashboardServer:
    db.connect(db_path).close()  # create or upgrade the database once, before read-only use
    runs = RunManager(db_path, experiments_dir or REPO / "experiments", command)
    return DashboardServer((host, port), db_path, runs, demo=demo)


def serve(db_path: str | Path, port: int = 8765, open_browser: bool = False, demo: bool = False) -> int:
    try:
        server = make_server(db_path, port=port, demo=demo)
    except OSError as e:
        print(f"Couldn't listen on 127.0.0.1:{port} ({e.strerror or e}). "
              "Is a dashboard already open? Pass --port to use another port.")
        return 2
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Dashboard: {url}\nDatabase:  {db_path}\nPress Ctrl+C to stop the dashboard.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
