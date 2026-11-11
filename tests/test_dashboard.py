"""Purpose: the dashboard shows what the database holds, follows live runs, and is safe to leave open.

Runs experiments against the stand-in engines while a live log records them, then
starts the dashboard server on a free port and checks its API: the numbers it serves,
starting and stopping a run, and the requests it must refuse (other hosts, other
sites, files outside experiments/).
"""

import re
import sys
import textwrap
import threading
import time
from pathlib import Path

import httpx
import pytest

from llmlab import db
from llmlab.dashboard.server import make_server
from llmlab.live import LiveLog, StopRequested, read_events, read_samples
from llmlab.runner import run_experiment

from conftest import FakeGpu, make_cloud_experiment, make_experiment

STATIC = Path(__file__).resolve().parents[1] / "llmlab" / "dashboard" / "static"


def _quiet(*args):
    pass


# ---------------------------------------------------------------- live log


def test_the_live_log_and_samples_are_read_incrementally(tmp_path):
    live = LiveLog(tmp_path / "live")
    live.emit("session", session_id=7)
    events = read_events(live.path)
    assert [e["type"] for e in events] == ["start", "session"] and events[1]["session_id"] == 7

    samples = tmp_path / "samples.jsonl"
    samples.write_text('{"t": 1}\n{"t": 2}\n{"t": 3', encoding="utf-8")  # the last line is still being written
    rows, offset = read_samples(samples, 0)
    assert [r["t"] for r in rows] == [1, 2]
    with open(samples, "a", encoding="utf-8") as f:
        f.write("}\n")
    rows, _ = read_samples(samples, offset)
    assert [r["t"] for r in rows] == [3]


def test_a_run_reports_each_step_to_the_live_log(fake_engine, tmp_path):
    live = LiveLog(tmp_path / "live")
    run_experiment(make_experiment(tmp_path), db.connect(":memory:"), gpu=FakeGpu(), echo=_quiet, live=live)
    kinds = [e["type"] for e in read_events(live.path)]
    assert kinds == ["start", "session", "variant", "samples", "phase", "loaded",
                     "trial", "trial", "trial", "trial", "phase", "variant_done", "done"]
    events = read_events(live.path)
    trial = next(e for e in events if e["type"] == "trial" and e["kind"] == "measured")
    assert trial["gen_tps"] > 0 and trial["t_send"] < trial["t_last"]
    done = next(e for e in events if e["type"] == "variant_done")
    assert done["status"] == "ok" and done["runs"] == 3 and done["gen_tps"] > 0


def test_a_cloud_run_reports_to_the_live_log_too(fake_claude, tmp_path):
    live = LiveLog(tmp_path / "live")
    run_experiment(make_cloud_experiment(tmp_path), db.connect(":memory:"), gpu=None, echo=_quiet, live=live)
    trials = [e for e in read_events(live.path) if e["type"] == "trial"]
    assert len(trials) == 3 and all(t["gen_tokens"] == 40 for t in trials)


class StopAfter(LiveLog):
    """Asks the run to stop once it has reported two runs, as the dashboard's Stop button would."""

    def stop_requested(self) -> bool:
        return sum(e["type"] == "trial" for e in read_events(self.path)) >= 2


def test_a_stop_request_ends_the_run_cleanly(fake_engine, tmp_path):
    conn = db.connect(":memory:")
    live = StopAfter(tmp_path / "live")
    with pytest.raises(StopRequested):
        run_experiment(make_experiment(tmp_path), conn, gpu=FakeGpu(), echo=_quiet, live=live)
    session = db.get_session(conn, 1)
    assert session["status"] == "aborted"
    (config,) = db.get_configs(conn, 1)
    assert config["error"] == "stopped from the dashboard" and len(db.get_trials(conn, config["id"])) == 2
    assert read_events(live.path)[-1]["type"] == "done"


# ---------------------------------------------------------------- the server

FAKE_RUN = textwrap.dedent("""
    import sys, time
    from llmlab.live import LiveLog
    live = LiveLog(sys.argv[1])
    live.emit("session", session_id=99, experiment="fake", file=sys.argv[2], args=sys.argv[3:], variants=[])
    for i in range(100):
        if live.stop_requested():
            live.emit("exit", code=130)
            sys.exit(130)
        time.sleep(0.05)
    live.emit("exit", code=0)
""")


@pytest.fixture
def dashboard(fake_engine, tmp_path):
    db_path = tmp_path / "lab.sqlite"
    conn = db.connect(db_path)
    run_experiment(make_experiment(tmp_path), conn, gpu=FakeGpu(), echo=_quiet)
    conn.close()
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    (experiments / "quick.yaml").write_text("name: quick\nmodel:\n  path: model.gguf\nworkload:\n  prompt: hi\n", encoding="utf-8")
    live = tmp_path / "live"
    server = make_server(db_path, port=0, experiments_dir=experiments,
                         command=lambda path: [sys.executable, "-c", FAKE_RUN, str(live), str(path)])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
        yield client, server
    server.shutdown()
    server.server_close()


def test_the_page_and_its_files_are_served(dashboard):
    client, _ = dashboard
    page = client.get("/")
    assert page.status_code == 200 and "LLM Performance Lab" in page.text
    js = client.get("/static/app.js")
    assert js.status_code == 200 and js.headers["content-type"].startswith("text/javascript")
    assert client.get("/static/..%2Fserver.py").status_code == 404
    assert client.get("/static/missing.js").status_code == 404


def test_the_page_never_animates_while_a_run_is_watched():
    # An endless animation repaints the page every frame. Watching a run with one cost 3% of
    # the CPU and slowed generation 3% (docs/measurement-findings.md, finding 10).
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert "infinite" not in css and "@keyframes" not in css
    assert not re.search(r"transition:[^;]*\bwidth\b", css), "animating a size re-lays out the page every frame"


def test_the_api_serves_what_the_database_holds(dashboard):
    client, _ = dashboard
    overview = client.get("/api/overview").json()
    assert overview["counts"]["sessions"] == 1 and overview["counts"]["runs"] == 3
    (latest,) = overview["latest"]
    assert latest["label"] == "base" and latest["runs"] == 3 and latest["gen_tps"] > 0

    (session,) = client.get("/api/sessions").json()
    assert session["status"] == "completed" and session["configs"][0]["status"] == "ok"
    detail = client.get("/api/sessions/1").json()
    config = detail["configs"][0]
    assert config["summary"]["gen_tps"]["median"] == pytest.approx(latest["gen_tps"])
    assert len(config["trials"]) == 4 and "itl_json" not in config["trials"][0]
    assert client.get("/api/sessions/9").status_code == 404

    timeline = client.get(f"/api/configs/{config['id']}/timeline").json()
    assert [p["name"] for p in timeline["phases"]] == ["load", "quiet"] and len(timeline["trials"]) == 4
    assert timeline["first_measured_s"] > 0
    assert "listening on" in client.get(f"/api/configs/{config['id']}/log").text

    (exp,) = client.get("/api/experiments").json()
    assert exp["file"] == "quick.yaml" and exp["requests"] == 6 and exp["variants"][0]["model_found"] is False

    # This configuration wasn't scored, so the quality routes have nothing to show.
    assert config["quality"] is None and client.get("/api/quality").json() == []
    assert client.get(f"/api/quality/{config['id']}").status_code == 404
    assert client.get(f"/api/quality/compare?a={config['id']}&b={config['id']}").json() is None
    assert client.get("/api/quality/compare?a=x&b=1").status_code == 400


def test_the_quality_routes_serve_a_scored_run(fake_engine, tmp_path):
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        'name: mini\nversion: 1\ntasks:\n'
        '  - {id: t, category: exact, prompt: "say: yes", answers: ["yes"], reference: "yes", wrong: "no"}\n',
        encoding="utf-8",
    )
    db_path = tmp_path / "lab.sqlite"
    conn = db.connect(db_path)
    exp = make_experiment(tmp_path, quality={"suite": str(suite), "repeats": 2})
    run_experiment(exp, conn, gpu=FakeGpu(), echo=_quiet)
    conn.close()
    server = make_server(db_path, port=0, experiments_dir=tmp_path)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_address[1]}", timeout=10) as client:
            (row,) = client.get("/api/quality").json()
            assert row["score"] == 1 and row["correct"] == 2 and row["cost"]["kind"] is None
            detail = client.get(f"/api/quality/{row['config_id']}").json()
            assert detail["prompts"]["t"] == "say: yes" and len(detail["answers"]) == 2
            both = client.get(f"/api/quality/compare?a={row['config_id']}&b={row['config_id']}").json()
            assert both["same_suite"] and both["difference"]["verdict"] == "no detectable difference"
            session = client.get(f"/api/sessions/{row['session_id']}").json()
            assert session["configs"][0]["quality"]["score"] == 1
    finally:
        server.shutdown()
        server.server_close()


def test_requests_from_elsewhere_are_refused(dashboard):
    client, _ = dashboard
    assert client.get("/api/sessions", headers={"Host": "evil.example"}).status_code == 403
    assert client.post("/api/runs", content='{"file": "quick.yaml"}').status_code == 415  # not JSON
    assert client.post("/api/runs", json={"file": "quick.yaml"}, headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.post("/api/runs", json={"file": "../lab.sqlite"}).status_code == 404
    assert client.post("/api/runs", json={"file": "missing.yaml"}).status_code == 404
    assert client.post("/api/runs/stop", json={}).status_code == 409  # nothing to stop


def _wait_for(client, states, events=0, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        live = client.get("/api/live").json()
        if live["state"] in states and len(live["events"]) >= events:
            return live
        time.sleep(0.05)
    raise AssertionError(f"the run never reached {states}: {live}")


def test_a_run_is_started_followed_and_stopped(dashboard):
    client, _ = dashboard
    assert client.post("/api/runs", json={"file": "quick.yaml"}).status_code == 202
    live = _wait_for(client, {"running"}, events=2)
    assert live["started_here"] and [e["type"] for e in live["events"]] == ["start", "session"]
    assert client.post("/api/runs", json={"file": "quick.yaml"}).status_code == 409  # one run at a time
    assert client.get("/api/status").json()["experiment"] == "fake"

    assert client.post("/api/runs/stop", json={}).status_code == 202
    ended = _wait_for(client, {"ended"})
    assert ended["exit_code"] == 130

    # A finished run frees the dashboard for the next one, and "run even if busy" reaches the run.
    assert client.post("/api/runs", json={"file": "quick.yaml", "allow_busy": True}).status_code == 202
    finished = _wait_for(client, {"ended"})
    assert finished["exit_code"] == 0 and finished["events"][1]["args"] == ["--allow-busy"]
    since = client.get("/api/live", params={"since": 2}).json()
    assert [e["type"] for e in since["events"]] == ["exit"] and since["next"] == 3
