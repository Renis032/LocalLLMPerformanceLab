"""Purpose: the demo works without a GPU or model, can't start runs, and publishes nothing private.

The demo database is real data from the author's machine, so these tests guard what
leaves it: rebuilding replaces home-folder and checkout paths (they contain the user
name) and refuses to write the file if any remain, and the shipped file is checked for
any home folder at all.
"""

import json
import re
import sqlite3
import threading
from pathlib import Path

import httpx
import pytest

from llmlab import db, demo
from llmlab.dashboard.server import make_server
from llmlab.runner import run_experiment

from conftest import FakeGpu, make_experiment

HOME_FOLDER = re.compile(r"[a-z]:(\\\\|\\|/)users(\\\\|\\|/)", re.IGNORECASE)  # C:\Users\..., in any spelling


def _quiet(*args):
    pass


def _text_values(conn):
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    for t in tables:
        cols = [c[1] for c in conn.execute(f'PRAGMA table_info("{t}")')]
        for col in cols:
            for (value,) in conn.execute(f'SELECT "{col}" FROM "{t}" WHERE typeof("{col}") = \'text\''):
                yield f"{t}.{col}", value


@pytest.fixture(scope="module")
def shipped(tmp_path_factory):
    if not demo.DEMO_DB.is_file():
        pytest.skip("demo/lab-demo.sqlite.gz isn't built (python -m llmlab demo --rebuild)")
    path = demo.unpack(tmp_path_factory.mktemp("demo"))
    conn = sqlite3.connect(path)
    yield conn
    conn.close()


def test_the_shipped_demo_holds_real_scored_sessions(shipped):
    sessions = shipped.execute("SELECT count(*) FROM sessions WHERE status = 'completed'").fetchone()[0]
    quality = shipped.execute("SELECT count(*) FROM quality_runs").fetchone()[0]
    engines = {r[0] for r in shipped.execute("SELECT DISTINCT engine FROM configs")}
    assert sessions >= 10 and quality >= 4 and {"llama.cpp", "claude-code"} <= engines


def test_the_shipped_demo_contains_no_home_folder(shipped):
    hits = sorted({where for where, text in _text_values(shipped) if HOME_FOLDER.search(text)})
    assert hits == []


def test_rebuilding_replaces_home_and_checkout_paths(fake_engine, tmp_path):
    src = tmp_path / "lab.sqlite"
    conn = db.connect(src)
    run_experiment(make_experiment(tmp_path), conn, gpu=FakeGpu(), echo=_quiet)
    conn.execute("UPDATE sessions SET source_path = ?", (r"C:\Users\alice\code\lab\experiments\x.yaml",))
    conn.execute("UPDATE configs SET config_json = ?", (json.dumps(
        {"prompt_file": r"C:\Users\alice\code\lab\prompts\p.txt", "cli": "c:/users/alice/AppData/claude.exe"}),))
    conn.commit()
    conn.close()

    made = demo.build(src, tmp_path / "out.sqlite.gz", home=Path("C:/Users/alice"))
    assert made["sessions"] == 1 and made["values_changed"] >= 2
    copy = sqlite3.connect(demo.unpack(tmp_path, tmp_path / "out.sqlite.gz"))
    assert copy.execute("SELECT source_path FROM sessions").fetchone()[0] == r"<repo>\experiments\x.yaml"
    assert json.loads(copy.execute("SELECT config_json FROM configs").fetchone()[0]) == {
        "prompt_file": r"<repo>\prompts\p.txt", "cli": "~/AppData/claude.exe"}
    copy.close()


def test_rebuilding_refuses_when_the_user_name_is_left(fake_engine, tmp_path):
    src = tmp_path / "lab.sqlite"
    conn = db.connect(src)
    run_experiment(make_experiment(tmp_path), conn, gpu=FakeGpu(), echo=_quiet)
    conn.execute("UPDATE sessions SET description = ?", (r"copied from D:\backup\alice\lab",))
    conn.commit()
    conn.close()
    out = tmp_path / "out.sqlite.gz"
    with pytest.raises(RuntimeError, match="sessions.description"):
        demo.build(src, out, home=Path("C:/Users/alice"))
    assert not out.exists()


def test_a_missing_demo_file_is_named(tmp_path):
    with pytest.raises(LookupError, match="demo database is missing"):
        demo.unpack(tmp_path, tmp_path / "gone.sqlite.gz")


def test_the_demo_server_shows_results_but_refuses_runs(fake_engine, tmp_path):
    db_path = tmp_path / "lab.sqlite"
    conn = db.connect(db_path)
    run_experiment(make_experiment(tmp_path), conn, gpu=FakeGpu(), echo=_quiet)
    conn.close()
    (tmp_path / "quick.yaml").write_text("name: quick\nmodel:\n  path: model.gguf\nworkload:\n  prompt: hi\n", encoding="utf-8")
    server = make_server(db_path, port=0, experiments_dir=tmp_path, demo=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_address[1]}", timeout=10) as client:
            assert client.get("/api/status").json()["demo"] is True
            assert len(client.get("/api/sessions").json()) == 1
            refused = client.post("/api/runs", json={"file": "quick.yaml"})
            assert refused.status_code == 403 and "demo" in refused.json()["error"]
    finally:
        server.shutdown()
        server.server_close()
