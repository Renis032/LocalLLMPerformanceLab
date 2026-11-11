"""Purpose: results are stored safely and stay readable as the lab evolves.

Covers round trips through SQLite, refusing unknown columns, reusing model file hashes
only while the file is unchanged, upgrading an older database in place without losing
data, and refusing a database written by a newer version.
"""

import os

import pytest

from llmlab import db


def _session(conn, **extra):
    row = {"experiment": "e", "started_at": "2026-11-07T12:00:00+01:00", "status": "completed",
           "host_json": {"gpu": {"name": "G"}}}
    row.update(extra)
    return db.insert(conn, "sessions", row)


def _config(conn, sid, idx=0, status="ok", config_hash="h"):
    return db.insert(conn, "configs", {"session_id": sid, "idx": idx, "label": f"c{idx}",
                                       "config_hash": config_hash, "config_json": {"a": 1}, "status": status})


def test_round_trip_and_json_columns():
    conn = db.connect(":memory:")
    sid = _session(conn)
    cid = _config(conn, sid)
    db.insert_many(conn, "trials", [
        {"config_id": cid, "kind": "measured", "idx": 0, "valid": 1, "problems_json": []},
        {"config_id": cid, "kind": "warmup", "idx": 0, "valid": 0, "problems_json": ["x"]},
    ])
    db.update(conn, "configs", cid, {"facts_json": {"n_ctx": 4096}})
    conn.commit()

    assert db.get_session(conn, sid)["host_json"] == {"gpu": {"name": "G"}}
    (c,) = db.get_configs(conn, sid)
    assert c["config_json"] == {"a": 1} and c["facts_json"] == {"n_ctx": 4096}
    trials = db.get_trials(conn, cid)
    assert [t["kind"] for t in trials] == ["warmup", "measured"]  # warm-ups first
    assert trials[0]["problems_json"] == ["x"]
    assert db.latest_session_id(conn) == sid


def test_list_sessions_counts_config_outcomes():
    conn = db.connect(":memory:")
    sid = _session(conn)
    _config(conn, sid, 0, "ok")
    _config(conn, sid, 1, "failed")
    (row,) = db.list_sessions(conn)
    assert (row["n_configs"], row["n_ok"], row["n_failed"]) == (2, 1, 1)


def test_unknown_columns_are_refused():
    conn = db.connect(":memory:")
    with pytest.raises(db.DatabaseError, match="unknown columns"):
        db.insert(conn, "sessions", {"experiment": "e", "started_at": "t", "status": "s", "nope": 1})


def test_file_hash_is_reused_until_the_file_changes(tmp_path):
    conn = db.connect(":memory:")
    f = tmp_path / "model.gguf"
    f.write_bytes(b"abc")
    assert db.known_sha256(conn, f) is None
    db.store_sha256(conn, f, "deadbeef")
    assert db.known_sha256(conn, f) == "deadbeef"
    f.write_bytes(b"abcd")
    assert db.known_sha256(conn, f) is None


def test_an_older_database_is_migrated_in_place(tmp_path):
    path = tmp_path / "lab.sqlite"
    conn = db.connect(path)
    # Turn it back into a version 1 database: drop every column a migration adds.
    added = []
    for statements in db.MIGRATIONS.values():
        for sql in statements:
            table, column = sql.split()[2], sql.split()[5]  # ALTER TABLE t ADD COLUMN c TYPE
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
            added.append(column)
    conn.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
    sid = _session(conn)
    conn.commit()
    conn.close()

    conn = db.connect(path)
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(trials)")}
    assert {"cpu_other_mean_pct", "api_calls", "cost_usd"} <= set(added)
    assert set(added) <= columns
    assert conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == str(db.SCHEMA_VERSION)
    assert db.get_session(conn, sid)["experiment"] == "e"  # existing data kept


def test_a_newer_schema_is_refused(tmp_path):
    path = tmp_path / "lab.sqlite"
    conn = db.connect(path)
    conn.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(db.DatabaseError, match="newer version"):
        db.connect(path)
    assert os.path.exists(path)
