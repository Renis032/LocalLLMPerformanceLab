"""SQLite storage. One file holds every session, so results stay comparable over time.

Tables:
  sessions     one `llmlab run` of an experiment file, with the file's text and the host
  configs      one variant of that experiment: resolved settings, server facts, checks
  trials       one request: speeds, latencies, and telemetry over its generation window
               (cloud runs fill the token and timing columns; telemetry stays empty)
  samples      raw telemetry, about five rows a second while a configuration runs
  quality_runs one suite of graded tasks for a configuration: score, uncertainty, cost
  quality_attempts  one answer to one task, with its grade and the answer's text
  file_hashes  SHA-256 of model files, reused while size and modification time match

Columns ending in _json hold JSON text; insert() and update() encode them, and
to_dict() decodes them.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 5

# Statements that bring a database from the previous version to this one.
# SCHEMA below always describes the latest version, for new databases.
MIGRATIONS = {
    2: ["ALTER TABLE trials ADD COLUMN cpu_other_mean_pct REAL"],
    3: [
        "ALTER TABLE trials ADD COLUMN startup_ms REAL",
        "ALTER TABLE trials ADD COLUMN reported_ttft_ms REAL",
        "ALTER TABLE trials ADD COLUMN chunks INTEGER",
        "ALTER TABLE trials ADD COLUMN thinking_tokens INTEGER",
        "ALTER TABLE trials ADD COLUMN cached_tokens INTEGER",
        "ALTER TABLE trials ADD COLUMN api_calls INTEGER",
        "ALTER TABLE trials ADD COLUMN model_used TEXT",
        "ALTER TABLE trials ADD COLUMN cost_usd REAL",
    ],
    4: ["ALTER TABLE trials ADD COLUMN first_content_ms REAL"],
    5: [],  # adds the quality tables, which SCHEMA creates
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment      TEXT NOT NULL,
    description     TEXT,
    source_path     TEXT,
    source_text     TEXT,
    source_sha256   TEXT,
    started_at      TEXT NOT NULL,   -- ISO 8601, local time with offset
    finished_at     TEXT,
    status          TEXT NOT NULL,   -- running, completed, failed, aborted
    error           TEXT,
    harness_version TEXT,
    host_json       TEXT
);

CREATE TABLE IF NOT EXISTS configs (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id             INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx                    INTEGER NOT NULL,
    label                  TEXT NOT NULL,
    config_hash            TEXT NOT NULL,
    config_json            TEXT NOT NULL,   -- resolved settings, prompt text included
    overrides_json         TEXT,            -- this variant's sweep values
    status                 TEXT NOT NULL,   -- running, ok, warning, failed
    error                  TEXT,
    engine                 TEXT,
    engine_version         TEXT,
    model_path             TEXT,            -- model file, or a cloud model's ID
    model_bytes            INTEGER,
    model_sha256           TEXT,
    prompt_tokens          INTEGER,
    command_json           TEXT,
    load_s                 REAL,
    idle_json              TEXT,            -- the idle check before loading
    vram_before_mib        REAL,
    vram_peak_mib          REAL,
    server_rss_max_mib     REAL,
    server_private_max_mib REAL,
    facts_json             TEXT,            -- what the server log says it loaded
    checks_json            TEXT,
    phases_json            TEXT,            -- timeline: load and other phases
    server_log             TEXT,
    started_at             REAL,            -- Unix time
    finished_at            REAL
);
CREATE INDEX IF NOT EXISTS configs_session ON configs(session_id);
CREATE INDEX IF NOT EXISTS configs_hash ON configs(config_hash);

CREATE TABLE IF NOT EXISTS trials (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id                INTEGER NOT NULL REFERENCES configs(id) ON DELETE CASCADE,
    kind                     TEXT NOT NULL,   -- warmup or measured
    idx                      INTEGER NOT NULL,
    t_send                   REAL,            -- Unix time
    t_first                  REAL,
    t_last                   REAL,
    prompt_tokens            INTEGER,
    gen_tokens               INTEGER,
    client_tokens            INTEGER,
    prompt_ms                REAL,
    prompt_tps               REAL,
    gen_tps                  REAL,            -- server timing
    client_gen_tps           REAL,            -- what arrived over HTTP
    ttft_ms                  REAL,
    total_ms                 REAL,
    itl_p50_ms               REAL,
    itl_p95_ms               REAL,
    itl_max_ms               REAL,
    itl_json                 TEXT,            -- every inter-token gap, ms
    gpu_util_mean_pct        REAL,
    gpu_power_mean_w         REAL,            -- NVML 1 s rolling average
    gpu_power_max_w          REAL,
    gpu_power_instant_mean_w REAL,            -- NVML instantaneous power
    gpu_temp_max_c           REAL,
    gpu_clock_mean_mhz       REAL,
    vram_used_max_mib        REAL,
    cpu_total_mean_pct       REAL,
    proc_cpu_mean_pct        REAL,
    proc_rss_max_mib         REAL,
    j_per_token              REAL,            -- GPU board only
    clock_events_json        TEXT,
    stop_type                TEXT,
    valid                    INTEGER NOT NULL,
    problems_json            TEXT,
    cpu_other_mean_pct       REAL,            -- CPU used by other programs, % of the whole CPU
    -- cloud runs only
    startup_ms               REAL,            -- Claude Code starting up, before the request
    reported_ttft_ms         REAL,            -- Claude Code's own time to first token
    chunks                   INTEGER,         -- streamed text chunks (several tokens each)
    thinking_tokens          INTEGER,         -- hidden thinking tokens, included in gen_tokens
    cached_tokens            INTEGER,         -- prompt tokens read from or written to the cache
    api_calls                INTEGER,         -- API requests Claude Code made for this run
    model_used               TEXT,            -- the model that answered
    cost_usd                 REAL,            -- the run's cost at API list prices
    first_content_ms         REAL             -- the first streamed content, thinking included
);
CREATE INDEX IF NOT EXISTS trials_config ON trials(config_id);

CREATE TABLE IF NOT EXISTS samples (
    config_id           INTEGER NOT NULL REFERENCES configs(id) ON DELETE CASCADE,
    t                   REAL NOT NULL,
    gpu_util_pct        REAL,
    gpu_mem_util_pct    REAL,
    vram_used_mib       REAL,
    gpu_power_w         REAL,
    gpu_power_instant_w REAL,
    gpu_temp_c          REAL,
    gpu_clock_mhz       REAL,
    mem_clock_mhz       REAL,
    pstate              INTEGER,
    clock_event_mask    INTEGER,
    cpu_pct             REAL,
    cpu_per_core_json   TEXT,
    ram_used_mib        REAL,
    proc_cpu_pct        REAL,
    proc_rss_mib        REAL,
    proc_private_mib    REAL
);
CREATE INDEX IF NOT EXISTS samples_config ON samples(config_id, t);

CREATE TABLE IF NOT EXISTS quality_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id     INTEGER NOT NULL UNIQUE REFERENCES configs(id) ON DELETE CASCADE,
    suite_name    TEXT NOT NULL,
    suite_version INTEGER,
    suite_sha256  TEXT NOT NULL,
    suite_text    TEXT,            -- the suite file as it was
    settings_json TEXT,            -- repeats, temperature, token limit, thinking
    quality_hash  TEXT NOT NULL,   -- suite plus settings; equal hashes are comparable
    status        TEXT NOT NULL,   -- ok, warning, failed
    error         TEXT,
    score         REAL,            -- mean over tasks of the share of repeats correct, 0 to 1
    score_lo      REAL,            -- 95% interval, resampling tasks
    score_hi      REAL,
    consistency   REAL,            -- how often repeats of a task say the same thing, 0 to 1
    attempts      INTEGER,
    correct       INTEGER,
    summary_json  TEXT,            -- by category and by task
    checks_json   TEXT,
    gen_tokens    INTEGER,
    cost_usd      REAL,            -- cloud: the whole suite at API list prices
    energy_j      REAL,            -- local: GPU board energy over the whole suite
    seconds       REAL,
    started_at    REAL,
    finished_at   REAL
);

CREATE TABLE IF NOT EXISTS quality_attempts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    quality_run_id  INTEGER NOT NULL REFERENCES quality_runs(id) ON DELETE CASCADE,
    task_id         TEXT NOT NULL,
    category        TEXT NOT NULL,
    repeat          INTEGER NOT NULL,
    seed            INTEGER,
    valid           INTEGER NOT NULL,
    problems_json   TEXT,
    passed          INTEGER NOT NULL,
    detail          TEXT,            -- why it passed or failed
    signature       TEXT,            -- the answer reduced to what matters, for consistency
    answer          TEXT,
    truncated       INTEGER NOT NULL,
    stop_type       TEXT,
    prompt_tokens   INTEGER,
    gen_tokens      INTEGER,
    thinking_tokens INTEGER,
    ttft_ms         REAL,
    total_ms        REAL,
    gen_tps         REAL,
    t_send          REAL,
    t_last          REAL,
    cost_usd        REAL
);
CREATE INDEX IF NOT EXISTS quality_attempts_run ON quality_attempts(quality_run_id);

CREATE TABLE IF NOT EXISTS file_hashes (
    path     TEXT PRIMARY KEY,
    size     INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256   TEXT NOT NULL
);
"""


class DatabaseError(RuntimeError):
    pass


def connect(path: str | Path) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    existing = None
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
    ).fetchone()
    if row:
        v = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        existing = int(v["value"]) if v else None
        if existing is not None and existing > SCHEMA_VERSION:
            raise DatabaseError(
                f"{path} was written by a newer version of the lab (schema {existing}, "
                f"this version reads up to {SCHEMA_VERSION})"
            )
    if existing is None:
        conn.executescript(SCHEMA)
    else:
        for version in range(existing + 1, SCHEMA_VERSION + 1):
            for statement in MIGRATIONS[version]:
                conn.execute(statement)
        conn.executescript(SCHEMA)  # creates anything missing; leaves existing tables alone
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------- writes

_columns_cache: dict[tuple[int, str], set[str]] = {}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    key = (id(conn), table)
    if key not in _columns_cache:
        _columns_cache[key] = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    return _columns_cache[key]


def _encode(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k.endswith("_json") and v is not None and not isinstance(v, str):
            v = json.dumps(v)
        out[k] = v
    return out


def _check_columns(conn, table: str, keys) -> None:
    unknown = set(keys) - _columns(conn, table)
    if unknown:
        raise DatabaseError(f"unknown columns for {table}: {', '.join(sorted(unknown))}")


def insert(conn: sqlite3.Connection, table: str, row: dict) -> int:
    row = _encode(row)
    _check_columns(conn, table, row)
    sql = f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})"
    return conn.execute(sql, list(row.values())).lastrowid


def insert_many(conn: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    rows = [_encode(r) for r in rows]
    keys = list(rows[0])
    _check_columns(conn, table, keys)
    sql = f"INSERT INTO {table} ({', '.join(keys)}) VALUES ({', '.join('?' * len(keys))})"
    conn.executemany(sql, [[r.get(k) for k in keys] for r in rows])


def update(conn: sqlite3.Connection, table: str, row_id: int, values: dict) -> None:
    values = _encode(values)
    _check_columns(conn, table, values)
    sets = ", ".join(f"{k} = ?" for k in values)
    conn.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*values.values(), row_id])


# ---------------------------------------------------------------- reads


def to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    out = {}
    for k in row.keys():
        v = row[k]
        if k.endswith("_json") and isinstance(v, str):
            v = json.loads(v)
        out[k] = v
    return out


def list_sessions(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        """
        SELECT s.id, s.experiment, s.started_at, s.finished_at, s.status, s.host_json,
               COUNT(c.id) AS n_configs,
               SUM(c.status = 'ok') AS n_ok,
               SUM(c.status = 'warning') AS n_warning,
               SUM(c.status = 'failed') AS n_failed
        FROM sessions s LEFT JOIN configs c ON c.session_id = s.id
        GROUP BY s.id ORDER BY s.id DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [to_dict(r) for r in rows]


def get_session(conn: sqlite3.Connection, session_id: int) -> dict | None:
    return to_dict(conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone())


def latest_session_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT MAX(id) AS id FROM sessions").fetchone()
    return row["id"] if row else None


def get_configs(conn: sqlite3.Connection, session_id: int) -> list[dict]:
    rows = conn.execute("SELECT * FROM configs WHERE session_id = ? ORDER BY idx", (session_id,))
    return [to_dict(r) for r in rows]


def get_trials(conn: sqlite3.Connection, config_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM trials WHERE config_id = ? ORDER BY kind = 'measured', idx", (config_id,)
    )
    return [to_dict(r) for r in rows]


def get_samples(conn: sqlite3.Connection, config_id: int) -> list[dict]:
    rows = conn.execute("SELECT * FROM samples WHERE config_id = ? ORDER BY t", (config_id,))
    return [to_dict(r) for r in rows]


def get_quality_run(conn: sqlite3.Connection, config_id: int) -> dict | None:
    return to_dict(conn.execute("SELECT * FROM quality_runs WHERE config_id = ?", (config_id,)).fetchone())


def get_quality_attempts(conn: sqlite3.Connection, run_id: int) -> list[dict]:
    rows = conn.execute("SELECT * FROM quality_attempts WHERE quality_run_id = ? ORDER BY id", (run_id,))
    return [to_dict(r) for r in rows]


# ---------------------------------------------------------------- file hashes


def _stat(path: str | Path) -> tuple[str, int, int]:
    p = os.path.abspath(path)
    st = os.stat(p)
    return p, st.st_size, st.st_mtime_ns


def known_sha256(conn: sqlite3.Connection, path: str | Path) -> str | None:
    """The stored hash, if the file still has the size and modification time it had then."""
    p, size, mtime = _stat(path)
    row = conn.execute("SELECT * FROM file_hashes WHERE path = ?", (p,)).fetchone()
    if row and row["size"] == size and row["mtime_ns"] == mtime:
        return row["sha256"]
    return None


def store_sha256(conn: sqlite3.Connection, path: str | Path, sha256: str) -> None:
    p, size, mtime = _stat(path)
    conn.execute(
        "INSERT OR REPLACE INTO file_hashes (path, size, mtime_ns, sha256) VALUES (?, ?, ?, ?)",
        (p, size, mtime, sha256),
    )
