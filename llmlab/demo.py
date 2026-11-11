"""Purpose: the demo. The dashboard on real recorded results, with no GPU, model or setup.

demo/lab-demo.sqlite.gz holds every session recorded while this project was built: SmolLM3
on an RTX 3060 Ti, and Claude Haiku, Sonnet and Opus as the cloud baseline. Nothing in it
is invented; failed and warned sessions are kept as they happened.

  python -m llmlab demo             unpack a fresh copy to a temporary folder and open the
                                    dashboard on it; the copy is deleted when you stop it
  python -m llmlab demo --rebuild   make demo/lab-demo.sqlite.gz again from your own
                                    database (--db), so you can publish your results too

Starting runs is turned off in the demo, because a run needs llama.cpp and a model
(docs/setup.md). Rebuilding replaces home-folder and checkout paths (they contain the
user name) and refuses to write the file if any are left.
"""

from __future__ import annotations

import gzip
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEMO_DB = REPO / "demo" / "lab-demo.sqlite.gz"


def unpack(dest_dir: Path, source: Path = DEMO_DB) -> Path:
    """Write the demo database into dest_dir and return its path."""
    if not source.is_file():
        raise LookupError(f"the demo database is missing: {source}")
    out = Path(dest_dir) / "lab-demo.sqlite"
    with gzip.open(source, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return out


def run(port: int = 8765, open_browser: bool = True, source: Path = DEMO_DB) -> int:
    from .dashboard.server import serve

    with tempfile.TemporaryDirectory(prefix="llmlab-demo-", ignore_cleanup_errors=True) as tmp:
        db_path = unpack(Path(tmp), source)
        print("Demo: results recorded while this project was built (RTX 3060 Ti, Claude as the cloud baseline).")
        print("Runs are off here; docs/setup.md explains how to run experiments on your own machine.\n")
        return serve(db_path, port=port, open_browser=open_browser, demo=True)


# ---------------------------------------------------------------- rebuilding


def _forms(path: str) -> list[str]:
    """How a Windows path can appear in stored text: escaped inside JSON, plain, or with slashes."""
    win = path.replace("/", "\\").rstrip("\\")
    return [win.replace("\\", "\\\\"), win, win.replace("\\", "/")]


def private_paths(conn: sqlite3.Connection, home: Path) -> list[tuple[str, str]]:
    """(path, replacement) pairs, longest first: each checkout runs were started from, then home."""
    roots = {str(REPO)}
    for (source,) in conn.execute("SELECT source_path FROM sessions WHERE source_path IS NOT NULL"):
        p = Path(source)
        if p.parent.name == "experiments":
            roots.add(str(p.parent.parent))
    pairs = [(r, "<repo>") for r in roots] + [(str(home), "~")]
    return sorted(pairs, key=lambda pr: -len(pr[0]))


def _text_columns(conn: sqlite3.Connection):
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
    for t in tables:
        for col in conn.execute(f'PRAGMA table_info("{t}")'):
            yield t, col[1]


def scrub(conn: sqlite3.Connection, home: Path) -> int:
    """Replace private paths in every text value; returns how many values changed."""
    rules = [(re.compile(re.escape(form), re.IGNORECASE), new)
             for old, new in private_paths(conn, home) for form in _forms(old)]

    def clean(value):
        if not isinstance(value, str):
            return value
        for pattern, new in rules:
            value = pattern.sub(new, value)
        return value

    conn.create_function("llmlab_scrub", 1, clean, deterministic=True)
    changed = 0
    for t, col in _text_columns(conn):
        cur = conn.execute(f'UPDATE "{t}" SET "{col}" = llmlab_scrub("{col}") '
                           f'WHERE typeof("{col}") = \'text\' AND "{col}" IS NOT llmlab_scrub("{col}")')
        changed += cur.rowcount
    conn.commit()
    return changed


def leftovers(conn: sqlite3.Connection, home: Path) -> list[str]:
    """Places where the home folder or the user name still appears, as 'table.column'."""
    needles = _forms(str(home)) + [f"\\{home.name}\\", f"/{home.name}/", f"\\\\{home.name}\\\\"]
    found = []
    for t, col in _text_columns(conn):
        for n in needles:
            if conn.execute(f'SELECT 1 FROM "{t}" WHERE typeof("{col}") = \'text\' AND instr(lower("{col}"), lower(?)) > 0 LIMIT 1',
                            (n,)).fetchone():
                found.append(f"{t}.{col}")
                break
    return found


def build(source_db: Path, out: Path = DEMO_DB, home: Path | None = None) -> dict:
    """Copy source_db, remove private paths, compact it and write it gzipped to out."""
    home = Path.home() if home is None else home
    source_db = Path(source_db)
    if not source_db.is_file():
        raise LookupError(f"no database at {source_db}")
    with tempfile.TemporaryDirectory(prefix="llmlab-demo-build-", ignore_cleanup_errors=True) as tmp:
        work = Path(tmp) / "demo.sqlite"
        src = sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True)
        dst = sqlite3.connect(work)
        try:
            src.backup(dst)
            changed = scrub(dst, home)
            left = leftovers(dst, home)
            if left:
                raise RuntimeError(f"private paths are still in {', '.join(left)}; the demo file was not written")
            sessions = dst.execute("SELECT count(*) FROM sessions").fetchone()[0]
            dst.execute("VACUUM")
        finally:
            src.close()
            dst.close()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(work, "rb") as f, gzip.open(out, "wb", compresslevel=9) as g:
            shutil.copyfileobj(f, g)
        return {"sessions": sessions, "values_changed": changed, "bytes": out.stat().st_size, "raw_bytes": work.stat().st_size}
