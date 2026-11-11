"""Command line: python -m llmlab <command>.

  run EXPERIMENT.yaml    run an experiment and store the results
  sessions               list recorded sessions
  show [ID]              show one session (default: the latest)
  compare A B            check two sessions agree on the configurations they share
  quality                speed, answer quality and cost for every scored configuration
  regrade [ID]           grade a session's stored answers again (after a grader fix)
  dashboard              open the dashboard in a browser: live runs, history, compare
  demo                   the dashboard on bundled real results: no GPU, model or setup needed

Exit codes: 0 success; 1 a configuration failed or a comparison didn't pass;
2 a bad experiment file or argument; 3 the GPU was busy; 130 stopped with Ctrl+C
or from the dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import claude_code, db, llamacpp, report
from .dashboard import api
from .config import ConfigError, default_db_path, load_experiment
from .live import LiveLog, NoLiveLog, StopRequested, live_dir
from .runner import GpuBusyError, run_experiment


def _session_id(conn, value: str) -> int:
    if value == "latest":
        sid = db.latest_session_id(conn)
        if sid is None:
            raise LookupError("no sessions recorded yet")
        return sid
    try:
        return int(value)
    except ValueError:
        raise LookupError(f"not a session id: {value!r} (use a number or 'latest')") from None


def _dry_run_cloud(v) -> None:
    c = v.config
    e = c["engine"]
    print(f"\n[{v.index + 1}] {v.label}  (config {v.config_hash})")
    print(f"  prompt {len(v.prompt):,} characters; effort {e['effort']}, thinking {e['thinking']}; "
          f"{c['run']['warmup']} warm-up + {c['run']['repeats']} runs")
    try:
        exe = claude_code.find_cli(e["cli"])
    except claude_code.ClaudeCodeError as err:
        print(f"  warning: {err}")
        exe = e["cli"]
    cmd = claude_code.build_command(exe, c["model"]["name"], e["effort"], c["workload"]["system_prompt"])
    print("  " + " ".join(f'"{a}"' if not a or " " in a else a for a in cmd) + "  < prompt")
    _dry_run_quality(c)


def _dry_run_quality(c: dict) -> None:
    q = c["quality"]
    if q["suite"]:
        print(f"  quality: {q['tasks']} tasks x {q['repeats']} repeats from {q['suite']}")


def cmd_run(args) -> int:
    exp = load_experiment(args.experiment)
    if args.dry_run:
        print(f"{exp.name}: {len(exp.variants)} configuration(s), file is valid")
        for v in exp.variants:
            c = v.config
            if c["engine"]["type"] == "claude-code":
                _dry_run_cloud(v)
                continue
            cmd = llamacpp.build_command(c["engine"]["server"], c["model"]["path"], c["server"], 0)
            print(f"\n[{v.index + 1}] {v.label}  (config {v.config_hash})")
            print(f"  prompt {len(v.prompt):,} characters, n_predict {c['workload']['n_predict']}, "
                  f"seed {c['workload']['seed']}; {c['run']['warmup']} warm-up + {c['run']['repeats']} runs")
            print("  " + " ".join(cmd).replace("--port 0", "--port <free port>"))
            _dry_run_quality(c)
            for path, what in ((c["engine"]["server"], "engine.server"), (c["model"]["path"], "model.path")):
                if not Path(path).is_file():
                    print(f"  warning: {what} not found: {path}")
        return 0

    conn = db.connect(args.db)
    # The dashboard follows the run through this log, wherever the run was started.
    live = NoLiveLog() if args.no_live else LiveLog(live_dir(args.db))

    def echo(line: str = "") -> None:
        print(line, flush=True)
        live.emit("log", text=line)

    code = 130
    try:
        echo(f"Database: {args.db}")
        sid, results = run_experiment(exp, conn, allow_busy=args.allow_busy, echo=echo, live=live)
        code = _finish_run(sid, results, echo)
        return code
    except GpuBusyError as e:
        echo(f"\nStopped: {e}")
        code = 3
        return code
    except StopRequested:
        echo("\nStopped from the dashboard. The session is recorded as aborted; the server has been shut down.")
        return code
    finally:
        conn.close()
        live.emit("exit", code=code)
        live.close()


def _finish_run(sid: int, results: list[dict], echo) -> int:
    bad = [r for r in results if r["status"] != "ok"]
    echo()
    if bad:
        echo(f"Session {sid} finished; {len(bad)} of {len(results)} configuration(s) need attention: "
             + ", ".join(f"{r['label']} ({r['status']})" for r in bad))
    else:
        echo(f"Session {sid} finished; all {len(results)} configuration(s) passed their checks.")
    echo(f"See it again with: python -m llmlab show {sid}")
    return 1 if any(r["status"] == "failed" for r in results) else 0


def cmd_dashboard(args) -> int:
    from .dashboard.server import serve

    return serve(args.db, port=args.port, open_browser=not args.no_browser)


def cmd_demo(args) -> int:
    from . import demo

    if args.rebuild:
        made = demo.build(args.db)
        print(f"Wrote {demo.DEMO_DB} from {args.db}: {made['sessions']} sessions, "
              f"{made['values_changed']} values with private paths replaced, "
              f"{made['raw_bytes'] / 2**20:.1f} MiB packed to {made['bytes'] / 2**20:.1f} MiB.")
        return 0
    return demo.run(port=args.port, open_browser=not args.no_browser)


def cmd_sessions(args) -> int:
    conn = db.connect(args.db)
    try:
        print(report.sessions_text(db.list_sessions(conn, args.limit)))
    finally:
        conn.close()
    return 0


def cmd_show(args) -> int:
    conn = db.connect(args.db)
    try:
        sid = _session_id(conn, args.session)
        if args.json:
            print(json.dumps(report.session_json(conn, sid), indent=2, default=str))
        else:
            print(report.session_text(conn, sid))
    finally:
        conn.close()
    return 0


def cmd_quality(args) -> int:
    conn = db.connect(args.db)
    try:
        print(report.quality_table(api.quality_results(conn)))
    finally:
        conn.close()
    return 0


def cmd_regrade(args) -> int:
    from .quality.run import regrade

    conn = db.connect(args.db)
    try:
        sid = _session_id(conn, args.session)
        found = False
        for c in db.get_configs(conn, sid):
            if db.get_quality_run(conn, c["id"]) is None:
                continue
            found = True
            before, after, changed = regrade(conn, c["id"])
            print(f"Session {sid}, {c['label']}: {report.pct(before['score'], 1)} -> {report.pct(after['score'], 1)} "
                  f"({changed} answer(s) changed between right and wrong)")
        if not found:
            print(f"Session {sid} has no quality runs.")
    finally:
        conn.close()
    return 0


def cmd_compare(args) -> int:
    conn = db.connect(args.db)
    try:
        text, verdict = report.compare_text(
            conn, _session_id(conn, args.a), _session_id(conn, args.b), args.tolerance
        )
    finally:
        conn.close()
    print(text)
    return 0 if verdict else 1


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--db", type=Path, default=default_db_path(),
        help="SQLite database (default: %(default)s; set LLMLAB_DB to change it)",
    )
    ap = argparse.ArgumentParser(
        prog="python -m llmlab",
        description="LLM Performance Lab: reproducible benchmarks for local LLM inference, with Claude as a cloud baseline.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", parents=[common], help="run an experiment file")
    p.add_argument("experiment", type=Path)
    p.add_argument("--allow-busy", action="store_true", help="run even if the GPU is busy (results are marked)")
    p.add_argument("--dry-run", action="store_true", help="check the file and print what would run")
    p.add_argument("--no-live", action="store_true", help="don't write the live log the dashboard follows")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("sessions", parents=[common], help="list recorded sessions")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_sessions)

    p = sub.add_parser("show", parents=[common], help="show one session")
    p.add_argument("session", nargs="?", default="latest", help="session id or 'latest' (default)")
    p.add_argument("--json", action="store_true", help="print everything except raw samples as JSON")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("compare", parents=[common], help="compare two sessions")
    p.add_argument("a", help="session id or 'latest'")
    p.add_argument("b", help="session id or 'latest'")
    p.add_argument("--tolerance", type=float, default=5.0, help="allowed median difference, %% (default 5)")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("quality", parents=[common], help="speed, answer quality and cost, side by side")
    p.set_defaults(func=cmd_quality)

    p = sub.add_parser("regrade", parents=[common], help="grade a session's stored answers again")
    p.add_argument("session", nargs="?", default="latest", help="session id or 'latest' (default)")
    p.set_defaults(func=cmd_regrade)

    p = sub.add_parser("dashboard", parents=[common], help="open the dashboard in a browser")
    p.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (default %(default)s)")
    p.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("demo", parents=[common], help="the dashboard on bundled real results (no GPU needed)")
    p.add_argument("--port", type=int, default=8765, help="port on 127.0.0.1 (default %(default)s)")
    p.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    p.add_argument("--rebuild", action="store_true", help="make demo/lab-demo.sqlite.gz from --db, with private paths removed")
    p.set_defaults(func=cmd_demo)
    return ap


def main(argv: list[str] | None = None) -> int:
    # Some Windows consoles can't print every character; never crash over that.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"Experiment file problem: {e}")
        return 2
    except (LookupError, db.DatabaseError) as e:
        print(f"Error: {e}")
        return 2
    except KeyboardInterrupt:
        print("\nStopped. The session is recorded as aborted; the server has been shut down.")
        return 130
