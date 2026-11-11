"""Run a generated Python function against hidden tests, in a guarded subprocess.

The code comes from a model, so it is treated as untrusted. There are four layers:

  1. A static gate reads the program before anything runs: only a short list of pure
     standard-library imports, and no file, eval or attribute tricks.
  2. A separate Python process, isolated from the project and its packages, in an empty
     temporary folder, with an almost empty environment.
  3. Inside it, an audit hook refuses files, sockets, processes and further imports once
     the program is about to run.
  4. Limits on time and memory, enforced from outside: the process is killed.

This contains ordinary mistakes and careless code. It is not a security boundary against
code written to escape, so only score models you would run anyway.
"""

from __future__ import annotations

import ast
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

# Modules that only compute: a correct solution may reasonably use any of them.
ALLOWED_IMPORTS = {
    "__future__", "abc", "array", "bisect", "cmath", "collections", "copy", "dataclasses", "datetime", "decimal",
    "enum", "fractions", "functools", "heapq", "itertools", "json", "math", "numbers", "operator", "random", "re",
    "statistics", "string", "textwrap", "typing", "unicodedata",
}
FORBIDDEN_NAMES = {
    "__import__", "breakpoint", "compile", "delattr", "dir", "eval", "exec", "exit", "getattr", "globals",
    "help", "input", "locals", "memoryview", "open", "quit", "setattr", "vars",
}
ALLOWED_DUNDERS = {"__name__", "__init__"}  # `if __name__ == "__main__":` and super().__init__() are harmless
TIMEOUT_S = 5.0
MEMORY_LIMIT_MIB = 512
MAX_OUTPUT_BYTES = 1 << 20

# The real interpreter: uv's python.exe is a launcher, and killing it would leave its child running.
PYTHON = getattr(sys, "_base_executable", sys.executable)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class SandboxResult:
    status: str  # ok, rejected, error, timeout or memory
    detail: str = ""
    results: list[dict] = field(default_factory=list)  # one {"value": ...} or {"error": ...} per test


def gate(source: str) -> str | None:
    """Why a program may not run, or None if it may."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as e:
        return f"not valid Python: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"imports {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"imports {node.module}"
        elif isinstance(node, ast.Name) and node.id not in ALLOWED_DUNDERS and (
            node.id in FORBIDDEN_NAMES or node.id.startswith("__")
        ):
            return f"uses {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ALLOWED_DUNDERS:
            return f"uses .{node.attr}"
    return None


# The program the subprocess runs. Results go to the real stdout behind a random marker,
# so anything the generated code prints can't be mistaken for them.
CHILD = r'''
import json, sys, io, builtins

job = json.load(open(sys.argv[1], encoding="utf-8"))
for name in job["allowed"]:  # loaded now, because imports are refused once the hook is armed
    __import__(name)
out = sys.stdout
sys.stdout = sys.stderr = io.StringIO()
allowed = set(job["allowed"])
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in allowed:
        raise ImportError("import not allowed: " + name)
    return real_import(name, globals, locals, fromlist, level)

safe = {k: v for k, v in vars(builtins).items() if k not in set(job["forbidden"]) or k == "__import__"}
safe["__import__"] = guarded_import
ns = {"__builtins__": safe, "__name__": "generated"}
program = compile(job["source"], "generated", "exec")
calls = [compile(c, "test", "eval") for c in job["calls"]]

armed = [False]
BLOCKED = ("open", "import", "socket.", "subprocess.", "os.", "shutil.", "ctypes.", "winreg.", "msvcrt.",
           "urllib.", "http.", "ftplib.", "smtplib.", "webbrowser.", "pty.", "mmap.", "signal.", "sqlite3.", "tempfile.", "glob.")

def hook(event, args):
    if armed[0] and event.startswith(BLOCKED):
        raise PermissionError("blocked: " + event)

sys.addaudithook(hook)
armed[0] = True

def plain(value):
    return json.loads(json.dumps(value))

report = {"results": []}
try:
    exec(program, ns)
except BaseException as e:
    report["load_error"] = type(e).__name__ + ": " + str(e)[:300]
else:
    for call in calls:
        try:
            report["results"].append({"value": plain(eval(call, ns))})
        except BaseException as e:
            report["results"].append({"error": type(e).__name__ + ": " + str(e)[:300]})
out.write(job["marker"] + json.dumps(report))
out.flush()
'''


def _kill(proc: subprocess.Popen) -> None:
    proc.kill()
    proc.wait(timeout=10)


def run_tests(source: str, calls: list[str], timeout_s: float = TIMEOUT_S, memory_mib: int = MEMORY_LIMIT_MIB) -> SandboxResult:
    """Run the calls against the program. Values come back as plain JSON types."""
    why = gate(source)
    if why:
        return SandboxResult("rejected", f"refused before running: it {why}")
    work = Path(tempfile.mkdtemp(prefix="llmlab-sandbox-"))
    marker = secrets.token_hex(16)
    try:
        (work / "child.py").write_text(CHILD, encoding="utf-8")
        job = {
            "source": source, "calls": calls, "marker": marker,
            "allowed": sorted(ALLOWED_IMPORTS), "forbidden": sorted(FORBIDDEN_NAMES),
        }
        (work / "job.json").write_text(json.dumps(job), encoding="utf-8")
        env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATHEXT") if k in os.environ}
        env["PYTHONIOENCODING"] = "utf-8"
        output_path = work / "output.txt"
        with open(output_path, "wb") as output:
            proc = subprocess.Popen(
                [PYTHON, "-I", "-S", "-B", "-X", "utf8", str(work / "child.py"), str(work / "job.json")],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.DEVNULL,
                cwd=str(work), env=env, creationflags=NO_WINDOW,
            )
            status = _watch(proc, timeout_s, memory_mib)
        if status == "timeout":
            return SandboxResult("timeout", f"still running after {timeout_s:g} s")
        if status == "memory":
            return SandboxResult("memory", f"used more than {memory_mib} MiB")
        with open(output_path, "rb") as f:
            text = f.read(MAX_OUTPUT_BYTES).decode("utf-8", "replace")
        if marker not in text:
            return SandboxResult("error", f"the program ended the test process (exit code {proc.returncode})")
        report = json.loads(text.split(marker, 1)[1])
        if report.get("load_error"):
            return SandboxResult("error", report["load_error"])
        return SandboxResult("ok", "", report["results"])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _watch(proc: subprocess.Popen, timeout_s: float, memory_mib: int) -> str:
    deadline = time.perf_counter() + timeout_s
    watcher = psutil.Process(proc.pid)
    while proc.poll() is None:
        if time.perf_counter() > deadline:
            _kill(proc)
            return "timeout"
        try:
            if watcher.memory_info().rss > memory_mib * 2**20:
                _kill(proc)
                return "memory"
        except psutil.Error:
            pass
        time.sleep(0.01)
    return "finished"
