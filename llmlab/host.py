"""The machine a benchmark ran on, recorded with every session.

No hostname or user name is recorded, so reports can be shared as they are.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import platform
import re
import subprocess
import sys
from pathlib import Path

import psutil

from . import __version__


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def power_plan() -> str | None:
    """The active Windows power plan, which affects CPU clocks (read only)."""
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(
            ["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"\(([^)]+)\)\s*$", out)
    return m.group(1) if m else (out or None)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def host_snapshot(gpu) -> dict:
    return {
        "captured_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": platform.python_version(),
        "harness_version": __version__,
        "cpu": cpu_name(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_mib": round(psutil.virtual_memory().total / 2**20),
        "power_plan": power_plan(),
        "gpu": gpu.info() if gpu is not None else None,
    }
