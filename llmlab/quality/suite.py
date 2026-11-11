"""Task suites: load a YAML file of tasks with known answers, and check it.

Mistakes are caught here, before a run starts. Each task also carries a correct answer
and a plausible wrong one, which verify_graders uses to test the graders themselves.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

CATEGORIES = ("exact", "json", "code", "reasoning")


class SuiteError(ValueError):
    """The suite file is invalid. The message names the task and the problem."""


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    prompt: str
    spec: dict  # what the category's grader needs: answers, expect, schema, tests...
    reference: str  # a correct answer: the grader must accept it
    wrong: str  # a plausible mistake: the grader must reject it


@dataclass(frozen=True)
class Suite:
    name: str
    version: int
    description: str
    text: str
    sha256: str
    tasks: list[Task]


SPEC_KEYS = {
    "exact": {"answers"},
    "json": {"expect", "schema"},
    "code": {"function", "tests"},
    "reasoning": {"answer"},
}
COMMON_KEYS = {"id", "category", "prompt", "reference", "wrong"}


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    if not path.is_file():
        raise SuiteError(f"suite file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise SuiteError(f"{path.name} is not valid YAML: {e}") from None
    return parse_suite(raw, text)


def parse_suite(raw, text: str = "") -> Suite:
    if not isinstance(raw, dict):
        raise SuiteError("a suite must be a set of settings with a list of tasks")
    name, version, tasks = raw.get("name"), raw.get("version"), raw.get("tasks")
    if not isinstance(name, str) or not name.strip():
        raise SuiteError("name: required, a short name for the suite")
    if isinstance(version, bool) or not isinstance(version, int):
        raise SuiteError(f"version: required, a whole number, got {version!r}")
    if not isinstance(tasks, list) or not tasks:
        raise SuiteError("tasks: required, a non-empty list")
    parsed = [_task(t, i) for i, t in enumerate(tasks)]
    ids = [t.id for t in parsed]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise SuiteError(f"tasks: repeated id: {', '.join(repeated)}")
    return Suite(
        name=name.strip(),
        version=version,
        description=str(raw.get("description") or "").strip(),
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        tasks=parsed,
    )


def _task(raw, index: int) -> Task:
    where = f"tasks[{index}]"
    if not isinstance(raw, dict):
        raise SuiteError(f"{where}: expected a task, got {raw!r}")
    tid = raw.get("id")
    if not isinstance(tid, str) or not tid.strip():
        raise SuiteError(f"{where}: id: required")
    where = f"task {tid}"
    category = raw.get("category")
    if category not in CATEGORIES:
        raise SuiteError(f"{where}: category: must be one of {', '.join(CATEGORIES)}, got {category!r}")
    unknown = set(raw) - COMMON_KEYS - SPEC_KEYS[category]
    if unknown:
        raise SuiteError(f"{where}: unknown setting for a {category} task: {', '.join(sorted(unknown))}")
    for key in ("prompt", "reference", "wrong"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise SuiteError(f"{where}: {key}: required text")
    spec = {k: raw[k] for k in SPEC_KEYS[category] if k in raw}
    _CHECKS[category](spec, where)
    return Task(tid, category, raw["prompt"].strip(), spec, raw["reference"], raw["wrong"])


def _check_exact(spec: dict, where: str) -> None:
    answers = spec.get("answers")
    if not isinstance(answers, list) or not answers:
        raise SuiteError(f"{where}: answers: required, a non-empty list")


def _check_json(spec: dict, where: str) -> None:
    if ("expect" in spec) == ("schema" in spec):
        raise SuiteError(f"{where}: set exactly one of expect or schema")
    if "schema" in spec and not isinstance(spec["schema"], dict):
        raise SuiteError(f"{where}: schema: expected a set of rules")


def _check_code(spec: dict, where: str) -> None:
    fn, tests = spec.get("function"), spec.get("tests")
    if not isinstance(fn, str) or not fn.isidentifier():
        raise SuiteError(f"{where}: function: required, a function name")
    if not isinstance(tests, list) or not tests:
        raise SuiteError(f"{where}: tests: required, a non-empty list")
    for t in tests:
        if not isinstance(t, dict) or set(t) != {"call", "expect"}:
            raise SuiteError(f"{where}: tests: each test needs exactly call and expect, got {t!r}")
        try:
            ast.parse(t["call"], mode="eval")
        except (SyntaxError, TypeError):
            raise SuiteError(f"{where}: tests: not a Python expression: {t['call']!r}") from None


def _check_reasoning(spec: dict, where: str) -> None:
    answer = spec.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, (int, float, str)) or answer == "":
        raise SuiteError(f"{where}: answer: required, a number or text")


_CHECKS = {"exact": _check_exact, "json": _check_json, "code": _check_code, "reasoning": _check_reasoning}
