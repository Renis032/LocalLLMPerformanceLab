"""Grade one answer against its task. Pure code: no model is involved.

Each grader returns whether the answer is right, why, and a signature: the answer
reduced to what matters, so consistency can count how often repeats say the same thing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from . import sandbox
from .suite import Suite, Task

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```[ \t]*([\w+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_ANSWER_LINE = re.compile(r"answer\s*[:=]\s*(.+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class Score:
    passed: bool
    detail: str
    signature: str


def visible(text: str) -> str:
    """The answer without a reasoning model's <think> section."""
    return _THINK_OPEN.sub("", _THINK_BLOCK.sub("", text)).strip()


def normalize(text) -> str:
    """Forgive formatting, never content: case, spacing, markdown, wrapping quotes, a final period."""
    t = re.sub(r"[*`]", "", str(text)).strip().strip("\"'").casefold()
    t = re.sub(r"\s*,\s*", ",", re.sub(r"\s+", " ", t))
    return t.rstrip(".;:! ")


def _short(text: str, limit: int = 80) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def same(a, b) -> bool:
    """Deep equality for JSON values, where true is not 1 and "34" is not 34."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def schema_errors(value, schema: dict, path: str = "$") -> list[str]:
    """Check a value against the small subset of JSON Schema the suites use."""
    kinds = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}
    errors: list[str] = []
    kind = schema.get("type")
    if kind == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif kind:
        ok = isinstance(value, kinds[kind])
    else:
        ok = True
    if not ok:
        return [f"{path} should be {kind}"]
    if "enum" in schema and not any(same(value, x) for x in schema["enum"]):
        errors.append(f"{path} is not one of the allowed values")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path} is below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path} is above {schema['maximum']}")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        errors.append(f"{path} is too short")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        errors += [f"{path}.{k} is missing" for k in schema.get("required", []) if k not in value]
        if schema.get("additionalProperties") is False:
            errors += [f"{path}.{k} is not allowed" for k in value if k not in props]
        for k, sub in props.items():
            if k in value:
                errors += schema_errors(value[k], sub, f"{path}.{k}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path} has too few items")
        if len(value) > schema.get("maxItems", math.inf):
            errors.append(f"{path} has too many items")
        if "uniqueBy" in schema:
            keys = [json.dumps(x.get(schema["uniqueBy"]), sort_keys=True) for x in value if isinstance(x, dict)]
            if len(set(keys)) != len(keys):
                errors.append(f"{path} repeats a {schema['uniqueBy']}")
        for i, item in enumerate(value):
            if "items" in schema:
                errors += schema_errors(item, schema["items"], f"{path}[{i}]")
    return errors


def _exact(task: Task, text: str) -> Score:
    got = normalize(text)
    wanted = {normalize(a) for a in task.spec["answers"]}
    return Score(got in wanted, "matches" if got in wanted else f"got {_short(text)!r}", got or "(empty)")


def _strip_fence(text: str) -> str:
    """A single fenced block around the whole answer is common and harmless."""
    t = text.strip()
    m = _FENCE.fullmatch(t)
    return m.group(2).strip() if m else t


def _reject_constant(name: str):
    raise ValueError(f"{name} is not valid JSON")


def _json(task: Task, text: str) -> Score:
    try:
        value = json.loads(_strip_fence(text), parse_constant=_reject_constant)
    except ValueError as e:
        return Score(False, f"not valid JSON: {_short(e, 60)}", "invalid json")
    if "schema" in task.spec:
        errors = schema_errors(value, task.spec["schema"])
        # The content is free by design, so only pass or fail can repeat.
        return Score(not errors, "fits the schema" if not errors else _short("; ".join(errors)), "fits" if not errors else "does not fit")
    canonical = json.dumps(value, sort_keys=True)
    ok = same(value, task.spec["expect"])
    return Score(ok, "matches" if ok else f"got {_short(canonical)}", canonical)


def extract_code(text: str) -> str:
    """The last fenced Python block, or the whole answer if it has no fence."""
    blocks = _FENCE.findall(text)
    python = [body for lang, body in blocks if lang.lower() in ("python", "py", "python3")]
    if python:
        return python[-1].strip()
    return blocks[-1][1].strip() if blocks else text.strip()


def _code(task: Task, text: str) -> Score:
    tests = task.spec["tests"]
    source = extract_code(text)
    r = sandbox.run_tests(source, [t["call"] for t in tests])
    if r.status != "ok":
        return Score(False, _short(r.detail), r.status)
    marks = [("error" not in got and same(got.get("value"), t["expect"])) for got, t in zip(r.results, tests)]
    passed = sum(marks)
    if all(marks):
        return Score(True, f"passes all {len(tests)} tests", "all pass")
    first = marks.index(False)
    got = r.results[first]
    seen = got.get("error") or json.dumps(got.get("value"))
    detail = f"{passed} of {len(tests)} tests pass; {tests[first]['call']} gave {_short(seen, 50)}"
    return Score(False, detail, "".join("1" if m else "0" for m in marks))


def _reasoning(task: Task, text: str) -> Score:
    lines = _ANSWER_LINE.findall(re.sub(r"[*`]", "", text))
    if not lines:
        return Score(False, "no final ANSWER line", "no answer")
    value = lines[-1].strip().strip("<>").strip()
    expected = task.spec["answer"]
    if isinstance(expected, (int, float)):
        found = _NUMBER.findall(value)
        if not found:
            return Score(False, f"not a number: {_short(value, 40)}", normalize(value))
        got = float(found[0].replace(",", ""))
        ok = math.isclose(got, float(expected), abs_tol=1e-6)
        return Score(ok, "correct" if ok else f"answered {got:g}", f"{got:g}")
    ok = normalize(value) == normalize(expected)
    return Score(ok, "correct" if ok else f"answered {_short(value, 40)}", normalize(value))


_GRADERS = {"exact": _exact, "json": _json, "code": _code, "reasoning": _reasoning}


def score(task: Task, text: str, *, truncated: bool = False) -> Score:
    """Grade an answer. One cut off at the token limit never counts, however it starts."""
    if truncated:
        return Score(False, "cut off at the token limit", "cut off")
    answer = visible(text)
    if not answer:
        return Score(False, "no answer after the thinking section", "(empty)")
    return _GRADERS[task.category](task, answer)


def verify_graders(suite: Suite) -> list[str]:
    """Problems found by grading each task's known-right and known-wrong answers."""
    problems = []
    for t in suite.tasks:
        right = score(t, t.reference)
        if not right.passed:
            problems.append(f"{t.id}: its reference answer was rejected ({right.detail})")
        if score(t, t.wrong).passed:
            problems.append(f"{t.id}: its wrong answer was accepted")
    return problems
