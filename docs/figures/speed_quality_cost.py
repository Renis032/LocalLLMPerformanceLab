"""Draw docs/figures/speed-quality-cost.svg from the lab database.

Purpose: the README's answer to "is the fast option any good?". One row per model, three
small charts side by side: answer quality with its 95% interval, generation speed, and
what a correct answer costs. Three charts, because the three measures have different units.

    D:\\LLMLab\\tools\\uv\\uv.exe run python docs/figures/speed_quality_cost.py --local 18 --cloud 19

--local is a session with the local model (its first configuration is used); --cloud is a
claude-quality session (every configuration is used).
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from llmlab.quality.summary import cost_per_correct, kwh_price  # noqa: E402

W, H = 860, 392
LABEL_X = 24
PANELS = ((210, 390), (430, 590), (632, 780))  # x range of each chart's marks
ROW_Y0, ROW_H, BAR_H = 172, 38, 20
NAMES = {
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-opus-5-5": "Claude Opus 5.5",
}


def row_for(conn, config_id: int, name: str, where: str) -> dict:
    q = dict(zip(
        ("score", "lo", "hi", "correct", "attempts", "cost_usd", "energy_j", "summary"),
        conn.execute(
            "SELECT score, score_lo, score_hi, correct, attempts, cost_usd, energy_j, summary_json FROM quality_runs WHERE config_id = ?",
            (config_id,),
        ).fetchone(),
    ))
    # The same rule as the dashboard: the latest speed runs of this configuration that passed their checks.
    speed_id = conn.execute(
        "SELECT MAX(id) FROM configs WHERE status = 'ok' AND config_hash = (SELECT config_hash FROM configs WHERE id = ?)",
        (config_id,),
    ).fetchone()[0] or config_id
    speeds = [r[0] for r in conn.execute(
        "SELECT gen_tps FROM trials WHERE config_id = ? AND kind = 'measured' AND valid = 1", (speed_id,)
    )]
    cost = cost_per_correct(q)
    return {"name": name, "where": where, "tps": statistics.median(speeds), "usd": cost["usd"],
            "tasks": len(json.loads(q["summary"])["tasks"]), **q}


def load(conn, local_session: int, cloud_session: int) -> list[dict]:
    (local_id,) = conn.execute(
        "SELECT id FROM configs WHERE session_id = ? ORDER BY idx LIMIT 1", (local_session,)
    ).fetchone()
    out = [row_for(conn, local_id, "SmolLM3 3B, Q4_K_M", "local")]
    for cid, model in conn.execute("SELECT id, model_path FROM configs WHERE session_id = ? ORDER BY idx", (cloud_session,)):
        out.append(row_for(conn, cid, NAMES.get(model, model), "cloud"))
    return out


def bar(x0: float, x1: float, y: float, h: float, r: float = 4) -> str:
    r = min(r, (x1 - x0) / 2, h / 2)
    return (
        f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
        f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z"
    )


def axis(x: float, text: str, n: int, zero: bool = False) -> str:
    end = ROW_Y0 + ROW_H * n - (ROW_H - BAR_H)
    return (
        f'<line class="{"axis" if zero else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{ROW_Y0 - 6}" y2="{end + 6}"/>'
        f'<text class="tick" x="{x:.1f}" y="{end + 22}" text-anchor="middle">{text}</text>'
    )


def quality_panel(rows, x_range) -> str:
    x0, x1 = x_range
    X = lambda v: x0 + v * (x1 - x0)  # noqa: E731
    parts = [f'<text class="ptitle" x="{x0}" y="{ROW_Y0 - 18}">Answer quality, % of tasks correct</text>']
    parts += [axis(X(t), f"{t * 100:.0f}%", len(rows), t == 0) for t in (0, 0.5, 1)]
    for i, r in enumerate(rows):
        y = ROW_Y0 + i * ROW_H
        cy = y + BAR_H / 2
        parts.append(f'<path class="{r["where"]}" d="{bar(x0, X(r["score"]), y, BAR_H)}"><title>{r["name"]}: {r["score"] * 100:.0f}% (95% interval {r["lo"] * 100:.0f}% to {r["hi"] * 100:.0f}%)</title></path>')
        parts.append(f'<line class="whisk" x1="{X(r["lo"]):.1f}" x2="{X(r["hi"]):.1f}" y1="{cy}" y2="{cy}"/>')
        for v in (r["lo"], r["hi"]):
            parts.append(f'<line class="whisk" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{cy - 4}" y2="{cy + 4}"/>')
        parts.append(f'<text class="val" x="{X(max(r["score"], r["hi"])) + 6:.1f}" y="{y + 15}">{r["score"] * 100:.0f}%</text>')
    return "".join(parts)


def speed_panel(rows, x_range) -> str:
    x0, x1 = x_range
    top = max(r["tps"] for r in rows)
    top = math.ceil(top / 50) * 50
    X = lambda v: x0 + v / top * (x1 - x0)  # noqa: E731
    parts = [f'<text class="ptitle" x="{x0}" y="{ROW_Y0 - 18}">Speed, tokens/s</text>']
    parts += [axis(X(t), f"{t}", len(rows), t == 0) for t in (0, top // 2, top)]
    for i, r in enumerate(rows):
        y = ROW_Y0 + i * ROW_H
        parts.append(f'<path class="{r["where"]}" d="{bar(x0, X(r["tps"]), y, BAR_H)}"><title>{r["name"]}: {r["tps"]:.0f} tokens/s</title></path>')
        parts.append(f'<text class="val" x="{X(r["tps"]) + 6:.1f}" y="{y + 15}">{r["tps"]:.0f}</text>')
    return "".join(parts)


def money(v: float) -> str:
    return f"${v:.3f}" if v >= 0.01 else f"${v:.{1 - math.floor(math.log10(v))}f}"


def cost_panel(rows, x_range) -> str:
    x0, x1 = x_range
    logs = [math.log10(r["usd"]) for r in rows]
    lo, hi = math.floor(min(logs)), math.ceil(max(logs))
    X = lambda v: x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
    parts = [f'<text class="ptitle" x="{x0}" y="{ROW_Y0 - 18}">Cost per correct answer (log)</text>']
    for e in range(lo, hi + 1):
        if (e - lo) % max(1, (hi - lo) // 3) == 0:
            parts.append(axis(X(10.0 ** e), money(10.0 ** e).rstrip("0") if e < -2 else f"${10.0 ** e:g}", len(rows)))
    for i, r in enumerate(rows):
        y = ROW_Y0 + i * ROW_H
        x = X(r["usd"])
        parts.append(f'<circle class="{r["where"]}" cx="{x:.1f}" cy="{y + BAR_H / 2}" r="7"><title>{r["name"]}: {money(r["usd"])}</title></circle>')
        parts.append(f'<text class="val" x="{x + 12:.1f}" y="{y + 15}">{money(r["usd"])}</text>')
    return "".join(parts)


def render(rows: list[dict]) -> str:
    local, cloud = rows[0], rows[1:]
    best = max(cloud, key=lambda r: r["score"])
    cheapest = min(cloud, key=lambda r: r["usd"])
    title = f"Claude is right far more often; the local model costs about {cheapest['usd'] / local['usd']:,.0f}× less per correct answer"
    if len(title) > 96:
        title = f"Local: {local['score'] * 100:.0f}% correct at a fraction of the cost. {best['name'].replace('Claude ', '')}: {best['score'] * 100:.0f}%"
    labels = []
    for i, r in enumerate(rows):
        y = ROW_Y0 + i * ROW_H + 15
        where = "this PC, RTX 3060 Ti" if r["where"] == "local" else "Anthropic cloud"
        labels.append(f'<text class="name" x="{LABEL_X}" y="{y - 1}">{r["name"]}</text>')
        labels.append(f'<text class="where" x="{LABEL_X}" y="{y + 14}">{where}</text>')
    foot_y = ROW_Y0 + ROW_H * len(rows) + 30
    n = rows[0]["tasks"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
<title id="t">{title}</title>
<desc id="d">{n} tasks, three repeats each, graded by code. {"; ".join(f"{r['name']}: {r['score'] * 100:.0f}% correct, {r['tps']:.0f} tokens per second, {money(r['usd'])} per correct answer" for r in rows)}.</desc>
<style>
  .root {{ --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
          --cloud: #2a78d6; --local: #eb6834; }}
  @media (prefers-color-scheme: dark) {{
    .root {{ --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
            --cloud: #3987e5; --local: #d95926; }}
  }}
  text {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink2); font-size: 13px; }}
  .bg {{ fill: var(--surface); }}
  .title {{ fill: var(--ink); font-size: 20px; font-weight: 600; }}
  .sub {{ font-size: 14px; }}
  .ptitle {{ fill: var(--ink); font-size: 13px; font-weight: 600; }}
  .name {{ fill: var(--ink); font-size: 13px; font-weight: 600; }}
  .where {{ fill: var(--muted); font-size: 12px; }}
  .val {{ fill: var(--ink); font-size: 13px; font-variant-numeric: tabular-nums; }}
  .tick {{ fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .axis {{ stroke: var(--axis); stroke-width: 1; }}
  .whisk {{ stroke: var(--ink2); stroke-width: 1.5; stroke-linecap: round; }}
  .local {{ fill: var(--local); }} .cloud {{ fill: var(--cloud); }}
  .note {{ fill: var(--muted); font-size: 12px; }}
</style>
<g class="root">
<rect class="bg" width="{W}" height="{H}" rx="8"/>
<text class="title" x="24" y="36">{title}</text>
<text class="sub" x="24" y="60">{n} original tasks, three repeats each, every answer graded by code against a known answer.</text>
<text class="sub" x="24" y="78">Exact match, JSON, coding against hidden tests, multi-step reasoning. Lines show the 95% interval.</text>
<rect class="local" x="24" y="96" width="14" height="14" rx="3"/><text x="44" y="108">On this PC (llama.cpp, RTX 3060 Ti)</text>
<rect class="cloud" x="300" y="96" width="14" height="14" rx="3"/><text x="320" y="108">Anthropic cloud, through Claude Code on a Claude plan</text>
{''.join(labels)}
{quality_panel(rows, PANELS[0])}
{speed_panel(rows, PANELS[1])}
{cost_panel(rows, PANELS[2])}
<text class="note" x="24" y="{foot_y}">Cost ratio is against {cheapest['name']}. Claude's cost is the API list price (these runs used a plan). The local cost is GPU electricity at ${kwh_price():.2f}/kWh:</text>
<text class="note" x="24" y="{foot_y + 17}">an estimate, GPU board only, excluding the price of the GPU. A 3B model is expected to score below Claude; later phases ask what each local setting gives up.</text>
</g>
</svg>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=r"D:\LLMLab\results\lab.sqlite")
    ap.add_argument("--local", type=int, required=True, help="session with the local model")
    ap.add_argument("--cloud", type=int, required=True, help="claude-quality session")
    ap.add_argument("--out", type=Path, default=HERE / "speed-quality-cost.svg")
    args = ap.parse_args()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = load(conn, args.local, args.cloud)
    args.out.write_text(render(rows), encoding="utf-8")
    for r in rows:
        print(f"{r['name']:<20} quality {r['score'] * 100:5.1f}%  {r['tps']:7.1f} tok/s  {money(r['usd'])} per correct answer")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
