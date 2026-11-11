"""Draw docs/figures/local-vs-cloud.svg from the lab database.

Purpose: the README's local-versus-cloud figure. It sets a 3B model on this PC's GPU
beside Claude Haiku, Sonnet and Opus on the same prompt: generation speed in one chart,
time to the first visible token in the other (two charts, because the two measures
have different units).

    D:\\LLMLab\\tools\\uv\\uv.exe run python docs/figures/local_vs_cloud.py --local 4 --cloud 6

--local is a session with the local model (its first configuration is used); --cloud is
a claude-baseline session (every configuration is used).
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

W, H = 820, 392
LABEL_X = 24
PANELS = ((214, 444), (548, 778))  # x range of each chart's bars
ROW_Y0, ROW_H, BAR_H = 172, 38, 20
NAMES = {
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-opus-5-5": "Claude Opus 5.5",
}


def medians(conn, config_id: int) -> dict:
    rows = conn.execute(
        "SELECT gen_tps, ttft_ms, thinking_tokens FROM trials WHERE config_id = ? AND kind = 'measured' AND valid = 1",
        (config_id,),
    ).fetchall()
    return {
        "tps": statistics.median(r[0] for r in rows),
        "ttft_s": statistics.median(r[1] for r in rows) / 1000,
        "thinking": statistics.median((r[2] or 0) for r in rows),
    }


def load(conn, local_session: int, cloud_session: int) -> list[dict]:
    (local_id,) = conn.execute(
        "SELECT id FROM configs WHERE session_id = ? ORDER BY idx LIMIT 1", (local_session,)
    ).fetchone()
    out = [{"name": "SmolLM3 3B, Q4_K_M", "where": "local", **medians(conn, local_id)}]
    for cid, model in conn.execute(
        "SELECT id, model_path FROM configs WHERE session_id = ? AND status = 'ok' ORDER BY idx", (cloud_session,)
    ):
        out.append({"name": NAMES.get(model, model), "where": "cloud", **medians(conn, cid)})
    return out


def bar(x0: float, x1: float, y: float, h: float, r: float = 4) -> str:
    """A bar anchored at x0, with a rounded data end at x1."""
    r = min(r, (x1 - x0) / 2, h / 2)
    return (
        f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
        f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z"
    )


def panel(rows, key, x_range, top, ticks, fmt_value, fmt_tick, title) -> str:
    x0, x1 = x_range
    scale = (x1 - x0) / top
    parts = [f'<text class="ptitle" x="{x0}" y="{ROW_Y0 - 18}">{title}</text>']
    y_axis_end = ROW_Y0 + ROW_H * len(rows) - (ROW_H - BAR_H)
    for t in ticks:
        x = x0 + t * scale
        parts.append(f'<line class="{"axis" if t == 0 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{ROW_Y0 - 6}" y2="{y_axis_end + 6}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{y_axis_end + 22}" text-anchor="middle">{fmt_tick(t)}</text>')
    for i, r in enumerate(rows):
        y = ROW_Y0 + i * ROW_H
        v = r[key]
        end = x0 + v * scale
        parts.append(
            f'<path class="{r["where"]}" d="{bar(x0, end, y, BAR_H)}"><title>{r["name"]}: {fmt_value(v)}</title></path>'
        )
        parts.append(f'<text class="val" x="{end + 6:.1f}" y="{y + 15}">{fmt_value(v)}</text>')
    return "".join(parts)


def render(rows: list[dict]) -> str:
    local = rows[0]
    cloud = rows[1:]
    sooner = sorted(r["ttft_s"] / local["ttft_s"] for r in cloud)
    slower = [r["name"].replace("Claude ", "").rsplit(" ", 1)[0] for r in cloud if r["tps"] < local["tps"]]
    title = f"A 3B model on a gaming GPU starts answering {sooner[0]:.0f}–{sooner[-1]:.0f}× sooner than Claude"
    labels = []
    for i, r in enumerate(rows):
        y = ROW_Y0 + i * ROW_H + 15
        where = "this PC, RTX 3060 Ti" if r["where"] == "local" else "Anthropic cloud"
        labels.append(f'<text class="name" x="{LABEL_X}" y="{y - 1}">{r["name"]}</text>')
        labels.append(f'<text class="where" x="{LABEL_X}" y="{y + 14}">{where}</text>')
    thinker = next((r for r in cloud if r["thinking"]), None)
    note = (
        f"{thinker['name'].replace('Claude ', '')} thinks for about {thinker['thinking']:.0f} hidden tokens before its first visible token. "
        if thinker else ""
    )
    gen = panel(rows, "tps", PANELS[0], 220, (0, 50, 100, 150, 200), lambda v: f"{v:.0f} tok/s", lambda t: f"{t}",
                "Generation speed, tokens/s (higher is faster)")
    ttft = panel(rows, "ttft_s", PANELS[1], 2.0, (0, 0.5, 1.0, 1.5, 2.0), lambda v: f"{v:.2f} s", lambda t: f"{t:g}",
                 "First visible token, seconds (lower is sooner)")
    foot_y = ROW_Y0 + ROW_H * len(rows) + 30
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
<title id="t">{title}</title>
<desc id="d">Median of 5 runs on the same 521-token prompt. {"; ".join(f"{r['name']}: {r['tps']:.0f} tokens per second, first visible token after {r['ttft_s']:.2f} seconds" for r in rows)}.</desc>
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
  .local {{ fill: var(--local); }} .cloud {{ fill: var(--cloud); }}
  .note {{ fill: var(--muted); font-size: 12px; }}
</style>
<g class="root">
<rect class="bg" width="{W}" height="{H}" rx="8"/>
<text class="title" x="24" y="36">{title}</text>
<text class="sub" x="24" y="60">It also streams faster than {" and ".join(slower)}.</text>
<text class="sub" x="24" y="78">Speed only: quality is in the figure at the top of the README. Median of 5 runs on the same 521-token prompt.</text>
<rect class="local" x="24" y="96" width="14" height="14" rx="3"/><text x="44" y="108">On this PC (llama.cpp, RTX 3060 Ti)</text>
<rect class="cloud" x="300" y="96" width="14" height="14" rx="3"/><text x="320" y="108">Anthropic cloud, through Claude Code on a Claude plan</text>
{''.join(labels)}
{gen}
{ttft}
<text class="note" x="24" y="{foot_y}">{note}Claude Code's start-up (about 0.55 s per request) is not counted.</text>
<text class="note" x="24" y="{foot_y + 17}">Each model family has its own tokenizer, so tokens/s compare only approximately across families.</text>
</g>
</svg>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=r"D:\LLMLab\results\lab.sqlite")
    ap.add_argument("--local", type=int, required=True, help="session with the local model")
    ap.add_argument("--cloud", type=int, required=True, help="claude-baseline session")
    ap.add_argument("--out", type=Path, default=HERE / "local-vs-cloud.svg")
    args = ap.parse_args()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = load(conn, args.local, args.cloud)
    args.out.write_text(render(rows), encoding="utf-8")
    for r in rows:
        print(f"{r['name']:<20} {r['tps']:7.1f} tok/s  {r['ttft_s']:.3f} s  thinking {r['thinking']:.0f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
