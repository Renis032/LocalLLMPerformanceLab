"""Draw docs/figures/clock-ramp.svg from the lab database.

Purpose: the README's headline figure. It shows the GPU clock during five measured
runs of the same model on the same GPU, measured two ways. With a 1 s pause between
runs, the GPU drops to a power-saving state and ramps back too slowly to reach full
speed, so generation runs at 97 tok/s instead of 152.

    D:\\LLMLab\\tools\\uv\\uv.exe run python docs/figures/clock_ramp.py --slow 1 --fast 4

--slow and --fast are session ids; the figure uses each session's first configuration.
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Plot geometry (SVG user units).
W, H = 820, 432
X0, X1, Y0, Y1 = 70, 640, 140, 372
Y_MAX = 2200


def series(conn: sqlite3.Connection, session_id: int) -> tuple[list[tuple[float, float]], float]:
    (config_id,) = conn.execute(
        "SELECT id FROM configs WHERE session_id = ? ORDER BY idx LIMIT 1", (session_id,)
    ).fetchone()
    trials = conn.execute(
        "SELECT t_send, t_last, gen_tps FROM trials WHERE config_id = ? AND kind = 'measured' AND valid = 1 "
        "ORDER BY t_send", (config_id,)
    ).fetchall()
    t0, t1 = trials[0][0], trials[-1][1]
    points = [
        (t - t0, mhz)
        for t, mhz in conn.execute(
            "SELECT t, gpu_clock_mhz FROM samples WHERE config_id = ? AND t BETWEEN ? AND ? "
            "AND gpu_clock_mhz IS NOT NULL ORDER BY t", (config_id, t0, t1)
        )
    ]
    return points, statistics.median(r[2] for r in trials)


def sx(t: float, t_max: float) -> float:
    return X0 + (X1 - X0) * t / t_max


def sy(mhz: float) -> float:
    return Y1 - (Y1 - Y0) * mhz / Y_MAX


def polyline(points, t_max, cls) -> str:
    coords = " ".join(f"{sx(t, t_max):.1f},{sy(m):.1f}" for t, m in points)
    return f'<polyline class="{cls}" points="{coords}"/>'


def end_marker(points, t_max, cls, label_lines: list[str]) -> str:
    t, m = points[-1]
    x, y = sx(t, t_max), sy(m)
    text = "".join(
        f'<tspan x="{x + 12:.1f}" dy="{0 if i == 0 else 17}" class="{"lbl-strong" if i == 0 else "lbl"}">{s}</tspan>'
        for i, s in enumerate(label_lines)
    )
    return (
        f'<circle class="ring" cx="{x:.1f}" cy="{y:.1f}" r="6"/>'
        f'<circle class="{cls}-dot" cx="{x:.1f}" cy="{y:.1f}" r="4"/>'
        f'<text y="{y - 4:.1f}">{text}</text>'
    )


def render(fast, fast_tps, slow, slow_tps) -> str:
    t_max = 5 * -(-max(fast[-1][0], slow[-1][0]) // 5)  # round up to 5 s
    grid = []
    for mhz in range(0, Y_MAX + 1, 500):
        y = sy(mhz)
        grid.append(f'<line class="{"axis" if mhz == 0 else "grid"}" x1="{X0}" x2="{X1}" y1="{y:.1f}" y2="{y:.1f}"/>')
        grid.append(f'<text class="tick" x="{X0 - 8}" y="{y + 4:.1f}" text-anchor="end">{mhz:,}</text>')
    for t in range(0, int(t_max) + 1, 5):
        x = sx(t, t_max)
        grid.append(f'<text class="tick" x="{x:.1f}" y="{Y1 + 20}" text-anchor="middle">{t}</text>')

    # One annotation: the first dip of the slow series, where the pause drops the clock.
    dip_t, dip_m = min((p for p in slow if 3 < p[0] < 9), key=lambda p: p[1])
    dx, dy = sx(dip_t, t_max), sy(dip_m)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
<title id="t">Same GPU, same model: {fast_tps:.0f} vs {slow_tps:.0f} tokens per second</title>
<desc id="d">GPU clock during five measured runs of SmolLM3 3B on an RTX 3060 Ti. Run back to back with a 0.1 second pause, the clock holds near 1,950 MHz and generation runs at {fast_tps:.0f} tokens per second. With a 1 second pause, the clock falls to about 780 MHz in every pause and climbs only to about 1,200 MHz during each run, so generation runs at {slow_tps:.0f} tokens per second.</desc>
<style>
  .root {{ --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
          --s1: #2a78d6; --s2: #eb6834; }}
  @media (prefers-color-scheme: dark) {{
    .root {{ --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
            --s1: #3987e5; --s2: #d95926; }}
  }}
  text {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink2); font-size: 13px; }}
  .bg {{ fill: var(--surface); }}
  .title {{ fill: var(--ink); font-size: 20px; font-weight: 600; }}
  .sub {{ font-size: 14px; }}
  .tick {{ fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }}
  .grid {{ stroke: var(--grid); stroke-width: 1; }}
  .axis {{ stroke: var(--axis); stroke-width: 1; }}
  .fast, .slow {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
  .fast {{ stroke: var(--s1); }} .slow {{ stroke: var(--s2); }}
  .fast-dot {{ fill: var(--s1); }} .slow-dot {{ fill: var(--s2); }}
  .ring {{ fill: var(--surface); }}
  .lbl-strong {{ fill: var(--ink); font-weight: 600; font-size: 14px; }}
  .lbl {{ fill: var(--ink2); font-size: 13px; }}
  .note {{ fill: var(--muted); font-size: 12px; }}
  .leader {{ stroke: var(--muted); stroke-width: 1; }}
</style>
<g class="root">
<rect class="bg" width="{W}" height="{H}" rx="8"/>
<text class="title" x="24" y="36">Same GPU, same model: {fast_tps:.0f} vs {slow_tps:.0f} tokens per second</text>
<text class="sub" x="24" y="60">The only difference is the pause between runs. After 1 s idle the GPU drops to a power-saving</text>
<text class="sub" x="24" y="78">state and ramps back too slowly, so every run is measured well below full speed.</text>
<line class="fast" x1="24" x2="44" y1="100" y2="100"/><text x="50" y="104">0.1 s pause between runs</text>
<line class="slow" x1="230" x2="250" y1="100" y2="100"/><text x="256" y="104">1 s pause between runs</text>
<text class="tick" x="{X0 - 8}" y="{Y0 - 10}" text-anchor="end">MHz</text>
{''.join(grid)}
<text class="tick" x="{(X0 + X1) / 2:.0f}" y="{Y1 + 42}" text-anchor="middle">GPU clock, seconds since the first measured run (RTX 3060 Ti, SmolLM3 3B Q4_K_M, 512 tokens per run)</text>
{polyline(slow, t_max, "slow")}
{polyline(fast, t_max, "fast")}
<line class="leader" x1="{dx:.1f}" y1="{dy + 6:.1f}" x2="{dx:.1f}" y2="{Y1 - 14}"/>
<text class="note" x="{dx + 6:.1f}" y="{Y1 - 16}">pause: the clock drops to {dip_m:,.0f} MHz</text>
{end_marker(fast, t_max, "fast", [f"{fast_tps:.0f} tok/s", "clock holds ~1,950 MHz"])}
{end_marker(slow, t_max, "slow", [f"{slow_tps:.0f} tok/s", "never above ~1,215 MHz"])}
</g>
</svg>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=r"D:\LLMLab\results\lab.sqlite")
    ap.add_argument("--slow", type=int, required=True, help="session measured with a 1 s pause")
    ap.add_argument("--fast", type=int, required=True, help="session measured with a 0.1 s pause")
    ap.add_argument("--out", type=Path, default=HERE / "clock-ramp.svg")
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)
    fast, fast_tps = series(conn, args.fast)
    slow, slow_tps = series(conn, args.slow)
    args.out.write_text(render(fast, fast_tps, slow, slow_tps), encoding="utf-8")
    print(f"wrote {args.out}: fast {fast_tps:.1f} tok/s ({len(fast)} samples), slow {slow_tps:.1f} tok/s ({len(slow)} samples)")


if __name__ == "__main__":
    main()
