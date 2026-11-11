// Hand-drawn SVG charts for the dashboard, with no libraries: line (with shaded run
// phases and a crosshair), horizontal bars, strip plots of repeated runs, per-core CPU
// bars and sparklines. Every chart redraws when its container changes width, and every
// value it shows is also in a table on the same page.

const NS = 'http://www.w3.org/2000/svg';

export function svg(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    // CSS variables only resolve reliably in style, not in presentation attributes.
    if ((k === 'fill' || k === 'stroke') && String(v).includes('var(')) e.style.setProperty(k, v);
    else e.setAttribute(k, v);
  }
  if (parent) parent.appendChild(e);
  return e;
}

function label(parent, x, y, str, cls, anchor = 'start') {
  const t = svg('text', { x, y, class: cls, 'text-anchor': anchor }, parent);
  t.textContent = str;
  return t;
}

export function fmtTick(v) {
  const a = Math.abs(v);
  if (a >= 1000) return Math.round(v).toLocaleString('en-US');
  if (a >= 10 || v === 0) return String(Math.round(v * 10) / 10);
  return String(Math.round(v * 100) / 100);
}

export function niceScale(min, max, count = 4) {
  if (!(max > min)) max = min + 1;
  const raw = (max - min) / count;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / p;
  const step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
  const lo = Math.floor(min / step + 1e-9) * step;
  const hi = Math.ceil(max / step - 1e-9) * step;
  const ticks = [];
  for (let v = lo; v <= hi + step * 1e-6; v += step) ticks.push(Math.round(v * 1e6) / 1e6);
  return { lo, hi: hi > lo ? hi : lo + step, ticks, step };
}

// ------------------------------------------------------------ tooltip

export function showTip(evt, title, rows) {
  const el = document.getElementById('tooltip');
  if (!el) return;
  el.replaceChildren();
  if (title) {
    const t = document.createElement('div');
    t.className = 'tt-title';
    t.textContent = title;
    el.appendChild(t);
  }
  for (const r of rows) {
    const row = document.createElement('div');
    row.className = 'tt-row';
    if (r.color) {
      const key = document.createElement('i');
      key.style.background = r.color;
      row.appendChild(key);
    }
    const value = document.createElement('strong');
    value.textContent = r.value;
    row.appendChild(value);
    if (r.label) {
      const name = document.createElement('span');
      name.textContent = r.label;
      row.appendChild(name);
    }
    el.appendChild(row);
  }
  el.hidden = false;
  const pad = 14;
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  if (x + el.offsetWidth > window.innerWidth - 8) x = evt.clientX - el.offsetWidth - pad;
  if (y + el.offsetHeight > window.innerHeight - 8) y = evt.clientY - el.offsetHeight - pad;
  el.style.left = `${Math.max(8, x)}px`;
  el.style.top = `${Math.max(8, y)}px`;
}

export function hideTip() {
  const el = document.getElementById('tooltip');
  if (el) el.hidden = true;
}

// ------------------------------------------------------------ mounting

const observed = new WeakSet();

function mount(host, render) {
  host.classList.add('chart');
  host._render = render;
  render();
  if (observed.has(host)) return;
  observed.add(host);
  let width = host.clientWidth;
  let frame = 0;
  new ResizeObserver(() => {
    if (host.clientWidth === width) return;
    width = host.clientWidth;
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => host._render && host._render());
  }).observe(host);
}

function legend(host, items, kind = 'line') {
  const d = document.createElement('div');
  d.className = 'legend';
  for (const it of items) {
    const s = document.createElement('span');
    const key = document.createElement(kind === 'line' ? 'i' : 'b');
    key.style.background = it.color;
    s.append(key, document.createTextNode(it.name));
    d.appendChild(s);
  }
  host.appendChild(d);
}

function nearest(points, x) {
  let lo = 0;
  let hi = points.length - 1;
  if (hi < 0) return -1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid][0] < x) lo = mid;
    else hi = mid;
  }
  return Math.abs(points[lo][0] - x) <= Math.abs(points[hi][0] - x) ? lo : hi;
}

function pointerX(evt, root, width) {
  const rect = root.getBoundingClientRect();
  return (evt.clientX - rect.left) * (width / rect.width);
}

// ------------------------------------------------------------ line chart

// series: [{name, color, points: [[x, y], ...]}] sorted by x; y may be null for a gap.
// bands: [{x0, x1, label, title, kind}] shaded behind the lines (load, warm-up, runs).
export function lineChart(host, opts) {
  mount(host, () => drawLine(host, opts));
}

function drawLine(host, o) {
  host.replaceChildren();
  const series = o.series.filter((s) => s.points.some((p) => p[1] != null));
  if (o.legend ?? o.series.length > 1) legend(host, o.series.map((s) => ({ name: s.name, color: s.color })));
  const W = Math.max(240, host.clientWidth);
  const H = o.height ?? 160;
  const bands = o.bands || [];
  const m = { l: 46, r: 14, t: bands.some((b) => b.label) ? 16 : 8, b: 22 };
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  let [x0, x1] = o.xDomain ?? [Math.min(...xs), Math.max(...xs)];
  if (!Number.isFinite(x0) || !Number.isFinite(x1)) [x0, x1] = [0, 1];
  if (x1 <= x0) x1 = x0 + 1;
  const ys = series.flatMap((s) => s.points.map((p) => p[1]).filter((v) => v != null));
  // yMax is a natural ceiling (100%, the GPU's top clock); without one, leave a little headroom.
  const dataTop = ys.length ? Math.max(...ys) : 1;
  const top = o.yMax != null ? Math.max(o.yMax, dataTop) : dataTop * 1.05;
  const ysc = niceScale(o.yMin ?? 0, top, o.yTicks ?? 5);
  const pw = W - m.l - m.r;
  const ph = H - m.t - m.b;
  const X = (x) => m.l + ((x - x0) / (x1 - x0)) * pw;
  const Y = (y) => m.t + (1 - (y - ysc.lo) / (ysc.hi - ysc.lo)) * ph;
  const fy = o.yFmt || fmtTick;
  const fx = o.xFmt || fmtTick;
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': o.label || '' }, host);

  for (const b of bands) {
    const bx0 = Math.max(X(b.x0), m.l);
    const bx1 = Math.min(X(b.x1), W - m.r);
    if (bx1 - bx0 < 0.5) continue;
    svg('rect', { x: bx0, y: m.t, width: bx1 - bx0, height: ph, class: `band ${b.kind || ''}` }, root);
    if (b.label && bx1 - bx0 >= 12) label(root, (bx0 + bx1) / 2, m.t - 5, b.label, 'band-label', 'middle');
  }
  for (const t of ysc.ticks) {
    const y = Y(t);
    svg('line', { x1: m.l, x2: W - m.r, y1: y, y2: y, class: t === ysc.lo ? 'axisline' : 'gridline' }, root);
    label(root, m.l - 6, y + 4, fy(t), 'tick', 'end');
  }
  const xsc = niceScale(x0, x1, Math.max(2, Math.floor(pw / 90)));
  for (const t of xsc.ticks) {
    if (t < x0 - 1e-9 || t > x1 + 1e-9) continue;
    label(root, X(t), H - 5, fx(t), 'tick', 'middle');
  }
  for (const s of series) {
    let d = '';
    let pen = false;
    for (const [x, y] of s.points) {
      if (y == null || x < x0 || x > x1) {
        pen = false;
        continue;
      }
      d += `${pen ? 'L' : 'M'}${X(x).toFixed(1)},${Y(y).toFixed(1)}`;
      pen = true;
    }
    svg('path', { d, class: 'series', stroke: s.color }, root);
    if (o.endDot) {
      const last = [...s.points].reverse().find((p) => p[1] != null && p[0] <= x1);
      if (last) {
        svg('circle', { cx: X(last[0]), cy: Y(last[1]), r: 6, class: 'dot-ring' }, root);
        svg('circle', { cx: X(last[0]), cy: Y(last[1]), r: 4, fill: s.color }, root);
      }
    }
  }
  if (!series.length) {
    label(root, m.l + pw / 2, m.t + ph / 2, o.empty || 'No data', 'tick', 'middle');
    return;
  }

  const cross = svg('line', { y1: m.t, y2: m.t + ph, class: 'crosshair', visibility: 'hidden', 'pointer-events': 'none' }, root);
  const dots = series.map((s) =>
    svg('circle', { r: 4, fill: s.color, stroke: 'var(--surface)', 'stroke-width': 2, visibility: 'hidden', 'pointer-events': 'none' }, root));
  const hit = svg('rect', { x: m.l, y: 0, width: pw, height: H, class: 'hit' }, root);
  const ref = series.reduce((a, b) => (b.points.length > a.points.length ? b : a));
  hit.addEventListener('pointermove', (e) => {
    const xv = x0 + ((pointerX(e, root, W) - m.l) / pw) * (x1 - x0);
    const at = ref.points[nearest(ref.points, xv)][0];
    cross.setAttribute('x1', X(at));
    cross.setAttribute('x2', X(at));
    cross.setAttribute('visibility', 'visible');
    const rows = series.map((s, k) => {
      const p = s.points[nearest(s.points, at)];
      const close = p && p[1] != null && Math.abs(p[0] - at) <= (o.snap ?? (x1 - x0) / 50);
      dots[k].setAttribute('visibility', close ? 'visible' : 'hidden');
      if (close) {
        dots[k].setAttribute('cx', X(p[0]));
        dots[k].setAttribute('cy', Y(p[1]));
      }
      return { color: s.color, value: close ? (o.valueFmt || fy)(p[1]) : 'N/A', label: s.name };
    });
    const band = bands.find((b) => at >= b.x0 && at <= b.x1 && b.title);
    showTip(e, `${fx(at)}${o.xUnit ? ` ${o.xUnit}` : ''}${band ? ` · ${band.title}` : ''}`, rows);
  });
  hit.addEventListener('pointerleave', () => {
    cross.setAttribute('visibility', 'hidden');
    dots.forEach((d) => d.setAttribute('visibility', 'hidden'));
    hideTip();
  });
}

// ------------------------------------------------------------ horizontal bars

function barPath(x0, x1, y, h, r = 4) {
  const w = x1 - x0;
  if (w < 0.5) return '';
  r = Math.min(r, w / 2, h / 2);
  return `M${x0},${y}H${x1 - r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - r},${y + h}H${x0}Z`;
}

// rows: [{label, sub, value, color, valueText, title, href}]
export function barChart(host, opts) {
  mount(host, () => drawBars(host, opts));
}

function drawBars(host, o) {
  host.replaceChildren();
  if (o.legend) legend(host, o.legend, 'box');
  const W = Math.max(240, host.clientWidth);
  const labelW = o.labelWidth ?? Math.min(170, Math.round(W * 0.38));
  const rowH = o.rowHeight ?? 38;
  const barH = Math.min(20, rowH - 14);
  const H = o.rows.length * rowH + 22;
  const values = o.rows.map((r) => r.value).filter((v) => v != null);
  const sc = niceScale(0, o.max ?? Math.max(1e-9, ...values), o.ticks ?? 4);
  const valueRoom = o.valueRoom ?? 70;
  const pw = W - labelW - valueRoom;
  const X = (v) => labelW + (v / sc.hi) * pw;
  const fmt = o.fmt || fmtTick;
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': o.label || '' }, host);
  const plotH = o.rows.length * rowH;
  for (const t of sc.ticks) {
    svg('line', { x1: X(t), x2: X(t), y1: 0, y2: plotH, class: t === 0 ? 'axisline' : 'gridline' }, root);
    label(root, X(t), plotH + 15, (o.tickFmt || fmtTick)(t), 'tick', 'middle');
  }
  o.rows.forEach((r, i) => {
    const y = i * rowH + (rowH - barH) / 2;
    label(root, 0, y + barH / 2 - (r.sub ? 2 : -4), r.label, 'bar-label');
    if (r.sub) label(root, 0, y + barH / 2 + 12, r.sub, 'bar-sub');
    if (r.value == null) {
      label(root, labelW + 6, y + barH / 2 + 4, 'N/A', 'bar-value');
      return;
    }
    const end = X(r.value);
    const bar = svg('path', { d: barPath(labelW, end, y, barH), fill: r.color, class: 'mark' }, root);
    label(root, end + 6, y + barH / 2 + 4, r.valueText ?? fmt(r.value), 'bar-value');
    const hit = svg('rect', { x: 0, y: i * rowH, width: W, height: rowH, class: 'hit' }, root);
    if (r.href) {
      hit.style.cursor = 'pointer';
      hit.addEventListener('click', () => { location.hash = r.href; });
    }
    hit.addEventListener('pointermove', (e) => {
      bar.classList.add('hover');
      showTip(e, r.title ?? r.label, [{ color: r.color, value: r.valueText ?? fmt(r.value), label: o.unitLabel || '' }]);
    });
    hit.addEventListener('pointerleave', () => {
      bar.classList.remove('hover');
      hideTip();
    });
  });
}

// ------------------------------------------------------------ aligned panels

// One row per configuration, one small chart per measure, each with its own axis (never
// two scales on one axis). rows: [{label, sub, color, href, title}]; each panel:
// {title, unit, read(row) -> {v, lo, hi, text}, log}. Bars start at zero; a log axis has
// no zero, so its panel uses dots instead.
export function alignedPanels(host, opts) {
  mount(host, () => drawPanels(host, opts));
}

function drawPanels(host, o) {
  host.replaceChildren();
  if (o.legend) legend(host, o.legend, 'box');
  const W = Math.max(280, host.clientWidth);
  // Side by side when there is room, one under another when there isn't.
  const groups = W >= 760 ? [o.panels] : o.panels.map((p) => [p]);
  for (const panels of groups) drawPanelRow(host, o, panels, W);
}

function logTicks(values) {
  const lo = Math.floor(Math.log10(Math.min(...values)));
  const hi = Math.ceil(Math.log10(Math.max(...values)) + 1e-9);
  const top = hi > lo ? hi : lo + 1;
  const ticks = [];
  for (let e = lo; e <= top; e += 1) ticks.push(e);
  return { lo, hi: top, ticks };
}

function drawPanelRow(host, o, panels, W) {
  const labelW = Math.min(158, Math.round(W * 0.3));
  const gap = 18;
  const rowH = 40;
  const top = 26;
  const H = top + o.rows.length * rowH + 22;
  const pw = (W - labelW - gap * (panels.length - 1)) / panels.length;
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': o.label || '' }, host);
  const barH = 18;
  o.rows.forEach((r, i) => {
    const y = top + i * rowH + (rowH - barH) / 2;
    label(root, 0, y + barH / 2 - 2, r.label, 'bar-label');
    if (r.sub) label(root, 0, y + barH / 2 + 12, r.sub, 'bar-sub');
  });
  panels.forEach((p, k) => {
    const x0 = labelW + k * (pw + gap);
    const valueRoom = 66;
    const inner = Math.max(40, pw - valueRoom);
    const t = label(root, x0, 13, p.title, 'panel-title');
    if (p.unit) {
      const u = svg('tspan', { class: 'panel-unit', dx: 6 }, t);
      u.textContent = p.unit;
    }
    const read = o.rows.map((r) => p.read(r));
    const vals = read.map((d) => d.v).filter((v) => v != null && (!p.log || v > 0));
    if (!vals.length) {
      label(root, x0, top + 18, 'N/A', 'bar-value');
      return;
    }
    let X;
    let ticks;
    if (p.log) {
      const sc = logTicks(vals);
      X = (v) => x0 + ((Math.log10(v) - sc.lo) / (sc.hi - sc.lo)) * inner;
      ticks = sc.ticks.map((e) => [10 ** e, (p.tickFmt || fmtTick)(10 ** e)]);
    } else {
      const sc = niceScale(0, p.max ?? Math.max(...vals), p.ticks ?? 3);
      X = (v) => x0 + (v / sc.hi) * inner;
      ticks = sc.ticks.map((v) => [v, (p.tickFmt || fmtTick)(v)]);
    }
    const plotH = o.rows.length * rowH;
    for (const [v, text] of ticks) {
      svg('line', { x1: X(v), x2: X(v), y1: top, y2: top + plotH, class: (!p.log && v === 0) ? 'axisline' : 'gridline' }, root);
      label(root, X(v), top + plotH + 15, text, 'tick', 'middle');
    }
    o.rows.forEach((r, i) => {
      const d = read[i];
      const y = top + i * rowH + (rowH - barH) / 2;
      if (d.v == null || (p.log && d.v <= 0)) {
        label(root, x0 + 6, y + barH / 2 + 4, 'N/A', 'bar-value');
        return;
      }
      let end = X(d.v);
      let mark;
      if (p.log) {
        mark = svg('circle', { cx: end, cy: y + barH / 2, r: 6, fill: r.color, class: 'mark' }, root);
        end += 6;
      } else {
        mark = svg('path', { d: barPath(x0, end, y, barH), fill: r.color, class: 'mark' }, root);
      }
      if (d.lo != null && d.hi != null) {
        const cy = y + barH / 2;
        svg('line', { x1: X(d.lo), x2: X(d.hi), y1: cy, y2: cy, class: 'whisker' }, root);
        svg('line', { x1: X(d.lo), x2: X(d.lo), y1: cy - 4, y2: cy + 4, class: 'whisker' }, root);
        svg('line', { x1: X(d.hi), x2: X(d.hi), y1: cy - 4, y2: cy + 4, class: 'whisker' }, root);
        end = Math.max(end, X(d.hi));
      }
      label(root, end + 6, y + barH / 2 + 4, d.text, 'bar-value');
      const hit = svg('rect', { x: x0, y: top + i * rowH, width: pw, height: rowH, class: 'hit' }, root);
      if (r.href) {
        hit.style.cursor = 'pointer';
        hit.addEventListener('click', () => { location.hash = r.href; });
      }
      hit.addEventListener('pointermove', (e) => {
        mark.classList.add('hover');
        showTip(e, r.title ?? r.label, [{ color: r.color, value: d.tip ?? d.text, label: p.title }]);
      });
      hit.addEventListener('pointerleave', () => {
        mark.classList.remove('hover');
        hideTip();
      });
    });
  });
}

// ------------------------------------------------------------ strip plot

// The spread of repeated runs: one row per group, one dot per run, a line at the median.
// groups: [{name, color, points: [{v, hollow, invalid, title}], median}]
export function stripPlot(host, opts) {
  mount(host, () => drawStrip(host, opts));
}

function drawStrip(host, o) {
  host.replaceChildren();
  const groups = o.groups.filter((g) => g.points.some((p) => p.v != null));
  if (o.legend) legend(host, o.groups.map((g) => ({ name: g.name, color: g.color })), 'box');
  const W = Math.max(240, host.clientWidth);
  const names = o.showNames ?? groups.length > 1;
  const labelW = names ? Math.min(150, Math.round(W * 0.3)) : 12;
  const rowH = o.rowHeight ?? 40;
  const H = Math.max(1, groups.length) * rowH + 22;
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': o.label || '' }, host);
  const vals = groups.flatMap((g) => g.points.map((p) => p.v).filter((v) => v != null));
  if (!vals.length) {
    label(root, W / 2, H / 2, o.empty || 'No runs', 'tick', 'middle');
    return;
  }
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  const pad = Math.max((hi - lo) * 0.25, Math.abs(hi) * 0.01, 1e-6);
  lo = o.min ?? Math.max(0, lo - pad);
  hi = o.max ?? hi + pad;
  const sc = niceScale(lo, hi, Math.max(2, Math.floor((W - labelW) / 90)));
  const pw = W - labelW - 16;
  const X = (v) => labelW + ((v - sc.lo) / (sc.hi - sc.lo)) * pw;
  const fmt = o.fmt || fmtTick;
  const plotH = groups.length * rowH;
  for (const t of sc.ticks) {
    svg('line', { x1: X(t), x2: X(t), y1: 0, y2: plotH, class: 'gridline' }, root);
    label(root, X(t), plotH + 15, (o.tickFmt || fmtTick)(t), 'tick', 'middle');
  }
  groups.forEach((g, i) => {
    const cy = i * rowH + rowH / 2 + 4;
    if (names) label(root, 0, cy + 4, g.name, 'bar-label');
    svg('line', { x1: labelW, x2: W - 16, y1: cy, y2: cy, class: 'axisline' }, root);
    if (g.median != null) {
      svg('line', { x1: X(g.median), x2: X(g.median), y1: cy - 13, y2: cy + 13, class: 'median-line' }, root);
      label(root, X(g.median) + 5, cy - 9, `median ${fmt(g.median)}`, 'median-label');
    }
    for (const p of g.points) {
      if (p.v == null) continue;
      const x = X(p.v);
      let mark;
      if (p.invalid) {
        mark = svg('path', { d: `M${x - 4},${cy - 4}L${x + 4},${cy + 4}M${x + 4},${cy - 4}L${x - 4},${cy + 4}`, stroke: 'var(--critical)', 'stroke-width': 2, 'stroke-linecap': 'round', class: 'mark' }, root);
      } else if (p.hollow) {
        mark = svg('circle', { cx: x, cy, r: 4, fill: 'var(--surface)', stroke: g.color, 'stroke-width': 2, class: 'mark' }, root);
      } else {
        svg('circle', { cx: x, cy, r: 6, class: 'dot-ring' }, root);
        mark = svg('circle', { cx: x, cy, r: 4.5, fill: g.color, class: 'mark' }, root);
      }
      const hit = svg('circle', { cx: x, cy, r: 12, class: 'hit' }, root);
      hit.addEventListener('pointermove', (e) => {
        mark.classList.add('hover');
        showTip(e, p.title ?? g.name, [{ color: g.color, value: fmt(p.v), label: o.unitLabel || '' }]);
      });
      hit.addEventListener('pointerleave', () => {
        mark.classList.remove('hover');
        hideTip();
      });
    }
  });
}

// ------------------------------------------------------------ CPU cores

export function coreBars(host, values, opts = {}) {
  mount(host, () => drawCores(host, values, opts));
}

function drawCores(host, values, o) {
  host.replaceChildren();
  const W = Math.max(200, host.clientWidth);
  const H = o.height ?? 120;
  const m = { l: 34, r: 6, t: 6, b: 18 };
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, height: H, role: 'img', 'aria-label': 'CPU use per core' }, host);
  const ph = H - m.t - m.b;
  const Y = (v) => m.t + (1 - v / 100) * ph;
  for (const t of [0, 50, 100]) {
    svg('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: t === 0 ? 'axisline' : 'gridline' }, root);
    label(root, m.l - 6, Y(t) + 4, `${t}%`, 'tick', 'end');
  }
  const n = values.length || 1;
  const slot = (W - m.l - m.r) / n;
  const bw = Math.min(24, slot - 4);
  values.forEach((v, i) => {
    const x = m.l + i * slot + (slot - bw) / 2;
    const val = Math.max(0, Math.min(100, v ?? 0));
    const y = Y(val);
    const h = m.t + ph - y;
    const r = Math.min(3, h / 2, bw / 2);
    const d = h < 0.5 ? '' : `M${x},${m.t + ph}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${m.t + ph}Z`;
    const bar = svg('path', { d, fill: o.color || 'var(--m-cpu)', class: 'mark' }, root);
    label(root, x + bw / 2, H - 4, String(i + 1), 'tick', 'middle');
    const hit = svg('rect', { x: m.l + i * slot, y: 0, width: slot, height: H, class: 'hit' }, root);
    hit.addEventListener('pointermove', (e) => {
      bar.classList.add('hover');
      showTip(e, `Logical core ${i + 1}`, [{ color: o.color || 'var(--m-cpu)', value: v == null ? 'N/A' : `${Math.round(v)}%`, label: 'busy' }]);
    });
    hit.addEventListener('pointerleave', () => {
      bar.classList.remove('hover');
      hideTip();
    });
  });
}

// ------------------------------------------------------------ sparkline

export function sparkline(host, values, color) {
  host.replaceChildren();
  const pts = values.filter((v) => v != null);
  if (pts.length < 2) return;
  const W = Math.max(60, host.clientWidth);
  const H = 28;
  const lo = Math.min(...pts);
  const hi = Math.max(...pts);
  const span = hi - lo || 1;
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, height: H, 'aria-hidden': 'true' }, host);
  const X = (i) => 2 + (i / (pts.length - 1)) * (W - 8);
  const Y = (v) => 4 + (1 - (v - lo) / span) * (H - 8);
  const d = pts.map((v, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join('');
  svg('path', { d, class: 'series', stroke: color, 'stroke-opacity': 0.55 }, root);
  svg('circle', { cx: X(pts.length - 1), cy: Y(pts[pts.length - 1]), r: 3, fill: color }, root);
}
