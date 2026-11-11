// The dashboard's pages: Overview, Run (live), History, a session's detail, Quality and Compare.
// Everything is read from the local API; text from the database is always inserted as
// text, never as HTML.

import { alignedPanels, barChart, coreBars, hideTip, lineChart, sparkline, stripPlot } from './charts.js';

// ------------------------------------------------------------ helpers

function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat(Infinity)) {
    if (c == null || c === false) continue;
    e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return e;
}

const NS = 'http://www.w3.org/2000/svg';
function icon(kind) {
  const s = document.createElementNS(NS, 'svg');
  s.setAttribute('viewBox', '0 0 16 16');
  s.setAttribute('aria-hidden', 'true');
  const add = (tag, attrs) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if ((k === 'fill' || k === 'stroke') && String(v).includes('var(')) e.style.setProperty(k, v);
      else e.setAttribute(k, v);
    }
    s.appendChild(e);
  };
  const ink = { stroke: 'var(--surface)', 'stroke-width': '1.8', 'stroke-linecap': 'round', 'stroke-linejoin': 'round', fill: 'none' };
  if (kind === 'warning') {
    add('path', { d: 'M8 1.5 15 14H1Z', fill: 'currentColor', 'stroke-linejoin': 'round' });
    add('path', { d: 'M8 6v3.5M8 11.6v.1', ...ink });
  } else if (kind === 'neutral') {
    add('circle', { cx: 8, cy: 8, r: 6.25, fill: 'none', stroke: 'currentColor', 'stroke-width': 1.5 });
    add('path', { d: 'M5.5 8h5', stroke: 'currentColor', 'stroke-width': 1.5, 'stroke-linecap': 'round' });
  } else {
    add('circle', { cx: 8, cy: 8, r: 7, fill: 'currentColor' });
    add('path', { d: kind === 'good' ? 'M5 8.2 7 10.2 11 6' : 'M5.6 5.6l4.8 4.8M10.4 5.6l-4.8 4.8', ...ink });
  }
  return s;
}

function infoIcon() {
  const s = document.createElementNS(NS, 'svg');
  s.setAttribute('viewBox', '0 0 16 16');
  s.setAttribute('aria-hidden', 'true');
  s.innerHTML = '<circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M8 7.2v4M8 4.8v.1" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>';
  return s;
}

const STATUS = {
  ok: ['good', 'OK'],
  completed: ['good', 'Completed'],
  warning: ['warning', 'Warning'],
  aborted: ['warning', 'Aborted'],
  failed: ['critical', 'Failed'],
  running: ['neutral', 'Running'],
};

function statusBadge(status) {
  const [tone, text] = STATUS[status] || ['neutral', status || 'Unknown'];
  return h('span', { class: `badge status-${tone}` }, icon(tone), text);
}

function checkBadge(passed) {
  const tone = passed === true ? 'good' : passed === false ? 'critical' : 'neutral';
  return icon(tone);
}

function engineBadge(engine) {
  const cloud = engine === 'claude-code';
  return h('span', { class: `badge ${cloud ? 'engine-cloud' : 'engine-local'}` },
    h('span', { class: 'dot' }), cloud ? 'Cloud · Claude Code' : 'Local · llama.cpp');
}

const isCloud = (engine) => engine === 'claude-code';

function fmt(v, d = 1) {
  if (v == null || Number.isNaN(v)) return 'N/A';
  return Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
}
const fmt0 = (v) => fmt(v, 0);
function fmtMs(v) {
  if (v == null) return 'N/A';
  return v >= 1000 ? `${fmt(v / 1000, 2)} s` : `${fmt(v, 0)} ms`;
}
function fmtMoney(v) {
  if (v == null) return 'N/A';
  return v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(3)}`;
}
function fmtWhen(iso) {
  if (!iso) return 'N/A';
  const d = new Date(iso);
  return d.toLocaleString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}
function fmtDuration(a, b) {
  if (!a || !b) return 'N/A';
  const s = Math.round((new Date(b) - new Date(a)) / 1000);
  return s >= 60 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${s} s`;
}
function pctDiff(b, a) {
  return a == null || b == null || a === 0 ? null : ((b - a) / a) * 100;
}
const signed = (p) => (p == null ? 'N/A' : `${p >= 0 ? '+' : '−'}${Math.abs(p).toFixed(1)}%`);
const pct = (v, d = 0) => (v == null ? 'N/A' : `${(v * 100).toFixed(d)}%`);
const points = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(1)}`;
const med = (s, k) => (s && s[k] ? s[k].median : null);

// "claude-haiku-5-5" -> "Claude Haiku 5.5"; "SmolLM3-Q4_K_M.gguf" -> "SmolLM3-Q4_K_M"
function prettyModel(name) {
  if (!name) return 'N/A';
  const m = /^claude-([a-z]+)-(\d+)(?:-(\d+))?$/.exec(name);
  if (m) return `Claude ${m[1][0].toUpperCase()}${m[1].slice(1)} ${m[2]}${m[3] ? `.${m[3]}` : ''}`;
  return name.replace(/\.gguf$/i, '');
}

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch (e) {
    return null;
  }
  return value;
}

async function api(url, options) {
  const r = await fetch(url, options);
  const body = r.headers.get('content-type')?.includes('json') ? await r.json() : await r.text();
  if (!r.ok) throw new Error(body?.error || `${r.status} ${r.statusText}`);
  return body;
}
const post = (url, data) => api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });

function tile({ label, color, value, unit, foot, spark }) {
  const sparkHost = spark ? h('div', { class: 'spark' }) : null;
  const el = h('div', { class: 'tile' },
    h('div', { class: 'label' }, color ? h('span', { class: 'key', style: `background:${color}` }) : null, label),
    h('div', { class: 'value' }, value, unit ? h('span', { class: 'unit' }, unit) : null),
    foot ? h('div', { class: 'foot' }, foot) : null,
    sparkHost);
  if (spark && spark.values.length > 1) requestAnimationFrame(() => sparkline(sparkHost, spark.values, color));
  return el;
}

function card(title, sub, ...body) {
  return h('section', { class: 'card' },
    title ? h('div', { class: 'card-head' }, h('h2', {}, title), sub ? h('span', { class: 'sub' }, sub) : null) : null,
    ...body);
}

function chartBox(title, sub, host, now) {
  return h('div', {},
    h('div', { class: 'chart-title' }, title, sub ? h('span', { class: 'sub' }, sub) : null, now ? h('span', { class: 'now' }, now) : null),
    host);
}

function notice(text, tone = 'neutral') {
  return h('div', { class: `notice status-${tone}` }, tone === 'neutral' ? infoIcon() : icon(tone), h('div', {}, text));
}

function dedupe(list, key) {
  const seen = new Set();
  return list.filter((x) => (seen.has(key(x)) ? false : seen.add(key(x))));
}

// ------------------------------------------------------------ router

const routes = [
  [/^#?\/?$/, 'overview', viewOverview],
  [/^#\/run$/, 'run', viewRun],
  [/^#\/history$/, 'history', viewHistory],
  [/^#\/session\/(\d+)$/, 'history', viewSession],
  [/^#\/quality$/, 'quality', viewQuality],
  [/^#\/quality\/(\d+)$/, 'quality', viewQualityDetail],
  [/^#\/compare$/, 'compare', viewCompare],
];
let cleanup = null;
let routeToken = 0;

async function route() {
  if (cleanup) cleanup();
  cleanup = null;
  hideTip();
  const [path, query] = (location.hash || '#/').split('?');
  const found = routes.find(([re]) => re.test(path)) || routes[0];
  const [re, tab, view] = found;
  for (const a of document.querySelectorAll('.tabs a')) {
    if (a.dataset.tab === tab) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  }
  const root = document.getElementById('view');
  root.replaceChildren(h('div', { class: 'muted' }, 'Loading…'));
  const token = ++routeToken;
  try {
    const done = await view(root, re.exec(path), new URLSearchParams(query || ''), () => token === routeToken);
    if (token === routeToken) cleanup = done || null;
    else if (done) done();
  } catch (e) {
    if (token === routeToken) root.replaceChildren(notice(`Couldn't load this page: ${e.message}`, 'critical'));
  }
  pollPill();
}

window.addEventListener('hashchange', route);

document.getElementById('theme-toggle').addEventListener('click', () => {
  const dark = getComputedStyle(document.documentElement).colorScheme.includes('dark');
  const next = dark ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  store('llmlab-theme', next);
  route();
});

// The pill in the top bar shows a run in progress on every page.
let pillTimer = 0;
async function pollPill() {
  clearTimeout(pillTimer);
  const pill = document.getElementById('run-pill');
  try {
    const s = await api('/api/status');
    document.getElementById('demo-badge').hidden = !s.demo;
    const active = s.state === 'running' || s.state === 'starting';
    pill.hidden = !active || location.hash === '#/run';
    pill.querySelector('.pill-text').textContent = s.state === 'starting'
      ? 'Starting a run'
      : `Running · ${s.experiment || ''}${s.total ? ` · ${s.index + 1}/${s.total}` : ''}`;
  } catch (e) {
    pill.hidden = true;
  }
  pillTimer = setTimeout(pollPill, 3000);
}

// ------------------------------------------------------------ overview

async function viewOverview(root) {
  const [ov, sessions] = await Promise.all([api('/api/overview'), api('/api/sessions')]);
  const latest = dedupe(ov.latest, (r) => `${r.engine}|${r.label}`);
  const local = latest.filter((r) => !isCloud(r.engine)).sort((a, b) => (b.gen_tps ?? 0) - (a.gen_tps ?? 0));
  const cloud = latest.filter((r) => isCloud(r.engine)).sort((a, b) => (b.gen_tps ?? 0) - (a.gen_tps ?? 0));
  const best = local[0];
  const c = ov.counts || {};
  root.replaceChildren();

  root.append(h('div', { class: 'page-head' },
    h('div', {},
      h('h1', {}, 'Local vs cloud, measured on this PC'),
      h('p', {}, 'Every number is the median of repeated runs that passed the lab’s trust checks: steady GPU clocks, a quiet CPU, the settings actually applied, and for Claude, one request per run with no hidden extras.')),
    h('a', { class: 'btn primary', href: '#/run' }, 'Start a run')));

  if (!latest.length) {
    root.append(h('div', { class: 'empty' }, 'No results yet. Start a run to fill the dashboard.'));
    return;
  }

  const ratios = best ? cloud.map((r) => r.ttft_ms / best.ttft_ms).filter((x) => Number.isFinite(x)) : [];
  const heroCard = h('section', { class: 'card' },
    h('div', { class: 'card-head' }, h('h2', {}, 'Fastest local configuration'), best ? engineBadge(best.engine) : null),
    best ? h('div', { class: 'hero' }, fmt(best.gen_tps, 1), h('span', { class: 'unit' }, 'tok/s')) : h('div', { class: 'muted' }, 'No local results yet'),
    best ? h('p', { class: 'ink-2', style: 'margin-top:10px' },
      `${prettyModel(best.model)}, ${best.label}. First token after ${fmtMs(best.ttft_ms)}; runs varied ${fmt(best.cv_pct, 2)}%.`) : null,
    ratios.length ? h('p', { class: 'ink-2', style: 'margin-top:6px' },
      `It starts answering ${fmt(Math.min(...ratios), 0)}–${fmt(Math.max(...ratios), 0)}× sooner than Claude on the same prompt.`) : null);

  const tiles = h('div', { class: 'tiles' },
    tile({ label: 'Sessions', value: fmt0(c.sessions), foot: `${fmt0(c.configs)} configurations` }),
    tile({ label: 'Measured runs', value: fmt0(c.runs), foot: `${fmt0(c.invalid_runs)} rejected by checks` }),
    tile({ label: 'Trust checks run', value: fmt0(c.checks_run), foot: `${fmt0(c.checks_failed)} failed and flagged` }),
    tile({ label: 'Configurations tracked', value: fmt0(latest.length), foot: `${local.length} local, ${cloud.length} cloud` }));

  root.append(h('div', { class: 'grid cols-2' }, heroCard, h('div', { class: 'stack' }, tiles,
    notice(['Claude runs in Anthropic’s cloud, so its first token includes the network and queueing. Speed is half the story: ', h('a', { href: '#/quality' }, 'see how good the answers are'), '.']))));

  const rows = [...local, ...cloud];
  const legendItems = [{ name: 'Local (llama.cpp)', color: 'var(--c-local)' }, { name: 'Cloud (Claude Code)', color: 'var(--c-cloud)' }];
  const toRow = (r, value, text) => ({
    label: isCloud(r.engine) ? prettyModel(r.model) : prettyModel(r.model),
    sub: isCloud(r.engine) ? 'Anthropic cloud' : r.label,
    value,
    valueText: text,
    color: isCloud(r.engine) ? 'var(--c-cloud)' : 'var(--c-local)',
    title: `${prettyModel(r.model)} · ${r.label} · session ${r.session_id}`,
    href: `#/session/${r.session_id}`,
  });
  const gen = h('div');
  const ttft = h('div');
  root.append(h('div', { class: 'section-title' }, 'Latest result of each configuration'));
  root.append(h('div', { class: 'grid cols-2' },
    card('Generation speed', 'tokens per second, higher is faster', gen),
    card('Time to first visible token', 'seconds, lower is sooner', ttft)));
  barChart(gen, { rows: rows.map((r) => toRow(r, r.gen_tps, `${fmt(r.gen_tps, 1)} tok/s`)), legend: legendItems, fmt: (v) => fmt(v, 1), unitLabel: 'tokens/s', label: 'Generation speed by configuration' });
  barChart(ttft, { rows: rows.map((r) => toRow(r, r.ttft_ms == null ? null : r.ttft_ms / 1000, `${fmt(r.ttft_ms / 1000, 2)} s`)), legend: legendItems, fmt: (v) => fmt(v, 2), unitLabel: 'seconds', label: 'Time to first token by configuration' });

  root.append(h('div', { class: 'section-title' }, 'All configurations'));
  root.append(card(null, null, h('div', { class: 'table-wrap' }, h('table', {},
    h('thead', {}, h('tr', {}, h('th', {}, 'Model'), h('th', {}, 'Configuration'), h('th', {}, 'Where'),
      h('th', { class: 'n' }, 'Generation'), h('th', { class: 'n' }, 'First token'), h('th', { class: 'n' }, 'Run-to-run CV'), h('th', { class: 'n' }, 'Quality'),
      h('th', { class: 'n' }, 'Cost per run'), h('th', {}, 'Session'))),
    h('tbody', {}, latest.map((r) => h('tr', { class: 'link', onclick: () => { location.hash = `#/session/${r.session_id}`; } },
      h('td', {}, prettyModel(r.model)), h('td', { class: 'ink-2' }, r.label), h('td', {}, engineBadge(r.engine)),
      h('td', { class: 'n' }, `${fmt(r.gen_tps, 1)} tok/s`), h('td', { class: 'n' }, fmtMs(r.ttft_ms)),
      h('td', { class: 'n' }, `${fmt(r.cv_pct, 2)}%`),
      h('td', { class: 'n' }, pct((ov.quality || {})[r.config_hash])),
      h('td', { class: 'n' }, isCloud(r.engine) ? fmtMoney(r.cost_usd) : `${fmt(r.j_per_token, 2)} J/token`),
      h('td', {}, `#${r.session_id}`))))))));

  root.append(h('div', { class: 'section-title' }, 'Recent sessions'));
  root.append(sessionsTable(sessions.slice(0, 6)));
}

// ------------------------------------------------------------ history

function sessionsTable(sessions) {
  if (!sessions.length) return h('div', { class: 'empty' }, 'No sessions recorded yet.');
  return card(null, null, h('div', { class: 'table-wrap' }, h('table', {},
    h('thead', {}, h('tr', {}, h('th', { class: 'n' }, 'ID'), h('th', {}, 'Started'), h('th', {}, 'Experiment'),
      h('th', {}, 'Status'), h('th', {}, 'Configurations'), h('th', {}, 'GPU'))),
    h('tbody', {}, sessions.map((s) => h('tr', { class: 'link', onclick: () => { location.hash = `#/session/${s.id}`; } },
      h('td', { class: 'n' }, s.id), h('td', {}, fmtWhen(s.started_at)), h('td', {}, s.experiment), h('td', {}, statusBadge(s.status)),
      h('td', {}, h('div', { class: 'row' }, s.configs.map((c) => h('span', { class: `badge status-${(STATUS[c.status] || ['neutral'])[0]}`, title: `${c.label}: ${c.status}` },
        icon((STATUS[c.status] || ['neutral'])[0]), `${c.label} · ${c.gen_tps == null ? 'N/A' : `${fmt(c.gen_tps, 1)} tok/s`}`)))),
      h('td', { class: 'muted' }, s.gpu || 'N/A')))))));
}

async function viewHistory(root) {
  const sessions = await api('/api/sessions');
  root.replaceChildren(
    h('div', { class: 'page-head' }, h('div', {}, h('h1', {}, 'History'),
      h('p', {}, 'Every session recorded in the database, newest first. Open one for its runs, telemetry and checks.'))),
    sessionsTable(sessions));
}

// ------------------------------------------------------------ one session

function trialName(t) {
  return t.kind === 'warmup' ? `warm-up ${t.idx + 1}` : `run ${t.idx + 1}`;
}

function runsStrip(host, trials, key, fmtFn, color, unitLabel) {
  const measuredValid = trials.filter((t) => t.kind === 'measured' && t.valid && t[key] != null).map((t) => t[key]);
  const sorted = [...measuredValid].sort((a, b) => a - b);
  const median = sorted.length ? (sorted.length % 2 ? sorted[(sorted.length - 1) / 2] : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2) : null;
  stripPlot(host, {
    groups: [{
      name: 'Runs', color, median,
      points: trials.map((t) => ({
        v: t[key], hollow: t.kind === 'warmup', invalid: !t.valid,
        title: `${trialName(t)}${t.kind === 'warmup' ? ' (not counted)' : ''}${!t.valid ? ` · rejected: ${(t.problems_json || []).join('; ')}` : ''}`,
      })),
    }],
    fmt: fmtFn, unitLabel, label: `${unitLabel} per run`,
  });
}

function timelineBands(tl) {
  const names = { load: ['load', 'loading the model'], quiet: ['quiet', 'quiet check after the runs'], quality: ['quality', 'answering the quality tasks'] };
  const bands = (tl.phases || []).map((p) => ({ x0: p.t0, x1: p.t1, kind: p.name, label: names[p.name]?.[0], title: names[p.name]?.[1] }));
  for (const t of tl.trials) {
    if (t.t_send == null || t.t_last == null) continue;
    bands.push({ x0: t.t_send, x1: t.t_last, kind: t.kind, label: t.kind === 'warmup' ? 'W' : String(t.idx + 1), title: trialName(t) });
  }
  return bands;
}

function configCard(c, timelinePromise) {
  const cloud = isCloud(c.engine);
  const s = c.summary || {};
  const g = s.gen_tps;
  const trials = c.trials || [];
  const valid = trials.filter((t) => t.kind === 'measured' && t.valid);
  const median = (key) => {
    const v = valid.map((t) => t[key]).filter((x) => x != null).sort((a, b) => a - b);
    return v.length ? (v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2) : null;
  };
  const tiles = cloud ? [
    tile({ label: 'Generation', color: 'var(--m-gen)', value: fmt(g?.median, 1), unit: 'tok/s', foot: `CV ${fmt(g?.cv_pct, 2)}% · ${s.n || 0} runs` }),
    tile({ label: 'First visible token', color: 'var(--m-ttft)', value: fmtMs(med(s, 'ttft_ms')), foot: `first content ${fmtMs(med(s, 'first_content_ms'))}` }),
    tile({ label: 'Hidden thinking', value: fmt0(med(s, 'thinking_tokens')), unit: 'tokens', foot: 'median per run' }),
    tile({ label: 'Tokens per run', value: fmt0(med(s, 'gen_tokens')), unit: 'out', foot: `${fmt0(med(s, 'prompt_tokens'))} in` }),
    tile({ label: 'Cost per run', value: fmtMoney(med(s, 'cost_usd')), foot: 'at API list prices' }),
    tile({ label: 'Chunk gap p95', value: fmt0(s.itl_pooled_ms?.p95), unit: 'ms', foot: `p50 ${fmt0(s.itl_pooled_ms?.p50)} ms` }),
  ] : [
    tile({ label: 'Generation', color: 'var(--m-gen)', value: fmt(g?.median, 1), unit: 'tok/s', foot: `CV ${fmt(g?.cv_pct, 2)}% · ${s.n || 0} runs` }),
    tile({ label: 'Prompt', value: fmt0(med(s, 'prompt_tps')), unit: 'tok/s', foot: `${fmt0(c.prompt_tokens)} prompt tokens` }),
    tile({ label: 'First token', color: 'var(--m-ttft)', value: fmtMs(med(s, 'ttft_ms')), foot: `inter-token p95 ${fmt(s.itl_pooled_ms?.p95, 2)} ms` }),
    tile({ label: 'GPU power', color: 'var(--m-power)', value: fmt0(med(s, 'gpu_power_instant_mean_w')), unit: 'W', foot: `${fmt(median('j_per_token'), 2)} J/token` }),
    tile({ label: 'VRAM added', color: 'var(--m-vram)', value: fmt0(c.vram_peak_mib != null && c.vram_before_mib != null ? c.vram_peak_mib - c.vram_before_mib : null), unit: 'MiB', foot: `server RAM ${fmt0(c.server_rss_max_mib)} MiB` }),
    tile({ label: 'GPU load', color: 'var(--m-load)', value: fmt0(med(s, 'gpu_util_mean_pct')), unit: '%', foot: `${fmt0(med(s, 'gpu_clock_mean_mhz'))} MHz` }),
  ];

  const strip = h('div');
  const right = h('div');
  const extra = h('div', { class: 'grid cols-2' });
  const checks = c.checks || [];
  const invalid = trials.filter((t) => !t.valid);
  const el = h('section', { class: 'card stack' },
    h('div', { class: 'card-head', style: 'margin-bottom:0' },
      h('h2', {}, `${c.idx + 1}. ${c.label}`), engineBadge(c.engine), statusBadge(c.status),
      h('span', { class: 'sub' }, `${prettyModel(c.model)} · config ${c.config_hash}`),
      h('a', { class: 'btn end', href: `#/compare?a=${c.id}` }, 'Compare…')),
    c.error ? notice(c.error, 'critical') : null,
    h('div', { class: 'tiles' }, tiles),
    c.quality ? qualityBrief(c) : null,
    h('div', { class: 'grid cols-2' },
      chartBox('Generation speed per run', 'dots: runs · hollow: warm-up · ×: rejected', strip),
      cloud ? chartBox('Time to first visible token per run', null, right) : chartBox('GPU clock', 'MHz, with load, warm-up and runs shaded', right)),
    cloud ? null : extra,
    h('div', { class: 'grid cols-2' },
      h('div', {}, h('div', { class: 'chart-title' }, 'Trust checks',
        h('span', { class: 'sub' }, `${checks.filter((x) => x.passed === true).length} of ${checks.length} passed · ${invalid.length} run(s) rejected`)),
      h('ul', { class: 'checks' },
        checks.map((x) => h('li', {}, h('span', { class: `status-${x.passed === true ? 'good' : x.passed === false ? 'critical' : 'neutral'}` }, checkBadge(x.passed)),
          h('div', {}, h('div', { class: 'name' }, x.check), h('div', { class: 'detail' }, x.detail)))),
        invalid.map((t) => h('li', {}, h('span', { class: 'status-critical' }, icon('critical')),
          h('div', {}, h('div', { class: 'name' }, `${trialName(t)} rejected`), h('div', { class: 'detail' }, (t.problems_json || []).join('; '))))))),
      runsTable(c)),
    details(c));

  requestAnimationFrame(() => {
    runsStrip(strip, trials, 'gen_tps', (v) => fmt(v, 1), 'var(--m-gen)', 'tokens/s');
    if (cloud) runsStrip(right, trials, 'ttft_ms', (v) => fmtMs(v), 'var(--m-ttft)', 'first visible token');
  });
  if (!cloud && timelinePromise) {
    timelinePromise.then((tl) => {
      if (!tl || !tl.samples.length) {
        right.replaceChildren(h('div', { class: 'empty' }, 'No telemetry for this configuration.'));
        return;
      }
      const bands = timelineBands(tl);
      // The quality tasks take minutes; the speed runs are what these charts are for.
      const qualityPhase = (tl.phases || []).find((p) => p.name === 'quality');
      const shown = qualityPhase ? tl.samples.filter((p) => p.t <= qualityPhase.t0) : tl.samples;
      const series = (key, name, color) => [{ name, color, points: shown.map((p) => [p.t, p[key]]) }];
      lineChart(right, { series: series('clock', 'GPU clock', 'var(--m-clock)'), bands, yFmt: fmtAxis, xFmt: (v) => fmt(v, 0), xUnit: 's', valueFmt: (v) => `${fmt0(v)} MHz`, label: 'GPU clock over the session' });
      const power = h('div');
      const util = h('div');
      extra.append(chartBox('GPU power', 'W, instantaneous', power), chartBox('GPU utilization', '%', util));
      lineChart(power, { series: series('power', 'GPU power', 'var(--m-power)'), bands, xFmt: (v) => fmt(v, 0), xUnit: 's', valueFmt: (v) => `${fmt0(v)} W`, label: 'GPU power over the session' });
      lineChart(util, { series: series('util', 'GPU utilization', 'var(--m-load)'), bands, yMax: 100, xFmt: (v) => fmt(v, 0), xUnit: 's', valueFmt: (v) => `${fmt0(v)}%`, label: 'GPU utilization over the session' });
    }).catch((e) => right.replaceChildren(notice(e.message, 'critical')));
  }
  return el;
}

const fmtAxis = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : String(Math.round(v * 10) / 10));

function qualityBrief(c) {
  const q = c.quality;
  const failed = (q.checks || []).filter((x) => !x.passed);
  return h('div', { class: 'stack', style: 'gap:6px' },
    h('div', { class: 'chart-title' }, 'Answer quality', h('span', { class: 'sub' }, `suite ${q.suite_name} v${q.suite_version}, graded by code`),
      h('a', { class: 'now', href: `#/quality/${c.id}` }, 'Every answer')),
    h('div', { class: 'tiles' },
      tile({ label: 'Quality', value: pct(q.score), foot: q.score == null ? (q.error || '') : `${pct(q.score_lo)} to ${pct(q.score_hi)} (95%)` }),
      tile({ label: 'Correct answers', value: fmt0(q.correct), unit: `of ${fmt0(q.attempts)}` }),
      tile({ label: 'Consistency', value: pct(q.consistency), foot: 'repeats that agree' }),
      tile({ label: 'Cost per correct answer', value: fmtCost(q.cost?.usd), foot: costFoot(q.cost) }),
      Object.entries(q.categories || {}).map(([name, x]) => tile({ label: CATEGORY_NAMES[name] || name, value: pct(x.score), foot: `${x.correct} of ${x.attempts} correct` }))),
    failed.map((x) => notice(`${x.check}: ${x.detail}`, 'warning')));
}

function fmtCost(v) {
  if (v == null) return 'N/A';
  if (v >= 0.01) return `$${v.toFixed(3)}`;
  return `$${v.toPrecision(2)}`;
}
const costFoot = (c) => (c?.kind === 'list' ? 'at API list prices' : c?.kind === 'electricity' ? 'electricity, estimate' : '');

function runsTable(c) {
  const cloud = isCloud(c.engine);
  const head = cloud
    ? ['', 'gen tok/s', 'first token', 'thinking', 'tokens out', 'requests', 'counted']
    : ['', 'gen tok/s', 'prompt tok/s', 'first token', 'W', 'J/token', 'counted'];
  const row = (t) => (cloud
    ? [fmt(t.gen_tps, 1), fmtMs(t.ttft_ms), fmt0(t.thinking_tokens), fmt0(t.gen_tokens), fmt0(t.api_calls)]
    : [fmt(t.gen_tps, 1), fmt0(t.prompt_tps), fmtMs(t.ttft_ms), fmt0(t.gpu_power_instant_mean_w), fmt(t.j_per_token, 2)]);
  return h('div', {}, h('div', { class: 'chart-title' }, 'Every run'), h('div', { class: 'table-wrap' }, h('table', {},
    h('thead', {}, h('tr', {}, head.map((x, i) => h('th', { class: i ? 'n' : null }, x)))),
    h('tbody', {}, (c.trials || []).map((t) => h('tr', { class: t.kind === 'warmup' ? 'dim' : null },
      h('td', {}, trialName(t)), row(t).map((v) => h('td', { class: 'n' }, v)),
      h('td', { class: 'n' }, t.kind === 'warmup' ? 'no (warm-up)' : t.valid ? 'yes' : 'no')))))));
}

function details(c) {
  const kv = (obj) => h('dl', { class: 'kv' }, Object.entries(obj || {}).filter(([, v]) => v != null && v !== '' && !(Array.isArray(v) && !v.length))
    .map(([k, v]) => [h('dt', {}, k), h('dd', {}, typeof v === 'object' ? JSON.stringify(v) : String(v))]));
  const cfg = c.config || {};
  const command = Array.isArray(c.command) ? c.command.map((a) => (a === '' || /\s/.test(a) ? `"${a}"` : a)).join(' ') : null;
  return h('details', { class: 'more' }, h('summary', {}, 'Settings, command and what the engine reported'),
    h('div', { class: 'grid cols-2' },
      h('div', { class: 'stack' },
        h('h3', {}, 'Settings'), kv({ ...cfg.engine, ...cfg.server, ...cfg.model, ...cfg.run, prompt: undefined }),
        command ? [h('h3', {}, 'Command'), h('pre', { class: 'log' }, command)] : null),
      h('div', { class: 'stack' },
        h('h3', {}, isCloud(c.engine) ? 'Claude Code' : 'Server log facts'), kv({ engine_version: c.engine_version, ...c.facts }),
        c.has_log ? h('a', { href: `/api/configs/${c.id}/log`, target: '_blank', rel: 'noopener' }, 'Open the full server log') : null)));
}

async function viewSession(root, match) {
  const id = Number(match[1]);
  const d = await api(`/api/sessions/${id}`);
  const s = d.session;
  const host = s.host || {};
  const gpu = host.gpu || {};
  const versions = [...new Set(d.configs.map((c) => `${isCloud(c.engine) ? 'Claude Code' : 'llama.cpp'} ${c.engine_version || ''}`.trim()))];
  root.replaceChildren(...[
    h('div', { class: 'page-head' }, h('div', {},
      h('div', { class: 'row' }, h('h1', {}, `Session ${s.id} · ${s.experiment}`), statusBadge(s.status)),
      s.description ? h('p', {}, s.description) : null),
      h('a', { class: 'btn', href: '#/history' }, 'All sessions')),
    card(null, null, h('dl', { class: 'kv' },
      h('dt', {}, 'Started'), h('dd', {}, `${fmtWhen(s.started_at)} · took ${fmtDuration(s.started_at, s.finished_at)}`),
      h('dt', {}, 'GPU'), h('dd', {}, `${gpu.name || 'N/A'} · driver ${gpu.driver || 'N/A'} · power limit ${fmt0(gpu.power_limit_w)} W`),
      h('dt', {}, 'CPU'), h('dd', {}, `${host.cpu || 'N/A'} · ${fmt0((host.ram_total_mib || 0) / 1024)} GiB RAM · power plan ${host.power_plan || 'N/A'}`),
      h('dt', {}, 'Engine'), h('dd', {}, versions.join('; ') || 'N/A'),
      h('dt', {}, 'Experiment file'), h('dd', {}, s.source_path || 'N/A'))),
    s.error ? notice(s.error, 'warning') : null].filter(Boolean));
  const list = h('div', { class: 'stack', style: 'margin-top:12px' });
  root.append(list);
  for (const c of d.configs) {
    const tl = isCloud(c.engine) ? null : api(`/api/configs/${c.id}/timeline`);
    list.append(configCard(c, tl));
  }
}

// ------------------------------------------------------------ quality

const CATEGORY_NAMES = { exact: 'Exact match', json: 'JSON generation', code: 'Coding', reasoning: 'Reasoning' };

function qualityName(r) {
  return isCloud(r.engine) ? prettyModel(r.model) : prettyModel(r.model);
}
const qualitySub = (r) => (isCloud(r.engine) ? 'Anthropic cloud' : r.label);

function heatCell(rate, text) {
  const mix = rate == null ? 0 : Math.round(rate * 55);
  return h('td', { class: 'heat', style: `background:color-mix(in srgb, var(--accent) ${mix}%, transparent)` }, text);
}

async function viewQuality(root) {
  const rows = await api('/api/quality');
  const head = h('div', { class: 'page-head' }, h('div', {}, h('h1', {}, 'Speed, quality and cost'),
    h('p', {}, 'Is a faster setting still as good? Every answer is graded by code against a known answer, so no model grades another or itself.')));
  if (!rows.length) {
    root.replaceChildren(head, h('div', { class: 'empty' }, 'No quality runs yet. Start an experiment with a quality section, such as smollm3-quality.yaml.'));
    return;
  }
  const legend = [{ name: 'Local (llama.cpp)', color: 'var(--c-local)' }, { name: 'Cloud (Claude Code)', color: 'var(--c-cloud)' }];
  const panelRows = rows.map((r) => ({
    label: qualityName(r), sub: qualitySub(r), color: isCloud(r.engine) ? 'var(--c-cloud)' : 'var(--c-local)',
    title: `${qualityName(r)} · ${r.label} · session ${r.session_id}`, href: `#/quality/${r.config_id}`, r,
  }));
  const chart = h('div');
  const suite = rows[0];
  root.replaceChildren(head,
    card('Speed, quality and cost, side by side', `suite ${suite.suite} v${suite.suite_version} · lines show the 95% interval · cost axis is logarithmic`, chart),
    h('div', { class: 'section-title' }, 'The numbers'),
    card(null, null, h('div', { class: 'table-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, 'Model'), h('th', {}, 'Where'), h('th', { class: 'n' }, 'Quality'), h('th', { class: 'n' }, '95% interval'),
        h('th', { class: 'n' }, 'Consistency'), h('th', { class: 'n' }, 'Correct'), h('th', { class: 'n' }, 'Generation'), h('th', { class: 'n' }, 'First token'),
        h('th', { class: 'n' }, 'Cost per correct answer'), h('th', {}, 'Checks'))),
      h('tbody', {}, rows.map((r) => h('tr', { class: 'link', onclick: () => { location.hash = `#/quality/${r.config_id}`; } },
        h('td', {}, qualityName(r), h('div', { class: 'muted small' }, qualitySub(r))), h('td', {}, engineBadge(r.engine)), h('td', { class: 'n' }, pct(r.score)),
        h('td', { class: 'n' }, `${pct(r.score_lo)} to ${pct(r.score_hi)}`), h('td', { class: 'n' }, pct(r.consistency)),
        h('td', { class: 'n' }, `${fmt0(r.correct)} of ${fmt0(r.attempts)}`), h('td', { class: 'n' }, `${fmt(r.gen_tps, 1)} tok/s`, r.speed_clean ? null : ' ⚠', r.speed_session_id !== r.session_id ? h('div', { class: 'muted small' }, `session ${r.speed_session_id}`) : null), h('td', { class: 'n' }, fmtMs(r.ttft_ms)),
        h('td', { class: 'n' }, `${fmtCost(r.cost?.usd)} ${r.cost?.kind === 'electricity' ? '(electricity)' : r.cost?.kind === 'list' ? '(list price)' : ''}`),
        h('td', {}, statusBadge(r.status)))))))),
    h('div', { style: 'height:12px' }),
    notice(`Quality is the share of tasks answered correctly, averaged over ${rows[0].attempts ? Object.keys(rows[0].summary.tasks || {}).length : 0} tasks; the interval comes from resampling the tasks. Consistency is how often repeats of a task say the same thing. Speed comes from the latest speed runs of the configuration that passed their checks. Cost is the whole suite divided by the correct answers: API list price for Claude (these runs used a plan), and GPU electricity at the assumed price for local models, GPU board only.`));

  const taskIds = Object.keys(rows[0].summary.tasks || {});
  const categories = [...new Set(taskIds.map((t) => rows[0].summary.tasks[t].category))];
  const colHead = h('tr', {}, h('th', {}, ''), rows.map((r) => h('th', {}, h('a', { href: `#/quality/${r.config_id}` }, qualityName(r)), h('div', { class: 'muted small' }, isCloud(r.engine) ? 'cloud' : r.label))));
  root.append(h('div', { class: 'section-title' }, 'By category and by task'),
    card('Where each model does well', 'share of repeats answered correctly', h('div', { class: 'table-wrap' }, h('table', { class: 'matrix' },
      h('thead', {}, colHead),
      h('tbody', {},
        categories.map((c) => h('tr', {}, h('td', {}, h('strong', {}, CATEGORY_NAMES[c] || c)), rows.map((r) => heatCell(r.summary.categories?.[c]?.score, pct(r.summary.categories?.[c]?.score))))),
        h('tr', { class: 'group' }, h('td', { colspan: rows.length + 1 }, 'Each task')),
        taskIds.map((t) => h('tr', {}, h('td', {}, t), rows.map((r) => {
          const x = r.summary.tasks?.[t];
          return heatCell(x?.rate, x ? `${x.correct}/${x.n}` : 'N/A');
        }))))))));

  requestAnimationFrame(() => alignedPanels(chart, {
    rows: panelRows,
    legend,
    label: 'Quality, generation speed and cost per correct answer for each configuration',
    panels: [
      { title: 'Quality', unit: '% correct', max: 1, ticks: 2, tickFmt: (v) => `${Math.round(v * 100)}%`,
        read: ({ r }) => ({ v: r.score, lo: r.score_lo, hi: r.score_hi, text: pct(r.score), tip: `${pct(r.score)} (${pct(r.score_lo)} to ${pct(r.score_hi)})` }) },
      { title: 'Speed', unit: 'tok/s',
        read: ({ r }) => ({ v: r.gen_tps, text: fmt(r.gen_tps, 0) }) },
      { title: 'Cost per correct answer', unit: 'lower is better', log: true, tickFmt: (v) => `$${v >= 1 ? v : v.toPrecision(1)}`,
        read: ({ r }) => ({ v: r.cost?.usd, text: fmtCost(r.cost?.usd), tip: `${fmtCost(r.cost?.usd)} ${costFoot(r.cost)}` }) },
    ],
  }));
}

async function viewQualityDetail(root, match) {
  const id = Number(match[1]);
  const r = await api(`/api/quality/${id}`);
  const byTask = new Map();
  for (const a of r.answers) {
    if (!byTask.has(a.task_id)) byTask.set(a.task_id, []);
    byTask.get(a.task_id).push(a);
  }
  const failed = (r.checks || []).filter((x) => !x.passed);
  root.replaceChildren(...[
    h('div', { class: 'page-head' }, h('div', {},
      h('div', { class: 'row' }, h('h1', {}, qualityName(r)), engineBadge(r.engine), statusBadge(r.status)),
      h('p', {}, `${r.label} · session ${r.session_id} · suite ${r.suite} v${r.suite_version}. Every answer and how the code graded it.`)),
    h('a', { class: 'btn', href: '#/quality' }, 'All models')),
    h('div', { class: 'tiles' },
      tile({ label: 'Quality', value: pct(r.score), foot: `95% interval ${pct(r.score_lo)} to ${pct(r.score_hi)}` }),
      tile({ label: 'Correct answers', value: fmt0(r.correct), unit: `of ${fmt0(r.attempts)}` }),
      tile({ label: 'Consistency', value: pct(r.consistency), foot: 'repeats that agree' }),
      tile({ label: 'Cost per correct answer', value: fmtCost(r.cost?.usd), foot: costFoot(r.cost) }),
      tile({ label: 'Whole suite took', value: fmt0(r.seconds), unit: 's', foot: `${fmt0(r.gen_tokens)} tokens out` })),
    failed.length ? h('div', { class: 'stack', style: 'margin-top:12px' }, failed.map((x) => notice(`${x.check}: ${x.detail}`, 'warning'))) : null,
    h('div', { class: 'stack', style: 'margin-top:12px' }, [...byTask.entries()].map(([task, rows]) => {
      const correct = rows.filter((a) => a.passed).length;
      return h('details', { class: 'card more' },
        h('summary', {}, h('strong', {}, task), h('span', { class: 'muted' }, ` · ${CATEGORY_NAMES[rows[0].category] || rows[0].category} · ${correct} of ${rows.length} correct`)),
        h('div', { class: 'stack' },
          h('div', {}, h('div', { class: 'chart-title' }, 'Prompt'), h('pre', { class: 'log answer' }, r.prompts?.[task] || 'N/A')),
          rows.map((a) => h('div', { class: 'stack', style: 'gap:4px' },
            h('div', { class: 'task-attempt' }, h('span', { class: `status-${a.passed ? 'good' : 'critical'}` }, icon(a.passed ? 'good' : 'critical')),
              h('div', {}, h('strong', {}, `Repeat ${a.repeat + 1}`), ` · ${a.detail}`, h('span', { class: 'muted' }, ` · ${fmt0(a.gen_tokens)} tokens in ${fmtMs(a.total_ms)}`))),
            h('pre', { class: 'log answer' }, a.answer || '(no text)')))));
    }))].filter(Boolean));
}

// ------------------------------------------------------------ compare

async function viewCompare(root, _m, q) {
  const sessions = await api('/api/sessions');
  const all = sessions.flatMap((s) => s.configs.map((c) => ({ ...c, session: s })));
  if (all.length < 2) {
    root.replaceChildren(h('div', { class: 'empty' }, 'Compare needs at least two recorded configurations.'));
    return;
  }
  let a = Number(q.get('a')) || null;
  let b = Number(q.get('b')) || null;
  if (!a || !b) {
    // By default, the newest configuration measured twice: a reproducibility check.
    let pair = null;
    for (const newer of all) {
      const older = all.find((x) => x.id !== newer.id && x.config_hash === newer.config_hash && x.id < newer.id && x.status !== 'failed');
      if (older && newer.status !== 'failed' && (!a || older.id === a || newer.id === a)) { pair = [older, newer]; break; }
    }
    if (a && !pair) pair = [all.find((x) => x.id === a), all.find((x) => x.id !== a)];
    pair = pair || [all[1], all[0]];
    location.replace(`#/compare?a=${pair[0].id}&b=${pair[1].id}`);
    return;
  }
  const ca = all.find((x) => x.id === a);
  const cb = all.find((x) => x.id === b);
  if (!ca || !cb) throw new Error('one of the configurations no longer exists');

  const picker = (current, which) => h('label', { class: 'field' }, which === 'a' ? 'A' : 'B',
    h('select', { onchange: (e) => { location.hash = `#/compare?a=${which === 'a' ? e.target.value : a}&b=${which === 'b' ? e.target.value : b}`; } },
      sessions.map((s) => h('optgroup', { label: `Session ${s.id} · ${s.experiment} · ${fmtWhen(s.started_at)}` },
        s.configs.map((c) => h('option', { value: c.id, selected: c.id === current }, `Session ${s.id} · ${c.label} (${c.status})`))))));
  root.replaceChildren(
    h('div', { class: 'page-head' }, h('div', {}, h('h1', {}, 'Compare two configurations'),
      h('p', {}, 'Pick any two configurations from any sessions. The same configuration measured twice is a reproducibility check: generation and prompt speed must agree within 5%.'))),
    h('div', { class: 'row', style: 'margin-bottom:12px;align-items:flex-end' }, picker(a, 'a'),
      h('button', { class: 'btn', type: 'button', onclick: () => { location.hash = `#/compare?a=${b}&b=${a}`; } }, 'Swap'),
      picker(b, 'b')));

  const [da, db] = await Promise.all([api(`/api/sessions/${ca.session.id}`), ca.session.id === cb.session.id ? null : api(`/api/sessions/${cb.session.id}`)]);
  const fa = da.configs.find((c) => c.id === a);
  const fb = (db || da).configs.find((c) => c.id === b);
  const [ta, tb] = await Promise.all([fa, fb].map((c) => (isCloud(c.engine) ? null : api(`/api/configs/${c.id}/timeline`))));
  const name = (c, which) => `${which} · session ${c.session_id} · ${c.label}`;
  const same = fa.config_hash === fb.config_hash;
  const sa = fa.summary || {};
  const sb = fb.summary || {};
  const anyCloud = isCloud(fa.engine) || isCloud(fb.engine);
  const rows = [
    ['Generation, tok/s (median)', med(sa, 'gen_tps'), med(sb, 'gen_tps'), (v) => fmt(v, 2), true],
    ['Prompt, tok/s (median)', med(sa, 'prompt_tps'), med(sb, 'prompt_tps'), fmt0, true],
    ['First token (median)', med(sa, 'ttft_ms'), med(sb, 'ttft_ms'), fmtMs, false],
    ['Run-to-run CV of generation', sa.gen_tps?.cv_pct, sb.gen_tps?.cv_pct, (v) => `${fmt(v, 2)}%`, false],
    [anyCloud ? 'Inter-token or chunk gap p95, ms' : 'Inter-token p95, ms', sa.itl_pooled_ms?.p95, sb.itl_pooled_ms?.p95, (v) => fmt(v, 2), false],
    ['GPU power, W (median)', med(sa, 'gpu_power_instant_mean_w'), med(sb, 'gpu_power_instant_mean_w'), fmt0, false],
    ['Energy, J/token (median)', med(sa, 'j_per_token'), med(sb, 'j_per_token'), (v) => fmt(v, 3), false],
    ['Tokens out per run (median)', med(sa, 'gen_tokens'), med(sb, 'gen_tokens'), fmt0, false],
    ['Cost per run at list prices', med(sa, 'cost_usd'), med(sb, 'cost_usd'), fmtMoney, false],
  ].filter(([, va, vb]) => va != null || vb != null);
  let verdict = true;
  const body = rows.map(([label, va, vb, f, gate]) => {
    const d = pctDiff(vb, va);
    let mark = null;
    if (gate && va != null && vb != null) {
      const ok = d != null && Math.abs(d) <= 5;
      verdict = verdict && ok;
      mark = h('span', { class: `badge status-${ok ? 'good' : 'critical'}` }, icon(ok ? 'good' : 'critical'), ok ? 'within 5%' : 'over 5%');
    }
    return h('tr', {}, h('td', {}, label), h('td', { class: 'n' }, f(va)), h('td', { class: 'n' }, f(vb)), h('td', { class: 'n' }, signed(d)), h('td', {}, mark));
  });
  const warnings = [fa, fb].filter((c) => c.status !== 'ok');
  const qc = await api(`/api/quality/compare?a=${a}&b=${b}`);
  const d = qc?.difference;
  const qualityNote = qc ? notice(`Answer quality: ${pct(qc.a.score)} (A) and ${pct(qc.b.score)} (B)${d
    ? `. B vs A is ${points(d.diff)} points, 95% interval ${points(d.lo)} to ${points(d.hi)}: ${d.verdict}.`
    : ', from different versions of the suite, so they don’t compare.'}`, d?.verdict === 'worse' ? 'warning' : 'neutral') : null;
  root.append(
    h('div', { class: 'stack' },
      notice(same
        ? `Same configuration measured twice, so this is a reproducibility check. Verdict: ${verdict ? 'they agree within 5%' : 'they differ by more than 5%'}.`
        : 'Different configurations, so differences are expected; the table shows how large they are.', same ? (verdict ? 'good' : 'warning') : 'neutral'),
      warnings.length ? notice(`${warnings.map((c) => `Session ${c.session_id} (${c.label})`).join(' and ')} measured with warnings or failures, so a difference may come from the measurement, not the model.`, 'warning') : null,
      qualityNote,
      card(null, null, h('div', { class: 'table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, h('th', {}, 'Measure'),
          h('th', { class: 'n' }, h('span', { class: 'legend', style: 'margin:0;justify-content:flex-end' }, h('span', {}, h('i', { style: 'background:var(--c-a)' }), name(fa, 'A')))),
          h('th', { class: 'n' }, h('span', { class: 'legend', style: 'margin:0;justify-content:flex-end' }, h('span', {}, h('i', { style: 'background:var(--c-b)' }), name(fb, 'B')))),
          h('th', { class: 'n' }, 'B vs A'), h('th', {}, ''))),
        h('tbody', {}, body))))));

  const strip = h('div');
  const ttft = h('div');
  root.append(h('div', { class: 'grid cols-2', style: 'margin-top:12px' },
    card('Generation speed per run', 'hollow: warm-up · ×: rejected', strip),
    card('First token per run', null, ttft)));
  const groups = (key) => [[fa, 'A', 'var(--c-a)'], [fb, 'B', 'var(--c-b)']].map(([c, w, color]) => {
    const vals = c.trials.filter((t) => t.kind === 'measured' && t.valid && t[key] != null).map((t) => t[key]).sort((x, y) => x - y);
    const m = vals.length ? (vals.length % 2 ? vals[(vals.length - 1) / 2] : (vals[vals.length / 2 - 1] + vals[vals.length / 2]) / 2) : null;
    return {
      name: `${w} · S${c.session_id}`, color, median: m,
      points: c.trials.map((t) => ({ v: t[key], hollow: t.kind === 'warmup', invalid: !t.valid, title: `${name(c, w)} · ${trialName(t)}` })),
    };
  });
  stripPlot(strip, { groups: groups('gen_tps'), fmt: (v) => fmt(v, 1), unitLabel: 'tokens/s', legend: true, label: 'Generation speed per run, A and B' });
  stripPlot(ttft, { groups: groups('ttft_ms'), fmt: fmtMs, unitLabel: 'first token', legend: true, label: 'First token per run, A and B' });

  if (ta?.samples.length && tb?.samples.length) {
    const shift = (tl) => tl.first_measured_s || 0;
    const series = (tl, key, nm, color) => ({ name: nm, color, points: tl.samples.map((p) => [p.t - shift(tl), p[key]]) });
    const end = Math.max(...[ta, tb].map((tl) => Math.max(...tl.trials.filter((t) => t.t_last != null).map((t) => t.t_last - shift(tl)))));
    const clock = h('div');
    const power = h('div');
    root.append(h('div', { class: 'grid cols-2', style: 'margin-top:12px' },
      card('GPU clock', 'MHz, both aligned at their first measured run', clock),
      card('GPU power', 'W, both aligned at their first measured run', power)));
    const opts = { xDomain: [-2, end + 1], xFmt: (v) => fmt(v, 0), xUnit: 's after the first measured run starts' };
    lineChart(clock, { ...opts, series: [series(ta, 'clock', name(fa, 'A'), 'var(--c-a)'), series(tb, 'clock', name(fb, 'B'), 'var(--c-b)')], yFmt: fmtAxis, valueFmt: (v) => `${fmt0(v)} MHz`, label: 'GPU clock, A and B' });
    lineChart(power, { ...opts, series: [series(ta, 'power', name(fa, 'A'), 'var(--c-a)'), series(tb, 'power', name(fb, 'B'), 'var(--c-b)')], valueFmt: (v) => `${fmt0(v)} W`, label: 'GPU power, A and B' });
  }
}

// ------------------------------------------------------------ run (live)

const WINDOW_S = 120;

async function viewRun(root) {
  const [exps, status] = await Promise.all([api('/api/experiments'), api('/api/status')]);
  const demo = !!status.demo; // recorded results only: the server refuses to start runs
  const usable = exps.filter((e) => !e.error);
  let selected = store('llmlab-experiment');
  if (!exps.some((e) => e.file === selected)) selected = usable[0]?.file ?? exps[0]?.file;
  let liveCharts = store('llmlab-live-charts') !== '0';
  let st = freshState();

  function freshState() {
    return { since: 0, samplesPath: null, samplesOffset: 0, samples: [], trials: [], variants: [], done: [], logs: [], quality: null, attemptsDone: 0, session: null, variant: null, phase: null, loadStart: null, loadEnd: null, t0: null, busy: null, state: 'idle', exitCode: null, output: null, stopping: false };
  }

  // --- rail
  let followRun = true; // the rail shows the current or last run's experiment until you pick another
  const select = h('select', { 'aria-label': 'Experiment file', onchange: (e) => { selected = e.target.value; followRun = false; store('llmlab-experiment', selected); renderRail(); } },
    exps.map((e) => h('option', { value: e.file, selected: e.file === selected },
      e.error ? `${e.file} (invalid)` : e.file.replace(/\.ya?ml$/, '') === e.name ? e.file : `${e.name} · ${e.file}`)));
  const railInfo = h('div', { class: 'stack' });
  const startBtn = h('button', { class: 'btn primary', type: 'button', onclick: start }, 'Start');
  const stopBtn = h('button', { class: 'btn danger', type: 'button', onclick: stop }, 'Stop');
  const liveToggle = h('input', { type: 'checkbox', checked: liveCharts, onchange: (e) => { liveCharts = e.target.checked; store('llmlab-live-charts', liveCharts ? '1' : '0'); render(); } });
  const railError = h('div');
  const busyToggle = h('input', { type: 'checkbox' });
  const rail = h('aside', { class: 'rail' },
    demo ? notice('This is the demo: it shows results recorded on an RTX 3060 Ti, so starting runs is off. A run needs llama.cpp and a model on your machine; docs/setup.md explains the setup.', 'warning') : null,
    h('section', { class: 'card stack' }, h('h2', {}, 'Experiment'), select, railInfo,
      h('div', { class: 'row' }, startBtn, stopBtn), railError,
      h('label', { class: 'switch' }, liveToggle, h('span', { class: 'track' }), 'Live charts'),
      h('label', { class: 'switch' }, busyToggle, h('span', { class: 'track' }), 'Run even if the GPU is busy'),
      h('p', { class: 'muted small' }, 'Drawing live charts takes about 0.3% of the CPU, which the “CPU quiet” check would catch if it grew. Turn them off for the cleanest numbers. A run on a busy GPU is recorded with a failed idle check.')),
    notice('A run started here is the same as “python -m llmlab run” in a terminal: same idle check, same trust checks, same database. Runs started from a terminal show up here too.'));

  function renderRail() {
    const e = exps.find((x) => x.file === selected);
    railInfo.replaceChildren();
    if (!e) return;
    if (e.error) {
      railInfo.append(notice(e.error, 'critical'));
      return;
    }
    railInfo.append(h('div', { class: 'stack' },
      e.description ? h('p', { class: 'ink-2' }, e.description) : null,
      h('ul', { class: 'variant-list' }, e.variants.map((v) => h('li', {},
        h('span', {}, isCloud(v.engine) ? prettyModel(v.model) : [v.label, h('span', { class: 'muted' }, ` · ${prettyModel(v.model)}`)]),
        engineBadge(v.engine)))),
      h('div', { class: 'muted small' }, `${e.variants.length} configuration(s), ${e.requests} requests in all (warm-ups included)`),
      e.variants.some((v) => v.model_found === false) ? notice('A model file this experiment needs is missing.', 'warning') : null,
      h('details', { class: 'more' }, h('summary', {}, `Prompt (${fmt0(e.prompt_chars)} characters)`), h('pre', {}, e.prompt_preview + (e.prompt_chars > e.prompt_preview.length ? ' …' : '')))));
  }

  // --- main
  const statusBadgeHost = h('span');
  const what = h('div', { class: 'what' });
  const meta = h('div', { class: 'muted' });
  const sessionLink = h('a', { class: 'btn', hidden: true }, 'Open session');
  const bar = h('div', { style: 'width:0%' });
  const statusExtra = h('div');
  const statusCard = h('section', { class: 'card stack' },
    h('div', { class: 'status-line' }, statusBadgeHost, what, h('span', { class: 'spacer' }), sessionLink),
    meta, h('div', { class: 'progress', role: 'progressbar', 'aria-label': 'Requests completed' }, bar), statusExtra);
  const tilesHost = h('div', { class: 'tiles' });
  const charts = { clock: h('div'), power: h('div'), util: h('div'), cores: h('div') };
  const nows = { clock: h('span'), power: h('span'), util: h('span'), cores: h('span') };
  const chartsCard = h('div', { class: 'grid cols-2' },
    card(null, null, chartBox('GPU clock', 'MHz · shaded: load, W = warm-up, 1, 2, … = runs', charts.clock, nows.clock)),
    card(null, null, chartBox('GPU power', 'watts · shaded: load, W = warm-up, 1, 2, … = runs', charts.power, nows.power)),
    card(null, null, chartBox('GPU utilization', '% · shaded: load, W = warm-up, 1, 2, … = runs', charts.util, nows.util)),
    card(null, null, chartBox('CPU per logical core', '%', charts.cores, nows.cores)));
  const chartsOff = notice('Live charts are off. Telemetry is still recorded and appears in the session’s page when each configuration finishes.');
  const cloudNote = notice('This configuration runs in Anthropic’s cloud, so there is no local GPU telemetry. Each request’s speed and first token appear in the runs table as they finish.');
  const endedNote = h('div', { hidden: true });
  const runsBody = h('tbody');
  const doneHost = h('div', { class: 'stack' });
  const logPre = h('pre', { class: 'log' });
  const main = h('section', { class: 'stack' }, statusCard, tilesHost, chartsCard, chartsOff, cloudNote, endedNote,
    card('Finished configurations', null, doneHost),
    card('Runs so far', 'newest first', h('div', { class: 'table-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, 'Configuration'), h('th', {}, 'Run'), h('th', { class: 'n' }, 'gen tok/s'), h('th', { class: 'n' }, 'prompt tok/s'), h('th', { class: 'n' }, 'first token'), h('th', { class: 'n' }, 'tokens out'))),
      runsBody))),
    h('details', { class: 'card more' }, h('summary', {}, 'Output'), logPre));

  root.replaceChildren(h('div', { class: 'run-layout' }, rail, main));
  renderRail();

  function apply(e) {
    switch (e.type) {
      case 'start': { const keep = st; st = freshState(); st.since = keep.since; st.t0 = e.time; break; }
      case 'session': st.session = e; break;
      case 'variant': st.variant = e; st.variants.push(e); st.phase = { name: 'starting' }; st.loadStart = null; st.loadEnd = null; break;
      case 'phase': st.phase = e; if (e.name === 'load') st.loadStart = e.time; break;
      case 'loaded': st.loadEnd = e.time; st.loaded = e; st.phase = { name: 'runs' }; break;
      case 'trial': st.trials.push(e); st.phase = { name: 'runs' }; break;
      case 'quality_start': st.quality = { total: e.total, done: 0, correct: 0 }; break;
      case 'attempt':
        st.attemptsDone += 1;
        if (st.quality) { st.quality.done = e.done; st.quality.correct += e.passed; }
        break;
      case 'variant_done': st.done.push(e); break;
      case 'busy': st.busy = e.message; break;
      case 'log': st.logs.push(e.text); break;
      default: break;
    }
  }

  async function poll() {
    try {
      const qs = new URLSearchParams({ since: st.since, samples: liveCharts ? '1' : '0', samples_offset: st.samplesOffset });
      if (st.samplesPath) qs.set('samples_path', st.samplesPath);
      const r = await api(`/api/live?${qs}`);
      if (r.next < st.since) {
        st = freshState();
        return poll();
      }
      for (const e of r.events) apply(e);
      st.since = r.next;
      if (r.samples_path !== st.samplesPath) st.samplesOffset = 0;
      st.samplesPath = r.samples_path;
      st.samplesOffset = r.samples_offset;
      if (r.samples.length) {
        st.samples.push(...r.samples);
        if (st.samples.length > 1500) st.samples.splice(0, st.samples.length - 1500);
      }
      const next = { state: r.state, exitCode: r.exit_code ?? null, output: r.output ?? null, stopping: !!r.stopping };
      // Redraw only when something changed: drawing costs CPU, which the run being watched competes for.
      const changed = r.events.length || r.samples.length || Object.keys(next).some((k) => next[k] !== st[k]);
      Object.assign(st, next);
      if (changed) render();
    } catch (e) {
      what.textContent = `Lost contact with the dashboard server: ${e.message}`;
    }
  }

  async function start() {
    railError.replaceChildren();
    startBtn.disabled = true;
    try {
      await post('/api/runs', { file: selected, allow_busy: busyToggle.checked });
      st = freshState();
      st.state = 'starting';
      render();
      poll();
    } catch (e) {
      railError.replaceChildren(notice(e.message, 'critical'));
      startBtn.disabled = false;
    }
  }

  async function stop() {
    stopBtn.disabled = true;
    try {
      await post('/api/runs/stop', {});
      st.stopping = true;
      render();
    } catch (e) {
      railError.replaceChildren(notice(e.message, 'critical'));
    }
  }

  function phaseText() {
    if (st.stopping) return 'stopping after the current request…';
    const v = st.variant;
    if (!v) return 'waiting for the GPU idle check';
    const mine = st.trials.filter((t) => t.index === v.index);
    const p = st.phase?.name;
    if (p === 'quiet') return 'checking the GPU stays quiet after the runs';
    if (p === 'quality' && st.quality) return `answering the quality tasks: ${st.quality.done} of ${st.quality.total} done, ${st.quality.correct} correct`;
    if (p === 'quality') return 'answering the quality tasks';
    if (!isCloud(v.engine) && (p === 'load' || p === 'starting')) return 'loading the model';
    const warm = mine.filter((t) => t.kind === 'warmup').length;
    const runs = mine.filter((t) => t.kind === 'measured').length;
    if (warm < v.warmup) return `warm-up ${warm + 1} of ${v.warmup}`;
    return `run ${Math.min(runs + 1, v.repeats)} of ${v.repeats}`;
  }

  function render() {
    const active = st.state === 'running' || st.state === 'starting';
    startBtn.disabled = demo || active || !exps.find((x) => x.file === selected && !x.error);
    stopBtn.disabled = !active || st.stopping;
    select.disabled = active;
    // A page opened mid-run, or after a run, shows that run's experiment in the rail.
    const info = exps.find((x) => st.session?.file && st.session.file.replace(/\\/g, '/').endsWith(`/${x.file}`));
    if ((active || followRun) && info && info.file !== selected) {
      selected = info.file;
      select.value = selected;
      renderRail();
    }

    // status
    let tone = 'neutral';
    let label = 'Ready';
    let headline = 'No run in progress';
    let line = demo ? 'Runs are off in the demo. The recorded results are under Overview, History, Quality and Compare.' : 'Pick an experiment and press Start.';
    const exp = st.session?.experiment;
    statusExtra.replaceChildren();
    if (st.state === 'starting') {
      label = 'Starting'; headline = 'Starting the run'; line = 'Checking the experiment file and the GPU…';
    } else if (st.state === 'running') {
      label = 'Running'; headline = exp || 'Running';
      line = st.variant ? `Configuration ${st.variant.index + 1} of ${st.variant.total}: ${st.variant.label} · ${phaseText()}` : phaseText();
    } else if (st.state === 'ended') {
      const code = st.exitCode;
      headline = exp || 'Last run';
      if (code === 0 && st.done.some((d) => d.status !== 'ok')) { tone = 'warning'; label = 'Finished'; line = 'Finished with warnings; see the finished configurations below.'; }
      else if (code === 0) { tone = 'good'; label = 'Finished'; line = 'Every configuration passed its checks.'; }
      else if (code === 1) { tone = 'warning'; label = 'Finished'; line = 'At least one configuration failed; see below.'; }
      else if (code === 3) { tone = 'warning'; label = 'GPU busy'; line = 'Nothing was recorded. Close what is using the GPU, or tick “Run even if the GPU is busy”.'; }
      else if (code === 130) { tone = 'warning'; label = 'Stopped'; line = 'The session is recorded as aborted.'; }
      else if (code === 2) { tone = 'critical'; label = 'Not started'; line = 'The experiment file has a problem.'; }
      else { tone = 'critical'; label = 'Ended unexpectedly'; line = 'The run process exited without finishing.'; }
      if (st.busy) statusExtra.append(notice(st.busy, 'warning'));
      if (st.output && code !== 0 && code !== 1) statusExtra.append(h('pre', { class: 'log' }, st.output));
    }
    statusBadgeHost.replaceChildren(st.state === 'running'
      ? h('span', { class: 'badge' }, h('span', { class: 'live-dot' }), label)
      : h('span', { class: `badge status-${tone}` }, icon(tone), label));
    what.textContent = headline;
    meta.textContent = line;
    if (st.session) {
      sessionLink.hidden = false;
      sessionLink.textContent = `Open session ${st.session.session_id}`;
      sessionLink.href = `#/session/${st.session.session_id}`;
    } else sessionLink.hidden = true;
    const total = info?.requests || 0;
    bar.style.width = total ? `${Math.min(100, ((st.trials.length + st.attemptsDone) / total) * 100)}%` : st.state === 'ended' ? '100%' : '0%';

    // tiles
    const last = st.trials[st.trials.length - 1];
    const sample = st.samples[st.samples.length - 1];
    const recent = st.samples.slice(-60);
    const trialSpark = st.trials.slice(-12).map((t) => t.gen_tps);
    const tel = (k) => (liveCharts && sample ? sample[k] : null);
    const cloudNow = !!st.variant && isCloud(st.variant.engine);
    const telFoot = liveCharts ? (sample ? 'now' : 'no telemetry yet') : 'live charts off';
    // Live telemetry exists while a run is in progress, or if this page watched it: the run
    // deletes its telemetry files when it ends, and the session page has it all from then on.
    const telemetry = !cloudNow && (active || st.samples.length > 0);
    const common = [
      tile({ label: 'Generation, last run', color: 'var(--m-gen)', value: fmt(last?.gen_tps, 1), unit: 'tok/s', foot: last ? trialName(last) : 'no runs yet', spark: { values: trialSpark } }),
      tile({ label: 'First token, last run', color: 'var(--m-ttft)', value: fmtMs(last?.ttft_ms), foot: last?.prompt_tokens ? `${fmt0(last.prompt_tokens)} prompt tokens` : '' }),
    ];
    if (cloudNow) {
      tilesHost.replaceChildren(...common,
        tile({ label: 'Tokens out, last run', value: fmt0(last?.gen_tokens), foot: 'Claude decides the length' }),
        tile({ label: 'Hidden thinking, last run', value: fmt0(last?.thinking_tokens), unit: 'tokens', foot: 'included in tokens out' }),
        tile({ label: 'First content, last run', value: fmtMs(last?.first_content_ms), foot: 'thinking included' }));
    } else if (!telemetry) tilesHost.replaceChildren(...common);
    else tilesHost.replaceChildren(...common,
      tile({ label: 'GPU utilization', color: 'var(--m-load)', value: fmt0(tel('util')), unit: '%', foot: telFoot, spark: liveCharts ? { values: recent.map((s) => s.util) } : null }),
      tile({ label: 'GPU clock', color: 'var(--m-clock)', value: fmt0(tel('clock')), unit: 'MHz', foot: telFoot, spark: liveCharts ? { values: recent.map((s) => s.clock) } : null }),
      tile({ label: 'GPU power', color: 'var(--m-power)', value: fmt0(tel('power')), unit: 'W', foot: telFoot, spark: liveCharts ? { values: recent.map((s) => s.power) } : null }),
      tile({ label: 'VRAM used', color: 'var(--m-vram)', value: fmt0(tel('vram')), unit: 'MiB', foot: telFoot, spark: liveCharts ? { values: recent.map((s) => s.vram) } : null }));

    // charts
    chartsCard.hidden = !liveCharts || !telemetry;
    cloudNote.hidden = !cloudNow;
    chartsOff.hidden = liveCharts || !telemetry;
    endedNote.hidden = !(st.state === 'ended' && !cloudNow && !telemetry && st.session);
    if (!endedNote.hidden) {
      endedNote.replaceChildren(notice(['Live telemetry shows while a run is in progress. This run’s full telemetry, for every configuration, is in ',
        h('a', { href: `#/session/${st.session.session_id}` }, `session ${st.session.session_id}`), '.']));
    }
    if (liveCharts && telemetry) {
      const t0 = st.t0 ?? sample?.t ?? 0;
      const tEnd = sample ? sample.t - t0 : 0;
      const domain = [Math.max(0, tEnd - WINDOW_S), Math.max(tEnd, 10)];
      const bands = [];
      if (st.loadStart) bands.push({ x0: st.loadStart - t0, x1: (st.loadEnd ?? sample?.t ?? st.loadStart) - t0, kind: 'load', label: 'load', title: 'loading the model' });
      for (const t of st.trials) {
        if (t.t_send && t.t_last) bands.push({ x0: t.t_send - t0, x1: t.t_last - t0, kind: t.kind, label: t.kind === 'warmup' ? 'W' : String(t.idx + 1), title: `${trialName(t)} of configuration ${t.index + 1}` });
      }
      const shown = st.samples.filter((s) => s.t - t0 >= domain[0]);
      const series = (k, name, color) => [{ name, color, points: shown.map((s) => [s.t - t0, s[k]]) }];
      const common = { bands, xDomain: domain, xFmt: (v) => fmt(v, 0), xUnit: 's since the run started', endDot: true, height: 150, empty: 'Waiting for telemetry' };
      lineChart(charts.clock, { ...common, series: series('clock', 'GPU clock', 'var(--m-clock)'), yMax: 2200, yFmt: fmtAxis, valueFmt: (v) => `${fmt0(v)} MHz`, label: 'Live GPU clock' });
      lineChart(charts.power, { ...common, series: series('power', 'GPU power', 'var(--m-power)'), valueFmt: (v) => `${fmt0(v)} W`, label: 'Live GPU power' });
      lineChart(charts.util, { ...common, series: series('util', 'GPU utilization', 'var(--m-load)'), yMax: 100, valueFmt: (v) => `${fmt0(v)}%`, label: 'Live GPU utilization' });
      coreBars(charts.cores, sample?.cores || []);
      nows.clock.textContent = sample ? `${fmt0(sample.clock)} MHz` : '';
      nows.power.textContent = sample ? `${fmt0(sample.power)} W` : '';
      nows.util.textContent = sample ? `${fmt0(sample.util)}%` : '';
      nows.cores.textContent = !sample ? ''
        : sample.server_cpu != null ? `${fmt0(sample.cpu)}% overall · server ${fmt0(sample.server_cpu)}% of one core` : `${fmt0(sample.cpu)}% overall`;
    }

    // tables
    const labels = Object.fromEntries(st.variants.map((v) => [v.index, v.label]));
    runsBody.replaceChildren(...[...st.trials].reverse().map((t) => h('tr', { class: t.kind === 'warmup' ? 'dim' : null },
      h('td', {}, labels[t.index] ?? t.index + 1), h('td', {}, trialName(t)), h('td', { class: 'n' }, fmt(t.gen_tps, 1)),
      h('td', { class: 'n' }, fmt0(t.prompt_tps)), h('td', { class: 'n' }, fmtMs(t.ttft_ms)), h('td', { class: 'n' }, fmt0(t.gen_tokens)))));
    if (!st.trials.length) runsBody.append(h('tr', {}, h('td', { colspan: 6, class: 'muted' }, 'No runs yet.')));
    doneHost.replaceChildren(...(st.done.length ? st.done.map((d) => {
      const failed = (d.checks || []).filter((c) => c.passed === false);
      return h('div', { class: 'stack', style: 'gap:6px' },
        h('div', { class: 'row' }, h('strong', {}, d.label), statusBadge(d.status), h('span', { class: 'spacer' }),
          d.config_id && st.session ? h('a', { href: `#/session/${st.session.session_id}` }, 'Details') : null),
        h('div', { class: 'ink-2' }, d.error ? d.error : `${fmt(d.gen_tps, 1)} tok/s median · first token ${fmtMs(d.ttft_ms)} · CV ${fmt(d.cv_pct, 2)}% · ${d.runs} runs`),
        d.quality ? h('div', { class: 'ink-2' }, d.quality.score == null ? `Quality: ${d.quality.error || 'not scored'}` : `Quality ${pct(d.quality.score)} · ${d.quality.correct} of ${d.quality.attempts} answers correct · consistency ${pct(d.quality.consistency)}`) : null,
        failed.map((c) => h('div', { class: 'small muted' }, `✕ ${c.check}: ${c.detail}`)),
        (d.invalid || []).map((x) => h('div', { class: 'small muted' }, `✕ ${x.run} rejected: ${x.problems.join('; ')}`)));
    }) : [h('div', { class: 'muted' }, 'None yet.')]));
    const atBottom = logPre.scrollTop + logPre.clientHeight >= logPre.scrollHeight - 4;
    logPre.textContent = st.logs.slice(-300).join('\n') || 'No output yet.';
    if (atBottom) logPre.scrollTop = logPre.scrollHeight;
  }

  render();
  await poll();
  const timer = setInterval(poll, 1000);
  return () => clearInterval(timer);
}

route();
