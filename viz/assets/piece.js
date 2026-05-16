/* Educational strategy page. The piece's index.html sets
 *   window.STRATEGY_ID = "01-dual-ema-momentum";
 * and includes chart.js then this file. Everything else — copy, tabs,
 * charts, metrics, the per-regime interpretation — is built from the JSON
 * that viz/build.py emitted, so the page can never drift from the engine. */

(function () {
  const ID = window.STRATEGY_ID;
  const app = document.getElementById('app');
  let DATA = null;
  let currentDs = null;

  // theme (shared key with the gallery)
  const saved = localStorage.getItem('theme');
  if (saved) document.documentElement.dataset.theme = saved;
  const toggle = document.getElementById('theme-toggle');
  if (toggle) toggle.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('theme', next);
    if (currentDs) renderDataset(currentDs); // re-theme canvases
  });

  let rz;
  window.addEventListener('resize', () => {
    clearTimeout(rz);
    rz = setTimeout(() => { if (currentDs) renderDataset(currentDs); }, 150);
  });

  function fail(msg, detail) {
    app.innerHTML = `<div class="err"><strong>${msg}</strong>
      <code>${detail || ''}</code>
      <p>Serve the gallery from a local server:</p>
      <code>cd viz &amp;&amp; python3 -m http.server</code></div>`;
  }

  if (window.location.protocol === 'file:') {
    return fail('Local file restriction',
      'Browsers block fetch() on file:// URLs — the JSON cannot load.');
  }

  fetch(`../../data/${ID}.json`)
    .then(r => { if (!r.ok) throw new Error(r.status + ' ' + r.statusText); return r.json(); })
    .then(d => { DATA = d; build(); })
    .catch(e => fail('Could not load strategy data', e.message +
      ' — has `python3 viz/build.py` been run?'));

  const pct = v => (v * 100).toFixed(1) + '%';
  const sgn = v => v > 0 ? 'pos' : (v < 0 ? 'neg' : '');

  function metricTile(label, value, cls) {
    return `<div class="metric"><div class="label">${label}</div>
      <div class="value ${cls || ''}">${value}</div></div>`;
  }

  function build() {
    document.title = DATA.title + ' — Strategy Lab';
    const dsNames = Object.keys(DATA.datasets);

    app.innerHTML = `
      <a class="back" href="../../index.html">← Strategy Lab</a>
      <h1 style="font-weight:400;letter-spacing:.03em;margin:.6rem 0 0">${DATA.title}</h1>
      <p class="lede">${DATA.idea}</p>

      <h2 class="section">The signal</h2>
      <p><span class="rule">${DATA.signal_rule}</span></p>
      <p style="color:var(--muted);font-size:.85rem">
        Fixed published parameters: ${Object.entries(DATA.params)
          .map(([k, v]) => `<code>${k}=${v}</code>`).join(', ')} ·
        source: <code>${DATA.source}</code> ·
        engine fill model: signal at close[t] executed at open[t+1].
      </p>

      <h2 class="section">Behaviour across market regimes</h2>
      <p style="color:var(--muted);font-size:.88rem">The same rule, unchanged,
        run on four synthetic worlds. Switch tabs to see where the edge holds
        and where it breaks — that contrast <em>is</em> the lesson.</p>
      <div class="tabs" id="tabs"></div>
      <div id="panel"></div>`;

    const tabs = document.getElementById('tabs');
    dsNames.forEach(ds => {
      const b = document.createElement('button');
      b.className = 'tab';
      b.textContent = ds;
      b.setAttribute('role', 'tab');
      b.onclick = () => { selectTab(ds); renderDataset(ds); };
      tabs.appendChild(b);
    });
    selectTab(dsNames[0]);
    renderDataset(dsNames[0]);
  }

  function selectTab(ds) {
    document.querySelectorAll('#tabs .tab').forEach(t =>
      t.setAttribute('aria-selected', String(t.textContent === ds)));
  }

  function chartCard(id, title) {
    return `<div class="chart-card"><p class="chart-title">${title}</p>
      <canvas class="chart" id="${id}"></canvas>
      <div class="legend" id="${id}-lg"></div></div>`;
  }

  function lg(items) {
    return items.map(it => it.dash
      ? `<span><i class="swatch dash" style="border-top-color:${it.c}"></i>${it.t}</span>`
      : it.box
        ? `<span><i class="swatch box" style="background:${it.c}"></i>${it.t}</span>`
        : `<span><i class="swatch" style="background:${it.c}"></i>${it.t}</span>`
    ).join('');
  }

  function renderDataset(ds) {
    currentDs = ds;
    const D = DATA.datasets[ds];
    const m = D.metrics;
    const wf = m.walk_forward;
    const n = D.close.length;

    const wins = D.trades.filter(t => t.ret > 0).length;
    const avg = D.trades.length
      ? D.trades.reduce((a, t) => a + t.ret, 0) / D.trades.length : 0;

    const ov = cssNames();
    const overlayKeys = Object.keys(D.price_overlays);
    const hasInd = D.indicator != null;

    const panel = document.getElementById('panel');
    panel.innerHTML = `
      <div class="interp">
        <div class="regime"><b>${ds}</b> — ${D.regime}
          <span class="verdict-pill v-${D.verdict}">${D.verdict}</span></div>
        ${D.interpretation}
      </div>
      ${chartCard('c-price', 'Price, signal &amp; trades — ' + ds)}
      ${hasInd ? chartCard('c-ind', D.indicator.name + ' — the rule\'s view') : ''}
      ${chartCard('c-eq', 'Strategy equity vs buy &amp; hold (start = ×1.00)')}
      <div class="metrics">
        ${metricTile('Sharpe (in-sample)', m.sharpe.toFixed(2), sgn(m.sharpe))}
        ${metricTile('CAGR', pct(m.cagr), sgn(m.cagr))}
        ${metricTile('Max drawdown', pct(m.max_drawdown), 'neg')}
        ${metricTile('Sortino', m.sortino.toFixed(2), sgn(m.sortino))}
        ${metricTile('Hit rate', pct(m.hit_rate))}
        ${metricTile('Exposure', pct(m.exposure))}
        ${metricTile('Walk-fwd OOS Sharpe', wf.oos_sharpe_mean.toFixed(2), sgn(wf.oos_sharpe_mean))}
        ${metricTile('OOS folds profitable', pct(wf.oos_consistency),
          wf.oos_consistency >= 0.6 ? 'pos' : 'neg')}
      </div>
      <p style="color:var(--muted);font-size:.85rem">
        ${D.trades.length} trades · ${D.trades.length ? Math.round(100 * wins / D.trades.length) : 0}% won ·
        avg trade ${(avg * 100).toFixed(2)}% ·
        deflated Sharpe ${m.deflated_sharpe.toFixed(2)} (1 = edge survives multiple-testing).
        <br/>In-sample Sharpe is the headline; the two walk-forward tiles are the
        honest out-of-sample test — compare them.
      </p>`;

    // price chart
    const priceSeries = [{ data: D.close, color: ov.price, width: 1.5, label: 'Close' }];
    const ovColors = [ov.ov1, ov.ov2];
    overlayKeys.forEach((k, i) =>
      priceSeries.push({ data: D.price_overlays[k], color: ovColors[i % 2], width: 1.3 }));
    const markers = [];
    D.trades.forEach(t => {
      markers.push({ i: t.entry_i, y: D.close[t.entry_i], type: 'entry' });
      markers.push({ i: t.exit_i, y: D.close[t.exit_i], type: 'exit' });
    });
    renderChart(document.getElementById('c-price'), {
      height: 240, xCount: n, dates: D.dates, series: priceSeries,
      shade: D.positions, markers, yfmt: v => v.toFixed(0),
    });
    document.getElementById('c-price-lg').innerHTML = lg([
      { c: ov.price, t: 'Close' },
      ...overlayKeys.map((k, i) => ({ c: ovColors[i % 2], t: k })),
      { c: ov.shadeSolid, box: true, t: 'In long position' },
      { c: ov.pos, t: '▲ entry' }, { c: ov.neg, t: '▼ exit' },
    ]);

    // indicator panel
    if (hasInd) {
      const I = D.indicator;
      renderChart(document.getElementById('c-ind'), {
        height: 150, xCount: n, dates: D.dates,
        yMin: I.min, yMax: I.max,
        series: [{ data: I.values, color: ov.ov1, width: 1.4 }],
        shade: D.positions,
        hbands: I.bands.map((b, i) => ({
          y: b.y, label: b.label, color: i === 0 ? ov.pos : ov.neg,
        })),
        yfmt: v => v.toFixed(I.max <= 2 ? 2 : 0),
      });
      document.getElementById('c-ind-lg').innerHTML = lg([
        { c: ov.ov1, t: I.name }, { c: ov.shadeSolid, box: true, t: 'In long position' },
      ]);
    }

    // equity chart
    renderChart(document.getElementById('c-eq'), {
      height: 200, xCount: n, dates: D.dates,
      series: [
        { data: D.buy_hold, color: ov.bench, width: 1.4, dash: [5, 4] },
        { data: D.equity, color: ov.equity, width: 2 },
      ],
      yfmt: v => '×' + v.toFixed(2),
    });
    document.getElementById('c-eq-lg').innerHTML = lg([
      { c: ov.equity, t: 'Strategy equity (net of costs)' },
      { c: ov.bench, dash: true, t: 'Buy &amp; hold' },
    ]);
  }

  function cssNames() {
    return {
      price: cssVar('--price'), equity: cssVar('--equity'), bench: cssVar('--bench'),
      ov1: cssVar('--ov1'), ov2: cssVar('--ov2'),
      pos: cssVar('--pos'), neg: cssVar('--neg'),
      shadeSolid: cssVar('--pos'),
    };
  }
})();
