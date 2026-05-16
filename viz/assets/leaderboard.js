/* The leaderboard dashboard. Loads leaderboard.json plus every strategy's
 * full JSON (for the equity small-multiples). Its single teaching goal: put
 * the in-sample Sharpe columns next to the walk-forward columns so the reader
 * sees green-on-the-left fade to red-on-the-right. */

(function () {
  const app = document.getElementById('app');

  const saved = localStorage.getItem('theme');
  if (saved) document.documentElement.dataset.theme = saved;
  let LB = null, STRATS = {};
  const toggle = document.getElementById('theme-toggle');
  if (toggle) toggle.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('theme', next);
    if (LB) render();
  });
  let rz;
  window.addEventListener('resize', () => {
    clearTimeout(rz);
    rz = setTimeout(() => { if (LB) render(); }, 150);
  });

  function fail(msg, detail) {
    app.innerHTML = `<div class="err"><strong>${msg}</strong><code>${detail || ''}</code>
      <p>Serve from a local server:</p><code>cd viz &amp;&amp; python3 -m http.server</code></div>`;
  }
  if (window.location.protocol === 'file:') {
    return fail('Local file restriction',
      'Browsers block fetch() on file:// URLs — the JSON cannot load.');
  }

  fetch('../../data/leaderboard.json')
    .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(lb => {
      LB = lb;
      return Promise.all(lb.strategies.map(s =>
        fetch(`../../data/${s.id}.json`).then(r => r.json())
          .then(d => { STRATS[s.id] = d; })));
    })
    .then(render)
    .catch(e => fail('Could not load leaderboard data',
      e.message + ' — has `python3 viz/build.py` been run?'));

  // value → background colour. Diverging green/red around 0; `cap` sets the
  // saturation point. consistency uses a 0..1 sequential green ramp.
  function heat(v, cap, sequential) {
    let a, rgb;
    if (sequential) {
      a = Math.max(0, Math.min(1, v));
      rgb = '63,185,80';
    } else {
      a = Math.min(1, Math.abs(v) / cap);
      rgb = v >= 0 ? '63,185,80' : '248,81,73';
    }
    const cls = sequential ? '' : (v > 0.02 ? 'cell-pos' : (v < -0.02 ? 'cell-neg' : ''));
    return { bg: `rgba(${rgb},${(a * 0.34).toFixed(3)})`, cls };
  }

  function cell(v, cap, seq, digits) {
    const h = heat(v, cap, seq);
    const txt = seq ? (v * 100).toFixed(0) + '%' : v.toFixed(digits != null ? digits : 2);
    return `<td><span class="heat ${h.cls}" style="background:${h.bg}">${txt}</span></td>`;
  }

  function sparkline(canvas, eq, bh) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.parentElement.clientWidth, h = 56;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const all = eq.concat(bh);
    const lo = Math.min(...all), hi = Math.max(...all);
    const X = i => 2 + (w - 4) * i / (eq.length - 1);
    const Y = v => 3 + (h - 6) * (1 - (v - lo) / (hi - lo || 1));
    const line = (arr, color, dash, lw) => {
      ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash);
      ctx.beginPath();
      arr.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)));
      ctx.stroke(); ctx.setLineDash([]);
    };
    line(bh, cssVar('--bench'), [4, 3], 1);
    line(eq, eq[eq.length - 1] >= 1 ? cssVar('--pos') : cssVar('--neg'), [], 1.6);
  }

  function render() {
    document.title = 'The Leaderboard — Strategy Lab';
    const ds = LB.datasets;

    let head = `<tr><th>Strategy</th><th>Thesis</th><th>#p</th>`;
    ds.forEach(d => head += `<th title="in-sample Sharpe on ${d}">IS ${d.replace('_', ' ')}</th>`);
    head += `<th>IS mean</th><th title="mean walk-forward OOS Sharpe">OOS Sharpe</th>
      <th title="mean fraction of profitable OOS folds">OOS folds&nbsp;✓</th></tr>`;

    let rows = '';
    LB.strategies.forEach(s => {
      rows += `<tr><td><a class="back" href="../${s.id}/index.html">${s.title}</a></td>
        <td>${s.thesis_tag}</td><td>${Object.keys(s.params).length}</td>`;
      ds.forEach(d => rows += cell(s.datasets[d].sharpe, 1.0));
      rows += cell(s.mean_sharpe, 1.0);
      rows += cell(s.mean_oos_sharpe, 0.6);
      rows += cell(s.mean_oos_consistency, null, true);
      rows += `</tr>`;
    });

    const sm = LB.strategies.map(s => {
      const cells = ds.map(d => {
        const D = STRATS[s.id].datasets[d];
        const fin = D.equity[D.equity.length - 1];
        const cls = fin >= 1 ? 'cell-pos' : 'cell-neg';
        return `<div class="sm-cell">
          <h4>${s.title.split('(')[0].trim()}</h4>
          <div class="sm-sub">${d} · final <span class="${cls}">×${fin.toFixed(2)}</span></div>
          <canvas data-strat="${s.id}" data-ds="${d}"></canvas></div>`;
      }).join('');
      return cells;
    }).join('');

    app.innerHTML = `
      <a class="back" href="../../index.html">← Strategy Lab</a>
      <h1 style="font-weight:400;letter-spacing:.03em;margin:.6rem 0 0">The Leaderboard</h1>
      <p class="lede">${LB.note}</p>

      <h2 class="section">How strategies are judged</h2>
      <ul class="criteria">
        ${LB.criteria.map(c => `<li><b>${c[0]}.</b> ${c[1]}</li>`).join('')}
      </ul>

      <h2 class="section">In-sample vs. out-of-sample</h2>
      <p style="color:var(--muted);font-size:.88rem">Each row is one strategy.
        The four <b>IS</b> columns are the full-window Sharpe per regime; the two
        right-hand columns are the walk-forward truth. Read left → right and watch
        the colour drain out.</p>
      <div style="overflow-x:auto"><table class="lb-table"><thead>${head}</thead>
        <tbody>${rows}</tbody></table></div>

      <h2 class="section">Equity curves — every strategy × every regime</h2>
      <p style="color:var(--muted);font-size:.88rem">Solid = strategy (net of
        costs), dashed = buy &amp; hold. Both normalised to ×1.00 at the start.</p>
      <div class="small-multiples">${sm}</div>`;

    app.querySelectorAll('canvas[data-strat]').forEach(cv => {
      const D = STRATS[cv.dataset.strat].datasets[cv.dataset.ds];
      sparkline(cv, D.equity, D.buy_hold);
    });
  }
})();
