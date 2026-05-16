/* Zero-dependency canvas charting shared by every piece page.
 *
 * One function, renderChart(canvas, opt), draws a line chart with optional
 * long/flat shading, trade markers, horizontal threshold bands, and date/price
 * axes. Colours are pulled from CSS custom properties at draw time, so a theme
 * toggle just needs to re-call renderChart — the chart re-themes itself.
 *
 * It is deliberately small and explicit (no library) to match the
 * zero-build, self-contained ethos of ~/Documents/git/computer_art. */

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
}

/* opt:
 *   height   css pixels (width fills the parent)
 *   xCount   number of x positions (all data arrays share this length)
 *   dates    array of "YYYY-MM-DD" for x tick labels
 *   yMin/yMax optional fixed y range (else auto from series, ignoring null)
 *   series   [{data:[num|null], color, width=1.5, dash=null, label}]
 *   shade    array(0|1) length xCount, or null — fills runs where 1
 *   markers  [{i, y, type:'entry'|'exit'}]
 *   hbands   [{y, color, label}]
 *   yfmt     v => string for y tick labels (default 2dp)
 */
function renderChart(canvas, opt) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.parentElement.clientWidth;
  const cssH = opt.height;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = cssW + 'px';
  canvas.style.height = cssH + 'px';

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const pad = { l: 52, r: 12, t: 10, b: 22 };
  const W = cssW - pad.l - pad.r;
  const Hh = cssH - pad.t - pad.b;
  const n = opt.xCount;

  // y range
  let lo = opt.yMin, hi = opt.yMax;
  if (lo == null || hi == null) {
    lo = Infinity; hi = -Infinity;
    for (const s of opt.series) {
      for (const v of s.data) {
        if (v == null || Number.isNaN(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    for (const b of (opt.hbands || [])) { lo = Math.min(lo, b.y); hi = Math.max(hi, b.y); }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    const padv = (hi - lo) * 0.06 || 0.5;
    lo -= padv; hi += padv;
  }

  const X = i => pad.l + (n <= 1 ? 0 : W * i / (n - 1));
  const Y = v => pad.t + Hh - Hh * (v - lo) / (hi - lo || 1);

  const grid = cssVar('--grid-line');
  const muted = cssVar('--muted');

  // long/flat shading (filled runs)
  if (opt.shade) {
    ctx.fillStyle = cssVar('--shade');
    let i = 0;
    while (i < n) {
      if (opt.shade[i] === 1) {
        let j = i;
        while (j < n && opt.shade[j] === 1) j++;
        const x0 = X(i), x1 = X(Math.min(j, n - 1));
        ctx.fillRect(x0, pad.t, Math.max(x1 - x0, 1), Hh);
        i = j;
      } else i++;
    }
  }

  // y gridlines + labels
  ctx.strokeStyle = grid;
  ctx.fillStyle = muted;
  ctx.font = '11px system-ui, sans-serif';
  ctx.lineWidth = 1;
  const yfmt = opt.yfmt || (v => v.toFixed(2));
  const TICKS = 4;
  for (let t = 0; t <= TICKS; t++) {
    const v = lo + (hi - lo) * t / TICKS;
    const y = Y(v);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + W, y);
    ctx.stroke();
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(yfmt(v), pad.l - 6, y);
  }

  // x date labels (~5)
  if (opt.dates) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const steps = 4;
    for (let t = 0; t <= steps; t++) {
      const i = Math.round((n - 1) * t / steps);
      ctx.fillText(opt.dates[i].slice(0, 7), X(i), pad.t + Hh + 5);
    }
  }

  // horizontal threshold bands
  for (const b of (opt.hbands || [])) {
    const y = Y(b.y);
    ctx.strokeStyle = b.color;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + W, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = b.color;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(b.label, pad.l + 4, y - 2);
  }

  // series
  for (const s of opt.series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width || 1.5;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < n; i++) {
      const v = s.data[i];
      if (v == null || Number.isNaN(v)) { pen = false; continue; }
      const x = X(i), y = Y(v);
      if (!pen) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // trade markers
  for (const mk of (opt.markers || [])) {
    const x = X(mk.i), y = Y(mk.y);
    const up = mk.type === 'entry';
    ctx.fillStyle = up ? cssVar('--pos') : cssVar('--neg');
    const d = 5, oy = up ? 9 : -9;
    ctx.beginPath();
    if (up) {
      ctx.moveTo(x, y + oy - d);
      ctx.lineTo(x - d, y + oy + d);
      ctx.lineTo(x + d, y + oy + d);
    } else {
      ctx.moveTo(x, y + oy + d);
      ctx.lineTo(x - d, y + oy - d);
      ctx.lineTo(x + d, y + oy - d);
    }
    ctx.closePath();
    ctx.fill();
  }

  // frame
  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.l, pad.t, W, Hh);
}
