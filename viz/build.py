#!/usr/bin/env python3
"""Build the static visualisation gallery data.

Run from the repo root (inside the project venv):
    python3 viz/build.py

For every strategy in ``strategies.json`` this runs the *real* backtest engine
on all four synthetic datasets and precomputes everything the static pages
render: the price path, the strategy's own indicator overlays, the long/flat
position track, the equity curve versus buy-and-hold, the individual trades,
the full metrics block, and a plain-English interpretation of why the edge
holds or breaks in each regime.

Nothing here is bespoke charting maths — the equity and position series come
straight out of ``engine.backtest._run_internal`` (the same code path
``scripts/regenerate_metrics.py`` uses), so the gallery can never disagree with
``metrics.json``. The output is plain JSON + SVG committed alongside the site,
which keeps the gallery a zero-build static site exactly like ~/Documents/git/
computer_art.

Swapping in real data later: drop real OHLCV CSVs into ``data/`` (same columns,
a DatetimeIndex) and point ``DATASETS`` / ``REGIME`` at them. No page code
changes — the pages only ever read the JSON this script emits.
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine import backtest  # noqa: E402
from engine import metrics as em  # noqa: E402
from engine.antioverfitting import compute_dsr  # noqa: E402

DATA_DIR = ROOT / "data"
STRATEGIES_DIR = ROOT / "strategies"
VIZ_DIR = Path(__file__).parent
OUT_DIR = VIZ_DIR / "data"
PIECES_DIR = VIZ_DIR / "pieces"

DATASETS = ["trend_gbm", "mean_rev_ou", "regime_switch", "fat_tail"]

# Plain-English description of what each synthetic dataset *is* (mirrors
# data/README.md) so a reader who has never seen the data understands the
# regime the strategy is being judged in.
REGIME = {
    "trend_gbm": "Geometric Brownian Motion, +15%/yr drift — a persistent uptrend.",
    "mean_rev_ou": "Ornstein–Uhlenbeck process oscillating around 100 — pure mean reversion.",
    "regime_switch": "200-bar blocks alternating trend and mean reversion — punishes regime overfitting.",
    "fat_tail": "Zero-drift Student-t(3) returns — extreme daily moves ~2–4× more often than normal.",
}

# Which regimes the *thesis* predicts should help (+1) or hurt (-1) each
# strategy family. This is the strategy's prior, written before any backtest —
# used only to phrase the interpretation honestly ("matched / contradicted the
# thesis"), never to grade.
THESIS_PRIOR = {
    "momentum": {"trend_gbm": +1, "mean_rev_ou": -1, "regime_switch": +1, "fat_tail": 0},
    "mean_reversion": {"trend_gbm": -1, "mean_rev_ou": +1, "regime_switch": +1, "fat_tail": +1},
    "anchoring": {"trend_gbm": 0, "mean_rev_ou": -1, "regime_switch": 0, "fat_tail": -1},
}


def _load_strategy_module(strategy_dir: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, strategy_dir / "strategy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_data(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)


# --------------------------------------------------------------------------
# Per-strategy indicator overlays
#
# Each function returns the indicator series the strategy *actually looks at*,
# so the chart shows the reader the same numbers the rule sees. All operators
# (ewm with adjust=False, rolling().max/min, shift) are strictly backward
# looking, so the vectorised full-series value at bar t equals the causal
# value the engine computed at t — no look-ahead is introduced for display.
# --------------------------------------------------------------------------


def _overlay_dual_ema(df, p):
    c = df["close"]
    return {
        "price_overlays": {
            f"EMA({p['fast_window']})": c.ewm(span=p["fast_window"], adjust=False).mean(),
            f"EMA({p['slow_window']})": c.ewm(span=p["slow_window"], adjust=False).mean(),
        },
        "indicator": None,
    }


def _rsi_series(closes: pd.Series, period: int) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi.fillna(50.0)


def _overlay_rsi(df, p):
    rsi = _rsi_series(df["close"], p["rsi_period"])
    return {
        "price_overlays": {},
        "indicator": {
            "name": f"RSI({p['rsi_period']})",
            "values": rsi,
            "min": 0,
            "max": 100,
            "bands": [
                {"label": f"oversold {p['oversold']:g} → enter long", "y": p["oversold"]},
                {"label": f"overbought {p['overbought']:g} → exit", "y": p["overbought"]},
            ],
        },
    }


def _overlay_donchian(df, p):
    c = df["close"]
    return {
        "price_overlays": {
            f"{p['entry_window']}-bar high (entry)": c.rolling(p["entry_window"]).max().shift(1),
            f"{p['exit_window']}-bar low (exit)": c.rolling(p["exit_window"]).min().shift(1),
        },
        "indicator": None,
    }


def _overlay_proximity(df, p):
    c = df["close"]
    ratio = c / c.rolling(252).max()
    return {
        "price_overlays": {},
        "indicator": {
            "name": "close / 52-week high",
            "values": ratio,
            "min": 0.5,
            "max": 1.02,
            "bands": [
                {"label": f"enter ≥ {p['proximity_threshold']:g} (& rising)",
                 "y": p["proximity_threshold"]},
                {"label": f"exit < {p['exit_threshold']:g}", "y": p["exit_threshold"]},
            ],
        },
    }


# Explicit registry, same pattern as scripts/regenerate_metrics.py: id, module
# name, class attribute, fixed published params, overlay fn, and the teaching
# copy (the one-line economic idea and the exact signal rule).
STRATEGIES = [
    {
        "id": "01-dual-ema-momentum",
        "module": "dual_ema_strategy",
        "cls": "DualEMAMomentum",
        "params": {"fast_window": 20, "slow_window": 60},
        "overlay": _overlay_dual_ema,
        "idea": (
            "Information diffuses slowly: when a fast moving average pulls above a "
            "slow one, a trend has started but is not yet fully priced in "
            "(Jegadeesh & Titman 1993, the momentum premium)."
        ),
        "signal_rule": "Hold long while EMA(20) > EMA(60); otherwise stay flat.",
    },
    {
        "id": "02-rsi-mean-reversion",
        "module": "rsi_strategy",
        "cls": "RSIMeanReversion",
        "params": {"rsi_period": 2, "oversold": 10.0, "overbought": 90.0},
        "overlay": _overlay_rsi,
        "idea": (
            "Crowds overreact to short-term moves and then revert (De Bondt & "
            "Thaler 1985; Jegadeesh 1990). A 2-day RSI extreme is a proxy for "
            "that overreaction."
        ),
        "signal_rule": "Enter long when RSI(2) < 10; exit when RSI(2) > 90; hold in between.",
    },
    {
        "id": "03-donchian-turtle-breakout",
        "module": "donchian_strategy",
        "cls": "DonchianTurtleBreakout",
        "params": {"entry_window": 20, "exit_window": 10, "atr_window": 20},
        "overlay": _overlay_donchian,
        "idea": (
            "A break above an N-bar high means buyers have absorbed every willing "
            "seller, leaving a supply vacuum that sustains the move (the Turtle "
            "experiment, Dennis & Eckhardt 1983)."
        ),
        "signal_rule": "Enter long when close breaks the prior 20-bar high; exit on the prior 10-bar low.",
    },
    {
        "id": "04-52wk-high-proximity",
        "module": "proximity_strategy",
        "cls": "FiftyTwoWeekHighProximity",
        "params": {"proximity_threshold": 0.95, "exit_threshold": 0.90},
        "overlay": _overlay_proximity,
        "idea": (
            "Investors anchor on the 52-week high and under-react to good news "
            "near it, so price keeps drifting up (George & Hwang 2004, J. of "
            "Finance). The effect is cross-sectional — weak on one isolated series."
        ),
        "signal_rule": "Enter long when close ≥ 95% of the 52-week high and rising; exit below 90%.",
    },
]


def _round_list(series, ndigits):
    """np.nan → None (valid JSON); everything else rounded."""
    out = []
    for v in np.asarray(series, dtype=float):
        out.append(None if np.isnan(v) else round(float(v), ndigits))
    return out


def _extract_trades(positions: np.ndarray, equity: np.ndarray, dates):
    """Turn the 0/1 position track into discrete long trades.

    A trade opens on the 0→1 bar and closes on the following 1→0 bar (or the
    last bar if still open). Return uses the net-of-cost equity curve, so it is
    exactly the P&L the metrics are computed from.
    """
    trades = []
    in_pos = False
    entry_i = 0
    for i, pos in enumerate(positions):
        if pos == 1 and not in_pos:
            in_pos, entry_i = True, i
        elif pos == 0 and in_pos:
            in_pos = False
            trades.append(_trade(entry_i, i, equity, dates))
    if in_pos:
        trades.append(_trade(entry_i, len(positions) - 1, equity, dates))
    return trades


def _trade(entry_i, exit_i, equity, dates):
    ret = float(equity[exit_i] / equity[entry_i] - 1.0) if equity[entry_i] else 0.0
    return {
        "entry_i": int(entry_i),
        "exit_i": int(exit_i),
        "entry_date": dates[entry_i],
        "exit_date": dates[exit_i],
        "ret": round(ret, 5),
    }


def _verdict(sharpe: float) -> str:
    if sharpe > 0.15:
        return "favourable"
    if sharpe < -0.15:
        return "adverse"
    return "marginal"


def _interpretation(thesis_tag, dataset, sharpe, cagr, oos_consistency):
    """Honest, computed sentence: measured result + thesis expectation."""
    verdict = _verdict(sharpe)
    prior = THESIS_PRIOR.get(thesis_tag, {}).get(dataset, 0)
    measured_sign = 1 if sharpe > 0.15 else (-1 if sharpe < -0.15 else 0)
    if prior == 0:
        match = "the thesis makes no strong prediction here"
    elif prior == measured_sign:
        match = "this matches what the thesis predicts"
    elif measured_sign == 0:
        match = "the result is too weak to confirm or reject the thesis"
    else:
        match = "this runs against what the thesis predicts"
    return (
        f"In-sample Sharpe {sharpe:+.2f} (CAGR {cagr*100:+.1f}%) — a {verdict} "
        f"regime for this edge; {match}. Out-of-sample, "
        f"{oos_consistency*100:.0f}% of walk-forward folds were profitable."
    )


def build_strategy(spec: dict) -> dict:
    sdir = STRATEGIES_DIR / spec["id"]
    mod = _load_strategy_module(sdir, spec["module"])
    cls = getattr(mod, spec["cls"])
    params = spec["params"]

    out = {
        "id": spec["id"],
        "title": REGISTRY_TITLE[spec["id"]],
        "thesis_tag": REGISTRY_TAG[spec["id"]],
        "params": params,
        "idea": spec["idea"],
        "signal_rule": spec["signal_rule"],
        "source": f"strategies/{spec['id']}/strategy.py",
        "datasets": {},
    }

    for name in DATASETS:
        df = _load_data(name)
        equity_s, _gross_s, pos_s, rfr = backtest._run_internal(cls(**params), df)
        m = em.compute_all(equity_s, pos_s, rfr)
        m["deflated_sharpe"] = compute_dsr(equity_s, n_trials=1)
        m["walk_forward"] = backtest.walk_forward_backtest(cls, params, df)
        m = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}
        m["walk_forward"] = {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in m["walk_forward"].items()
        }

        equity = equity_s.to_numpy()
        positions = pos_s.to_numpy()
        close = df["close"].to_numpy()
        dates = [d.strftime("%Y-%m-%d") for d in df.index]

        ov = spec["overlay"](df, params)
        price_overlays = {
            k: _round_list(v, 4) for k, v in ov["price_overlays"].items()
        }
        indicator = None
        if ov["indicator"] is not None:
            ind = ov["indicator"]
            indicator = {
                "name": ind["name"],
                "values": _round_list(ind["values"], 4),
                "min": ind["min"],
                "max": ind["max"],
                "bands": ind["bands"],
            }

        oos_c = m["walk_forward"]["oos_consistency"]
        out["datasets"][name] = {
            "dates": dates,
            "close": _round_list(close, 4),
            "equity": _round_list(equity / equity[0], 5),
            "buy_hold": _round_list(close / close[0], 5),
            "positions": [int(p) for p in positions],
            "trades": _extract_trades(positions, equity, dates),
            "price_overlays": price_overlays,
            "indicator": indicator,
            "metrics": m,
            "regime": REGIME[name],
            "verdict": _verdict(m["sharpe"]),
            "interpretation": _interpretation(
                out["thesis_tag"], name, m["sharpe"], m["cagr"], oos_c
            ),
        }
    return out


# --------------------------------------------------------------------------
# SVG thumbnails (pure string building, zero deps — mirrors computer_art's
# generate_thumbnail.py approach: a static snapshot committed with the piece).
# --------------------------------------------------------------------------

W, H, PAD = 320, 240, 14


def _poly(values, x0, x1, y0, y1, lo, hi):
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = x0 + (x1 - x0) * i / (n - 1)
        y = y1 - (y1 - y0) * (v - lo) / (hi - lo if hi != lo else 1)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def _thumbnail_svg(strat_json: dict) -> str:
    d = strat_json["datasets"]["regime_switch"]
    eq, bh = d["equity"], d["buy_hold"]
    lo = min(min(eq), min(bh))
    hi = max(max(eq), max(bh))
    x0, x1, y0, y1 = PAD, W - PAD, PAD + 26, H - PAD
    final = eq[-1]
    col = "#3fb950" if final >= 1 else "#f85149"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">
  <rect width="{W}" height="{H}" fill="#0f1117"/>
  <text x="{PAD}" y="22" fill="#e8e8e8" font-size="13" font-weight="600">{strat_json['title']}</text>
  <polyline points="{_poly(bh, x0, x1, y0, y1, lo, hi)}" fill="none" stroke="#6e7681" stroke-width="1.5" stroke-dasharray="3 3"/>
  <polyline points="{_poly(eq, x0, x1, y0, y1, lo, hi)}" fill="none" stroke="{col}" stroke-width="2"/>
  <text x="{W-PAD}" y="{H-PAD}" fill="{col}" font-size="12" text-anchor="end">×{final:.2f} on regime_switch</text>
</svg>
"""


def _leaderboard_svg(board: list) -> str:
    bars = []
    bw = (W - 2 * PAD) / len(board)
    for i, row in enumerate(board):
        v = max(0.0, min(1.0, row["mean_oos_consistency"]))
        bh = (H - 60) * v
        x = PAD + i * bw + 6
        y = H - PAD - bh
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-12:.1f}" height="{bh:.1f}" '
            f'fill="#58a6ff"/>'
            f'<text x="{x+(bw-12)/2:.1f}" y="{H-PAD+12:.0f}" fill="#8b949e" '
            f'font-size="10" text-anchor="middle">{row["id"][:2]}</text>'
            f'<text x="{x+(bw-12)/2:.1f}" y="{y-4:.1f}" fill="#e8e8e8" '
            f'font-size="10" text-anchor="middle">{v:.2f}</text>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">
  <rect width="{W}" height="{H}" fill="#0f1117"/>
  <text x="{PAD}" y="22" fill="#e8e8e8" font-size="13" font-weight="600">Leaderboard — mean OOS consistency</text>
  {''.join(bars)}
</svg>
"""


def main():
    registry = json.loads((ROOT / "strategies.json").read_text())
    global REGISTRY_TITLE, REGISTRY_TAG
    REGISTRY_TITLE = {e["name"]: e["title"] for e in registry}
    REGISTRY_TAG = {e["name"]: e["thesis_tag"] for e in registry}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    board = []

    for spec in STRATEGIES:
        data = build_strategy(spec)
        (OUT_DIR / f"{spec['id']}.json").write_text(json.dumps(data) + "\n")
        print(f"Written: {OUT_DIR / (spec['id'] + '.json')}")

        # thumbnail next to the piece page
        piece_dir = PIECES_DIR / spec["id"]
        piece_dir.mkdir(parents=True, exist_ok=True)
        (piece_dir / "thumbnail.svg").write_text(_thumbnail_svg(data))

        per_ds = {
            ds: {
                "sharpe": data["datasets"][ds]["metrics"]["sharpe"],
                "cagr": data["datasets"][ds]["metrics"]["cagr"],
                "max_drawdown": data["datasets"][ds]["metrics"]["max_drawdown"],
                "oos_sharpe_mean":
                    data["datasets"][ds]["metrics"]["walk_forward"]["oos_sharpe_mean"],
                "oos_consistency":
                    data["datasets"][ds]["metrics"]["walk_forward"]["oos_consistency"],
            }
            for ds in DATASETS
        }
        board.append({
            "id": spec["id"],
            "title": data["title"],
            "thesis_tag": data["thesis_tag"],
            "params": data["params"],
            "datasets": per_ds,
            "mean_sharpe": round(
                float(np.mean([per_ds[d]["sharpe"] for d in DATASETS])), 4
            ),
            "mean_oos_consistency": round(
                float(np.mean([per_ds[d]["oos_consistency"] for d in DATASETS])), 4
            ),
            "mean_oos_sharpe": round(
                float(np.mean([per_ds[d]["oos_sharpe_mean"] for d in DATASETS])), 4
            ),
        })

    leaderboard = {
        "datasets": DATASETS,
        "regime": REGIME,
        "criteria": [
            ["Thesis", "Is the exploited inefficiency coherent, documented, and falsifiable?"],
            ["Robustness", "Does the edge survive across all four regimes and parameter nudges?"],
            ["Simplicity", "Fewer parameters and shorter lookbacks are less likely to be overfit."],
            ["Walk-forward", "Out-of-sample Sharpe and the fraction of profitable OOS folds (the real test)."],
            ["Raw performance", "Sharpe, CAGR and drawdown on the full window — tiebreaker only."],
        ],
        "note": (
            "Rows are shown in registry order, which mirrors the editorial ranking "
            "in RANKING.md. The lesson of this dashboard: a green in-sample Sharpe "
            "(left) routinely turns red out-of-sample (right). Judge strategies by "
            "the walk-forward columns, not the headline number."
        ),
        "strategies": board,
    }
    (OUT_DIR / "leaderboard.json").write_text(json.dumps(leaderboard) + "\n")
    print(f"Written: {OUT_DIR / 'leaderboard.json'}")

    lb_dir = PIECES_DIR / "leaderboard"
    lb_dir.mkdir(parents=True, exist_ok=True)
    (lb_dir / "thumbnail.svg").write_text(_leaderboard_svg(board))
    print("Done.")


if __name__ == "__main__":
    main()
