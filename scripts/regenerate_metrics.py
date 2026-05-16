#!/usr/bin/env python3
"""Regenerate metrics.json for all strategies.

Run from the repo root:
    python3 scripts/regenerate_metrics.py

Produces engine-consistent metrics.json files including the new deflated_sharpe
field (n_trials=1, conservative lower bound for single-strategy evaluation).
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from engine import backtest  # noqa: E402
from engine import metrics as em  # noqa: E402
from engine.antioverfitting import compute_dsr  # noqa: E402

DATA_DIR = ROOT / "data"
STRATEGIES_DIR = ROOT / "strategies"
DATASETS = ["trend_gbm", "mean_rev_ou", "regime_switch", "fat_tail"]


def _load_strategy_module(strategy_dir: Path, module_name: str):
    strategy_file = strategy_dir / "strategy.py"
    spec = importlib.util.spec_from_file_location(module_name, strategy_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_data(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)


def compute_strategy_metrics(strategy_cls, params: dict) -> dict:
    result = {}
    for dataset in DATASETS:
        df = _load_data(dataset)
        equity_series, positions_series, rfr = backtest._run_internal(
            strategy_cls(**params), df
        )
        base = em.compute_all(equity_series, positions_series, rfr)
        base["deflated_sharpe"] = compute_dsr(equity_series, n_trials=1)

        wf = backtest.walk_forward_backtest(strategy_cls, params, df)
        base["walk_forward"] = wf

        # Round to 6 decimal places for stable diffs
        rounded = {k: round(v, 6) if isinstance(v, float) else v for k, v in base.items()}
        rounded["walk_forward"] = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in rounded["walk_forward"].items()
        }
        result[dataset] = rounded
    return result


def main():
    # Strategy 01: Dual EMA Momentum
    s01_dir = STRATEGIES_DIR / "01-dual-ema-momentum"
    mod01 = _load_strategy_module(s01_dir, "dual_ema_strategy")
    metrics01 = compute_strategy_metrics(
        mod01.DualEMAMomentum, {"fast_window": 20, "slow_window": 60}
    )
    out01 = s01_dir / "metrics.json"
    out01.write_text(json.dumps(metrics01, indent=2) + "\n")
    print(f"Written: {out01}")

    # Strategy 02: RSI Mean-Reversion
    s02_dir = STRATEGIES_DIR / "02-rsi-mean-reversion"
    mod02 = _load_strategy_module(s02_dir, "rsi_strategy")
    metrics02 = compute_strategy_metrics(
        mod02.RSIMeanReversion,
        {"rsi_period": 2, "oversold": 10.0, "overbought": 90.0},
    )
    out02 = s02_dir / "metrics.json"
    out02.write_text(json.dumps(metrics02, indent=2) + "\n")
    print(f"Written: {out02}")

    # Strategy 03: Donchian Turtle Breakout
    s03_dir = STRATEGIES_DIR / "03-donchian-turtle-breakout"
    mod03 = _load_strategy_module(s03_dir, "donchian_strategy")
    metrics03 = compute_strategy_metrics(
        mod03.DonchianTurtleBreakout,
        {"entry_window": 20, "exit_window": 10, "atr_window": 20},
    )
    out03 = s03_dir / "metrics.json"
    out03.write_text(json.dumps(metrics03, indent=2) + "\n")
    print(f"Written: {out03}")

    # Strategy 04: 52-Week High Proximity
    s04_dir = STRATEGIES_DIR / "04-52wk-high-proximity"
    mod04 = _load_strategy_module(s04_dir, "proximity_strategy")
    metrics04 = compute_strategy_metrics(
        mod04.FiftyTwoWeekHighProximity,
        {"proximity_threshold": 0.95, "exit_threshold": 0.90},
    )
    out04 = s04_dir / "metrics.json"
    out04.write_text(json.dumps(metrics04, indent=2) + "\n")
    print(f"Written: {out04}")

    print("Done.")


if __name__ == "__main__":
    main()
