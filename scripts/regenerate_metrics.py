#!/usr/bin/env python3
"""Regenerate metrics.json for all strategies.

Run from the repo root:
    python3 scripts/regenerate_metrics.py

Produces engine-consistent metrics.json files including the deflated_sharpe
and regime_sharpe fields.
"""
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from engine import backtest  # noqa: E402
from engine import metrics as em  # noqa: E402
from engine.antioverfitting import compute_dsr  # noqa: E402
from engine.sensitivity import build_param_grid, build_trials_matrix  # noqa: E402

DATA_DIR = ROOT / "data"
STRATEGIES_DIR = ROOT / "strategies"
DATASETS = ["trend_gbm", "mean_rev_ou", "regime_switch", "fat_tail"]

_MIN_TRIALS_FOR_SWEEP = 8


def _load_strategy_module(strategy_dir: Path, module_name: str):
    strategy_file = strategy_dir / "strategy.py"
    spec = importlib.util.spec_from_file_location(module_name, strategy_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_data(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)


def _round_float(v):
    """Round float to 6 decimal places; convert NaN and inf to None for valid JSON."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 6)
    return v


def _find_strategy_class(mod):
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ == mod.__name__ and callable(obj):
            return obj
    raise AttributeError(f"No strategy class found in module {mod.__name__}")


def _block_bootstrap_trials(
    returns,
    n_trials: int = 16,
    block_length: int = 21,
    seed: int = 42,
):
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(returns)
    padded = np.concatenate([returns, returns])
    result = np.empty((n, n_trials))
    n_blocks = int(np.ceil(n / block_length))
    for trial in range(n_trials):
        starts = rng.integers(0, n, size=n_blocks)
        bootstrapped = np.concatenate([padded[s: s + block_length] for s in starts])
        result[:, trial] = bootstrapped[:n]
    return result


def compute_strategy_metrics(
    strategy_cls,
    params: dict,
    strategy_dir: Path = None,
    datasets: list = None,
    backtest_config: dict = None,
) -> dict:
    result = {}
    sensitivity = None
    param_grid = None
    if strategy_dir is not None:
        sens_path = strategy_dir / "sensitivity.json"
        if sens_path.exists():
            sensitivity = json.loads(sens_path.read_text())
            param_grid = build_param_grid(params)

    active_datasets = datasets if datasets is not None else DATASETS

    def factory(p):
        return strategy_cls(**p)

    for dataset in active_datasets:
        df = _load_data(dataset)
        equity_series, gross_equity_series, positions_series, rfr = backtest._run_internal(
            strategy_cls(**params), df, backtest_config
        )

        trials_matrix = None
        if sensitivity is not None and param_grid is not None and dataset in sensitivity:
            n_trials_in_sweep = sensitivity[dataset]["n_trials"]
            if n_trials_in_sweep >= _MIN_TRIALS_FOR_SWEEP:
                trials_matrix = build_trials_matrix(factory, param_grid, df)
            else:
                returns = equity_series.pct_change().dropna().values
                trials_matrix = _block_bootstrap_trials(returns)

        base = em.compute_all(equity_series, positions_series, rfr, trials_matrix)
        base["deflated_sharpe"] = compute_dsr(equity_series, n_trials=1)
        base["cost_to_alpha_ratio"] = em.cost_to_alpha_ratio(gross_equity_series, equity_series)

        daily_returns = equity_series.pct_change().dropna()
        ci_lo, _ci_pt, ci_hi = em.sharpe_ci(daily_returns)
        base["sharpe_ci_lower"] = ci_lo
        base["sharpe_ci_upper"] = ci_hi
        base["sharpe_ci_confidence"] = 0.95

        rs_raw = em.regime_conditional_sharpe(equity_series.pct_change(), df)
        rs_clean = {}
        for k, v in rs_raw.items():
            if k == "regime_counts":
                rs_clean[k] = v
            else:
                rs_clean[k] = _round_float(v)
        base["regime_sharpe"] = rs_clean

        wf = backtest.walk_forward_backtest(strategy_cls, params, df, config=backtest_config)
        base["walk_forward"] = wf

        rounded = {k: _round_float(v) if k not in ("walk_forward", "regime_sharpe") else v for k, v in base.items()}
        rounded["walk_forward"] = {
            k: _round_float(v)
            for k, v in rounded["walk_forward"].items()
        }
        result[dataset] = rounded
    return result


def main():
    # Strategy 01: Dual EMA Momentum
    s01_dir = STRATEGIES_DIR / "01-dual-ema-momentum"
    mod01 = _load_strategy_module(s01_dir, "dual_ema_strategy")
    metrics01 = compute_strategy_metrics(
        mod01.DualEMAMomentum, {"fast_window": 20, "slow_window": 60}, s01_dir
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
        s02_dir,
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
        s03_dir,
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
        s04_dir,
    )
    out04 = s04_dir / "metrics.json"
    out04.write_text(json.dumps(metrics04, indent=2) + "\n")
    print(f"Written: {out04}")

    # Strategy 05: Turn-of-Month Calendar Effect
    s05_dir = STRATEGIES_DIR / "05-turn-of-month"
    mod05 = _load_strategy_module(s05_dir, "tom_strategy")
    metrics05 = compute_strategy_metrics(
        mod05.TurnOfMonth, {"tail_days": 2, "head_days": 3}, s05_dir
    )
    out05 = s05_dir / "metrics.json"
    out05.write_text(json.dumps(metrics05, indent=2) + "\n")
    print(f"Written: {out05}")

    # Strategy 06: Bollinger Band Mean-Reversion
    s06_dir = STRATEGIES_DIR / "06-bollinger-mean-reversion"
    mod06 = _load_strategy_module(s06_dir, "bollinger_strategy")
    metrics06 = compute_strategy_metrics(
        mod06.BollingerMeanReversion, {"window": 20, "nstd": 2.0, "exit_window": 20}, s06_dir
    )
    out06 = s06_dir / "metrics.json"
    out06.write_text(json.dumps(metrics06, indent=2) + "\n")
    print(f"Written: {out06}")

    # Strategy 07: Absolute Momentum (Trend Filter)
    s07_dir = STRATEGIES_DIR / "07-absolute-momentum"
    mod07 = _load_strategy_module(s07_dir, "abs_mom_strategy")
    metrics07 = compute_strategy_metrics(
        mod07.AbsoluteMomentum, {"lookback": 252, "threshold": 0.0}, s07_dir
    )
    out07 = s07_dir / "metrics.json"
    out07.write_text(json.dumps(metrics07, indent=2) + "\n")
    print(f"Written: {out07}")

    # Strategy 08: NR7 Volatility-Contraction Breakout
    s08_dir = STRATEGIES_DIR / "08-nr7-breakout"
    mod08 = _load_strategy_module(s08_dir, "nr7_strategy")
    metrics08 = compute_strategy_metrics(
        mod08.NR7Breakout, {"n_bars": 7, "exit_bars": 4}, s08_dir
    )
    out08 = s08_dir / "metrics.json"
    out08.write_text(json.dumps(metrics08, indent=2) + "\n")
    print(f"Written: {out08}")

    # Strategy 09: Volatility-Managed Portfolio
    s09_dir = STRATEGIES_DIR / "09-volatility-managed"
    mod09 = _load_strategy_module(s09_dir, "volmgd_strategy")
    metrics09 = compute_strategy_metrics(
        mod09.VolatilityManagedPortfolio, {"window": 21, "target_vol": 0.12}, s09_dir
    )
    out09 = s09_dir / "metrics.json"
    out09.write_text(json.dumps(metrics09, indent=2) + "\n")
    print(f"Written: {out09}")

    # Strategy 10: Low-Volatility Anomaly
    s10_dir = STRATEGIES_DIR / "10-low-volatility-anomaly"
    mod10 = _load_strategy_module(s10_dir, "lva_strategy")
    metrics10 = compute_strategy_metrics(
        mod10.LowVolatilityAnomaly,
        {"vol_window": 60, "ranking_window": 252, "exit_percentile": 75.0},
        s10_dir,
    )
    out10 = s10_dir / "metrics.json"
    out10.write_text(json.dumps(metrics10, indent=2) + "\n")
    print(f"Written: {out10}")

    # Strategy 11: Statistical Pairs Mean-Reversion (paired_cointegrated dataset only)
    # allow_short=True is required: the strategy generates -1 (short spread) signals
    # that must be executed; without it the strategy degenerates to long-only.
    s11_dir = STRATEGIES_DIR / "11-pairs-mean-reversion"
    mod11 = _load_strategy_module(s11_dir, "pairs_strategy")
    metrics11 = compute_strategy_metrics(
        mod11.PairsMeanReversion,
        {"z_entry": 2.0, "z_exit": 0.5, "window": 60},
        s11_dir,
        datasets=["paired_cointegrated"],
        backtest_config={"allow_short": True},
    )
    out11 = s11_dir / "metrics.json"
    out11.write_text(json.dumps(metrics11, indent=2) + "\n")
    print(f"Written: {out11}")

    # Strategy 12: Volatility-Conditioned RSI Mean-Reversion
    # allow_short=True is required: the strategy takes short positions when RSI > 90
    # in high-vol regimes; without it only the long side is evaluated.
    s12_dir = STRATEGIES_DIR / "12-vol-conditioned-rsi"
    mod12 = _load_strategy_module(s12_dir, "vol_rsi_strategy")
    metrics12 = compute_strategy_metrics(
        mod12.VolConditionedRSI,
        {
            "vol_window": 21,
            "vol_lookback": 252,
            "vol_threshold": 0.75,
            "rsi_window": 2,
            "rsi_entry_long": 10.0,
            "rsi_exit_long": 70.0,
            "rsi_entry_short": 90.0,
            "rsi_exit_short": 30.0,
        },
        s12_dir,
        backtest_config={"allow_short": True},
    )
    out12 = s12_dir / "metrics.json"
    out12.write_text(json.dumps(metrics12, indent=2) + "\n")
    print(f"Written: {out12}")

    print("Done.")


if __name__ == "__main__":
    main()
