#!/usr/bin/env python3
"""Backfill the 'pbo' key into each strategy's metrics.json.

For each strategy / dataset pair:
- If the parameter sweep had n_trials >= 8 (from sensitivity.json), re-run the
  sweep via build_trials_matrix to obtain the full (n_bars-1 x n_trials) return
  matrix, then compute PBO via CSCV.
- Otherwise (n_trials < 8 — typically zero-param or single-param strategies),
  fall back to a block bootstrap: generate 16 synthetic trials by resampling
  the strategy's own return series in blocks of 21 bars (monthly), fixed
  seed=42 for full determinism.

Run from the repo root:
    python3 scripts/backfill_pbo.py
"""
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.antioverfitting import pbo  # noqa: E402
from engine.backtest import _run_internal  # noqa: E402
from engine.sensitivity import build_param_grid, build_trials_matrix  # noqa: E402

DATA_DIR = ROOT / "data"
STRATEGIES_DIR = ROOT / "strategies"
DATASETS = ["trend_gbm", "mean_rev_ou", "regime_switch", "fat_tail"]

_MIN_TRIALS_FOR_SWEEP = 8
_BOOTSTRAP_N_TRIALS = 16
_BOOTSTRAP_BLOCK = 21
_BOOTSTRAP_SEED = 42


def _load_strategy_module(strategy_dir: Path, module_name: str):
    strategy_file = strategy_dir / "strategy.py"
    spec = importlib.util.spec_from_file_location(module_name, strategy_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_strategy_class(mod):
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ == mod.__name__ and callable(obj):
            return obj
    raise AttributeError(f"No strategy class found in module {mod.__name__}")


def _load_data(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)


def _block_bootstrap_trials(
    returns: np.ndarray,
    n_trials: int = _BOOTSTRAP_N_TRIALS,
    block_length: int = _BOOTSTRAP_BLOCK,
    seed: int = _BOOTSTRAP_SEED,
) -> np.ndarray:
    """Generate (len(returns) × n_trials) matrix via circular block bootstrap.

    Each column is an independent block-bootstrapped realization of the input
    return series. Block boundaries are drawn with replacement from a uniform
    distribution over valid start positions. The circular wrap ensures the last
    block never runs off the end of the series.

    Fixed seed guarantees byte-for-byte identical output across runs.
    """
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


def compute_pbo_for_strategy(
    strategy_dir: Path,
    module_name: str,
) -> dict:
    """Return a dict mapping dataset name → PBO float for one strategy.

    Args:
        strategy_dir: Path to the strategy directory (contains strategy.py,
            sensitivity.json, and metrics.json).
        module_name: Unique importlib module name (avoids collisions).

    Returns:
        Dict like {"trend_gbm": 0.42, "mean_rev_ou": 0.31, ...}.
    """
    mod = _load_strategy_module(strategy_dir, module_name)
    strategy_cls = _find_strategy_class(mod)
    default_params = mod.DEFAULT_PARAMS
    param_grid = build_param_grid(default_params)

    sensitivity_path = strategy_dir / "sensitivity.json"
    sensitivity = json.loads(sensitivity_path.read_text())

    def factory(params):
        return strategy_cls(**params)

    result = {}
    for dataset in DATASETS:
        df = _load_data(dataset)
        n_trials_in_sweep = sensitivity[dataset]["n_trials"]

        if n_trials_in_sweep >= _MIN_TRIALS_FOR_SWEEP:
            matrix = build_trials_matrix(factory, param_grid, df)
        else:
            equity, _, _, _ = _run_internal(strategy_cls(**default_params), df)
            returns = equity.pct_change().dropna().values
            matrix = _block_bootstrap_trials(returns)

        result[dataset] = round(pbo(matrix), 6)
        print(f"    {dataset}: pbo={result[dataset]:.4f}  (n_trials={n_trials_in_sweep})")

    return result


def main():
    strategies = [
        ("01-dual-ema-momentum",   "dual_ema_strategy"),
        ("02-rsi-mean-reversion",  "rsi_strategy"),
        ("03-donchian-turtle-breakout", "donchian_strategy"),
        ("04-52wk-high-proximity", "proximity_strategy"),
        ("05-turn-of-month",       "tom_strategy"),
        ("06-bollinger-mean-reversion", "bollinger_strategy"),
        ("07-absolute-momentum",   "abs_mom_strategy"),
        ("08-nr7-breakout",        "nr7_strategy"),
        ("09-volatility-managed",  "volmgd_strategy"),
    ]

    for name, module_name in strategies:
        strategy_dir = STRATEGIES_DIR / name
        print(f"\n{name} ...")
        pbo_values = compute_pbo_for_strategy(strategy_dir, module_name)

        metrics_path = strategy_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        for dataset, pbo_val in pbo_values.items():
            metrics[dataset]["pbo"] = pbo_val

        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"  Updated: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
