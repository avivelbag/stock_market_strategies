"""Parameter sensitivity analysis: sweep a grid of parameter combinations and
measure how much the Sharpe ratio disperses across them.

The coefficient of variation (sensitivity_score = std / |mean|) is used as the
robustness signal: a lower score means performance barely changes when parameters
shift, which is evidence against overfitting to the default values.
"""

import itertools
from typing import Callable

import numpy as np
import pandas as pd

from engine.backtest import _run_internal, run


def _grid_values(value, multipliers=(0.8, 0.9, 1.0, 1.1, 1.2)):
    """Generate a ±20% grid around a single default parameter value.

    For integer parameters the grid is rounded and deduplicated (preserving
    insertion order) so that small defaults like 2 produce a compact set rather
    than five identical entries.  For float parameters the grid is returned as-is.

    Args:
        value: The default parameter value (int or float).
        multipliers: Scaling factors to apply; defaults to a 5-point ±20% sweep.

    Returns:
        List of parameter values to sweep, deduplicated for integers.
    """
    if isinstance(value, bool):
        return [value]
    if isinstance(value, int):
        seen = {}
        for m in multipliers:
            v = round(value * m)
            if v not in seen:
                seen[v] = None
        return list(seen)
    return [value * m for m in multipliers]


def build_param_grid(default_params: dict) -> dict:
    """Build a ±20% sweep grid from a DEFAULT_PARAMS dict.

    Each parameter gets five candidate values (fewer if integer deduplication
    collapses them).  The caller passes this grid to parameter_sweep.

    Args:
        default_params: Mapping of parameter name → default value, e.g.
            ``{"fast_window": 20, "slow_window": 60}``.

    Returns:
        Dict mapping each parameter name to its list of sweep values.
    """
    return {k: _grid_values(v) for k, v in default_params.items()}


def parameter_sweep(
    strategy_factory: Callable,
    param_grid: dict,
    price_data: pd.DataFrame,
    engine_kwargs: dict = {},
) -> dict:
    """Run the backtest for every combination in param_grid and return dispersion stats.

    Iterates over the Cartesian product of param_grid values.  For each
    combination it creates a fresh strategy via strategy_factory(params_dict),
    runs the full backtest, and collects the resulting Sharpe ratio.  Combinations
    that fail strategy construction (invalid params) are silently skipped.

    sensitivity_score = std_sharpe / abs(mean_sharpe) is the coefficient of
    variation: lower means more robust.  When mean_sharpe is effectively zero
    the score is capped at 99.0 rather than returning infinity (avoids JSON
    serialisation issues and signals "uninformative" rather than "fragile").

    Args:
        strategy_factory: Callable that accepts a single dict of parameters and
            returns a strategy instance (callable) ready to pass to
            engine.backtest.run.  Must create a fresh stateful instance on each
            call.
        param_grid: Dict mapping parameter names to lists of candidate values,
            as returned by build_param_grid.
        price_data: OHLCV DataFrame with DatetimeIndex; passed verbatim to
            engine.backtest.run for each combination.
        engine_kwargs: Optional config dict forwarded to engine.backtest.run
            (commission_bps, slippage_bps, etc.).

    Returns:
        Dict with keys:
            ``mean_sharpe``: mean Sharpe across valid combinations.
            ``std_sharpe``: population std of Sharpe across valid combinations.
            ``min_sharpe``: minimum observed Sharpe.
            ``max_sharpe``: maximum observed Sharpe.
            ``n_trials``: number of valid (non-skipped) combinations run.
            ``sensitivity_score``: std_sharpe / abs(mean_sharpe), capped at 99.0.
                Lower = more robust.  Returns 0.0 when n_trials == 0.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    config = engine_kwargs if engine_kwargs else None

    sharpes: list[float] = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        try:
            strategy = strategy_factory(params)
        except (ValueError, TypeError):
            continue
        try:
            result = run(strategy, price_data, config)
            sharpes.append(result["sharpe"])
        except Exception:
            continue

    if not sharpes:
        return {
            "mean_sharpe": 0.0,
            "std_sharpe": 0.0,
            "min_sharpe": 0.0,
            "max_sharpe": 0.0,
            "n_trials": 0,
            "sensitivity_score": 0.0,
        }

    mean_sharpe = float(np.mean(sharpes))
    std_sharpe = float(np.std(sharpes))
    min_sharpe = float(np.min(sharpes))
    max_sharpe = float(np.max(sharpes))
    n_trials = len(sharpes)

    if abs(mean_sharpe) < 1e-10:
        sensitivity_score = 99.0
    else:
        sensitivity_score = min(std_sharpe / abs(mean_sharpe), 99.0)

    return {
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "min_sharpe": min_sharpe,
        "max_sharpe": max_sharpe,
        "n_trials": n_trials,
        "sensitivity_score": sensitivity_score,
    }


def build_trials_matrix(
    strategy_factory: Callable,
    param_grid: dict,
    prices_df: pd.DataFrame,
    engine_kwargs: dict = {},
) -> np.ndarray:
    """Build an (n_bars-1 × n_trials) daily-return matrix from a parameter sweep.

    Runs the full backtest for every combination in param_grid and collects the
    daily return series for each trial. The resulting matrix has one column per
    valid (non-skipped) trial, aligned to a common time axis. The column order
    matches itertools.product(*param_grid.values()), which is the same
    deterministic order used by parameter_sweep — so trial k in this matrix
    corresponds to trial k in the sensitivity summary.

    Args:
        strategy_factory: Callable that accepts a single dict of parameters and
            returns a strategy instance ready to pass to engine.backtest.run.
        param_grid: Dict mapping parameter names to lists of candidate values,
            as returned by build_param_grid.
        prices_df: OHLCV DataFrame with DatetimeIndex; passed verbatim to
            engine.backtest._run_internal for each combination.
        engine_kwargs: Optional config dict forwarded to _run_internal.

    Returns:
        np.ndarray of shape (n_bars-1, n_trials) where n_bars = len(prices_df).
        Returns an empty array of shape (n_bars-1, 0) when all trials fail.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    config = engine_kwargs if engine_kwargs else None
    n_bars = len(prices_df)

    trial_cols: list[np.ndarray] = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        try:
            strategy = strategy_factory(params)
        except (ValueError, TypeError):
            continue
        try:
            equity, _, _, _ = _run_internal(strategy, prices_df, config)
            returns = equity.pct_change().dropna().values
            if len(returns) == n_bars - 1:
                trial_cols.append(returns)
        except Exception:
            continue

    if not trial_cols:
        return np.empty((n_bars - 1, 0))

    return np.column_stack(trial_cols)
