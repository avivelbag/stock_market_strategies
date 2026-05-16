"""Convenience wrappers for anti-overfitting metrics.

The Deflated Sharpe Ratio requires n_trials (the number of independent parameter
combinations or strategies evaluated before selecting the winner), which is
caller-supplied context that the engine cannot infer automatically. This module
exposes helpers that pass n_trials=1 as the conservative lower bound for a single
prior-specified strategy — meaning no multiple-testing penalty is applied, but the
finite-sample and non-normality corrections still hold.

Pass n_trials > 1 whenever multiple parameter sets or strategies were evaluated and
this equity curve belongs to the one that was selected as the best.
"""

import pandas as pd

from engine import backtest as _backtest
from engine import metrics as _metrics


def compute_dsr(equity: pd.Series, n_trials: int = 1) -> float:
    """Compute the Deflated Sharpe Ratio for a pre-computed equity curve.

    Convenience wrapper that calls sharpe_distribution_stats then deflated_sharpe
    so the caller only needs to supply the equity series and n_trials.

    Args:
        equity: Portfolio value series (at least 2 points).
        n_trials: Number of strategies or parameter sets evaluated. Default 1
            applies no multiple-testing correction (conservative lower bound).

    Returns:
        DSR in [0, 1].
    """
    skewness, kurtosis = _metrics.sharpe_distribution_stats(equity)
    return _metrics.deflated_sharpe(equity, n_trials, skewness, kurtosis)


def run_with_dsr(
    strategy_fn,
    prices_df: pd.DataFrame,
    config: dict = None,
    n_trials: int = 1,
) -> dict:
    """Run a backtest and append the Deflated Sharpe Ratio to the metrics dict.

    Like engine.backtest.run(), but also computes deflated_sharpe with the given
    n_trials and includes it in the returned dict under the key "deflated_sharpe".
    compute_all() is not modified so the core backtest API stays stable.

    Args:
        strategy_fn: Strategy callable (same contract as backtest.run).
        prices_df: OHLCV DataFrame with DatetimeIndex.
        config: Optional backtest config dict.
        n_trials: Number of strategies/parameter sets evaluated. Default 1.

    Returns:
        Dict from compute_all() plus key "deflated_sharpe".
    """
    equity_series, positions_series, risk_free_rate = _backtest._run_internal(
        strategy_fn, prices_df, config
    )
    result = _metrics.compute_all(equity_series, positions_series, risk_free_rate)
    result["deflated_sharpe"] = compute_dsr(equity_series, n_trials)
    return result
