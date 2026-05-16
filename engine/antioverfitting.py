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

from itertools import combinations

import numpy as np
import pandas as pd

from engine import backtest as _backtest
from engine import metrics as _metrics

_ANNUALIZE = 252


def _col_sharpes(data: np.ndarray) -> np.ndarray:
    """Annualized Sharpe ratio for each column of a 2D daily-return array.

    Uses ddof=1 standard deviation. Columns with near-zero std return 0.0 to
    avoid division-by-zero — a degenerate trial contributes nothing to ranking.
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=1)
    std = np.where(std < 1e-12, 1.0, std)
    return mean / std * np.sqrt(_ANNUALIZE)


def pbo(trials_matrix: np.ndarray, n_splits: int = 16) -> float:
    """Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

    PBO answers: given that we selected the best-performing trial in-sample (IS),
    how often does that trial underperform the median trial out-of-sample (OOS)?
    A PBO of 0.0 means the IS-winner is always the OOS-winner — selection is
    perfectly informative. A PBO of 0.5 means IS selection is no better than
    chance — the strategy was likely chosen by luck from the trial space.

    This measure resists gaming unlike raw Sharpe: inflating the best Sharpe by
    adding more trials raises PBO rather than lowering it, because more diverse
    trials make the IS-optimal trial less likely to win OOS as well.

    Implements CSCV (Bailey & López de Prado 2014, "The Deflated Sharpe Ratio"):
    1. Split the T-bar return series into n_splits equal sub-periods.
    2. For each of C(n_splits, n_splits//2) complementary IS/OOS partitions:
       a. IS: concatenation of n_splits//2 sub-periods; OOS: the remaining half.
       b. Identify k* = trial with highest IS Sharpe.
       c. Compute OOS Sharpe for all trials; count how many beat k* OOS.
       d. If k*'s OOS rank is in the bottom half (≥ n_trials/2 trials beat it),
          increment the underperform counter.
    3. PBO = underperform_count / total_splits.

    The split enumeration order follows itertools.combinations(range(n_splits),
    n_splits//2), which is fully deterministic — results are byte-for-byte
    reproducible across Python versions that preserve itertools ordering.

    Args:
        trials_matrix: (n_bars × n_trials) array of daily returns. Each column
            is one trial (parameter combination or bootstrap realization). Must
            have at least n_splits rows and at least 2 columns.
        n_splits: Number of equal-length sub-periods. Must be even. Default 16
            follows Bailey & López de Prado (2014) stability recommendation.

    Returns:
        PBO in [0.0, 1.0]. Returns 0.0 when trials_matrix has fewer than
        n_splits rows (too short for meaningful sub-period splits) or fewer
        than 2 columns (single trial cannot be ranked).
    """
    matrix = np.asarray(trials_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    n_bars, n_trials = matrix.shape

    if n_trials < 2 or n_bars < n_splits:
        return 0.0

    if n_splits % 2 != 0:
        n_splits = n_splits - 1

    sub_size = n_bars // n_splits
    trim = sub_size * n_splits
    matrix = matrix[:trim]

    sub_periods = matrix.reshape(n_splits, sub_size, n_trials)

    half = n_splits // 2
    all_is = list(range(n_splits))

    n_underperform = 0
    n_total = 0

    for is_idx in combinations(all_is, half):
        is_set = set(is_idx)
        oos_idx = [i for i in all_is if i not in is_set]

        is_arr = sub_periods[list(is_idx)].reshape(-1, n_trials)
        oos_arr = sub_periods[oos_idx].reshape(-1, n_trials)

        is_sharpes = _col_sharpes(is_arr)
        k_star = int(np.argmax(is_sharpes))

        oos_sharpes = _col_sharpes(oos_arr)
        n_better_oos = int(np.sum(oos_sharpes > oos_sharpes[k_star]))

        if n_better_oos >= n_trials / 2:
            n_underperform += 1
        n_total += 1

    return float(n_underperform / n_total)


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
    equity_series, _gross, positions_series, risk_free_rate = _backtest._run_internal(
        strategy_fn, prices_df, config
    )
    result = _metrics.compute_all(equity_series, positions_series, risk_free_rate)
    result["deflated_sharpe"] = compute_dsr(equity_series, n_trials)
    return result
