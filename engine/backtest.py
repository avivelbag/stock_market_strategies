"""Backtest engine: runs a strategy function against OHLCV price data.

Fill model: a signal computed from data up to and including close[t] is executed
at open[t+1]. The strategy captures the open[t+1]-to-close[t+1] intraday return,
not the overnight gap from close[t] to open[t+1].

Usage (module entrypoint):
    python -m engine.backtest

Usage (as library):
    from engine.backtest import run, walk_forward_backtest
    metrics = run(my_strategy_fn, prices_df, config)
    oos = walk_forward_backtest(MyStrategyClass, params, prices_df)
"""

import sys

import numpy as np
import pandas as pd

from engine import metrics as _metrics


class LookAheadError(Exception):
    """Raised when a strategy attempts to access future price data."""


class _GuardedIloc:
    """Intercepts iloc access and converts out-of-bounds IndexError to LookAheadError."""

    def __init__(self, iloc_indexer):
        self._iloc = iloc_indexer

    def __getitem__(self, key):
        try:
            return self._iloc[key]
        except IndexError:
            raise LookAheadError(
                "Strategy accessed a bar beyond the current time step. "
                "Signals must be computed from prices up to and including bar t only."
            ) from None


class _LookAheadGuardedFrame:
    """Wraps a DataFrame slice to raise LookAheadError on out-of-bounds iloc access.

    The strategy receives this object instead of a raw DataFrame. All attribute
    access is delegated to the underlying slice. If the strategy tries to index
    beyond the slice boundary via .iloc[], LookAheadError is raised instead of
    a plain IndexError, making the violation explicit.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __getattr__(self, name: str):
        return getattr(self._df, name)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, key):
        return self._df[key]

    @property
    def iloc(self):
        return _GuardedIloc(self._df.iloc)

    @property
    def loc(self):
        return self._df.loc

    @property
    def values(self):
        return self._df.values

    @property
    def index(self):
        return self._df.index

    @property
    def columns(self):
        return self._df.columns


def _run_internal(
    strategy_fn, prices_df: pd.DataFrame, config: dict = None
) -> tuple:
    """Core backtest computation shared by run() and antioverfitting helpers.

    Validates inputs, simulates the t→t+1 fill model, and returns the raw
    equity and position series so callers can apply any metrics they need.

    Args:
        strategy_fn: Strategy callable (same contract as run()).
        prices_df: OHLCV DataFrame with DatetimeIndex.
        config: Optional backtest config dict.

    Returns:
        (equity_series, positions_series, risk_free_rate) — all needed by
        metrics.compute_all() and by metrics.deflated_sharpe().

    Raises:
        LookAheadError: If the strategy accesses prices beyond the current bar.
        ValueError: If prices_df has fewer than 2 rows or missing required columns.
    """
    cfg = config or {}
    commission_frac = cfg.get("commission_bps", 5) / 10_000
    slippage_frac = cfg.get("slippage_bps", 5) / 10_000
    allow_short = cfg.get("allow_short", False)
    initial_capital = float(cfg.get("initial_capital", 10_000))
    risk_free_rate = float(cfg.get("risk_free_rate", 0.0))

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(prices_df.columns)
    if missing:
        raise ValueError(f"prices_df missing columns: {missing}")
    if len(prices_df) < 2:
        raise ValueError("prices_df must have at least 2 rows")

    n = len(prices_df)
    closes = prices_df["close"].values.astype(float)
    opens = prices_df["open"].values.astype(float)

    equity = np.empty(n, dtype=float)
    equity[0] = initial_capital
    pos_arr = np.zeros(n, dtype=int)

    prev_pos = 0

    for t in range(n - 1):
        view = _LookAheadGuardedFrame(prices_df.iloc[: t + 1])
        raw = strategy_fn(view)

        if raw > 0:
            target = 1
        elif raw < 0 and allow_short:
            target = -1
        else:
            target = 0

        bar_return = (closes[t + 1] - opens[t + 1]) / opens[t + 1]

        if target != prev_pos:
            cost = abs(target - prev_pos) * (commission_frac + slippage_frac)
        else:
            cost = 0.0

        equity[t + 1] = equity[t] * (1.0 + target * bar_return - cost)
        pos_arr[t] = target
        prev_pos = target

    pos_arr[n - 1] = prev_pos

    equity_series = pd.Series(equity, index=prices_df.index)
    positions_series = pd.Series(pos_arr, index=prices_df.index)
    return equity_series, positions_series, risk_free_rate


def run(strategy_fn, prices_df: pd.DataFrame, config: dict = None) -> dict:
    """Run a vectorized backtest with t→t+1 fill and look-ahead protection.

    Signal semantics:
        - Signal at bar t (computed from close[t] and earlier) is filled at
          open[t+1] — the strategy cannot profit from knowing open[t+1] in advance.
        - The strategy receives ``prices_df.iloc[:t+1]`` wrapped in a guard that
          raises ``LookAheadError`` if the strategy tries to access bar t+1 or beyond.

    Return model:
        Open[t+1]-to-close[t+1] returns capture the intraday move from the fill price.
        equity[t+1] = equity[t] × (1 + position × (close[t+1]−open[t+1])/open[t+1] − cost)

    Cost model:
        trading_cost = |Δposition| × (commission_frac + slippage_frac)
        Costs are expressed as a fraction of current portfolio value.

    Args:
        strategy_fn: callable(view: _LookAheadGuardedFrame) → float
            Receives a guarded view of prices up to and including bar t.
            Return value: positive = long, negative = short (if allow_short),
            zero = flat. Any non-zero magnitude is treated as a full position.
        prices_df: pd.DataFrame with columns [open, high, low, close, volume]
            and a DatetimeIndex. Must have at least 2 rows.
        config: dict with optional keys:
            ``commission_bps`` (int, default 5): one-way commission in basis points.
            ``slippage_bps`` (int, default 5): half-spread slippage in basis points.
            ``allow_short`` (bool, default False): allow negative position signals.
            ``initial_capital`` (float, default 10_000): starting portfolio value.
            ``risk_free_rate`` (float, default 0.0): annualized risk-free rate.

    Returns:
        Dict of metric scalars from engine.metrics (see metrics.compute_all).
        Does not include deflated_sharpe — use engine.antioverfitting.run_with_dsr
        for a result that also includes the DSR.

    Raises:
        LookAheadError: If the strategy accesses prices beyond the current bar.
        ValueError: If prices_df has fewer than 2 rows or missing required columns.
    """
    equity_series, positions_series, risk_free_rate = _run_internal(
        strategy_fn, prices_df, config
    )
    return _metrics.compute_all(equity_series, positions_series, risk_free_rate)


def walk_forward_backtest(
    strategy_cls,
    params_default: dict,
    prices_df: pd.DataFrame,
    n_splits: int = 5,
    train_frac: float = 0.7,
    config: dict = None,
) -> dict:
    """Honest OOS evaluation: fixed params across all folds, no per-fold re-fitting.

    Partitions prices_df into n_splits anchored (expanding-train) windows and
    evaluates strategy_cls(**params_default) on each held-out test slice.

    Why no re-fitting per fold: params_default are the author's prior, not values
    derived from the data in each window. Re-fitting per fold would conflate OOS
    evaluation with in-sample optimisation — a strategy that needs re-tuning every
    sub-period does not have a robust edge; it is curve-fitting each sub-period.

    Args:
        strategy_cls: Callable returning a strategy callable when invoked with
            **params_default (e.g. a class with __call__, or a factory function).
        params_default: Fixed parameter dict used unchanged on every fold.
        prices_df: Full price history with DatetimeIndex and OHLCV columns.
        n_splits: Number of OOS folds.
        train_frac: Upper bound on the training window as a fraction of the series.
        config: Optional backtest config dict (commission_bps, slippage_bps, etc.).

    Returns:
        Dict with keys:
            ``oos_sharpe_mean``: mean Sharpe across OOS folds.
            ``oos_sharpe_std``: std of Sharpe across OOS folds.
            ``oos_cagr_mean``: mean CAGR across OOS folds.
            ``oos_max_drawdown_mean``: mean max drawdown across OOS folds.
            ``oos_consistency``: fraction of OOS folds with positive Sharpe.
    """
    n = len(prices_df)
    split_size = n // (n_splits + 1)
    fold_results = []

    for i in range(1, n_splits + 1):
        train_end = int(n * train_frac * (i / n_splits))
        test_start = train_end
        test_end = test_start + split_size
        if test_end > n:
            break
        oos_prices = prices_df.iloc[test_start:test_end]
        if len(oos_prices) < 2:
            continue
        strategy = strategy_cls(**params_default)
        fold_results.append(run(strategy, oos_prices, config))

    if not fold_results:
        return {
            "oos_sharpe_mean": 0.0,
            "oos_sharpe_std": 0.0,
            "oos_cagr_mean": 0.0,
            "oos_max_drawdown_mean": 0.0,
            "oos_consistency": 0.0,
        }

    sharpes = [m["sharpe"] for m in fold_results]
    cagrs = [m["cagr"] for m in fold_results]
    drawdowns = [m["max_drawdown"] for m in fold_results]

    return {
        "oos_sharpe_mean": float(np.mean(sharpes)),
        "oos_sharpe_std": float(np.std(sharpes)),
        "oos_cagr_mean": float(np.mean(cagrs)),
        "oos_max_drawdown_mean": float(np.mean(drawdowns)),
        "oos_consistency": _metrics.walk_forward_consistency(sharpes),
    }


if __name__ == "__main__":
    print("engine.backtest: backtest runner")
    print("Usage: python -m engine.backtest <strategy_module> <data_file> [config_file]")
    print("No strategies configured — pass a strategy module to run.")
    sys.exit(0)
