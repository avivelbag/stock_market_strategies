"""Backtest engine: runs a strategy function against OHLCV price data.

Usage (module entrypoint):
    python -m engine.backtest

Usage (as library):
    from engine.backtest import run
    metrics = run(my_strategy_fn, prices_df, config)
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


def run(strategy_fn, prices_df: pd.DataFrame, config: dict = None) -> dict:
    """Run a vectorized backtest with t→t+1 fill and look-ahead protection.

    Signal semantics:
        - Signal at bar t (computed from close[t] and earlier) is filled at
          open[t+1] — the strategy cannot profit from knowing open[t+1] in advance.
        - The strategy receives ``prices_df.iloc[:t+1]`` wrapped in a guard that
          raises ``LookAheadError`` if the strategy tries to access bar t+1 or beyond.

    Return model:
        Close-to-close returns are used (standard academic approximation). The
        slippage term partially compensates for the actual open[t+1] fill price.
        equity[t+1] = equity[t] × (1 + position × bar_return − trading_cost)

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

        bar_return = (closes[t + 1] - closes[t]) / closes[t]

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

    return _metrics.compute_all(equity_series, positions_series, risk_free_rate)


if __name__ == "__main__":
    print("engine.backtest: backtest runner")
    print("Usage: python -m engine.backtest <strategy_module> <data_file> [config_file]")
    print("No strategies configured — pass a strategy module to run.")
    sys.exit(0)
