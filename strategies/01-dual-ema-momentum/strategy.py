"""Dual EMA Crossover Momentum strategy.

Signal: long (1.0) when EMA(fast_window) > EMA(slow_window), flat (0.0) otherwise.
The crossover detects when short-term momentum has overtaken long-term trend, capturing
the momentum risk premium documented by Jegadeesh & Titman (1993).
"""



class DualEMAMomentum:
    """Long-only momentum strategy based on dual exponential moving average crossover.

    Generates a long signal when the fast EMA crosses above the slow EMA, indicating
    that recent price momentum is outpacing the longer-term trend. Returns to flat
    (no position) when the fast EMA falls back below the slow EMA.

    Args:
        fast_window: EMA span for the short-term trend (default 20 days).
        slow_window: EMA span for the long-term trend (default 60 days).

    Raises:
        ValueError: If fast_window >= slow_window or either window is non-positive.
    """

    def __init__(self, fast_window: int = 20, slow_window: int = 60):
        if fast_window <= 0 or slow_window <= 0:
            raise ValueError("fast_window and slow_window must be positive integers")
        if fast_window >= slow_window:
            raise ValueError("fast_window must be strictly less than slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def __call__(self, view) -> float:
        """Compute the position signal for the current bar.

        Args:
            view: A guarded DataFrame slice with OHLCV columns up to and including
                the current bar t. Must support view["close"] returning a pd.Series.

        Returns:
            1.0 if EMA(fast) > EMA(slow), 0.0 otherwise.
            Returns 0.0 (flat) when fewer bars are available than slow_window,
            since the slow EMA has not yet reached its effective lookback.
        """
        closes = view["close"]
        if len(closes) < self.slow_window:
            return 0.0
        ema_fast = closes.ewm(span=self.fast_window, adjust=False).mean().iloc[-1]
        ema_slow = closes.ewm(span=self.slow_window, adjust=False).mean().iloc[-1]
        return 1.0 if ema_fast > ema_slow else 0.0
