"""Absolute Momentum (Trend Filter) strategy.

Thesis: Jegadeesh and Titman (1993) documented that intermediate-horizon (3–12 month)
past returns predict future returns. Antonacci (2014) showed that absolute momentum —
whether the asset's own trailing return is positive, not relative to peers — is a
reliable crash filter. The mechanism: investors underreact to slow-moving fundamental
deterioration (they hold losers too long due to loss aversion), creating positive
autocorrelation at intermediate horizons as prices gradually adjust. The same loss
aversion causes overreaction at long horizons, producing negative autocorrelation.

Differs from 01-dual-ema-momentum (which uses a fast/slow EMA crossover within the
same asset) and 03-donchian-turtle-breakout (which uses a channel breakout above the
N-bar high): absolute momentum uses a single trailing return comparison against a
fixed threshold, making it the simplest trend filter in the library with the fewest
free parameters.

This strategy acts as a crash filter: it exits bear markets early enough to avoid
most of the drawdown, at the cost of some bull-market whipsaw in choppy regimes.
"""

DEFAULT_PARAMS = {"lookback": 252, "threshold": 0.0}


class AbsoluteMomentum:
    """Long-only absolute momentum strategy: long when trailing return exceeds threshold, else flat.

    At each bar t, the trailing return is:
        trailing_return = close[t] / close[max(0, t - lookback)] - 1

    If trailing_return > threshold the strategy goes long (position = 1.0);
    otherwise it holds cash (position = 0.0). No short positions are taken.

    When fewer than ``lookback`` bars of history are available (t < lookback), the
    reference price is close[0], so the trailing return is computed over all available
    history. This avoids the warm-up period that longer-lookback strategies suffer.

    The lookback default of 252 bars (one trading year) is the canonical value from
    Jegadeesh and Titman (1993), independently replicated across asset classes. The
    threshold default of 0.0 means "any positive trailing return triggers a long
    position." Both defaults are prior-specified, not derived from the test datasets.

    Args:
        lookback: Number of bars used to compute the trailing return. Default 252.
            Very short lookbacks (< 20 bars) degrade to noise; very long lookbacks
            (> 500 bars) react too slowly to regime changes.
        threshold: Minimum trailing return required to go long. Default 0.0 (any
            positive trailing return triggers a position). Higher values require
            stronger momentum before entering.

    Raises:
        ValueError: If lookback is not a positive integer.
    """

    def __init__(self, lookback: int = 252, threshold: float = 0.0):
        if lookback <= 0:
            raise ValueError("lookback must be a positive integer")
        self.lookback = lookback
        self.threshold = threshold

    def __call__(self, view) -> float:
        """Compute position signal for the current bar.

        The signal is stateless — it depends only on the trailing return computable
        at bar t, with no memory of prior position state. The engine handles
        position transitions and associated transaction costs.

        Args:
            view: A guarded DataFrame slice with OHLCV columns up to and including
                the current bar t. Must support view["close"] returning a pd.Series.

        Returns:
            1.0 if trailing_return > threshold, 0.0 otherwise.
        """
        closes = view["close"]
        if len(closes) < 1:
            return 0.0

        t = len(closes) - 1
        ref_idx = max(0, t - self.lookback)
        ref_close = float(closes.iloc[ref_idx])

        if ref_close == 0.0:
            return 0.0

        trailing_return = float(closes.iloc[-1]) / ref_close - 1.0
        return 1.0 if trailing_return > self.threshold else 0.0
