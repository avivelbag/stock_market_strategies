"""RSI Mean-Reversion strategy (Connors RSI-2 variant).

Thesis: short-term RSI extremes proxy for crowd overreaction — a behavioral
finance result (De Bondt & Thaler 1985 on reversals; Jegadeesh 1990 on
short-horizon reversals). At daily timeframes the mean-reversion effect is
strongest in the 2–10 day window. Structurally opposite to 01-dual-ema-momentum:
profits in mean-reverting regimes where momentum strategies suffer.
"""

import pandas as pd


def _rsi(closes: pd.Series, period: int) -> float:
    """Compute the most recent RSI value using Wilder's exponential smoothing.

    Wilder's smoothing is equivalent to EWM with alpha=1/period (adjust=False).
    Returns 50.0 (neutral) when insufficient history is available.

    Args:
        closes: Price series; must have len >= period + 1 for a valid result.
        period: RSI lookback period.

    Returns:
        RSI value in [0.0, 100.0], or 50.0 when insufficient history.
    """
    if len(closes) < period + 1:
        return 50.0

    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean().iloc[-1]
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean().iloc[-1]

    if pd.isna(avg_gain) or pd.isna(avg_loss):
        return 50.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


class RSIMeanReversion:
    """Long-only mean-reversion strategy based on RSI(2) extremes (Connors RSI-2).

    Enters long when RSI drops below the oversold threshold, signaling crowd
    overreaction to the downside. Exits when RSI rises above the overbought
    threshold. Between signals the position is held — this is not a
    per-bar signal strategy.

    Default parameters follow Connors & Alvarez (2009) "Short Term Trading
    Strategies That Work" and were not derived from any backtest on these datasets.

    Args:
        rsi_period: RSI lookback window (default 2, from Connors literature).
        oversold: RSI level below which a long entry is signaled (default 10).
        overbought: RSI level above which the long position is exited (default 90).

    Raises:
        ValueError: If parameters are out of valid range.
    """

    def __init__(
        self,
        rsi_period: int = 2,
        oversold: float = 10.0,
        overbought: float = 90.0,
    ):
        if rsi_period <= 0:
            raise ValueError("rsi_period must be a positive integer")
        if not (0 <= oversold < overbought <= 100):
            raise ValueError(
                "oversold and overbought must satisfy 0 <= oversold < overbought <= 100"
            )
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self._in_position = False

    def __call__(self, view) -> float:
        """Compute the position signal for the current bar.

        State is updated in-place: the instance tracks whether a long position
        is currently held across sequential bar-by-bar calls from the engine.

        Args:
            view: A guarded DataFrame slice with OHLCV columns up to and including
                the current bar t. Must support view["close"] returning a pd.Series.

        Returns:
            1.0 if in a long position, 0.0 otherwise.
        """
        closes = view["close"]
        if len(closes) < self.rsi_period + 1:
            return 0.0

        rsi_val = _rsi(closes, self.rsi_period)

        if not self._in_position and rsi_val < self.oversold:
            self._in_position = True
        elif self._in_position and rsi_val > self.overbought:
            self._in_position = False

        return 1.0 if self._in_position else 0.0
