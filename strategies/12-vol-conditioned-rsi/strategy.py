"""Volatility-Conditioned RSI Mean-Reversion strategy (Strategy 12).

Thesis: De Bondt & Thaler (1985) and Jegadeesh (1990) document that mean-reversion
is strongest following overreaction episodes. Overreaction is amplified when
uncertainty is high (elevated realized volatility). Strategy 02 (RSI-2) enters
mean-reversion trades regardless of volatility regime. This strategy synthesizes
RSI-2 (Connors 2009) with the volatility-regime awareness of strategy 09
(Moreira & Muir 2017): RSI-2 entries are only taken when realized volatility is
in the top quartile of its trailing 252-bar distribution.

The falsifiable hypothesis: the RSI-2 mean-reversion edge is concentrated in
high-volatility regimes where overreaction is most extreme.

Relates to:
  - 02-rsi-mean-reversion: same RSI thresholds; adds a vol-regime gate
  - 09-volatility-managed: shares the 21-bar realized vol computation
  - 10-low-volatility-anomaly: inverse thesis — this strategy ENTERS in high vol
"""

import pandas as pd

DEFAULT_PARAMS = {
    "vol_window": 21,
    "vol_lookback": 252,
    "vol_threshold": 0.75,
    "rsi_window": 2,
    "rsi_entry_long": 10.0,
    "rsi_exit_long": 70.0,
    "rsi_entry_short": 90.0,
    "rsi_exit_short": 30.0,
}


def _rsi(closes: pd.Series, period: int) -> float:
    """Compute the most recent RSI value using Wilder's exponential smoothing.

    Identical to strategy 02 implementation: Wilder smoothing is EWM with
    alpha=1/period (adjust=False). Returns 50.0 when insufficient history.

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


class VolConditionedRSI:
    """Long/short RSI(2) mean-reversion gated by a realized-volatility percentile filter.

    Mechanism: two-step filter before entry.

    Step 1 — vol regime filter: compute 21-bar annualized realized vol; rank it
    against its own trailing 252-bar history. Entry is only allowed when the
    percentile rank is >= vol_threshold (default 0.75 = top quartile). Once in a
    position, the vol filter does NOT apply to exits — the exit is RSI-only. This
    avoids the pathological case where a mean-reversion trade is closed out simply
    because vol normalizes, even if price has not yet reverted.

    Step 2 — RSI(2) signal: the standard Connors (2009) RSI-2 thresholds applied
    to both long and short sides.

    Parameter provenance:
      Prior-specified (Connors 2009): rsi_window=2, rsi_entry_long=10,
        rsi_exit_long=70, rsi_entry_short=90, rsi_exit_short=30.
      Prior-specified (Moreira & Muir 2017): vol_window=21.
      Prior-specified (standard percentile lookback): vol_lookback=252.
      Single free choice: vol_threshold=0.75 (Engle 2004 high-vol definition).

    Position semantics (engine must be run with allow_short=True):
        +1.0  long  — RSI-2 oversold in high-vol regime
        -1.0  short — RSI-2 overbought in high-vol regime
         0.0  flat  — no signal or warm-up period

    Args:
        vol_window: Bars for rolling annualized realized vol. Default 21.
        vol_lookback: Bars for rolling percentile rank of vol. Default 252.
        vol_threshold: Minimum vol percentile for entry. Default 0.75 (top quartile).
        rsi_window: RSI lookback period. Default 2 (Connors 2009).
        rsi_entry_long: Enter long when RSI < this. Default 10 (Connors 2009).
        rsi_exit_long: Exit long when RSI > this. Default 70 (Connors 2009).
        rsi_entry_short: Enter short when RSI > this. Default 90 (Connors 2009).
        rsi_exit_short: Exit short when RSI < this. Default 30 (Connors 2009).

    Raises:
        ValueError: If parameters are out of valid range.
    """

    def __init__(
        self,
        vol_window: int = 21,
        vol_lookback: int = 252,
        vol_threshold: float = 0.75,
        rsi_window: int = 2,
        rsi_entry_long: float = 10.0,
        rsi_exit_long: float = 70.0,
        rsi_entry_short: float = 90.0,
        rsi_exit_short: float = 30.0,
    ):
        if vol_window < 2:
            raise ValueError("vol_window must be at least 2")
        if vol_lookback < 1:
            raise ValueError("vol_lookback must be at least 1")
        if not (0.0 < vol_threshold < 1.0):
            raise ValueError("vol_threshold must be in (0, 1)")
        if rsi_window < 1:
            raise ValueError("rsi_window must be at least 1")
        if not (0.0 < rsi_entry_long < rsi_exit_long < rsi_entry_short < 100.0):
            raise ValueError(
                "RSI thresholds must satisfy 0 < rsi_entry_long < rsi_exit_long"
                " < rsi_entry_short < 100"
            )
        if not (0.0 < rsi_exit_short < rsi_entry_short):
            raise ValueError(
                "rsi_exit_short must satisfy 0 < rsi_exit_short < rsi_entry_short"
            )
        self.vol_window = vol_window
        self.vol_lookback = vol_lookback
        self.vol_threshold = vol_threshold
        self.rsi_window = rsi_window
        self.rsi_entry_long = rsi_entry_long
        self.rsi_exit_long = rsi_exit_long
        self.rsi_entry_short = rsi_entry_short
        self.rsi_exit_short = rsi_exit_short
        self._position = 0  # -1 = short, 0 = flat, 1 = long

    def __call__(self, view) -> float:
        """Compute the position signal for the current bar.

        The vol percentile is computed as: what fraction of the last vol_lookback
        non-NaN vol values are <= the current vol. This is equivalent to
        pd.Series.rolling(vol_lookback).rank(pct=True) but O(vol_lookback) per bar
        instead of O(n * vol_lookback), matching the efficiency of strategy 10.

        The exit condition is checked before entry so that a single bar can
        transition from long to short (or short to long) without a separate flat bar
        in between — consistent with the spec's "(and flat/long)" annotation.

        Args:
            view: A guarded DataFrame slice with OHLCV columns up to and
                including the current bar t.

        Returns:
            1.0 if long, -1.0 if short, 0.0 if flat. Negative values are only
            acted upon by the backtest engine when allow_short=True is set.
        """
        closes = view["close"]

        rets = closes.pct_change()
        rolling_vol = rets.rolling(self.vol_window).std() * (252 ** 0.5)

        # Build the vol history for percentile ranking from non-NaN values only,
        # taking the last vol_lookback of them. Need vol_lookback valid vol values.
        vol_series = rolling_vol.dropna()
        high_vol = False
        if len(vol_series) >= self.vol_lookback:
            vol_history = vol_series.iloc[-self.vol_lookback:]
            current_vol = float(vol_history.iloc[-1])
            hist_vals = vol_history.values
            # Average-method rank: consistent with pd.Series.rolling().rank(pct=True).
            # Strictly-below count + half the tied count (including self), divided by n.
            strictly_less = float((hist_vals < current_vol).sum())
            tied = float((hist_vals == current_vol).sum())
            vol_pct = (strictly_less + (tied + 1.0) / 2.0) / len(vol_history)
            high_vol = vol_pct >= self.vol_threshold

        rsi_val = _rsi(closes, self.rsi_window)

        # Exit conditions: RSI-only, no vol filter (avoid premature exits on vol drop)
        if self._position == 1 and rsi_val > self.rsi_exit_long:
            self._position = 0
        if self._position == -1 and rsi_val < self.rsi_exit_short:
            self._position = 0

        # Entry conditions: both vol regime and RSI must confirm
        if self._position != 1 and high_vol and rsi_val < self.rsi_entry_long:
            self._position = 1
        elif self._position != -1 and high_vol and rsi_val > self.rsi_entry_short:
            self._position = -1

        return float(self._position)
