"""Volatility-Managed Portfolio strategy.

Thesis: Moreira & Muir (2017, Journal of Finance) document that scaling equity
exposure inversely to recent realized variance harvests the variance risk premium.
Risk-averse investors demand a higher expected return to hold equity during high-
variance periods; by reducing size when markets are volatile, this strategy earns
that premium at low realized cost and holds full (or leveraged) size when markets
are calm. The edge is in optimal position sizing, not direction timing.

Differs from strategy 08 (NR7): NR7 seeks breakout entries after compression
episodes; this strategy continuously scales a constant-long-equity exposure by
the ratio of target volatility to realized volatility. Neither strategy takes
short positions, but NR7 is a signal-timing strategy while this is a risk-sizing
strategy.

Differs from strategy 07 (Absolute Momentum): absolute momentum exits the market
when trailing return is negative (crash filter); this strategy stays long but
reduces leverage proportionally to realized variance. Both are long-only; they
compose naturally as a combined overlay.
"""

import numpy as np

DEFAULT_PARAMS = {"window": 21, "target_vol": 0.12}


class VolatilityManagedPortfolio:
    """Volatility-managed long-only position sizing strategy (Moreira & Muir 2017).

    At each bar t, computes a position scalar:

        scalar = clip(target_vol / realized_vol, 0, 2)

    where realized_vol is the annualized standard deviation of daily returns over
    the prior ``window`` bars (lagged: the current bar's return is excluded).
    The scalar represents the desired fractional equity exposure — 1.0 is full
    exposure, 2.0 is double leverage (low-vol environment), and values approaching
    zero represent near-flat (very high-vol environment).

    Important engine note: the backtest engine maps any positive return to a full
    long position (1x) and 0.0 to flat. This means the strategy's fractional
    sizing is reduced to a binary long/flat signal in this engine — it is long
    during and after the warm-up (since scalar is always > 0 for positive target_vol
    and non-zero realized_vol). The true variance-risk-premium benefit of fractional
    position sizing requires a portfolio engine that supports non-binary position
    weights. The metrics.json therefore reflects a long-after-warmup binary
    approximation, not the true risk-managed performance.

    The two parameters are both prior-specified from Moreira & Muir (2017):
    - window=21: one calendar month of trading days (paper's baseline lookback)
    - target_vol=0.12: 12% annualized target volatility (paper's baseline)

    Args:
        window: Rolling lookback in bars for realized volatility computation.
            Default 21 is the Moreira & Muir (2017) prior-specified value (one
            calendar month). Must be at least 2 to compute a meaningful std.
        target_vol: Annualized target volatility (fractional). The position scalar
            equals target_vol / realized_vol. Default 0.12 (12% annualized) is the
            Moreira & Muir (2017) paper baseline. Must be positive.

    Raises:
        ValueError: If window < 2 or target_vol <= 0.
    """

    def __init__(self, window: int = 21, target_vol: float = 0.12):
        if window < 2:
            raise ValueError("window must be at least 2")
        if target_vol <= 0:
            raise ValueError("target_vol must be positive")
        self.window = window
        self.target_vol = target_vol

    def __call__(self, view) -> float:
        """Compute the position scalar for the current bar.

        Uses the ``window`` lagged daily returns (pct_change over bars t-window
        to t-1, exclusive of the current bar t) to compute realized annualized
        volatility. Returns 0.0 during warm-up, 1.0 when realized_vol is
        negligible, and target_vol / realized_vol clipped to [0, 2] otherwise.

        No look-ahead: the realized_vol computation uses only close prices at or
        before bar t. The engine fills any resulting position at open[t+1].

        Args:
            view: A guarded DataFrame slice with OHLCV columns up to and
                including the current bar t.

        Returns:
            Float in [0.0, 2.0]: fractional position scalar. 0.0 during warm-up.
            Positive values map to long in the binary engine.
        """
        closes = view["close"]
        n = len(closes)
        t = n - 1

        if t < self.window:
            return 0.0

        # Lagged window: bars t-window..t-1 (excludes current bar's return)
        rets = closes.pct_change().iloc[t - self.window : t]
        realized_vol = float(rets.std() * (252 ** 0.5))

        if realized_vol < 1e-9:
            return 1.0

        scalar = self.target_vol / realized_vol
        return float(np.clip(scalar, 0.0, 2.0))
