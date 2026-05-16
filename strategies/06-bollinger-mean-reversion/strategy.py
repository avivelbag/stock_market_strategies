"""Bollinger Band Mean-Reversion strategy.

Thesis: price that closes below the lower Bollinger Band has deviated more than
``nstd`` standard deviations from its recent mean — a distributional overextension
that tends to revert.  Bollinger & (2001) codified this signal; the statistical
justification is that daily log-returns are approximately i.i.d. over short windows,
so a close at z-score < -2 is an improbable observation under the local null and
often precedes a reversal.

Differs from 02-rsi-mean-reversion: RSI is a momentum-oscillator (rank of close
relative to recent highs/lows); Bollinger Bands are a distributional signal (z-score
from rolling mean).  Both exploit crowd overreaction, but the entry condition is
continuous and volatility-normalised here, versus nonlinear and bounded in RSI-2.
"""

import pandas as pd

DEFAULT_PARAMS = {"window": 20, "nstd": 2.0, "exit_window": 20}


class BollingerMeanReversion:
    """Long-only mean-reversion strategy using Bollinger Band entry/middle-band exit.

    Entry: close at bar t is below the lower Bollinger Band
        ``mean(close, window) - nstd * std(close, window)``.
    Exit: close at bar t is above the middle band (simple moving average
        ``mean(close, exit_window)``).

    Position is held between entry and exit signals.  No short side — unlimited-risk
    argument avoided; ThinkorSwim port kept simple.

    Default parameters are not grid-searched from the synthetic datasets.  Window=20
    and nstd=2.0 are the Bollinger (2001) published defaults that have been the
    market-practitioner standard for over two decades.

    Args:
        window: Lookback period for the rolling mean and standard deviation used
            at entry (lower-band computation).  Default 20.
        nstd: Number of standard deviations below the mean that defines the lower
            band.  Default 2.0 (Bollinger, 2001 standard).
        exit_window: Lookback period for the rolling mean used at exit (middle
            band).  Default 20 — shares the same band as entry.

    Raises:
        ValueError: If window or exit_window is not a positive integer, or if
            nstd is not a positive number.
    """

    def __init__(
        self,
        window: int = 20,
        nstd: float = 2.0,
        exit_window: int = 20,
    ):
        if window <= 0:
            raise ValueError("window must be a positive integer")
        if exit_window <= 0:
            raise ValueError("exit_window must be a positive integer")
        if nstd <= 0:
            raise ValueError("nstd must be a positive number")
        self.window = window
        self.nstd = nstd
        self.exit_window = exit_window
        self._in_position = False

    def __call__(self, view) -> float:
        """Compute position signal for the current bar.

        State is updated in-place: the instance tracks whether a long position
        is currently held across sequential bar-by-bar calls from the engine.

        Args:
            view: A guarded DataFrame slice with OHLCV columns up to and including
                the current bar t.  Must support ``view["close"]`` returning a
                ``pd.Series``.

        Returns:
            1.0 if in a long position, 0.0 otherwise.
        """
        closes = view["close"]
        if len(closes) < self.window:
            return 0.0

        rolling_mean = closes.rolling(self.window).mean().iloc[-1]
        rolling_std = closes.rolling(self.window).std(ddof=1).iloc[-1]

        if pd.isna(rolling_mean) or pd.isna(rolling_std) or rolling_std == 0:
            return 0.0

        lower_band = rolling_mean - self.nstd * rolling_std

        exit_bars = max(self.exit_window, self.window)
        if len(closes) >= exit_bars:
            middle_band = closes.rolling(exit_bars).mean().iloc[-1]
        else:
            middle_band = rolling_mean

        last_close = float(closes.iloc[-1])

        if not self._in_position and last_close < lower_band:
            self._in_position = True
        elif self._in_position and last_close > middle_band:
            self._in_position = False

        return 1.0 if self._in_position else 0.0
