"""Statistical Pairs Mean-Reversion strategy (Gatev, Goetzmann & Rouwenhorst 2006).

Thesis: two co-integrated assets share a common non-stationary factor. Their
log-price spread (log A − log B) is stationary — it oscillates around a
long-run equilibrium with mean-reverting dynamics. When the spread deviates
beyond z_entry standard deviations from its rolling mean, the co-integration
relationship implies reversion, providing a tradeable signal. This is a
portfolio-level structural thesis distinct from single-asset mean-reversion:
the equilibrium is theoretically grounded in the co-integration vector, not
merely in a distributional or momentum argument about one price series.

Contrast with single-asset counterparts in this library:
  - 02-rsi-mean-reversion: exploits behavioral overreaction in a *single*
    series via a momentum oscillator (RSI); no equilibrium relationship.
  - 06-bollinger-mean-reversion: exploits distributional overextension in a
    *single* series via a z-score from rolling mean; also no equilibrium.
  Both are agnostic to a second asset. This strategy's edge requires the
  co-integration structure — a collapsed co-integration relationship (regime
  break) immediately invalidates the signal.

Parameters follow Gatev et al. (2006, Review of Financial Studies) defaults
and were not grid-searched from the synthetic dataset.
"""

import numpy as np
import pandas as pd

DEFAULT_PARAMS = {"z_entry": 2.0, "z_exit": 0.5, "window": 60}


class PairsMeanReversion:
    """Long-short pairs strategy using the rolling z-score of the log-spread.

    The strategy is fed a spread instrument whose OHLCV ``close`` column equals
    close_A / close_B (always positive). log(close) = log_A − log_B is the
    log-price spread. Stationarity of this spread under co-integration implies
    it reverts to zero; the z-score measures the number of rolling standard
    deviations from the rolling mean.

    Position semantics (engine must be run with allow_short=True):
        +1  long the spread  (long-A / short-B) — spread expected to rise.
        -1  short the spread (short-A / long-B) — spread expected to fall.
         0  flat.

    Entry rule (Gatev et al. 2006 defaults):
        Enter long  (+1) when z < -z_entry.
        Enter short (-1) when z > +z_entry.

    Exit rule:
        Exit long  when z > -z_exit  (spread has recovered toward mean).
        Exit short when z <  z_exit  (spread has recovered toward mean).
        z_exit < z_entry ensures the position is held until at least partial
        mean-reversion has occurred.

    Args:
        z_entry: Z-score threshold for opening a position. Default 2.0
            (Gatev et al. 2006 standard).
        z_exit: Z-score threshold for closing a position. Default 0.5
            (Gatev et al. 2006 standard).
        window: Rolling window in bars for computing the spread mean and
            standard deviation. Default 60.

    Raises:
        ValueError: If z_entry <= 0, z_exit < 0, z_exit >= z_entry, or
            window <= 1.
    """

    def __init__(
        self,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        window: int = 60,
    ):
        if z_entry <= 0:
            raise ValueError("z_entry must be positive")
        if z_exit < 0:
            raise ValueError("z_exit must be non-negative")
        if z_exit >= z_entry:
            raise ValueError("z_exit must be strictly less than z_entry")
        if window <= 1:
            raise ValueError("window must be greater than 1")
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.window = window
        self._position = 0

    def __call__(self, view) -> float:
        """Compute position signal for the current bar.

        State is updated in-place across sequential bar-by-bar calls.

        Args:
            view: A guarded DataFrame slice with OHLCV columns up to and
                including bar t. The ``close`` column must equal
                close_A / close_B (a positive spread-ratio value).

        Returns:
            1.0 if long the spread, -1.0 if short the spread, 0.0 if flat.
        """
        closes = view["close"]
        if len(closes) < self.window:
            return 0.0

        log_spread = np.log(closes.values.astype(float))
        log_spread_s = pd.Series(log_spread, index=closes.index)

        rolling_mean = log_spread_s.rolling(self.window).mean().iloc[-1]
        rolling_std = log_spread_s.rolling(self.window).std(ddof=1).iloc[-1]

        if pd.isna(rolling_mean) or pd.isna(rolling_std) or rolling_std == 0:
            return 0.0

        z = (log_spread[-1] - rolling_mean) / rolling_std

        if self._position == 0:
            if z < -self.z_entry:
                self._position = 1
            elif z > self.z_entry:
                self._position = -1
        elif self._position == 1:
            if z > -self.z_exit:
                self._position = 0
        else:
            if z < self.z_exit:
                self._position = 0

        return float(self._position)
