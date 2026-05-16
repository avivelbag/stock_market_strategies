"""Low-Volatility Anomaly strategy (time-series version).

Thesis: Baker, Bradley & Wurgler (2011) and Blitz & van Vliet (2007) document
that low-beta / low-volatility securities produce superior risk-adjusted returns
despite CAPM's prediction to the contrary. Two reinforcing mechanisms drive this:

  Behavioral (lottery demand): investors overpay for high-vol "lottery ticket"
  stocks, bidding up their prices and depressing their expected returns. Low-vol
  assets are overlooked, leaving their expected return intact or elevated.

  Institutional (leverage constraint): benchmark-constrained fund managers cannot
  use leverage to hit return targets, so they tilt toward high-vol stocks to boost
  raw expected returns without explicit leverage. This systematic demand overprices
  high-vol assets and underprices low-vol assets.

Cross-sectional to time-series adaptation: the published anomaly ranks stocks in a
universe and buys the lowest-vol decile. Adapted here to a single asset: the asset
is held when its OWN realized volatility is below its own rolling median (calm,
predictable, "low-vol anomaly" regime), and exited when vol spikes above the
rolling 75th percentile (high-vol, lottery-like regime). The economic rationale is
preserved: hold the asset during its calm periods, step aside during volatile ones.

Differs from strategy 09 (Volatility-Managed Portfolio): strategy 09 scales
position SIZE continuously by target_vol / realized_vol (always invested, just more
or less). This strategy makes a binary SELECTION decision (fully in or fully out)
based on RELATIVE volatility (vol vs own history), not absolute vol level. An asset
can pass strategy 09's size threshold while failing this strategy's selection
criterion if its current vol is high relative to its own past even if low in
absolute terms.

Differs from momentum strategies 02, 07, 08: those strategies go long during strong
recent price movement — often the highest-vol periods. This strategy enters in the
opposite regime: when the asset is calm and predictable.
"""

import numpy as np

DEFAULT_PARAMS = {"vol_window": 60, "ranking_window": 252, "exit_percentile": 75}


class LowVolatilityAnomaly:
    """Time-series Low-Volatility Anomaly strategy (Baker/Blitz entry criterion).

    At each bar t, computes the trailing vol_window-bar annualized realized
    volatility (std of daily pct_change returns × sqrt(252)). The strategy holds
    a long position while that volatility is below its own rolling median over the
    prior ranking_window bars, and exits when vol rises above the rolling
    exit_percentile-th percentile of the same window.

    Hysteresis prevents churn around the median boundary: once long, the position
    is held until vol explicitly exceeds the exit_percentile threshold, even if vol
    temporarily rises above the median during the hold.

    Entry condition: current_vol < rolling_median(vol_history[-ranking_window:])
    Exit condition:  current_vol > rolling_percentile(exit_percentile, same window)

    Warm-up: the strategy returns 0.0 for the first vol_window + ranking_window bars
    because no meaningful percentile can be computed before the full history is
    available.

    Args:
        vol_window: Lookback in bars for realized volatility. Default 60 (roughly one
            trading quarter). Must be at least 2.
        ranking_window: Lookback in bars for the rolling median / percentile of vol
            history. Default 252 (one trading year). Must be >= vol_window so that
            the percentile is computed over a meaningful vol series.
        exit_percentile: Percentile of the rolling vol history used as the exit
            threshold. Default 75. Must be in (0, 100).

    Raises:
        ValueError: If vol_window < 2, ranking_window < vol_window, or
            exit_percentile not in (0, 100).
    """

    def __init__(
        self,
        vol_window: int = 60,
        ranking_window: int = 252,
        exit_percentile: float = 75.0,
    ):
        if vol_window < 2:
            raise ValueError("vol_window must be at least 2")
        if ranking_window < vol_window:
            raise ValueError("ranking_window must be >= vol_window")
        if not (0 < exit_percentile < 100):
            raise ValueError("exit_percentile must be in (0, 100)")
        self.vol_window = vol_window
        self.ranking_window = ranking_window
        self.exit_percentile = exit_percentile
        self._in_position = False

    def __call__(self, view) -> float:
        """Compute entry/exit signal for the current bar.

        Requires at least vol_window + ranking_window bars of close history.
        During warm-up, returns 0.0 (flat).

        The realized vol series is computed as a rolling vol_window std of
        pct_change returns, annualized by sqrt(252). The last ranking_window
        values of that series form the vol history used for the median and
        exit_percentile thresholds.

        No look-ahead: all computations use close[t] and earlier. The engine
        fills at open[t+1].

        Args:
            view: A guarded DataFrame slice with OHLCV columns up to and
                including bar t.

        Returns:
            1.0 if long, 0.0 if flat.
        """
        closes = view["close"]
        n = len(closes)
        min_bars = self.vol_window + self.ranking_window
        if n < min_bars:
            self._in_position = False
            return 0.0

        returns = closes.pct_change()
        # Rolling annualized vol; iloc[-1] is current bar's vol estimate
        rolling_vol = returns.rolling(self.vol_window).std() * (252 ** 0.5)
        current_vol = float(rolling_vol.iloc[-1])

        if np.isnan(current_vol):
            self._in_position = False
            return 0.0

        # Use the last ranking_window values of the vol series as the reference
        vol_history = rolling_vol.dropna().iloc[-self.ranking_window :]
        if len(vol_history) < 2:
            self._in_position = False
            return 0.0

        rolling_median = float(vol_history.median())
        rolling_exit = float(np.percentile(vol_history.values, self.exit_percentile))

        if self._in_position:
            if current_vol > rolling_exit:
                self._in_position = False
                return 0.0
            return 1.0
        else:
            if current_vol < rolling_median:
                self._in_position = True
                return 1.0
            return 0.0
