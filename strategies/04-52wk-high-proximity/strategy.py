"""52-Week High Proximity Strategy (investor anchoring bias).

Thesis: George & Hwang (2004, Journal of Finance) document that stocks trading
near their 52-week high outperform over the following month. The mechanism is
behavioural: investors anchor on the 52-week high as a reference price and
under-react to positive news when price is near that level, causing delayed
price discovery that the strategy exploits.
"""


DEFAULT_PARAMS = {"proximity_threshold": 0.95, "exit_threshold": 0.90}


class FiftyTwoWeekHighProximity:
    """Long-only strategy based on proximity to the 52-week (252-bar) closing high.

    Enters long when the ratio (close / rolling_max(close, 252)) is at or above
    proximity_threshold AND is rising relative to the prior bar — confirming that
    the approach toward the anchor level is gaining speed. Exits when the ratio
    falls below exit_threshold.

    The 252-bar lookback requires at least 253 bars of history before any position
    is taken: 252 bars to form the current ratio, plus one additional bar to form
    the prior ratio for the increasing-ratio filter.

    Args:
        proximity_threshold: Minimum ratio (close / 52-week high) to enter long.
            Default 0.95 captures "approaching but not yet exceeding" the anchor.
        exit_threshold: Ratio below which the long position is closed.
            Default 0.90 provides a buffer below the entry zone.

    Raises:
        ValueError: If thresholds are not in the order
            0 < exit_threshold < proximity_threshold <= 1.0.
    """

    def __init__(
        self,
        proximity_threshold: float = 0.95,
        exit_threshold: float = 0.90,
    ):
        if not (0.0 < exit_threshold < proximity_threshold <= 1.0):
            raise ValueError(
                "Must satisfy 0 < exit_threshold < proximity_threshold <= 1.0; "
                f"got exit_threshold={exit_threshold}, "
                f"proximity_threshold={proximity_threshold}"
            )
        self.proximity_threshold = proximity_threshold
        self.exit_threshold = exit_threshold
        self._in_position = False

    def __call__(self, view) -> float:
        """Compute position signal for the current bar.

        State is updated in-place: the instance tracks whether a long position
        is held across sequential bar-by-bar calls from the engine.

        Entry: ratio[t] >= proximity_threshold AND ratio[t] > ratio[t-1].
        Exit:  ratio[t] < exit_threshold.

        Args:
            view: Guarded DataFrame slice with OHLCV columns up to and including
                bar t. Must support view["close"] returning a pd.Series.

        Returns:
            1.0 if in a long position, 0.0 otherwise.
        """
        closes = view["close"]
        n = len(closes)

        # 252 bars for current ratio + 1 more bar for prior ratio comparison
        if n < 253:
            return 0.0

        current_close = float(closes.iloc[-1])
        high52_current = float(closes.iloc[-252:].max())
        ratio = current_close / high52_current

        prev_close = float(closes.iloc[-2])
        # Prior 52-week high: max of 252 bars ending at t-1
        high52_prev = float(closes.iloc[-253:-1].max())
        prev_ratio = prev_close / high52_prev

        if not self._in_position:
            if ratio >= self.proximity_threshold and ratio > prev_ratio:
                self._in_position = True
        else:
            if ratio < self.exit_threshold:
                self._in_position = False

        return 1.0 if self._in_position else 0.0
