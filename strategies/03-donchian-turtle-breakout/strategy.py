"""Donchian Channel Turtle Breakout Strategy.

Thesis: price breakouts above multi-week extremes signal that informed buyers
have absorbed all willing sellers, creating a structural supply vacuum that
historically sustains trend momentum for days to weeks. Based on the Turtle
Trader experiment (Dennis & Eckhardt, 1983) as documented in Covel (2007).
"""


class DonchianTurtleBreakout:
    """Long-only Turtle System 1 breakout strategy based on Donchian channels.

    Enters long when today's close breaks above the prior N-bar (default 20) channel
    high. Exits when today's close breaks below the prior M-bar (default 10) channel
    low. Position sizing follows 1 ATR unit; the binary engine clips this to 1.0
    exposure. All parameter defaults are from the published Dennis/Eckhardt 1983 rules.

    Args:
        entry_window: Lookback bars for the channel high breakout entry (default 20).
        exit_window: Lookback bars for the channel low breakout exit (default 10).
        atr_window: Lookback bars for ATR computation (default 20). Stored for
            parity with the TOS implementation; does not affect the binary signal.

    Raises:
        ValueError: If any window is non-positive or exit_window >= entry_window.
    """

    def __init__(
        self,
        entry_window: int = 20,
        exit_window: int = 10,
        atr_window: int = 20,
    ):
        if entry_window <= 0 or exit_window <= 0 or atr_window <= 0:
            raise ValueError("All windows must be positive integers")
        if exit_window >= entry_window:
            raise ValueError("exit_window must be strictly less than entry_window")
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.atr_window = atr_window
        self._in_position = False

    def __call__(self, view) -> float:
        """Compute the position signal for the current bar.

        State is updated in-place: the instance tracks whether a long position
        is currently held across sequential bar-by-bar calls from the engine.

        Entry: today's close > max(close) of the prior entry_window bars.
        Exit:  today's close < min(close) of the prior exit_window bars.

        Args:
            view: A guarded DataFrame slice with OHLCV columns up to and including
                the current bar t. Must support view["close"] returning a pd.Series.

        Returns:
            1.0 if in a long position, 0.0 otherwise.
        """
        closes = view["close"]
        n = len(closes)

        if n < self.entry_window + 1:
            return 0.0

        current_close = float(closes.iloc[-1])
        channel_high = float(closes.iloc[-(self.entry_window + 1):-1].max())
        channel_low = float(closes.iloc[-(self.exit_window + 1):-1].min())

        if not self._in_position:
            if current_close > channel_high:
                self._in_position = True
        else:
            if current_close < channel_low:
                self._in_position = False

        return 1.0 if self._in_position else 0.0
