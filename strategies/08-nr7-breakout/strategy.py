"""NR7 Volatility-Contraction Breakout Strategy.

Thesis: markets alternate between compression (low-range, low-volume consolidation) and
expansion (breakout). The NR7 pattern (Narrowest Range in 7 bars) identifies the extreme
of compression — bar t has the smallest high-minus-low range of the last n_bars. During
consolidation, stop orders accumulate beyond the edges of the tight range; the first move
outside that range triggers those stops en masse, creating a self-reinforcing price move in
the breakout direction.

Based on Crabel (1990) — both the n_bars=7 label and the exit_bars=4 time-based exit are
published defaults from that work. True range is defined here as high - low (the intraday
range only), not the wider ATR definition that also incorporates prior-close gaps. Python
and ThinkorSwim use this identical definition; the README parity section documents the
divergence from ThinkorSwim's built-in TrueRange() function.
"""

DEFAULT_PARAMS = {"n_bars": 7, "exit_bars": 4}


class NR7Breakout:
    """NR7 Volatility-Contraction Breakout strategy.

    Fires an entry signal when bar t has the narrowest high-minus-low range of
    the last n_bars bars (including bar t). A direction filter selects long or
    short: if close[t] is above the bar midpoint, go long; otherwise go short.
    The position is held for exactly exit_bars bars then unconditionally closed.
    No stop-loss parameter is used — the time-based exit follows Crabel's (1990)
    original design, which deliberately avoids introducing a tuned stop parameter.

    At most one position is held at a time. If a new NR7 signal fires while a
    position is already open, it is ignored until the current position exits.

    True-range definition: high - low. The ATR variant
    (max(H-L, |H-prevClose|, |L-prevClose|)) is NOT used. Both strategy.py and
    strategy.ts use high - low identically for cross-platform parity.

    Args:
        n_bars: Rolling window for the minimum range comparison. Bar t must have
            the narrowest range of the last n_bars bars (including itself) to
            qualify. Default 7 is the Crabel (1990) published value.
        exit_bars: Number of bars to hold the position before unconditional exit.
            Default 4 is the Crabel (1990) published value.

    Raises:
        ValueError: If n_bars < 2 or exit_bars < 1.
    """

    def __init__(self, n_bars: int = 7, exit_bars: int = 4):
        if n_bars < 2:
            raise ValueError("n_bars must be at least 2")
        if exit_bars < 1:
            raise ValueError("exit_bars must be at least 1")
        self.n_bars = n_bars
        self.exit_bars = exit_bars
        self._in_position = False
        self._bars_held = 0
        self._position_direction = 0

    def __call__(self, view) -> float:
        """Compute position signal for the current bar.

        When in a position the hold counter is incremented each bar; once
        exit_bars consecutive holds have been signalled, the position is closed
        and 0.0 is returned. New NR7 signals are ignored while in a position.

        When flat, bar t is tested for the NR7 condition: true range must equal
        the minimum of the last n_bars true ranges (including bar t itself).
        Entry direction is determined by close[t] versus the bar midpoint;
        close above midpoint → long (1.0), at or below → short (-1.0).

        No look-ahead: high[t] and low[t] are fully known at bar t's close.
        The fill executes at bar t+1 open in the engine.

        Args:
            view: A guarded DataFrame slice with OHLCV columns up to and
                including bar t.

        Returns:
            1.0 (long), -1.0 (short), or 0.0 (flat).
        """
        highs = view["high"]
        lows = view["low"]
        closes = view["close"]
        n = len(closes)

        if n < self.n_bars:
            return 0.0

        if self._in_position:
            self._bars_held += 1
            if self._bars_held < self.exit_bars:
                return float(self._position_direction)
            self._in_position = False
            self._bars_held = 0
            self._position_direction = 0
            return 0.0

        true_range = highs - lows
        window_min = float(true_range.iloc[-self.n_bars:].min())
        current_tr = float(true_range.iloc[-1])

        if current_tr != window_min:
            return 0.0

        midpoint = (float(highs.iloc[-1]) + float(lows.iloc[-1])) / 2.0
        current_close = float(closes.iloc[-1])
        direction = 1.0 if current_close > midpoint else -1.0

        self._in_position = True
        self._bars_held = 0
        self._position_direction = direction
        return direction
