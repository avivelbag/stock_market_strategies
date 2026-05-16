"""Turn-of-Month (TOM) Calendar Effect Strategy.

Thesis: Lakonishok & Smidt (1988, Journal of Finance) and Ariel (1987) document
that equity returns are concentrated in the last few and first few trading days
of each calendar month.  The mechanism is institutional: mutual funds and pension
managers window-dress portfolios before month-end statements; fresh cash from
monthly payroll savers arrives at the start of the new month, creating predictable
demand around the month boundary.

The window [tail_days=2, head_days=3] is the Lakonishok & Smidt (1988) specification
— not grid-searched from this data.  The strategy has essentially zero free parameters
relative to the literature; both values are published prior-set defaults.
"""

import pandas as pd
from pandas.tseries.offsets import BMonthEnd

DEFAULT_PARAMS = {"tail_days": 2, "head_days": 3}


class TurnOfMonth:
    """Long-only strategy based on the turn-of-month calendar effect.

    Generates a +1 signal on the last ``tail_days`` trading days of each calendar
    month and on the first ``head_days`` trading days of the following month.
    Signal is 0 on all other bars.

    **Head-days computation (no price look-ahead)**: counts how many trading bars
    in the current month precede or include bar t from ``view.index``.  Only data
    ≤ t is used.

    **Tail-days computation (calendar look-ahead, not price look-ahead)**: uses
    ``pandas.tseries.offsets.BMonthEnd`` and ``pd.bdate_range`` to count the
    remaining business days from bar t to the last business day of the month.
    This requires knowing when the month ends (calendar arithmetic), but accesses
    no future price data.  For synthetic data with no market holidays, business
    days and actual trading bars are identical, so the count is exact.  For real
    equity data with observed holidays the count may differ by at most one day in
    a given month; that edge case is documented in README.md.

    Args:
        tail_days: Trading days at month-end to include in the TOM window.
            Default 2, per Lakonishok & Smidt (1988).
        head_days: Trading days at month-start to include in the TOM window.
            Default 3, per Lakonishok & Smidt (1988).

    Raises:
        ValueError: If tail_days or head_days is not a positive integer.
    """

    def __init__(self, tail_days: int = 2, head_days: int = 3):
        if tail_days <= 0 or head_days <= 0:
            raise ValueError(
                f"tail_days and head_days must be positive integers; "
                f"got tail_days={tail_days}, head_days={head_days}"
            )
        self.tail_days = tail_days
        self.head_days = head_days

    def __call__(self, view) -> float:
        """Compute position signal for the current bar.

        Entry logic (no state — signal is stateless/calendrical):
          - HEAD: bar t is the 1st through head_days-th trading bar of its month.
          - TAIL: fewer than tail_days business days remain in bar t's month
            (including bar t itself), computed via calendar arithmetic.

        Args:
            view: Guarded DataFrame slice with OHLCV columns up to and including
                bar t.  Must expose ``view.index`` (DatetimeIndex).

        Returns:
            1.0 if bar t is in the TOM window, 0.0 otherwise.
        """
        idx = view.index
        current_date = idx[-1]
        current_period = current_date.to_period("M")

        # HEAD check: position of bar t within its calendar month.
        # idx.to_period('M') == current_period gives all bars of current month seen so far.
        # len() of that slice is bar t's 1-indexed position in the month.
        bars_this_month = idx[idx.to_period("M") == current_period]
        if len(bars_this_month) <= self.head_days:
            return 1.0

        # TAIL check: remaining business days to month end (inclusive of today).
        # BMonthEnd(0) returns the last business day of the current month;
        # it is idempotent when applied to the last business day itself.
        last_bday = (current_date + BMonthEnd(0)).normalize()
        remaining_bdays = len(pd.bdate_range(current_date, last_bday))
        if remaining_bdays <= self.tail_days:
            return 1.0

        return 0.0
