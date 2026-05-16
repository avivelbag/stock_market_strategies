"""Tests for walk-forward validation, open[t+1] fill timing, and turnover seeding.

Covers:
  - walk_forward_backtest happy path on a trending series
  - OOS consistency == 1.0 for a trivially correct always-long strategy
  - Required return keys from walk_forward_backtest
  - Edge cases: too-short series, zero valid folds
  - Open[t+1]-to-close[t+1] fill timing (distinguishes from close-to-close)
  - Turnover seeded from flat-0 prior (opening entry is counted)
  - walk_forward_consistency helper in metrics
"""

from pathlib import Path

import numpy as np
import pandas as pd

from engine.backtest import run, walk_forward_backtest
import engine.metrics as metrics

DATA_DIR = Path(__file__).parent.parent / "data"

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, index_col=0, parse_dates=True)


def _make_df(opens, closes, *, n=None) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from open and close arrays."""
    if n is None:
        n = len(closes)
    dates = pd.bdate_range("2020-01-02", periods=n)
    highs = np.maximum(opens, closes) + 0.01
    lows = np.minimum(opens, closes) - 0.01
    lows = np.maximum(lows, 0.01)
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(n, dtype=int) * 1000,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------


class _AlwaysLong:
    """Class-based always-long strategy for use with walk_forward_backtest."""

    def __call__(self, view) -> float:
        return 1.0


class _AlwaysFlat:
    """Class-based always-flat strategy."""

    def __call__(self, view) -> float:
        return 0.0


def _always_long_fn(view) -> float:
    return 1.0


# ---------------------------------------------------------------------------
# walk_forward_backtest — happy path
# ---------------------------------------------------------------------------


class TestWalkForwardHappyPath:
    def test_returns_all_required_keys(self):
        df = _load("trend_gbm.csv")
        result = walk_forward_backtest(_AlwaysLong, {}, df)
        expected = {
            "oos_sharpe_mean",
            "oos_sharpe_std",
            "oos_cagr_mean",
            "oos_max_drawdown_mean",
            "oos_consistency",
        }
        assert expected == set(result.keys())

    def _make_monotone_trending(self, n: int = 500) -> pd.DataFrame:
        """Synthetic series where every bar has a guaranteed positive open→close return.

        close[t] = 100 * 1.002^t  (0.2% per bar compound growth)
        open[t]  = close[t] * 0.999  (open is 0.1% below close every bar)

        Every OOS window with an always-long strategy will have positive CAGR
        and positive Sharpe, so oos_consistency == 1.0 is guaranteed.
        """
        t = np.arange(n, dtype=float)
        closes = 100.0 * (1.002 ** t)
        opens = closes * 0.999
        return _make_df(opens, closes, n=n)

    def test_always_long_on_trending_series_oos_consistency_one(self):
        """Always-long on a monotone trending series: all OOS folds profitable."""
        df = self._make_monotone_trending(n=500)
        result = walk_forward_backtest(
            _AlwaysLong, {}, df, n_splits=5, train_frac=0.7,
            config={"commission_bps": 0, "slippage_bps": 0},
        )
        assert result["oos_consistency"] == 1.0

    def test_always_long_oos_cagr_positive_on_trending_series(self):
        df = self._make_monotone_trending(n=500)
        result = walk_forward_backtest(
            _AlwaysLong, {}, df,
            config={"commission_bps": 0, "slippage_bps": 0},
        )
        assert result["oos_cagr_mean"] > 0.0

    def test_deterministic_across_calls(self):
        df = _load("trend_gbm.csv")
        r1 = walk_forward_backtest(_AlwaysLong, {}, df)
        r2 = walk_forward_backtest(_AlwaysLong, {}, df)
        assert r1 == r2

    def test_flat_strategy_oos_sharpe_zero(self):
        df = _load("trend_gbm.csv")
        result = walk_forward_backtest(_AlwaysFlat, {}, df)
        assert result["oos_sharpe_mean"] == 0.0
        assert result["oos_cagr_mean"] == 0.0

    def test_params_default_passed_to_strategy_cls(self):
        """Verify strategy_cls receives params_default at each fold."""

        class _Recorder:
            received = []

            def __init__(self, multiplier=1):
                self.multiplier = multiplier
                _Recorder.received.append(multiplier)

            def __call__(self, view):
                return float(self.multiplier)

        _Recorder.received.clear()
        df = _load("trend_gbm.csv")
        walk_forward_backtest(_Recorder, {"multiplier": 7}, df, n_splits=3)
        assert all(v == 7 for v in _Recorder.received)
        assert len(_Recorder.received) == 3


# ---------------------------------------------------------------------------
# walk_forward_backtest — edge cases
# ---------------------------------------------------------------------------


class TestWalkForwardEdgeCases:
    def test_zero_valid_folds_returns_zeros(self):
        """When split_size is so large that test_end always exceeds n, return zeros."""
        # 10 bars with 5 splits: split_size = 10//6 = 1; train_frac * i/n_splits windows
        # will overflow quickly
        df = _load("trend_gbm.csv").iloc[:5]  # very short slice
        result = walk_forward_backtest(_AlwaysLong, {}, df, n_splits=100, train_frac=0.99)
        assert result["oos_consistency"] == 0.0
        assert result["oos_sharpe_mean"] == 0.0

    def test_single_split_returns_scalar_std_zero(self):
        """With n_splits=1 a single fold gives std=0."""
        df = _load("trend_gbm.csv")
        result = walk_forward_backtest(_AlwaysLong, {}, df, n_splits=1, train_frac=0.5)
        assert result["oos_sharpe_std"] == 0.0

    def test_all_four_synthetic_datasets_run_cleanly(self):
        """Regression: walk_forward_backtest completes on all four datasets."""
        for name in ("trend_gbm.csv", "mean_rev_ou.csv", "regime_switch.csv", "fat_tail.csv"):
            df = _load(name)
            result = walk_forward_backtest(_AlwaysLong, {}, df)
            assert isinstance(result["oos_consistency"], float)
            assert 0.0 <= result["oos_consistency"] <= 1.0


# ---------------------------------------------------------------------------
# Open[t+1] fill timing
# ---------------------------------------------------------------------------


class TestOpenFillTiming:
    def _make_declining_close_rising_intraday(self) -> pd.DataFrame:
        """Series where close falls each bar but open[t+1] < close[t+1] (positive intraday).

        close-to-close return is negative (old model → losing long position).
        open[t+1]-to-close[t+1] return is positive (new model → winning long position).
        """
        n = 10
        # close decreases: 10, 9, 8, ..., 1
        # open[t+1] is well below close[t+1]: e.g. close[t+1] - 0.9
        closes = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        opens = np.empty(n)
        opens[0] = closes[0]
        # Each open[t+1] is 0.5 below its close, ensuring positive intraday return
        opens[1:] = closes[1:] - 0.5
        return _make_df(opens, closes, n=n)

    def test_always_long_positive_cagr_with_open_fill(self):
        """With open[t+1] fill, always-long earns positive intraday returns even
        when close-to-close returns are negative."""
        df = self._make_declining_close_rising_intraday()
        result = run(_always_long_fn, df, {"commission_bps": 0, "slippage_bps": 0})
        assert result["cagr"] > 0.0, (
            "Expected positive CAGR when intraday (open→close) returns are positive, "
            "even though close-to-close returns are negative"
        )

    def test_bar_return_uses_open_not_prior_close(self):
        """Verify: equity growth comes from open[t+1]-to-close[t+1], not close[t]-to-close[t+1].

        Construct a 3-bar series where:
          Bar 0: open=100, close=100
          Bar 1: open=50,  close=80   (open-to-close return = +60%; close-to-close = -20%)
          Bar 2: open=70,  close=90   (open-to-close return = +28.6%; close-to-close = +12.5%)

        Always-long with old model (close-to-close): net return ≈ -20% then +12.5% → loses
        Always-long with new model (open-to-close): net return ≈ +60% then +28.6% → wins
        """
        opens = np.array([100.0, 50.0, 70.0])
        closes = np.array([100.0, 80.0, 90.0])
        df = _make_df(opens, closes, n=3)
        result = run(_always_long_fn, df, {"commission_bps": 0, "slippage_bps": 0})
        # With open[t+1] fill: equity[1]=10000*(1+30/50)=16000, equity[2]=16000*(1+20/70)≈19657
        assert result["cagr"] > 0.0


# ---------------------------------------------------------------------------
# Turnover seeded from flat-0 prior
# ---------------------------------------------------------------------------


class TestTurnoverSeeding:
    def test_always_long_turnover_nonzero(self):
        """Opening trade from flat→long is now counted: turnover must be > 0."""
        df = _load("trend_gbm.csv")
        result = run(_always_long_fn, df, {})
        assert result["turnover"] > 0.0, (
            "Opening trade (flat→long) must contribute to turnover"
        )

    def test_always_flat_turnover_still_zero(self):
        """No position changes; turnover is still exactly 0."""
        df = _load("trend_gbm.csv")
        result = run(lambda view: 0.0, df, {})
        assert result["turnover"] == 0.0

    def test_turnover_value_for_known_positions(self):
        """Single entry at bar 0 from flat: turnover = 1 / n_bars."""
        n = 100
        positions = pd.Series(np.ones(n))
        # prev = [0, 1, 1, ..., 1]; diff = [1, 0, 0, ..., 0]
        computed = metrics.turnover(positions)
        expected = 1.0 / n
        assert abs(computed - expected) < 1e-12

    def test_full_flip_every_bar_turnover(self):
        """Flipping +1→-1 every bar: each step has |diff|=2, mean=2."""
        positions = pd.Series([1.0, -1.0, 1.0, -1.0])
        # prev = [0, 1, -1, 1]; diff = [1, 2, 2, 2]; mean = 7/4 = 1.75
        computed = metrics.turnover(positions)
        expected = (1 + 2 + 2 + 2) / 4
        assert abs(computed - expected) < 1e-12


# ---------------------------------------------------------------------------
# walk_forward_consistency helper
# ---------------------------------------------------------------------------


class TestWalkForwardConsistency:
    def test_all_positive_gives_one(self):
        assert metrics.walk_forward_consistency([0.5, 1.2, 0.1]) == 1.0

    def test_all_negative_gives_zero(self):
        assert metrics.walk_forward_consistency([-0.5, -1.2, -0.1]) == 0.0

    def test_half_positive_gives_half(self):
        result = metrics.walk_forward_consistency([1.0, -1.0])
        assert abs(result - 0.5) < 1e-12

    def test_empty_gives_zero(self):
        assert metrics.walk_forward_consistency([]) == 0.0

    def test_exact_zero_sharpe_is_not_positive(self):
        assert metrics.walk_forward_consistency([0.0]) == 0.0
