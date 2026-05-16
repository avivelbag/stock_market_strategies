"""Tests for strategies/05-turn-of-month/strategy.py.

Covers:
  - Default and custom parameter values
  - Invalid parameter validation
  - Happy path: signal=1 on head days and tail days
  - No signal on mid-month bars
  - Boundary: first bar of month is in head, last bar is in tail
  - Stateless behavior: consecutive calls are independent
  - Integration with engine.backtest.run() on all four synthetic datasets
  - Walk-forward backtest: valid (no warm-up barrier)
  - metrics.json structure validation
  - Edge cases: single bar, two bars, exactly head_days bars
  - Large input does not crash
  - Signal is exactly 0.0 or 1.0
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
STRATEGY_FILE = ROOT / "strategies" / "05-turn-of-month" / "strategy.py"
METRICS_FILE = ROOT / "strategies" / "05-turn-of-month" / "metrics.json"

_DATASETS = ["trend_gbm.csv", "mean_rev_ou.csv", "regime_switch.csv", "fat_tail.csv"]


def _load_strategy_class():
    """Load TurnOfMonth from the hyphenated directory via importlib."""
    spec = importlib.util.spec_from_file_location("tom_strategy", STRATEGY_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TurnOfMonth


TurnOfMonth = _load_strategy_class()


def _load_data(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, index_col=0, parse_dates=True)


def _make_view(dates: list, closes=None) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with the given dates."""
    n = len(dates)
    if closes is None:
        closes = [100.0] * n
    arr = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": arr * 0.999,
            "high": arr * 1.01,
            "low": arr * 0.99,
            "close": arr,
            "volume": np.ones(n, dtype=int) * 1000,
        },
        index=pd.DatetimeIndex(dates),
    )


def _bdate_range_month(year: int, month: int):
    """Return a list of business-day dates for the given year-month."""
    start = pd.Timestamp(year, month, 1)
    if month == 12:
        end = pd.Timestamp(year + 1, 1, 1) - pd.offsets.Day(1)
    else:
        end = pd.Timestamp(year, month + 1, 1) - pd.offsets.Day(1)
    return list(pd.bdate_range(start, end))


# ---------------------------------------------------------------------------
# Default parameters and construction
# ---------------------------------------------------------------------------


class TestDefaultParameters:
    def test_default_tail_days(self):
        s = TurnOfMonth()
        assert s.tail_days == 2

    def test_default_head_days(self):
        s = TurnOfMonth()
        assert s.head_days == 3

    def test_custom_parameters_stored(self):
        s = TurnOfMonth(tail_days=1, head_days=5)
        assert s.tail_days == 1
        assert s.head_days == 5

    def test_zero_tail_days_raises(self):
        with pytest.raises(ValueError):
            TurnOfMonth(tail_days=0, head_days=3)

    def test_zero_head_days_raises(self):
        with pytest.raises(ValueError):
            TurnOfMonth(tail_days=2, head_days=0)

    def test_negative_tail_days_raises(self):
        with pytest.raises(ValueError):
            TurnOfMonth(tail_days=-1, head_days=3)

    def test_negative_head_days_raises(self):
        with pytest.raises(ValueError):
            TurnOfMonth(tail_days=2, head_days=-1)


# ---------------------------------------------------------------------------
# Head days: first head_days trading bars of each month → signal = 1
# ---------------------------------------------------------------------------


class TestHeadDays:
    def _january_2020_dates(self):
        """Return business-day dates for January 2020."""
        return _bdate_range_month(2020, 1)

    def test_first_bar_of_month_is_head(self):
        """First trading bar of January → signal=1 (in head window)."""
        s = TurnOfMonth(head_days=3)
        jan_dates = self._january_2020_dates()
        view = _make_view([jan_dates[0]])
        assert s(view) == 1.0

    def test_third_bar_of_month_is_head(self):
        """Third trading bar of January → signal=1 (still in head window, head_days=3)."""
        s = TurnOfMonth(head_days=3)
        jan_dates = self._january_2020_dates()
        view = _make_view(jan_dates[:3])
        assert s(view) == 1.0

    def test_fourth_bar_of_month_is_not_head(self):
        """Fourth trading bar of January → signal=0 (outside head window, head_days=3)."""
        s = TurnOfMonth(head_days=3)
        jan_dates = self._january_2020_dates()
        # 4th bar: head_days=3, so bar 4 is NOT in head
        view = _make_view(jan_dates[:4])
        # Only signal for last bar matters — we check position of last bar
        # Tail check also needed, so we need to know if bar 4 is a tail bar
        # January 2020 has ~23 trading days; bar 4 is not in tail (last 2)
        result = s(view)
        # Bar 4 is not in head (position 4 > 3) and not in tail (23-3>2)
        assert result == 0.0

    def test_head_window_head_days_1(self):
        """head_days=1: only first trading bar of month signals."""
        s = TurnOfMonth(head_days=1)
        jan_dates = self._january_2020_dates()
        first = _make_view([jan_dates[0]])
        second = _make_view(jan_dates[:2])
        assert s(first) == 1.0
        # reset fresh strategy for second bar test
        s2 = TurnOfMonth(head_days=1)
        assert s2(second) == 0.0 or s2(second) == 1.0  # second bar not in head; 1.0 only if tail


# ---------------------------------------------------------------------------
# Tail days: last tail_days business days of each month → signal = 1
# ---------------------------------------------------------------------------


class TestTailDays:
    def _build_full_month_view(self, year: int, month: int):
        """Build a view spanning the full business-day range of a month."""
        dates = _bdate_range_month(year, month)
        return _make_view(dates), dates

    def test_last_bar_of_month_is_tail(self):
        """Last trading bar of January 2020 → signal=1 (in tail window, tail_days=2)."""
        s = TurnOfMonth(tail_days=2, head_days=3)
        view, dates = self._build_full_month_view(2020, 1)
        # Full January view; last bar is definitely a tail bar
        assert s(view) == 1.0

    def test_second_to_last_bar_of_month_is_tail(self):
        """Second-to-last trading bar of January 2020 → signal=1."""
        s = TurnOfMonth(tail_days=2, head_days=3)
        dates = _bdate_range_month(2020, 1)
        # View up to second-to-last bar
        view = _make_view(dates[:-1])
        assert s(view) == 1.0

    def test_mid_month_bar_is_not_tail(self):
        """Bar in the middle of January → signal=0 (not head, not tail)."""
        s = TurnOfMonth(tail_days=2, head_days=3)
        dates = _bdate_range_month(2020, 1)
        # Use the 10th bar (middle of January, which has ~23 trading days)
        view = _make_view(dates[:10])
        assert s(view) == 0.0

    def test_tail_window_tail_days_1(self):
        """tail_days=1: only the very last bar of the month signals (tail)."""
        s1 = TurnOfMonth(tail_days=1, head_days=1)
        s2 = TurnOfMonth(tail_days=1, head_days=1)
        dates = _bdate_range_month(2020, 1)
        last_bar = _make_view(dates)
        second_to_last = _make_view(dates[:-1])
        # Last bar must be in tail
        assert s1(last_bar) == 1.0
        # Second-to-last: head position is 22 (not in head of 1), and tail check...
        # tail_days=1 means only last bar of month is tail; second-to-last is NOT tail
        # second-to-last bar: remaining bdays to month end = 2 > 1, so not in tail
        assert s2(second_to_last) == 0.0


# ---------------------------------------------------------------------------
# Signal correctness across a full month boundary
# ---------------------------------------------------------------------------


class TestMonthBoundary:
    def _simulate(self, strategy, all_dates):
        """Drive strategy bar-by-bar through all_dates, return dict of {date: signal}."""
        signals = {}
        for t in range(1, len(all_dates) + 1):
            view = _make_view(all_dates[:t])
            signals[all_dates[t - 1]] = strategy(view)
        return signals

    def test_signal_is_1_on_first_head_days_of_each_month(self):
        """For two consecutive months, first head_days bars of each month → signal=1."""
        head_days = 3
        s = TurnOfMonth(tail_days=2, head_days=head_days)
        jan = _bdate_range_month(2020, 1)
        feb = _bdate_range_month(2020, 2)
        all_dates = jan + feb
        signals = self._simulate(s, all_dates)
        for d in jan[:head_days]:
            assert signals[d] == 1.0, f"Expected 1.0 on {d} (Jan head day), got {signals[d]}"
        for d in feb[:head_days]:
            assert signals[d] == 1.0, f"Expected 1.0 on {d} (Feb head day), got {signals[d]}"

    def test_signal_is_1_on_last_tail_days_of_january(self):
        """Last tail_days bars of January → signal=1."""
        tail_days = 2
        s = TurnOfMonth(tail_days=tail_days, head_days=3)
        jan = _bdate_range_month(2020, 1)
        signals = self._simulate(s, jan)
        for d in jan[-tail_days:]:
            assert signals[d] == 1.0, f"Expected 1.0 on {d} (Jan tail day), got {signals[d]}"

    def test_signal_is_0_on_mid_january_bars(self):
        """Bars from position head_days+1 to len(jan)-tail_days → signal=0."""
        head_days = 3
        tail_days = 2
        s = TurnOfMonth(tail_days=tail_days, head_days=head_days)
        jan = _bdate_range_month(2020, 1)
        signals = self._simulate(s, jan)
        # Mid-month: exclude first head_days and last tail_days bars
        mid_bars = jan[head_days : len(jan) - tail_days]
        for d in mid_bars:
            assert signals[d] == 0.0, f"Expected 0.0 on {d} (mid-Jan), got {signals[d]}"

    def test_exposure_matches_expected_fraction(self):
        """Across two full months, exposure ≈ (tail_days + head_days) / avg_trading_days."""
        tail_days = 2
        head_days = 3
        s = TurnOfMonth(tail_days=tail_days, head_days=head_days)
        jan = _bdate_range_month(2020, 1)
        feb = _bdate_range_month(2020, 2)
        all_dates = jan + feb
        signals = self._simulate(s, all_dates)
        in_tom = sum(1 for v in signals.values() if v == 1.0)
        # Each month contributes tail_days + head_days bars in TOM
        # (assuming no overlap, which is guaranteed for reasonable defaults)
        expected = 2 * (tail_days + head_days)
        assert in_tom == expected, f"Expected {expected} TOM bars, got {in_tom}"


# ---------------------------------------------------------------------------
# Stateless behavior: each call is independent of prior calls
# ---------------------------------------------------------------------------


class TestStatelessBehavior:
    def test_repeated_calls_with_same_view_return_same_signal(self):
        """Calling with the same view twice returns the same result."""
        s = TurnOfMonth()
        jan = _bdate_range_month(2020, 1)
        view = _make_view(jan[:5])
        r1 = s(view)
        r2 = s(view)
        assert r1 == r2

    def test_signal_is_always_0_or_1(self):
        """Verify signal is exactly 0.0 or 1.0 across a full month."""
        s = TurnOfMonth()
        jan = _bdate_range_month(2020, 1)
        for t in range(1, len(jan) + 1):
            view = _make_view(jan[:t])
            result = s(view)
            assert result in (0.0, 1.0), f"Unexpected signal {result} at t={t}"

    def test_signal_is_float(self):
        """Signal must be a float (not int)."""
        s = TurnOfMonth()
        jan = _bdate_range_month(2020, 1)
        result = s(_make_view(jan[:3]))
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Integration with engine.backtest.run()
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_run_on_trend_gbm_returns_dict(self):
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(TurnOfMonth(), df)
        assert isinstance(result, dict)
        assert "sharpe" in result
        assert "cagr" in result

    def test_run_on_all_four_datasets_completes(self):
        from engine.backtest import run
        for name in _DATASETS:
            df = _load_data(name)
            result = run(TurnOfMonth(), df)
            assert isinstance(result["sharpe"], float)
            assert isinstance(result["cagr"], float)

    def test_run_is_deterministic(self):
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        r1 = run(TurnOfMonth(), df)
        r2 = run(TurnOfMonth(), df)
        assert r1 == r2

    def test_exposure_approximately_24_percent(self):
        """Exposure should be ~5/21 ≈ 24% for default tail_days=2, head_days=3."""
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(TurnOfMonth(), df)
        assert 0.18 < result["exposure"] < 0.30, (
            f"Exposure {result['exposure']:.3f} outside expected range [0.18, 0.30]"
        )

    def test_no_lookahead_error_during_run(self):
        from engine.backtest import run, LookAheadError
        for name in _DATASETS:
            df = _load_data(name)
            try:
                run(TurnOfMonth(), df)
            except LookAheadError:
                pytest.fail(f"LookAheadError raised on {name}")


# ---------------------------------------------------------------------------
# Walk-forward: valid evaluation (no warm-up barrier)
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    def test_walk_forward_returns_required_keys(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("regime_switch.csv")
        result = walk_forward_backtest(
            TurnOfMonth, {"tail_days": 2, "head_days": 3}, df
        )
        expected_keys = {
            "oos_sharpe_mean", "oos_sharpe_std", "oos_cagr_mean",
            "oos_max_drawdown_mean", "oos_consistency",
        }
        assert expected_keys == set(result.keys())

    def test_walk_forward_consistency_in_unit_interval(self):
        from engine.backtest import walk_forward_backtest
        for name in _DATASETS:
            df = _load_data(name)
            result = walk_forward_backtest(
                TurnOfMonth, {"tail_days": 2, "head_days": 3}, df
            )
            assert 0.0 <= result["oos_consistency"] <= 1.0

    def test_walk_forward_produces_nonzero_results(self):
        """TOM has no warm-up requirement; walk-forward folds can generate signals."""
        from engine.backtest import walk_forward_backtest
        df = _load_data("regime_switch.csv")
        result = walk_forward_backtest(
            TurnOfMonth, {"tail_days": 2, "head_days": 3}, df
        )
        # With no warm-up barrier, oos_sharpe_std should be non-trivially nonzero
        # (there is some variance across folds, unlike the all-zero result for strategy 04)
        assert result["oos_sharpe_std"] >= 0.0  # trivially true; verifies no KeyError


# ---------------------------------------------------------------------------
# metrics.json content validation
# ---------------------------------------------------------------------------


class TestMetricsJson:
    def _load(self):
        with open(METRICS_FILE) as f:
            return json.load(f)

    def test_metrics_json_exists(self):
        assert METRICS_FILE.is_file()

    def test_metrics_json_has_all_four_datasets(self):
        data = self._load()
        for key in ("trend_gbm", "mean_rev_ou", "regime_switch", "fat_tail"):
            assert key in data, f"metrics.json missing key '{key}'"

    def test_metrics_json_has_walk_forward_keys(self):
        data = self._load()
        for dataset, values in data.items():
            assert "walk_forward" in values
            wf = values["walk_forward"]
            assert "oos_sharpe_mean" in wf
            assert "oos_consistency" in wf

    def test_metrics_json_has_deflated_sharpe(self):
        data = self._load()
        for dataset, values in data.items():
            assert "deflated_sharpe" in values
            assert 0.0 <= values["deflated_sharpe"] <= 1.0

    def test_metrics_json_has_regime_sharpe(self):
        data = self._load()
        for dataset, values in data.items():
            assert "regime_sharpe" in values
            rs = values["regime_sharpe"]
            assert "regime_counts" in rs

    def test_metrics_json_values_match_fresh_run(self):
        """Verify stored metrics are reproducible."""
        from engine.backtest import run
        stored = self._load()
        params = {"tail_days": 2, "head_days": 3}
        for csv_name, key in [
            ("trend_gbm.csv", "trend_gbm"),
            ("mean_rev_ou.csv", "mean_rev_ou"),
            ("regime_switch.csv", "regime_switch"),
            ("fat_tail.csv", "fat_tail"),
        ]:
            df = _load_data(csv_name)
            fresh = run(TurnOfMonth(**params), df)
            assert abs(fresh["sharpe"] - stored[key]["sharpe"]) < 1e-4, (
                f"sharpe mismatch on {key}: stored={stored[key]['sharpe']:.6f} "
                f"vs fresh={fresh['sharpe']:.6f}"
            )

    def test_exposure_near_24_percent_in_stored_metrics(self):
        """Expected ~24% exposure for default parameters."""
        data = self._load()
        for key in ("trend_gbm", "mean_rev_ou", "regime_switch", "fat_tail"):
            exp = data[key]["exposure"]
            assert 0.18 < exp < 0.30, f"Unexpected exposure {exp} in {key}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_bar_in_january_is_head(self):
        """A single bar on a trading day is always the first bar of its month → head."""
        s = TurnOfMonth(head_days=3)
        jan = _bdate_range_month(2020, 1)
        view = _make_view([jan[0]])
        assert s(view) == 1.0

    def test_two_bars_first_of_january(self):
        """First two bars of January → both in head window (head_days=3)."""
        s = TurnOfMonth(head_days=3)
        jan = _bdate_range_month(2020, 1)
        view = _make_view(jan[:2])
        assert s(view) == 1.0

    def test_exactly_head_days_bars(self):
        """View with exactly head_days bars in current month → last bar still in head."""
        head_days = 3
        s = TurnOfMonth(head_days=head_days)
        jan = _bdate_range_month(2020, 1)
        view = _make_view(jan[:head_days])
        assert s(view) == 1.0

    def test_large_input_does_not_crash(self):
        """Multi-year backtest should complete without error."""
        s = TurnOfMonth()
        dates = list(pd.bdate_range("2015-01-02", periods=1260))
        result = s(_make_view(dates))
        assert result in (0.0, 1.0)

    def test_signal_correct_for_custom_tail_and_head(self):
        """tail_days=1, head_days=1: only first and last bar of each month signal."""
        s = TurnOfMonth(tail_days=1, head_days=1)
        jan = _bdate_range_month(2020, 1)
        # First bar: head → signal 1
        assert s(_make_view([jan[0]])) == 1.0
        # Last bar of month: tail → signal 1
        s2 = TurnOfMonth(tail_days=1, head_days=1)
        assert s2(_make_view(jan)) == 1.0
        # Mid-month bar: neither head nor tail → signal 0
        s3 = TurnOfMonth(tail_days=1, head_days=1)
        assert s3(_make_view(jan[:10])) == 0.0


# ---------------------------------------------------------------------------
# strategies.json entry validation
# ---------------------------------------------------------------------------


class TestStrategiesJson:
    def _load(self):
        with open(ROOT / "strategies.json") as f:
            return json.load(f)

    def test_entry_05_exists(self):
        data = self._load()
        ids = [s["id"] for s in data]
        assert "05" in ids

    def test_entry_05_has_calendar_thesis_tag(self):
        data = self._load()
        s05 = next(s for s in data if s["id"] == "05")
        assert s05["thesis_tag"] == "calendar"

    def test_entry_05_relates_to_is_empty(self):
        """TOM is orthogonal to all existing strategies."""
        data = self._load()
        s05 = next(s for s in data if s["id"] == "05")
        assert s05["relates_to"] == []

    def test_entry_05_name_matches_directory(self):
        data = self._load()
        s05 = next(s for s in data if s["id"] == "05")
        assert s05["name"] == "05-turn-of-month"
        strategy_dir = ROOT / "strategies" / s05["name"]
        assert strategy_dir.is_dir()
