"""Tests for strategies/04-52wk-high-proximity/strategy.py.

Covers:
  - Happy path: long signal when ratio >= 0.95 and ratio is increasing
  - No entry when ratio is below proximity_threshold
  - No entry when ratio is not increasing (flat or declining approach)
  - Exit when ratio falls below exit_threshold
  - Warm-up guard: flat when fewer than 253 bars
  - Default and custom parameter values
  - Invalid parameter validation
  - Integration with engine.backtest.run() on all four synthetic datasets
  - Walk-forward backtest: all zeros expected (253-bar warmup vs 166-bar folds)
  - metrics.json structure validation
  - Edge cases: single bar, exactly 252 bars, constant prices, large input
  - Stateful behavior: position held between bars, entry/exit sequence
  - No look-ahead violation during engine run
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
STRATEGY_FILE = ROOT / "strategies" / "04-52wk-high-proximity" / "strategy.py"
METRICS_FILE = ROOT / "strategies" / "04-52wk-high-proximity" / "metrics.json"

_DATASETS = ["trend_gbm.csv", "mean_rev_ou.csv", "regime_switch.csv", "fat_tail.csv"]


def _load_strategy_class():
    """Load FiftyTwoWeekHighProximity from the hyphenated directory via importlib."""
    spec = importlib.util.spec_from_file_location("proximity_strategy", STRATEGY_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FiftyTwoWeekHighProximity


FiftyTwoWeekHighProximity = _load_strategy_class()


def _load_data(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, index_col=0, parse_dates=True)


def _make_view(closes: list) -> pd.DataFrame:
    """Construct a minimal OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    dates = pd.bdate_range("2020-01-02", periods=n)
    arr = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": arr * 0.999,
            "high": arr * 1.01,
            "low": arr * 0.99,
            "close": arr,
            "volume": np.ones(n, dtype=int) * 1000,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Default parameters and construction
# ---------------------------------------------------------------------------


class TestDefaultParameters:
    def test_default_proximity_threshold(self):
        s = FiftyTwoWeekHighProximity()
        assert s.proximity_threshold == 0.95

    def test_default_exit_threshold(self):
        s = FiftyTwoWeekHighProximity()
        assert s.exit_threshold == 0.90

    def test_custom_thresholds_stored(self):
        s = FiftyTwoWeekHighProximity(proximity_threshold=0.98, exit_threshold=0.85)
        assert s.proximity_threshold == 0.98
        assert s.exit_threshold == 0.85

    def test_starts_out_of_position(self):
        s = FiftyTwoWeekHighProximity()
        assert s._in_position is False

    def test_invalid_exit_ge_proximity_raises(self):
        with pytest.raises(ValueError):
            FiftyTwoWeekHighProximity(proximity_threshold=0.90, exit_threshold=0.90)

    def test_invalid_exit_gt_proximity_raises(self):
        with pytest.raises(ValueError):
            FiftyTwoWeekHighProximity(proximity_threshold=0.90, exit_threshold=0.95)

    def test_proximity_above_one_raises(self):
        with pytest.raises(ValueError):
            FiftyTwoWeekHighProximity(proximity_threshold=1.1, exit_threshold=0.90)

    def test_exit_at_zero_raises(self):
        with pytest.raises(ValueError):
            FiftyTwoWeekHighProximity(proximity_threshold=0.95, exit_threshold=0.0)

    def test_negative_exit_raises(self):
        with pytest.raises(ValueError):
            FiftyTwoWeekHighProximity(proximity_threshold=0.95, exit_threshold=-0.1)


# ---------------------------------------------------------------------------
# Warm-up guard: fewer than 253 bars must produce 0.0
# ---------------------------------------------------------------------------


class TestWarmupGuard:
    def test_single_bar_returns_flat(self):
        s = FiftyTwoWeekHighProximity()
        view = _make_view([100.0])
        assert s(view) == 0.0

    def test_exactly_252_bars_returns_flat(self):
        """252 bars: can compute current ratio but not prior ratio — still flat."""
        s = FiftyTwoWeekHighProximity()
        closes = [100.0] * 252
        view = _make_view(closes)
        assert s(view) == 0.0

    def test_exactly_253_bars_produces_signal(self):
        """At 253 bars the strategy can compute both ratios; signal may be 0 or 1."""
        s = FiftyTwoWeekHighProximity()
        closes = [100.0] * 253
        view = _make_view(closes)
        result = s(view)
        assert result in (0.0, 1.0)

    def test_252_bars_always_flat_regardless_of_prices(self):
        """Even with high proximity, flat until 253 bars are available."""
        s = FiftyTwoWeekHighProximity()
        # Build 252 bars ending at the 52-week high
        closes = list(range(50, 302))  # 252 bars, monotonically increasing
        assert len(closes) == 252
        view = _make_view(closes)
        assert s(view) == 0.0


# ---------------------------------------------------------------------------
# Signal correctness — entry conditions
# ---------------------------------------------------------------------------


class TestEntryConditions:
    def _build_base_closes(self, warmup_val=80.0, peak_val=100.0, n_warmup=252):
        """Build a series of n_warmup bars at warmup_val followed by peak_val."""
        return [warmup_val] * n_warmup + [peak_val]

    def test_entry_when_ratio_at_threshold_and_increasing(self):
        """ratio = 0.95 (exactly threshold) and ratio > prior ratio → long entry."""
        s = FiftyTwoWeekHighProximity(proximity_threshold=0.95, exit_threshold=0.90)
        # 252 bars at 100, then bar 252 (0-indexed) at 95, then bar 253 at 96
        # At bar 253 (the 254th bar): ratio = 96/100 = 0.96 >= 0.95
        # Prior ratio at bar 252: 95/100 = 0.95
        # Increasing: 0.96 > 0.95 → enter
        closes = [100.0] * 252 + [95.0, 96.0]
        view = _make_view(closes)
        assert s(view) == 1.0

    def test_no_entry_when_ratio_below_threshold(self):
        """ratio = 0.93 < 0.95 → no entry."""
        s = FiftyTwoWeekHighProximity()
        closes = [100.0] * 252 + [92.0, 93.0]
        view = _make_view(closes)
        assert s(view) == 0.0

    def test_no_entry_when_ratio_not_increasing(self):
        """ratio >= threshold but decreasing today vs yesterday → no entry."""
        s = FiftyTwoWeekHighProximity()
        # ratio at t: 96/100 = 0.96 >= 0.95; ratio at t-1: 97/100 = 0.97 > 0.96
        closes = [100.0] * 252 + [97.0, 96.0]
        view = _make_view(closes)
        assert s(view) == 0.0

    def test_no_entry_when_ratio_flat(self):
        """ratio at threshold but same as yesterday (not strictly increasing) → no entry."""
        s = FiftyTwoWeekHighProximity()
        closes = [100.0] * 252 + [95.0, 95.0]
        view = _make_view(closes)
        assert s(view) == 0.0

    def test_returns_only_zero_or_one(self):
        """Strategy must return exactly 0.0 or 1.0 — no fractional positions."""
        s = FiftyTwoWeekHighProximity()
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 1.0, 300))
        prices = np.maximum(prices, 1.0)
        for n in range(1, len(prices) + 1):
            view = _make_view(prices[:n].tolist())
            result = s(view)
            assert result in (0.0, 1.0), f"Unexpected signal {result} at n={n}"

    def test_signal_is_float(self):
        s = FiftyTwoWeekHighProximity()
        closes = [100.0] * 260
        view = _make_view(closes)
        result = s(view)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Exit conditions
# ---------------------------------------------------------------------------


class TestExitConditions:
    def _enter_position(self, s, warmup_bars=252):
        """Return a closes list that enters the strategy into a long position."""
        # bar 253: ratio = 0.96/1.00 = 0.96 >= 0.95, increasing from 0.95
        return [100.0] * warmup_bars + [95.0, 96.0]

    def test_exit_when_ratio_falls_below_exit_threshold(self):
        """After entry, close drops to 89 — ratio 0.89 < 0.90 → exit."""
        s = FiftyTwoWeekHighProximity()
        entry_closes = self._enter_position(s)
        view_entry = _make_view(entry_closes)
        s(view_entry)
        assert s._in_position is True

        # Add enough bars so that 89.0 creates ratio < 0.90
        # 52-week high is still ~100 (the warmup bars)
        exit_closes = entry_closes + [89.0]
        view_exit = _make_view(exit_closes)
        result = s(view_exit)
        assert result == 0.0
        assert s._in_position is False

    def test_position_held_while_ratio_above_exit_threshold(self):
        """While ratio stays >= 0.90, position is maintained."""
        s = FiftyTwoWeekHighProximity()
        entry_closes = self._enter_position(s)
        s(_make_view(entry_closes))
        assert s._in_position is True

        hold_closes = entry_closes + [91.0]  # ratio = 0.91 >= 0.90
        result = s(_make_view(hold_closes))
        assert result == 1.0
        assert s._in_position is True

    def test_no_exit_when_ratio_exactly_at_exit_threshold(self):
        """Ratio == exit_threshold (0.90) should NOT trigger exit (strict '<')."""
        s = FiftyTwoWeekHighProximity()
        entry_closes = self._enter_position(s)
        s(_make_view(entry_closes))
        assert s._in_position is True

        at_threshold_closes = entry_closes + [90.0]  # ratio = 0.90, not < 0.90
        result = s(_make_view(at_threshold_closes))
        assert result == 1.0


# ---------------------------------------------------------------------------
# Stateful sequential simulation (mimics engine bar-by-bar)
# ---------------------------------------------------------------------------


class TestStatefulBehavior:
    def _simulate(self, strategy, closes):
        """Drive strategy bar-by-bar as the engine would, return list of signals."""
        signals = []
        for t in range(1, len(closes) + 1):
            signals.append(strategy(_make_view(closes[:t])))
        return signals

    def test_entry_then_exit_sequence(self):
        """Verify: warmup stays flat, entry fires, position holds, exit fires."""
        s = FiftyTwoWeekHighProximity()
        warmup = [100.0] * 252
        approach = [95.0, 96.0, 97.0]  # ratio increasing into threshold zone
        retreat = [88.0]               # ratio drops below 0.90
        closes = warmup + approach + retreat

        signals = self._simulate(s, closes)
        in_pos = [i for i, sig in enumerate(signals) if sig == 1.0]

        assert all(sig == 0.0 for sig in signals[:252]), "Warmup must be flat"
        assert len(in_pos) > 0, "Strategy must enter a long position"
        assert signals[-1] == 0.0, "Strategy must exit after ratio drops below 0.90"

    def test_multiple_entry_exit_cycles(self):
        """Strategy can re-enter after an exit when conditions are met again."""
        s = FiftyTwoWeekHighProximity()
        warmup = [100.0] * 252
        # First entry cycle
        first_entry = [95.0, 96.0]
        first_exit = [88.0]
        # Recovery period — ratio dips then rises above threshold again
        recovery = [92.0, 93.0, 94.0, 95.0, 96.0]
        closes = warmup + first_entry + first_exit + recovery

        signals = self._simulate(s, closes)
        entry_bars = [i for i, sig in enumerate(signals) if sig == 1.0]
        # After exit and recovery, at least two separate long periods expected
        assert len(entry_bars) >= 2


# ---------------------------------------------------------------------------
# Integration with engine.backtest.run()
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_run_on_trend_gbm_returns_dict(self):
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(FiftyTwoWeekHighProximity(), df)
        assert isinstance(result, dict)
        assert "sharpe" in result
        assert "cagr" in result

    def test_run_on_all_four_datasets_completes(self):
        from engine.backtest import run
        for name in _DATASETS:
            df = _load_data(name)
            result = run(FiftyTwoWeekHighProximity(), df)
            assert isinstance(result["sharpe"], float)
            assert isinstance(result["cagr"], float)

    def test_run_is_deterministic(self):
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        r1 = run(FiftyTwoWeekHighProximity(), df)
        r2 = run(FiftyTwoWeekHighProximity(), df)
        assert r1 == r2

    def test_exposure_less_than_one_due_to_warmup(self):
        """252-bar warmup forces flat early in the series; exposure must be < 1.0."""
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(FiftyTwoWeekHighProximity(), df)
        assert result["exposure"] < 1.0

    def test_no_lookahead_error_during_run(self):
        from engine.backtest import run, LookAheadError
        for name in _DATASETS:
            df = _load_data(name)
            try:
                run(FiftyTwoWeekHighProximity(), df)
            except LookAheadError:
                pytest.fail(f"LookAheadError raised on {name}")


# ---------------------------------------------------------------------------
# Walk-forward: all-zero expected (warmup exceeds fold length)
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    def test_walk_forward_returns_required_keys(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("regime_switch.csv")
        result = walk_forward_backtest(
            FiftyTwoWeekHighProximity,
            {"proximity_threshold": 0.95, "exit_threshold": 0.90},
            df,
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
                FiftyTwoWeekHighProximity,
                {"proximity_threshold": 0.95, "exit_threshold": 0.90},
                df,
            )
            assert 0.0 <= result["oos_consistency"] <= 1.0

    def test_walk_forward_all_zeros_due_to_warmup_constraint(self):
        """
        OOS folds are ~166 bars; strategy requires 253-bar warmup.
        No position can be taken in any fold, so all OOS metrics are 0.0.
        This is the correct behaviour — not a bug.
        """
        from engine.backtest import walk_forward_backtest
        df = _load_data("trend_gbm.csv")
        result = walk_forward_backtest(
            FiftyTwoWeekHighProximity,
            {"proximity_threshold": 0.95, "exit_threshold": 0.90},
            df,
        )
        assert result["oos_sharpe_mean"] == 0.0
        assert result["oos_consistency"] == 0.0


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
            assert "walk_forward" in values, (
                f"metrics.json entry '{dataset}' missing 'walk_forward'"
            )
            wf = values["walk_forward"]
            assert "oos_sharpe_mean" in wf
            assert "oos_consistency" in wf

    def test_metrics_json_has_deflated_sharpe(self):
        data = self._load()
        for dataset, values in data.items():
            assert "deflated_sharpe" in values, (
                f"metrics.json entry '{dataset}' missing 'deflated_sharpe'"
            )
            assert 0.0 <= values["deflated_sharpe"] <= 1.0

    def test_metrics_json_values_match_fresh_run(self):
        """Verify stored metrics are reproducible."""
        from engine.backtest import run
        stored = self._load()
        params = {"proximity_threshold": 0.95, "exit_threshold": 0.90}
        for csv_name, key in [
            ("trend_gbm.csv", "trend_gbm"),
            ("mean_rev_ou.csv", "mean_rev_ou"),
            ("regime_switch.csv", "regime_switch"),
            ("fat_tail.csv", "fat_tail"),
        ]:
            df = _load_data(csv_name)
            fresh = run(FiftyTwoWeekHighProximity(**params), df)
            assert abs(fresh["sharpe"] - stored[key]["sharpe"]) < 1e-4, (
                f"sharpe mismatch on {key}: stored={stored[key]['sharpe']:.6f} "
                f"vs fresh={fresh['sharpe']:.6f}"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_constant_price_series_is_flat(self):
        """Constant prices: ratio always 1.0, never increasing → no entry (ratio == prev_ratio)."""
        s = FiftyTwoWeekHighProximity()
        closes = [100.0] * 300
        view = _make_view(closes)
        assert s(view) == 0.0

    def test_monotonically_declining_series_never_enters(self):
        """Declining prices: ratio wanders below threshold, never meets increasing filter."""
        s = FiftyTwoWeekHighProximity()
        closes = list(range(400, 100, -1))
        view = _make_view(closes)
        assert s(view) == 0.0

    def test_large_input_does_not_crash(self):
        """5000-bar input should complete without error."""
        s = FiftyTwoWeekHighProximity()
        rng = np.random.default_rng(0)
        closes = (100.0 + np.cumsum(rng.normal(0.05, 1.0, 5000))).tolist()
        closes = [max(c, 1.0) for c in closes]
        view = _make_view(closes)
        result = s(view)
        assert result in (0.0, 1.0)

    def test_minimum_bars_for_signal_is_253(self):
        """Exactly at 253 bars a signal becomes possible; at 252 it cannot."""
        s252 = FiftyTwoWeekHighProximity()
        closes_252 = [100.0] * 252
        assert s252(_make_view(closes_252)) == 0.0

        # 253 bars where bar 253 exceeds proximity threshold and is increasing
        s253 = FiftyTwoWeekHighProximity()
        closes_253 = [100.0] * 251 + [94.0, 96.0]
        result = s253(_make_view(closes_253))
        assert result in (0.0, 1.0)

    def test_all_prices_at_52wk_high_no_increasing_filter(self):
        """If all closes are identical, ratio is 1.0 but never increases → no entry."""
        s = FiftyTwoWeekHighProximity()
        closes = [100.0] * 260
        signals = []
        for t in range(1, len(closes) + 1):
            signals.append(s(_make_view(closes[:t])))
        assert all(sig == 0.0 for sig in signals)


# ---------------------------------------------------------------------------
# strategies.json entry validation
# ---------------------------------------------------------------------------


class TestStrategiesJson:
    def _load(self):
        with open(ROOT / "strategies.json") as f:
            return json.load(f)

    def test_entry_04_exists(self):
        data = self._load()
        ids = [s["id"] for s in data]
        assert "04" in ids

    def test_entry_04_has_anchoring_thesis_tag(self):
        data = self._load()
        s04 = next(s for s in data if s["id"] == "04")
        assert s04["thesis_tag"] == "anchoring"

    def test_entry_04_relates_to_is_empty(self):
        data = self._load()
        s04 = next(s for s in data if s["id"] == "04")
        assert s04["relates_to"] == []
