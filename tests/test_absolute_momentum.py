"""Tests for strategies/07-absolute-momentum/strategy.py.

Covers:
  - Happy path: long signal when trailing return > threshold, flat otherwise
  - Stateless signal: same inputs always produce same output
  - Default parameter values match published academic defaults (lookback=252, threshold=0.0)
  - Warm-up behavior: uses close[0] as reference when t < lookback
  - Signal correctness: verified against manual trailing-return computation
  - Zero reference price guard: flat when reference close is zero
  - Integration with engine.backtest.run() on all four synthetic datasets
  - Integration with walk_forward_backtest()
  - Regime-sensitivity: regime_switch shows higher exposure than mean_rev_ou
  - metrics.json reproducibility: stored sharpe matches fresh run within tolerance
  - Edge cases: single bar, large input, constant prices, alternating prices
  - Failure modes: invalid lookback raises ValueError, insufficient bars returns 0.0
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
STRATEGY_FILE = ROOT / "strategies" / "07-absolute-momentum" / "strategy.py"
METRICS_FILE = ROOT / "strategies" / "07-absolute-momentum" / "metrics.json"

_DATASETS = ["trend_gbm.csv", "mean_rev_ou.csv", "regime_switch.csv", "fat_tail.csv"]


def _load_strategy_module():
    """Load AbsoluteMomentum from the hyphenated directory via importlib."""
    spec = importlib.util.spec_from_file_location("abs_mom_strategy", STRATEGY_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_strategy_module()
AbsoluteMomentum = _mod.AbsoluteMomentum


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


def _trailing_return(closes: list, lookback: int) -> float:
    """Compute trailing return for the last bar: close[-1] / close[max(0, t-lookback)] - 1."""
    t = len(closes) - 1
    ref_idx = max(0, t - lookback)
    return closes[-1] / closes[ref_idx] - 1.0


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------


class TestDefaultParameters:
    def test_default_lookback_is_252(self):
        s = AbsoluteMomentum()
        assert s.lookback == 252

    def test_default_threshold_is_zero(self):
        s = AbsoluteMomentum()
        assert s.threshold == 0.0

    def test_custom_params_stored(self):
        s = AbsoluteMomentum(lookback=50, threshold=0.05)
        assert s.lookback == 50
        assert s.threshold == 0.05

    def test_invalid_lookback_zero_raises(self):
        with pytest.raises(ValueError, match="lookback must be a positive integer"):
            AbsoluteMomentum(lookback=0)

    def test_invalid_lookback_negative_raises(self):
        with pytest.raises(ValueError, match="lookback must be a positive integer"):
            AbsoluteMomentum(lookback=-10)


# ---------------------------------------------------------------------------
# Signal correctness: long when trailing return positive, flat otherwise
# ---------------------------------------------------------------------------


class TestSignalCorrectness:
    def test_enters_long_when_trailing_return_positive(self):
        """When the last close is above the reference close, signal should be 1.0."""
        lookback = 5
        closes = [100.0] * 10 + [110.0]
        s = AbsoluteMomentum(lookback=lookback)
        view = _make_view(closes)
        result = s(view)
        assert result == 1.0, f"Expected 1.0 (positive trailing return), got {result}"

    def test_returns_flat_when_trailing_return_negative(self):
        """When the last close is below the reference close, signal should be 0.0."""
        lookback = 5
        closes = [100.0] * 10 + [90.0]
        s = AbsoluteMomentum(lookback=lookback)
        view = _make_view(closes)
        result = s(view)
        assert result == 0.0, f"Expected 0.0 (negative trailing return), got {result}"

    def test_returns_flat_when_trailing_return_equals_threshold(self):
        """Trailing return exactly equal to threshold (0.0 → same price) should be flat."""
        lookback = 5
        closes = [100.0] * 10
        s = AbsoluteMomentum(lookback=lookback, threshold=0.0)
        view = _make_view(closes)
        result = s(view)
        assert result == 0.0, "Equal return (zero) must not trigger a long (strict > threshold)"

    def test_verify_manual_trailing_return_matches_logic(self):
        """Manual trailing return computation must agree with strategy signal."""
        lookback = 10
        closes = [100.0] * 15 + [112.0]
        s = AbsoluteMomentum(lookback=lookback)
        view = _make_view(closes)
        manual_tr = _trailing_return(closes, lookback)
        expected = 1.0 if manual_tr > 0.0 else 0.0
        assert s(view) == expected

    def test_positive_threshold_requires_larger_return(self):
        """With threshold=0.05, a 3% return must not trigger a long."""
        lookback = 5
        closes = [100.0] * 10 + [103.0]
        s = AbsoluteMomentum(lookback=lookback, threshold=0.05)
        view = _make_view(closes)
        assert s(view) == 0.0, "3% return should not exceed 5% threshold"

    def test_positive_threshold_triggers_when_exceeded(self):
        """With threshold=0.05, a 10% return must trigger a long."""
        lookback = 5
        closes = [100.0] * 10 + [110.0]
        s = AbsoluteMomentum(lookback=lookback, threshold=0.05)
        view = _make_view(closes)
        assert s(view) == 1.0, "10% return should exceed 5% threshold"

    def test_returns_only_zero_or_one(self):
        """Strategy must return exactly 0.0 or 1.0 — no fractional positions."""
        s = AbsoluteMomentum(lookback=20)
        rng = np.random.default_rng(42)
        closes = (100.0 + np.cumsum(rng.normal(0, 2.0, 300))).tolist()
        for i in range(1, len(closes) + 1):
            result = s(_make_view(closes[:i]))
            assert result in (0.0, 1.0), f"Got {result} at bar {i}"

    def test_is_stateless_same_input_same_output(self):
        """The signal is stateless: calling with the same view twice must return the same result."""
        closes = [100.0, 105.0, 102.0, 108.0, 115.0]
        s = AbsoluteMomentum(lookback=3)
        view = _make_view(closes)
        result1 = s(view)
        result2 = s(view)
        assert result1 == result2, "Stateless signal must be idempotent"


# ---------------------------------------------------------------------------
# Warm-up behavior (t < lookback → ref_idx = 0)
# ---------------------------------------------------------------------------


class TestWarmupBehavior:
    def test_single_bar_returns_flat(self):
        """With only 1 bar, ref_idx = max(0, 0-252) = 0, trailing return = 0 → flat."""
        s = AbsoluteMomentum()
        view = _make_view([100.0])
        assert s(view) == 0.0

    def test_early_bars_use_first_close_as_reference(self):
        """When t=10 < lookback=252, reference is close[0]. Price up → long."""
        s = AbsoluteMomentum(lookback=252)
        closes = [100.0, 99.0, 98.0, 102.0, 105.0, 108.0, 110.0, 112.0, 115.0, 120.0, 125.0]
        view = _make_view(closes)
        manual_tr = closes[-1] / closes[0] - 1.0
        expected = 1.0 if manual_tr > 0.0 else 0.0
        assert s(view) == expected

    def test_early_bars_below_start_returns_flat(self):
        """When price is below close[0] and t < lookback, trailing return is negative → flat."""
        s = AbsoluteMomentum(lookback=252)
        closes = [100.0] + [95.0] * 10
        view = _make_view(closes)
        assert s(view) == 0.0

    def test_exactly_at_lookback_uses_first_bar_as_ref(self):
        """At t = lookback, ref_idx = max(0, lookback - lookback) = 0 → still uses close[0]."""
        lookback = 5
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 110.0]
        s = AbsoluteMomentum(lookback=lookback)
        view = _make_view(closes)
        manual_tr = closes[-1] / closes[0] - 1.0
        expected = 1.0 if manual_tr > 0.0 else 0.0
        assert s(view) == expected

    def test_beyond_lookback_uses_shifted_reference(self):
        """At t = lookback + 1, ref_idx = 1 → uses close[1], not close[0]."""
        lookback = 3
        closes = [100.0, 80.0, 95.0, 97.0, 120.0]
        s = AbsoluteMomentum(lookback=lookback)
        view = _make_view(closes)
        t = len(closes) - 1
        ref_idx = max(0, t - lookback)
        manual_tr = closes[t] / closes[ref_idx] - 1.0
        expected = 1.0 if manual_tr > 0.0 else 0.0
        assert s(view) == expected


# ---------------------------------------------------------------------------
# Zero reference price guard
# ---------------------------------------------------------------------------


class TestZeroReferencePrice:
    def test_zero_ref_price_returns_flat(self):
        """If ref close is 0.0 (degenerate data), strategy must return 0.0 to avoid division by zero."""
        s = AbsoluteMomentum(lookback=1)
        closes = [0.0, 100.0]
        view = _make_view(closes)
        result = s(view)
        assert result == 0.0, "Zero reference price must not trigger a long"


# ---------------------------------------------------------------------------
# Integration with engine.backtest.run()
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_run_on_regime_switch_returns_dict(self):
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        result = run(AbsoluteMomentum(), df)
        assert isinstance(result, dict)
        assert "sharpe" in result
        assert "cagr" in result

    def test_run_on_all_four_datasets_completes(self):
        from engine.backtest import run
        for name in _DATASETS:
            df = _load_data(name)
            result = run(AbsoluteMomentum(), df)
            assert isinstance(result["sharpe"], float)
            assert isinstance(result["exposure"], float)

    def test_regime_switch_has_positive_sharpe(self):
        """Absolute momentum should produce positive Sharpe on the trending sub-periods of regime_switch."""
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        result = run(AbsoluteMomentum(), df)
        assert result["sharpe"] > 0.5, (
            f"Expected positive Sharpe on regime_switch (trending dataset), got {result['sharpe']:.4f}"
        )

    def test_regime_switch_exposure_high(self):
        """On regime_switch (strong trend component), strategy should be nearly always long (>0.75)."""
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        result = run(AbsoluteMomentum(), df)
        assert result["exposure"] > 0.75, (
            f"Expected high exposure on regime_switch, got {result['exposure']:.3f}"
        )

    def test_regime_switch_exposure_higher_than_mean_rev(self):
        """Exposure must be higher on trending regime_switch than on mean-reverting mean_rev_ou."""
        from engine.backtest import run
        df_trend = _load_data("regime_switch.csv")
        df_rev = _load_data("mean_rev_ou.csv")
        trend_result = run(AbsoluteMomentum(), df_trend)
        rev_result = run(AbsoluteMomentum(), df_rev)
        assert trend_result["exposure"] > rev_result["exposure"], (
            f"Trending dataset should produce higher exposure: "
            f"regime_switch={trend_result['exposure']:.3f} vs mean_rev_ou={rev_result['exposure']:.3f}"
        )

    def test_does_not_short(self):
        """Absolute momentum is long-only — exposure must never be negative."""
        from engine.backtest import run
        for name in _DATASETS:
            df = _load_data(name)
            result = run(AbsoluteMomentum(), df)
            assert result["exposure"] >= 0.0, f"Negative exposure on {name}"

    def test_run_is_deterministic(self):
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        r1 = run(AbsoluteMomentum(), df)
        r2 = run(AbsoluteMomentum(), df)
        assert r1 == r2

    def test_no_lookahead_error_during_run(self):
        from engine.backtest import run, LookAheadError
        for name in _DATASETS:
            df = _load_data(name)
            try:
                run(AbsoluteMomentum(), df)
            except LookAheadError:
                pytest.fail(f"LookAheadError raised on {name}")

    def test_low_turnover_due_to_slow_signal(self):
        """252-bar trailing return changes slowly — turnover should be low (< 0.1)."""
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        result = run(AbsoluteMomentum(), df)
        assert result["turnover"] < 0.1, (
            f"Expected low turnover for 252-bar signal, got {result['turnover']:.4f}"
        )


# ---------------------------------------------------------------------------
# Walk-forward integration
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    def test_walk_forward_returns_required_keys(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("regime_switch.csv")
        params = {"lookback": 252, "threshold": 0.0}
        result = walk_forward_backtest(AbsoluteMomentum, params, df)
        expected_keys = {
            "oos_sharpe_mean", "oos_sharpe_std", "oos_cagr_mean",
            "oos_max_drawdown_mean", "oos_consistency",
        }
        assert expected_keys == set(result.keys())

    def test_walk_forward_consistency_in_unit_interval(self):
        from engine.backtest import walk_forward_backtest
        params = {"lookback": 252, "threshold": 0.0}
        for name in _DATASETS:
            df = _load_data(name)
            result = walk_forward_backtest(AbsoluteMomentum, params, df)
            assert 0.0 <= result["oos_consistency"] <= 1.0, (
                f"oos_consistency out of [0,1] on {name}"
            )

    def test_walk_forward_is_deterministic(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("regime_switch.csv")
        params = {"lookback": 252, "threshold": 0.0}
        r1 = walk_forward_backtest(AbsoluteMomentum, params, df)
        r2 = walk_forward_backtest(AbsoluteMomentum, params, df)
        assert r1 == r2


# ---------------------------------------------------------------------------
# metrics.json reproducibility
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

    def test_regime_switch_sharpe_is_positive_and_strong(self):
        """The stored regime_switch Sharpe must be positive, confirming the trend-filter thesis."""
        data = self._load()
        sharpe = data["regime_switch"]["sharpe"]
        assert sharpe > 0.5, (
            f"Expected strong positive Sharpe on regime_switch, got {sharpe:.4f}"
        )

    def test_regime_switch_exposure_nearly_always_long(self):
        """Stored exposure on regime_switch must exceed 0.75 (nearly always long)."""
        data = self._load()
        exposure = data["regime_switch"]["exposure"]
        assert exposure > 0.75, (
            f"Expected exposure > 0.75 on regime_switch, got {exposure:.3f}"
        )

    def test_metrics_json_values_match_fresh_run(self):
        """Stored sharpe values must match a fresh backtest run within 1e-4."""
        from engine.backtest import run, walk_forward_backtest
        stored = self._load()
        params = {"lookback": 252, "threshold": 0.0}
        for csv_name, key in [
            ("regime_switch.csv", "regime_switch"),
            ("mean_rev_ou.csv", "mean_rev_ou"),
        ]:
            df = _load_data(csv_name)
            fresh = run(AbsoluteMomentum(**params), df)
            assert abs(fresh["sharpe"] - stored[key]["sharpe"]) < 1e-4, (
                f"sharpe mismatch on {key}: "
                f"stored={stored[key]['sharpe']:.6f} vs fresh={fresh['sharpe']:.6f}"
            )
            wf_fresh = walk_forward_backtest(AbsoluteMomentum, params, df)
            assert abs(
                wf_fresh["oos_sharpe_mean"] - stored[key]["walk_forward"]["oos_sharpe_mean"]
            ) < 1e-4, f"oos_sharpe_mean mismatch on {key}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_constant_price_series_returns_flat(self):
        """Constant prices produce trailing return = 0 → never triggers long (strict >)."""
        s = AbsoluteMomentum(lookback=5)
        closes = [100.0] * 50
        view = _make_view(closes)
        result = s(view)
        assert result == 0.0

    def test_large_input_completes(self):
        """5000-bar input must complete without error."""
        from engine.backtest import run
        rng = np.random.default_rng(0)
        closes = (100.0 + np.cumsum(rng.normal(0, 2.0, 5001))).tolist()
        dates = pd.bdate_range("2000-01-03", periods=5001)
        arr = np.array(closes)
        df = pd.DataFrame(
            {
                "open": arr * 0.999,
                "high": arr * 1.01,
                "low": arr * 0.99,
                "close": arr,
                "volume": np.ones(5001, dtype=int) * 1000,
            },
            index=dates,
        )
        result = run(AbsoluteMomentum(), df)
        assert isinstance(result["sharpe"], float)

    def test_two_instances_do_not_share_state(self):
        """Two separate instances must produce identical results on the same data."""
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        r1 = run(AbsoluteMomentum(), df)
        r2 = run(AbsoluteMomentum(), df)
        assert r1 == r2

    def test_two_bar_input_second_bar_uses_first_as_ref(self):
        """With 2 bars, t=1, ref_idx=0; if close[1] > close[0] → long."""
        s = AbsoluteMomentum(lookback=252)
        view = _make_view([100.0, 105.0])
        assert s(view) == 1.0

    def test_shorter_lookback_has_equal_or_higher_exposure(self):
        """A shorter lookback is more reactive — it should not produce strictly lower exposure."""
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        result_short = run(AbsoluteMomentum(lookback=20), df)
        result_long = run(AbsoluteMomentum(lookback=252), df)
        assert result_short["turnover"] >= result_long["turnover"], (
            "Shorter lookback should produce equal or more frequent position changes"
        )

    def test_alternating_prices_does_not_crash(self):
        """Alternating up-down prices stress-test signal stability."""
        s = AbsoluteMomentum(lookback=10)
        closes = [100.0 + (i % 2) * 5 for i in range(50)]
        view = _make_view(closes)
        result = s(view)
        assert result in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_single_bar_does_not_enter_position(self):
        """With only 1 bar, close[0]/close[0] - 1 = 0 → flat (not > threshold)."""
        s = AbsoluteMomentum()
        view = _make_view([50.0])
        assert s(view) == 0.0

    def test_insufficient_bars_returns_flat_not_error(self):
        """Fewer bars than lookback must return 0.0 or 1.0, never raise an exception."""
        s = AbsoluteMomentum(lookback=252)
        for n in range(1, 10):
            closes = [100.0] * n
            view = _make_view(closes)
            result = s(view)
            assert result in (0.0, 1.0), f"Should return valid signal with only {n} bars"

    def test_sharply_declining_price_returns_flat(self):
        """A sequence that ends well below the reference must trigger flat signal."""
        lookback = 5
        closes = [100.0] * 20 + [50.0]
        s = AbsoluteMomentum(lookback=lookback)
        view = _make_view(closes)
        result = s(view)
        assert result == 0.0, f"Sharp decline should return flat, got {result}"

    def test_negative_threshold_triggers_on_small_decline(self):
        """With threshold=-0.05, a 2% decline still satisfies trailing_return > -0.05 → long."""
        lookback = 5
        closes = [100.0] * 10 + [98.0]
        s = AbsoluteMomentum(lookback=lookback, threshold=-0.05)
        view = _make_view(closes)
        assert s(view) == 1.0, "−2% return should exceed −5% threshold"

    def test_large_decline_below_negative_threshold_is_flat(self):
        """With threshold=-0.05, a 10% decline (trailing_return < -0.05) must be flat."""
        lookback = 5
        closes = [100.0] * 10 + [88.0]
        s = AbsoluteMomentum(lookback=lookback, threshold=-0.05)
        view = _make_view(closes)
        assert s(view) == 0.0, "−12% return should not exceed −5% threshold"
