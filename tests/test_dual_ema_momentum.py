"""Tests for strategies/01-dual-ema-momentum/strategy.py.

Covers:
  - Happy path: long signal when fast EMA > slow EMA on trending series
  - Flat signal when fast EMA < slow EMA (downtrend)
  - Warm-up guard: flat when fewer bars than slow_window
  - Default parameter values
  - Integration with engine.backtest.run() on all four synthetic datasets
  - Integration with walk_forward_backtest() and metrics.json structure
  - Edge cases: single bar, exactly slow_window bars, alternating prices
  - Error cases: invalid window arguments
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
STRATEGY_FILE = ROOT / "strategies" / "01-dual-ema-momentum" / "strategy.py"
METRICS_FILE = ROOT / "strategies" / "01-dual-ema-momentum" / "metrics.json"

_DATASETS = ["trend_gbm.csv", "mean_rev_ou.csv", "regime_switch.csv", "fat_tail.csv"]


def _load_strategy_class():
    """Load DualEMAMomentum from the hyphenated directory via importlib."""
    spec = importlib.util.spec_from_file_location("dual_ema_strategy", STRATEGY_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DualEMAMomentum


DualEMAMomentum = _load_strategy_class()


def _load_data(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, index_col=0, parse_dates=True)


def _make_view(closes: list) -> pd.DataFrame:
    """Construct a minimal OHLCV-style DataFrame from a list of close prices."""
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
    def test_default_fast_window_is_20(self):
        s = DualEMAMomentum()
        assert s.fast_window == 20

    def test_default_slow_window_is_60(self):
        s = DualEMAMomentum()
        assert s.slow_window == 60

    def test_custom_windows_stored(self):
        s = DualEMAMomentum(fast_window=5, slow_window=15)
        assert s.fast_window == 5
        assert s.slow_window == 15

    def test_invalid_fast_ge_slow_raises(self):
        with pytest.raises(ValueError, match="fast_window must be strictly less"):
            DualEMAMomentum(fast_window=60, slow_window=20)

    def test_invalid_equal_windows_raises(self):
        with pytest.raises(ValueError, match="fast_window must be strictly less"):
            DualEMAMomentum(fast_window=20, slow_window=20)

    def test_nonpositive_fast_window_raises(self):
        with pytest.raises(ValueError, match="positive integers"):
            DualEMAMomentum(fast_window=0, slow_window=60)

    def test_nonpositive_slow_window_raises(self):
        with pytest.raises(ValueError, match="positive integers"):
            DualEMAMomentum(fast_window=20, slow_window=-1)


# ---------------------------------------------------------------------------
# Warm-up guard (fewer bars than slow_window → flat)
# ---------------------------------------------------------------------------


class TestWarmupGuard:
    def test_single_bar_returns_flat(self):
        s = DualEMAMomentum(fast_window=5, slow_window=20)
        view = _make_view([100.0])
        assert s(view) == 0.0

    def test_slow_window_minus_one_bars_returns_flat(self):
        s = DualEMAMomentum(fast_window=5, slow_window=20)
        closes = [100.0 + i for i in range(19)]
        view = _make_view(closes)
        assert s(view) == 0.0

    def test_exactly_slow_window_bars_produces_signal(self):
        """At exactly slow_window bars, a strongly trending series should go long."""
        s = DualEMAMomentum(fast_window=5, slow_window=20)
        closes = list(range(100, 120))
        view = _make_view(closes)
        assert s(view) == 1.0


# ---------------------------------------------------------------------------
# Signal correctness
# ---------------------------------------------------------------------------


class TestSignalCorrectness:
    def test_long_signal_on_uptrend(self):
        """Strongly rising series: fast EMA must be above slow EMA."""
        s = DualEMAMomentum(fast_window=5, slow_window=20)
        closes = list(range(100, 200))
        view = _make_view(closes)
        assert s(view) == 1.0

    def test_flat_signal_on_downtrend(self):
        """Strongly falling series: fast EMA must be below slow EMA."""
        s = DualEMAMomentum(fast_window=5, slow_window=20)
        closes = list(range(200, 100, -1))
        view = _make_view(closes)
        assert s(view) == 0.0

    def test_returns_only_zero_or_one(self):
        """Strategy must return exactly 0.0 or 1.0 — no fractional positions."""
        s = DualEMAMomentum(fast_window=5, slow_window=20)
        for n in range(1, 50):
            closes = [100.0 + i * (1 if n % 2 == 0 else -1) for i in range(n)]
            view = _make_view(closes)
            result = s(view)
            assert result in (0.0, 1.0), f"Got unexpected signal {result} for n={n}"

    def test_signal_is_float(self):
        s = DualEMAMomentum(fast_window=5, slow_window=20)
        view = _make_view(list(range(100, 150)))
        result = s(view)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# EMA arithmetic — verify crossover logic against pandas directly
# ---------------------------------------------------------------------------


class TestEMAArithmetic:
    def test_signal_matches_manual_ema_calculation(self):
        """Cross-verify the strategy signal against a direct pandas EMA computation."""
        fast_w, slow_w = 10, 30
        s = DualEMAMomentum(fast_window=fast_w, slow_window=slow_w)
        rng = np.random.default_rng(42)
        closes = 100.0 + np.cumsum(rng.normal(0.1, 1.0, 60))
        view = _make_view(closes.tolist())
        close_series = pd.Series(closes)
        expected_fast = close_series.ewm(span=fast_w, adjust=False).mean().iloc[-1]
        expected_slow = close_series.ewm(span=slow_w, adjust=False).mean().iloc[-1]
        expected_signal = 1.0 if expected_fast > expected_slow else 0.0
        assert s(view) == expected_signal


# ---------------------------------------------------------------------------
# Integration with engine.backtest.run()
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_run_on_trend_gbm_returns_dict(self):
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        s = DualEMAMomentum()
        result = run(s, df)
        assert isinstance(result, dict)
        assert "sharpe" in result
        assert "cagr" in result

    def test_run_on_all_four_datasets_completes(self):
        from engine.backtest import run
        for name in _DATASETS:
            df = _load_data(name)
            s = DualEMAMomentum()
            result = run(s, df)
            assert isinstance(result["sharpe"], float)
            assert isinstance(result["cagr"], float)

    def test_run_is_deterministic(self):
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        r1 = run(DualEMAMomentum(), df)
        r2 = run(DualEMAMomentum(), df)
        assert r1 == r2

    def test_exposure_less_than_one_due_to_warmup(self):
        """The warmup period forces flat positions, so exposure < 1.0."""
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(DualEMAMomentum(), df)
        assert result["exposure"] < 1.0

    def test_no_lookahead_error_during_run(self):
        """Strategy must not trigger LookAheadError on any dataset."""
        from engine.backtest import run, LookAheadError
        for name in _DATASETS:
            df = _load_data(name)
            try:
                run(DualEMAMomentum(), df)
            except LookAheadError:
                pytest.fail(f"LookAheadError raised on {name}")


# ---------------------------------------------------------------------------
# Walk-forward backtest integration
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    def test_walk_forward_returns_required_keys(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("trend_gbm.csv")
        result = walk_forward_backtest(
            DualEMAMomentum, {"fast_window": 20, "slow_window": 60}, df
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
                DualEMAMomentum, {"fast_window": 20, "slow_window": 60}, df
            )
            assert 0.0 <= result["oos_consistency"] <= 1.0, (
                f"oos_consistency out of range on {name}"
            )

    def test_walk_forward_is_deterministic(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("trend_gbm.csv")
        params = {"fast_window": 20, "slow_window": 60}
        r1 = walk_forward_backtest(DualEMAMomentum, params, df)
        r2 = walk_forward_backtest(DualEMAMomentum, params, df)
        assert r1 == r2


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
                f"metrics.json entry '{dataset}' missing 'walk_forward' key"
            )
            wf = values["walk_forward"]
            assert "oos_sharpe_mean" in wf
            assert "oos_consistency" in wf

    def test_metrics_json_values_match_fresh_run(self):
        """Verify stored metrics are reproducible: rerun backtest and compare."""
        from engine.backtest import run, walk_forward_backtest
        stored = self._load()
        params = {"fast_window": 20, "slow_window": 60}
        for csv_name, key in [
            ("trend_gbm.csv", "trend_gbm"),
            ("mean_rev_ou.csv", "mean_rev_ou"),
            ("regime_switch.csv", "regime_switch"),
            ("fat_tail.csv", "fat_tail"),
        ]:
            df = _load_data(csv_name)
            fresh = run(DualEMAMomentum(**params), df)
            assert abs(fresh["sharpe"] - stored[key]["sharpe"]) < 1e-4, (
                f"sharpe mismatch on {key}: stored={stored[key]['sharpe']:.6f} "
                f"vs fresh={fresh['sharpe']:.6f}"
            )
            wf_fresh = walk_forward_backtest(DualEMAMomentum, params, df)
            assert abs(
                wf_fresh["oos_sharpe_mean"] - stored[key]["walk_forward"]["oos_sharpe_mean"]
            ) < 1e-4, f"oos_sharpe_mean mismatch on {key}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_constant_price_series_is_flat_after_warmup(self):
        """When fast and slow EMA are equal (flat price), signal must be flat."""
        s = DualEMAMomentum(fast_window=5, slow_window=20)
        closes = [100.0] * 40
        view = _make_view(closes)
        assert s(view) == 0.0

    def test_alternating_prices_does_not_crash(self):
        """Alternating up-down prices stress-test the EMA calculation."""
        s = DualEMAMomentum(fast_window=5, slow_window=20)
        closes = [100.0 + (i % 2) * 5 for i in range(50)]
        view = _make_view(closes)
        result = s(view)
        assert result in (0.0, 1.0)

    def test_very_short_fast_window(self):
        """fast_window=1 with slow_window=2 is the smallest valid configuration."""
        s = DualEMAMomentum(fast_window=1, slow_window=2)
        closes = [100.0, 110.0]
        view = _make_view(closes)
        result = s(view)
        assert result in (0.0, 1.0)

    def test_large_input_does_not_crash(self):
        """5000-bar input should complete without error."""
        s = DualEMAMomentum()
        rng = np.random.default_rng(0)
        closes = (100.0 + np.cumsum(rng.normal(0.05, 1.0, 5000))).tolist()
        view = _make_view(closes)
        result = s(view)
        assert result in (0.0, 1.0)
