"""Tests for strategies/02-rsi-mean-reversion/strategy.py.

Covers:
  - Happy path: long signal when RSI drops below oversold threshold
  - Exit signal: flat when RSI rises above overbought threshold
  - Position hold: strategy stays long between entry and exit signals
  - Warm-up guard: flat when fewer bars than rsi_period + 1
  - Default parameter values
  - RSI arithmetic: verified against manual Wilder EWM computation
  - Integration with engine.backtest.run() on all four synthetic datasets
  - Integration with walk_forward_backtest() and metrics.json reproducibility
  - Edge cases: constant prices, single bar, large input, alternating prices
  - Error cases: invalid parameter arguments
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
STRATEGY_FILE = ROOT / "strategies" / "02-rsi-mean-reversion" / "strategy.py"
METRICS_FILE = ROOT / "strategies" / "02-rsi-mean-reversion" / "metrics.json"

_DATASETS = ["trend_gbm.csv", "mean_rev_ou.csv", "regime_switch.csv", "fat_tail.csv"]


def _load_strategy_module():
    """Load RSIMeanReversion and _rsi from the hyphenated directory via importlib."""
    spec = importlib.util.spec_from_file_location("rsi_strategy", STRATEGY_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_strategy_module()
RSIMeanReversion = _mod.RSIMeanReversion
_rsi = _mod._rsi


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
    def test_default_rsi_period_is_2(self):
        s = RSIMeanReversion()
        assert s.rsi_period == 2

    def test_default_oversold_is_10(self):
        s = RSIMeanReversion()
        assert s.oversold == 10.0

    def test_default_overbought_is_90(self):
        s = RSIMeanReversion()
        assert s.overbought == 90.0

    def test_custom_params_stored(self):
        s = RSIMeanReversion(rsi_period=5, oversold=20.0, overbought=80.0)
        assert s.rsi_period == 5
        assert s.oversold == 20.0
        assert s.overbought == 80.0

    def test_starts_flat(self):
        """A fresh instance must begin with no position."""
        s = RSIMeanReversion()
        assert s._in_position is False

    def test_invalid_rsi_period_zero_raises(self):
        with pytest.raises(ValueError, match="rsi_period must be a positive integer"):
            RSIMeanReversion(rsi_period=0)

    def test_invalid_rsi_period_negative_raises(self):
        with pytest.raises(ValueError, match="rsi_period must be a positive integer"):
            RSIMeanReversion(rsi_period=-1)

    def test_invalid_oversold_ge_overbought_raises(self):
        with pytest.raises(ValueError, match="oversold and overbought"):
            RSIMeanReversion(oversold=90.0, overbought=10.0)

    def test_invalid_equal_thresholds_raises(self):
        with pytest.raises(ValueError, match="oversold and overbought"):
            RSIMeanReversion(oversold=50.0, overbought=50.0)


# ---------------------------------------------------------------------------
# Warm-up guard
# ---------------------------------------------------------------------------


class TestWarmupGuard:
    def test_single_bar_returns_flat(self):
        s = RSIMeanReversion()
        view = _make_view([100.0])
        assert s(view) == 0.0

    def test_rsi_period_bars_returns_flat(self):
        """Need rsi_period + 1 bars for a valid RSI; with exactly rsi_period bars → flat."""
        s = RSIMeanReversion(rsi_period=3)
        view = _make_view([100.0, 99.0, 98.0])
        assert s(view) == 0.0

    def test_rsi_period_plus_one_bars_produces_signal(self):
        """With rsi_period + 1 bars, RSI can be computed and a signal generated."""
        s = RSIMeanReversion(rsi_period=2)
        view = _make_view([100.0, 50.0, 1.0])
        result = s(view)
        assert result in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Signal correctness: entry, hold, exit
# ---------------------------------------------------------------------------


class TestSignalCorrectness:
    def test_enters_long_when_rsi_below_oversold(self):
        """A sharp price drop must push RSI(2) below 10, triggering a long entry."""
        s = RSIMeanReversion()
        closes = [100.0] * 10 + [80.0, 60.0]
        view = _make_view(closes)
        result = s(view)
        assert result == 1.0, f"Expected long entry (RSI extreme drop), got {result}"

    def test_stays_long_between_signals(self):
        """After entering, the strategy must hold long while RSI is between thresholds."""
        s = RSIMeanReversion()
        closes = [100.0] * 10 + [80.0, 60.0]
        view_entry = _make_view(closes)
        s(view_entry)
        assert s._in_position is True

        closes_hold = closes + [62.0]
        view_hold = _make_view(closes_hold)
        result = s(view_hold)
        assert result == 1.0, "Strategy must hold long when RSI is between thresholds"

    def test_exits_long_when_rsi_above_overbought(self):
        """After a strong recovery, RSI(2) should exceed 90 and trigger an exit."""
        s = RSIMeanReversion()
        closes = [100.0] * 10 + [80.0, 60.0, 62.0, 80.0, 100.0]
        for i in range(len(closes)):
            view = _make_view(closes[: i + 1])
            s(view)

        closes_exit = closes + [120.0, 140.0]
        for i in range(len(closes), len(closes_exit)):
            view = _make_view(closes_exit[: i + 1])
            result = s(view)

        assert result == 0.0 or s._in_position is False, (
            "Strategy should have exited long on strong RSI recovery"
        )

    def test_returns_only_zero_or_one(self):
        """Strategy must return exactly 0.0 or 1.0 — no fractional positions."""
        s = RSIMeanReversion()
        rng = np.random.default_rng(42)
        closes = (100.0 + np.cumsum(rng.normal(0, 2.0, 100))).tolist()
        for i in range(1, len(closes) + 1):
            view = _make_view(closes[:i])
            result = s(view)
            assert result in (0.0, 1.0), f"Got {result} at bar {i}"


# ---------------------------------------------------------------------------
# RSI arithmetic
# ---------------------------------------------------------------------------


class TestRSIArithmetic:
    def test_rsi_returns_neutral_on_insufficient_history(self):
        closes = pd.Series([100.0, 99.0])
        assert _rsi(closes, 3) == 50.0

    def test_rsi_all_up_days_returns_100(self):
        """When every move is upward, RSI must be 100 (no losses)."""
        closes = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        result = _rsi(closes, 2)
        assert result == 100.0

    def test_rsi_all_down_days_returns_zero(self):
        """When every move is downward, RSI must be 0 (no gains)."""
        closes = pd.Series([104.0, 103.0, 102.0, 101.0, 100.0])
        result = _rsi(closes, 2)
        assert result == 0.0

    def test_rsi_matches_manual_wilder_calculation(self):
        """Cross-verify against a manual EWM computation with alpha=1/period."""
        period = 2
        closes = pd.Series([100.0, 98.0, 97.0, 96.0, 94.0])
        delta = closes.diff().dropna()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        if avg_loss == 0:
            expected = 100.0
        else:
            rs = avg_gain / avg_loss
            expected = 100.0 - (100.0 / (1.0 + rs))
        assert abs(_rsi(closes, period) - expected) < 1e-9

    def test_rsi_in_range_0_to_100(self):
        """RSI must always be in [0, 100] for any valid input."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            closes = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, 30)))
            val = _rsi(closes, 2)
            assert 0.0 <= val <= 100.0, f"RSI out of range: {val}"


# ---------------------------------------------------------------------------
# Integration with engine.backtest.run()
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_run_on_trend_gbm_returns_dict(self):
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(RSIMeanReversion(), df)
        assert isinstance(result, dict)
        assert "sharpe" in result
        assert "cagr" in result

    def test_run_on_all_four_datasets_completes(self):
        from engine.backtest import run
        for name in _DATASETS:
            df = _load_data(name)
            result = run(RSIMeanReversion(), df)
            assert isinstance(result["sharpe"], float)
            assert isinstance(result["cagr"], float)

    def test_run_is_deterministic(self):
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        r1 = run(RSIMeanReversion(), df)
        r2 = run(RSIMeanReversion(), df)
        assert r1 == r2

    def test_no_lookahead_error_during_run(self):
        from engine.backtest import run, LookAheadError
        for name in _DATASETS:
            df = _load_data(name)
            try:
                run(RSIMeanReversion(), df)
            except LookAheadError:
                pytest.fail(f"LookAheadError raised on {name}")

    def test_exposure_less_than_one_due_to_selectivity(self):
        """RSI-2 trades infrequently; exposure must be well below 1.0."""
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        result = run(RSIMeanReversion(), df)
        assert result["exposure"] < 0.9, f"Exposure unexpectedly high: {result['exposure']}"


# ---------------------------------------------------------------------------
# Walk-forward integration
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    def test_walk_forward_returns_required_keys(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("regime_switch.csv")
        params = {"rsi_period": 2, "oversold": 10.0, "overbought": 90.0}
        result = walk_forward_backtest(RSIMeanReversion, params, df)
        expected_keys = {
            "oos_sharpe_mean", "oos_sharpe_std", "oos_cagr_mean",
            "oos_max_drawdown_mean", "oos_consistency",
        }
        assert expected_keys == set(result.keys())

    def test_walk_forward_consistency_in_unit_interval(self):
        from engine.backtest import walk_forward_backtest
        params = {"rsi_period": 2, "oversold": 10.0, "overbought": 90.0}
        for name in _DATASETS:
            df = _load_data(name)
            result = walk_forward_backtest(RSIMeanReversion, params, df)
            assert 0.0 <= result["oos_consistency"] <= 1.0, (
                f"oos_consistency out of range on {name}"
            )

    def test_walk_forward_is_deterministic(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("regime_switch.csv")
        params = {"rsi_period": 2, "oversold": 10.0, "overbought": 90.0}
        r1 = walk_forward_backtest(RSIMeanReversion, params, df)
        r2 = walk_forward_backtest(RSIMeanReversion, params, df)
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
                f"metrics.json entry '{dataset}' missing 'walk_forward' key"
            )
            wf = values["walk_forward"]
            assert "oos_sharpe_mean" in wf
            assert "oos_consistency" in wf

    def test_metrics_json_values_match_fresh_run(self):
        """Stored metrics must be reproducible: rerun backtest and compare."""
        from engine.backtest import run, walk_forward_backtest
        stored = self._load()
        params = {"rsi_period": 2, "oversold": 10.0, "overbought": 90.0}
        for csv_name, key in [
            ("trend_gbm.csv", "trend_gbm"),
            ("mean_rev_ou.csv", "mean_rev_ou"),
            ("regime_switch.csv", "regime_switch"),
            ("fat_tail.csv", "fat_tail"),
        ]:
            df = _load_data(csv_name)
            fresh = run(RSIMeanReversion(**params), df)
            assert abs(fresh["sharpe"] - stored[key]["sharpe"]) < 1e-4, (
                f"sharpe mismatch on {key}: stored={stored[key]['sharpe']:.6f} "
                f"vs fresh={fresh['sharpe']:.6f}"
            )
            wf_fresh = walk_forward_backtest(RSIMeanReversion, params, df)
            assert abs(
                wf_fresh["oos_sharpe_mean"] - stored[key]["walk_forward"]["oos_sharpe_mean"]
            ) < 1e-4, f"oos_sharpe_mean mismatch on {key}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_constant_price_series_stays_flat(self):
        """Flat prices produce RSI = neutral (50) — never an entry signal."""
        s = RSIMeanReversion()
        closes = [100.0] * 50
        view = _make_view(closes)
        result = s(view)
        assert result == 0.0

    def test_alternating_prices_does_not_crash(self):
        """Alternating up-down prices stress-test RSI stability."""
        s = RSIMeanReversion()
        closes = [100.0 + (i % 2) * 5 for i in range(50)]
        view = _make_view(closes)
        result = s(view)
        assert result in (0.0, 1.0)

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
        result = run(RSIMeanReversion(), df)
        assert isinstance(result["sharpe"], float)

    def test_each_new_instance_starts_flat(self):
        """Two separate instances must not share state."""
        df = _load_data("regime_switch.csv")
        from engine.backtest import run
        r1 = run(RSIMeanReversion(), df)
        r2 = run(RSIMeanReversion(), df)
        assert r1 == r2

    def test_two_bar_input_returns_valid_signal(self):
        """Minimum valid input for RSI(2): 3 bars (rsi_period + 1)."""
        s = RSIMeanReversion()
        view = _make_view([100.0, 90.0, 80.0])
        result = s(view)
        assert result in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_single_bar_does_not_enter_position(self):
        """With only 1 bar, the strategy must never take a position."""
        s = RSIMeanReversion()
        view = _make_view([50.0])
        assert s(view) == 0.0
        assert s._in_position is False

    def test_does_not_short(self):
        """RSI-2 is long-only; signal must never be negative."""
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(RSIMeanReversion(), df)
        assert result.get("exposure", 0.0) >= 0.0

    def test_invalid_oversold_above_100_raises(self):
        with pytest.raises(ValueError):
            RSIMeanReversion(oversold=110.0, overbought=120.0)
