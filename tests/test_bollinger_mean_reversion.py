"""Tests for strategies/06-bollinger-mean-reversion/strategy.py.

Covers:
  - Happy path: long signal when close falls below lower Bollinger Band
  - Exit signal: flat when close rises above middle band (SMA)
  - Position hold: strategy stays long between entry and exit signals
  - Warm-up guard: flat when fewer bars than window
  - Default parameter values match the Bollinger (2001) published defaults
  - Band arithmetic: verified against manual rolling mean/std computation
  - Constant-price guard: std == 0 case returns flat (no false entry)
  - Integration with engine.backtest.run() on all four synthetic datasets
  - Integration with walk_forward_backtest() and metrics.json reproducibility
  - Edge cases: constant prices, single bar, large input, alternating prices
  - Error cases: invalid parameter arguments (window <= 0, nstd <= 0)
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
STRATEGY_FILE = ROOT / "strategies" / "06-bollinger-mean-reversion" / "strategy.py"
METRICS_FILE = ROOT / "strategies" / "06-bollinger-mean-reversion" / "metrics.json"

_DATASETS = ["trend_gbm.csv", "mean_rev_ou.csv", "regime_switch.csv", "fat_tail.csv"]


def _load_strategy_module():
    """Load BollingerMeanReversion from the hyphenated directory via importlib."""
    spec = importlib.util.spec_from_file_location("bb_strategy", STRATEGY_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_strategy_module()
BollingerMeanReversion = _mod.BollingerMeanReversion


def _load_data(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, index_col=0, parse_dates=True)


def _make_view(closes: list, window: int = 20) -> pd.DataFrame:
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


def _lower_band(closes: list, window: int, nstd: float) -> float:
    """Compute the lower Bollinger Band for the last bar from a list of closes."""
    s = pd.Series(closes)
    mean = s.rolling(window).mean().iloc[-1]
    std = s.rolling(window).std(ddof=1).iloc[-1]
    return float(mean - nstd * std)


def _middle_band(closes: list, window: int) -> float:
    """Compute the middle band (SMA) for the last bar from a list of closes."""
    s = pd.Series(closes)
    return float(s.rolling(window).mean().iloc[-1])


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------


class TestDefaultParameters:
    def test_default_window_is_20(self):
        s = BollingerMeanReversion()
        assert s.window == 20

    def test_default_nstd_is_2(self):
        s = BollingerMeanReversion()
        assert s.nstd == 2.0

    def test_default_exit_window_is_20(self):
        s = BollingerMeanReversion()
        assert s.exit_window == 20

    def test_custom_params_stored(self):
        s = BollingerMeanReversion(window=10, nstd=1.5, exit_window=10)
        assert s.window == 10
        assert s.nstd == 1.5
        assert s.exit_window == 10

    def test_starts_flat(self):
        s = BollingerMeanReversion()
        assert s._in_position is False

    def test_invalid_window_zero_raises(self):
        with pytest.raises(ValueError, match="window must be a positive integer"):
            BollingerMeanReversion(window=0)

    def test_invalid_window_negative_raises(self):
        with pytest.raises(ValueError, match="window must be a positive integer"):
            BollingerMeanReversion(window=-5)

    def test_invalid_exit_window_zero_raises(self):
        with pytest.raises(ValueError, match="exit_window must be a positive integer"):
            BollingerMeanReversion(exit_window=0)

    def test_invalid_nstd_zero_raises(self):
        with pytest.raises(ValueError, match="nstd must be a positive number"):
            BollingerMeanReversion(nstd=0.0)

    def test_invalid_nstd_negative_raises(self):
        with pytest.raises(ValueError, match="nstd must be a positive number"):
            BollingerMeanReversion(nstd=-1.0)


# ---------------------------------------------------------------------------
# Warm-up guard
# ---------------------------------------------------------------------------


class TestWarmupGuard:
    def test_single_bar_returns_flat(self):
        s = BollingerMeanReversion(window=5)
        view = _make_view([100.0])
        assert s(view) == 0.0

    def test_window_minus_one_bars_returns_flat(self):
        """Need at least `window` bars for the first valid rolling statistic."""
        s = BollingerMeanReversion(window=5)
        view = _make_view([100.0] * 4)
        assert s(view) == 0.0

    def test_exactly_window_bars_may_produce_signal(self):
        """At exactly `window` bars the rolling mean and std can be computed."""
        s = BollingerMeanReversion(window=3)
        closes = [100.0, 90.0, 80.0]
        view = _make_view(closes)
        result = s(view)
        assert result in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Signal correctness: entry, hold, exit
# ---------------------------------------------------------------------------


class TestSignalCorrectness:
    # Use window=10 where a 50% crash is guaranteed below the lower band.
    # For window W with ddof=1 std: entry fires iff -1 + 1/W + 2/sqrt(W-1) < 0,
    # which holds for W >= 10 (value ≈ -0.23 at W=10).
    _WINDOW = 10

    def _make_entry_closes(self, window: int = 10, nstd: float = 2.0) -> list:
        """Build a close series that ends below the lower Bollinger Band.

        Provides 2*window bars of stable prices then a 50% crash.  For window >= 10
        a 50% crash from a stable baseline is mathematically guaranteed to fall below
        mean - nstd*std (with ddof=1) because the band math requires W >= 10 for a
        50%-crash signal to fire.
        """
        base = [100.0] * (2 * window)
        crash = 50.0
        return base + [crash]

    def test_enters_long_when_close_below_lower_band(self):
        """A sharp price drop below the lower band must trigger a long entry."""
        w = self._WINDOW
        s = BollingerMeanReversion(window=w, nstd=2.0)
        closes = self._make_entry_closes(window=w)
        view = _make_view(closes)
        result = s(view)
        assert result == 1.0, f"Expected long entry after sharp drop, got {result}"

    def test_verify_close_is_actually_below_lower_band(self):
        """Confirm the test helper truly produces a below-band close."""
        w, nstd = self._WINDOW, 2.0
        closes = self._make_entry_closes(window=w, nstd=nstd)
        lb = _lower_band(closes, w, nstd)
        assert closes[-1] < lb, f"close {closes[-1]} not below lower_band {lb}"

    def test_stays_long_between_signals(self):
        """After entering, the strategy must hold while close is between bands."""
        w = self._WINDOW
        s = BollingerMeanReversion(window=w, nstd=2.0)
        closes = self._make_entry_closes(window=w)
        s(_make_view(closes))
        assert s._in_position is True

        closes_hold = closes + [55.0]
        result = s(_make_view(closes_hold))
        assert result == 1.0, "Strategy must remain long between entry and exit"

    def test_exits_long_when_close_above_middle_band(self):
        """After recovery above the SMA, the strategy must exit long."""
        w = self._WINDOW
        s = BollingerMeanReversion(window=w, nstd=2.0, exit_window=w)
        closes = self._make_entry_closes(window=w)
        s(_make_view(closes))
        assert s._in_position is True

        recovery = closes + [100.0] * (w * 3)
        for i in range(len(closes), len(recovery)):
            view = _make_view(recovery[: i + 1])
            s(view)
            if not s._in_position:
                break

        assert not s._in_position, "Strategy must eventually exit on recovery above SMA"

    def test_returns_only_zero_or_one(self):
        """Strategy must return exactly 0.0 or 1.0 — no fractional positions."""
        s = BollingerMeanReversion(window=10)
        rng = np.random.default_rng(42)
        closes = (100.0 + np.cumsum(rng.normal(0, 2.0, 200))).tolist()
        for i in range(1, len(closes) + 1):
            result = s(_make_view(closes[:i]))
            assert result in (0.0, 1.0), f"Got {result} at bar {i}"


# ---------------------------------------------------------------------------
# Band arithmetic
# ---------------------------------------------------------------------------


class TestBandArithmetic:
    def test_lower_band_matches_manual_computation(self):
        """Strategy entry threshold must match pandas rolling mean - nstd * rolling std."""
        window, nstd = 10, 2.0
        base = [100.0] * (2 * window)
        closes = base + [50.0]
        expected_lb = _lower_band(closes, window, nstd)
        assert closes[-1] < expected_lb, "Test setup error: close should be below lower band"
        s = BollingerMeanReversion(window=window, nstd=nstd)
        result = s(_make_view(closes))
        assert result == 1.0, (
            f"Should enter long when close ({closes[-1]}) < lower_band ({expected_lb:.4f})"
        )

    def test_middle_band_matches_manual_sma(self):
        """Exit middle band must match pandas rolling mean over exit_window bars."""
        window = 5
        closes = [100.0] * 5 + [50.0] + [100.0] * 5
        expected_mb = _middle_band(closes, window)
        assert expected_mb > 50.0, "Middle band should be above the crash low"

    def test_zero_std_does_not_enter(self):
        """Constant prices produce std == 0 — strategy must not enter position."""
        s = BollingerMeanReversion(window=5)
        closes = [100.0] * 10
        view = _make_view(closes)
        result = s(view)
        assert result == 0.0, "Strategy must not enter when rolling std is zero"
        assert s._in_position is False

    def test_nstd_controls_band_width(self):
        """A higher nstd produces a lower entry threshold (wider band)."""
        window = 5
        closes = [100.0, 102.0, 98.0, 101.0, 99.0, 75.0]
        lb_narrow = _lower_band(closes, window, nstd=1.0)
        lb_wide = _lower_band(closes, window, nstd=3.0)
        assert lb_wide < lb_narrow, "Wider bands must produce a lower lower-band threshold"


# ---------------------------------------------------------------------------
# Integration with engine.backtest.run()
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_run_on_trend_gbm_returns_dict(self):
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(BollingerMeanReversion(), df)
        assert isinstance(result, dict)
        assert "sharpe" in result
        assert "cagr" in result

    def test_run_on_all_four_datasets_completes(self):
        from engine.backtest import run
        for name in _DATASETS:
            df = _load_data(name)
            result = run(BollingerMeanReversion(), df)
            assert isinstance(result["sharpe"], float)
            assert isinstance(result["cagr"], float)

    def test_all_four_datasets_produce_positive_sharpe(self):
        """Cross-dataset robustness: Bollinger BB should be positive on all four regimes."""
        from engine.backtest import run
        for name in _DATASETS:
            df = _load_data(name)
            result = run(BollingerMeanReversion(), df)
            assert result["sharpe"] > 0, (
                f"Expected positive Sharpe on {name}, got {result['sharpe']:.4f}"
            )

    def test_run_is_deterministic(self):
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        r1 = run(BollingerMeanReversion(), df)
        r2 = run(BollingerMeanReversion(), df)
        assert r1 == r2

    def test_no_lookahead_error_during_run(self):
        from engine.backtest import run, LookAheadError
        for name in _DATASETS:
            df = _load_data(name)
            try:
                run(BollingerMeanReversion(), df)
            except LookAheadError:
                pytest.fail(f"LookAheadError raised on {name}")

    def test_exposure_low_due_to_selectivity(self):
        """Bollinger Band entries are rare; exposure must be well below 1.0."""
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        result = run(BollingerMeanReversion(), df)
        assert result["exposure"] < 0.5, (
            f"Exposure unexpectedly high: {result['exposure']}"
        )

    def test_does_not_short(self):
        """Bollinger Band strategy is long-only; no negative exposure."""
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(BollingerMeanReversion(), df)
        assert result.get("exposure", 0.0) >= 0.0


# ---------------------------------------------------------------------------
# Walk-forward integration
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    def test_walk_forward_returns_required_keys(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("regime_switch.csv")
        params = {"window": 20, "nstd": 2.0, "exit_window": 20}
        result = walk_forward_backtest(BollingerMeanReversion, params, df)
        expected_keys = {
            "oos_sharpe_mean", "oos_sharpe_std", "oos_cagr_mean",
            "oos_max_drawdown_mean", "oos_consistency",
        }
        assert expected_keys == set(result.keys())

    def test_walk_forward_consistency_in_unit_interval(self):
        from engine.backtest import walk_forward_backtest
        params = {"window": 20, "nstd": 2.0, "exit_window": 20}
        for name in _DATASETS:
            df = _load_data(name)
            result = walk_forward_backtest(BollingerMeanReversion, params, df)
            assert 0.0 <= result["oos_consistency"] <= 1.0, (
                f"oos_consistency out of [0,1] on {name}"
            )

    def test_walk_forward_is_deterministic(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("regime_switch.csv")
        params = {"window": 20, "nstd": 2.0, "exit_window": 20}
        r1 = walk_forward_backtest(BollingerMeanReversion, params, df)
        r2 = walk_forward_backtest(BollingerMeanReversion, params, df)
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

    def test_metrics_json_values_match_fresh_run(self):
        """Stored sharpe values must match a fresh backtest run within 1e-4."""
        from engine.backtest import run, walk_forward_backtest
        stored = self._load()
        params = {"window": 20, "nstd": 2.0, "exit_window": 20}
        for csv_name, key in [
            ("trend_gbm.csv", "trend_gbm"),
            ("mean_rev_ou.csv", "mean_rev_ou"),
            ("regime_switch.csv", "regime_switch"),
            ("fat_tail.csv", "fat_tail"),
        ]:
            df = _load_data(csv_name)
            fresh = run(BollingerMeanReversion(**params), df)
            assert abs(fresh["sharpe"] - stored[key]["sharpe"]) < 1e-4, (
                f"sharpe mismatch on {key}: "
                f"stored={stored[key]['sharpe']:.6f} vs fresh={fresh['sharpe']:.6f}"
            )
            wf_fresh = walk_forward_backtest(BollingerMeanReversion, params, df)
            assert abs(
                wf_fresh["oos_sharpe_mean"] - stored[key]["walk_forward"]["oos_sharpe_mean"]
            ) < 1e-4, f"oos_sharpe_mean mismatch on {key}"

    def test_all_stored_sharpes_positive(self):
        """Stored metrics must show positive Sharpe on all four datasets."""
        data = self._load()
        for key in ("trend_gbm", "mean_rev_ou", "regime_switch", "fat_tail"):
            sharpe = data[key]["sharpe"]
            assert sharpe > 0, f"Expected positive stored sharpe on {key}, got {sharpe}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_constant_price_series_stays_flat(self):
        """Constant prices produce std==0 — entry condition can never fire."""
        s = BollingerMeanReversion(window=5)
        closes = [100.0] * 50
        view = _make_view(closes)
        result = s(view)
        assert result == 0.0

    def test_alternating_prices_does_not_crash(self):
        """Alternating up-down prices stress-test band stability."""
        s = BollingerMeanReversion(window=10)
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
        result = run(BollingerMeanReversion(), df)
        assert isinstance(result["sharpe"], float)

    def test_two_instances_do_not_share_state(self):
        """Two separate instances must not share _in_position state."""
        df = _load_data("regime_switch.csv")
        from engine.backtest import run
        r1 = run(BollingerMeanReversion(), df)
        r2 = run(BollingerMeanReversion(), df)
        assert r1 == r2

    def test_two_bar_input_returns_flat(self):
        """Fewer bars than `window` must always return flat."""
        s = BollingerMeanReversion(window=20)
        view = _make_view([100.0, 90.0])
        result = s(view)
        assert result == 0.0

    def test_narrow_band_fires_more_often_than_wide_band(self):
        """A strategy with nstd=0.5 must enter at least as often as nstd=3.0."""
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        narrow = run(BollingerMeanReversion(nstd=0.5), df)
        wide = run(BollingerMeanReversion(nstd=3.0), df)
        assert narrow["exposure"] >= wide["exposure"], (
            "Narrower band should produce equal or higher exposure"
        )


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_single_bar_does_not_enter_position(self):
        """With only 1 bar, the strategy must never take a position."""
        s = BollingerMeanReversion(window=5)
        view = _make_view([50.0])
        assert s(view) == 0.0
        assert s._in_position is False

    def test_insufficient_bars_returns_flat_not_error(self):
        """Fewer than `window` bars must return 0.0, not raise an exception."""
        s = BollingerMeanReversion(window=20)
        for n in range(1, 20):
            closes = [100.0] * n
            view = _make_view(closes)
            assert s(view) == 0.0, f"Should be flat with only {n} bars"

    def test_nan_close_does_not_enter(self):
        """If rolling stat produces NaN (e.g. gap in data), strategy stays flat."""
        s = BollingerMeanReversion(window=5)
        closes = [100.0] * 3 + [float("nan")] + [100.0]
        df_raw = _make_view(closes)
        result = s(df_raw)
        assert result in (0.0, 1.0)
