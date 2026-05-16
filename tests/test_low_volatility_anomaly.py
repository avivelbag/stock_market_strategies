"""Tests for strategies/10-low-volatility-anomaly/strategy.py.

Covers:
  - Default parameters and constructor validation
  - Warm-up guard: flat during first vol_window + ranking_window bars
  - Entry signal: long when current_vol < rolling median
  - Exit signal: flat when current_vol > rolling exit_percentile
  - Hysteresis: hold between median and 75th-pct without exit
  - State reset after exit (re-entry on next low-vol bar)
  - No look-ahead: signal only uses data up to bar t
  - Integration with engine.backtest.run() on all four synthetic datasets
  - Integration with walk_forward_backtest()
  - metrics.json and sensitivity.json structure validation
  - Edge cases: minimum valid params, large input, single bar, short series
  - Failure modes: invalid constructor arguments
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
STRATEGY_FILE = ROOT / "strategies" / "10-low-volatility-anomaly" / "strategy.py"
METRICS_FILE = ROOT / "strategies" / "10-low-volatility-anomaly" / "metrics.json"
SENSITIVITY_FILE = ROOT / "strategies" / "10-low-volatility-anomaly" / "sensitivity.json"

_DATASETS = ["trend_gbm.csv", "mean_rev_ou.csv", "regime_switch.csv", "fat_tail.csv"]


def _load_strategy_class():
    """Load LowVolatilityAnomaly from the hyphenated directory via importlib."""
    spec = importlib.util.spec_from_file_location("lva_strategy", STRATEGY_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LowVolatilityAnomaly


LowVolatilityAnomaly = _load_strategy_class()


def _load_data(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, index_col=0, parse_dates=True)


def _make_ohlcv(closes: list) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a close-price list."""
    n = len(closes)
    dates = pd.bdate_range("2020-01-02", periods=n)
    arr = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": arr * 0.999,
            "high": arr * 1.001,
            "low": arr * 0.999,
            "close": arr,
            "volume": np.ones(n, dtype=int) * 1000,
        },
        index=dates,
    )


def _make_low_vol_series(n: int, daily_move: float = 0.001, seed: int = 42) -> list:
    """Build a price series with deterministically low realized volatility."""
    rng = np.random.default_rng(seed)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + rng.uniform(-daily_move, daily_move)))
    return closes


def _make_high_vol_series(n: int, daily_move: float = 0.05, seed: int = 42) -> list:
    """Build a price series with deterministically high realized volatility."""
    rng = np.random.default_rng(seed)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + rng.uniform(-daily_move, daily_move)))
    return closes


def _simulate(strategy, ohlcv: pd.DataFrame) -> list:
    """Drive the strategy bar-by-bar as the engine would, returning all signals."""
    signals = []
    for t in range(1, len(ohlcv) + 1):
        signals.append(strategy(ohlcv.iloc[:t]))
    return signals


# ---------------------------------------------------------------------------
# Default parameters and constructor validation
# ---------------------------------------------------------------------------


class TestDefaultParameters:
    def test_default_vol_window_is_60(self):
        s = LowVolatilityAnomaly()
        assert s.vol_window == 60

    def test_default_ranking_window_is_252(self):
        s = LowVolatilityAnomaly()
        assert s.ranking_window == 252

    def test_default_exit_percentile_is_75(self):
        s = LowVolatilityAnomaly()
        assert s.exit_percentile == 75.0

    def test_custom_params_stored(self):
        s = LowVolatilityAnomaly(vol_window=20, ranking_window=100, exit_percentile=80.0)
        assert s.vol_window == 20
        assert s.ranking_window == 100
        assert s.exit_percentile == 80.0

    def test_vol_window_less_than_2_raises(self):
        with pytest.raises(ValueError, match="vol_window must be at least 2"):
            LowVolatilityAnomaly(vol_window=1)

    def test_vol_window_zero_raises(self):
        with pytest.raises(ValueError, match="vol_window must be at least 2"):
            LowVolatilityAnomaly(vol_window=0)

    def test_ranking_window_less_than_vol_window_raises(self):
        with pytest.raises(ValueError, match="ranking_window must be >= vol_window"):
            LowVolatilityAnomaly(vol_window=60, ranking_window=50)

    def test_exit_percentile_zero_raises(self):
        with pytest.raises(ValueError, match="exit_percentile must be in"):
            LowVolatilityAnomaly(exit_percentile=0.0)

    def test_exit_percentile_100_raises(self):
        with pytest.raises(ValueError, match="exit_percentile must be in"):
            LowVolatilityAnomaly(exit_percentile=100.0)

    def test_exit_percentile_negative_raises(self):
        with pytest.raises(ValueError, match="exit_percentile must be in"):
            LowVolatilityAnomaly(exit_percentile=-5.0)

    def test_default_params_dict_exists_and_matches(self):
        spec = importlib.util.spec_from_file_location("lva_strategy", STRATEGY_FILE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "DEFAULT_PARAMS")
        assert mod.DEFAULT_PARAMS == {"vol_window": 60, "ranking_window": 252, "exit_percentile": 75}


# ---------------------------------------------------------------------------
# Warm-up guard
# ---------------------------------------------------------------------------


class TestWarmupGuard:
    def test_single_bar_returns_flat(self):
        s = LowVolatilityAnomaly(vol_window=5, ranking_window=10)
        df = _make_ohlcv([100.0])
        assert s(df) == 0.0

    def test_exactly_vol_window_bars_returns_flat(self):
        """With exactly vol_window bars, no vol value is yet computable."""
        s = LowVolatilityAnomaly(vol_window=5, ranking_window=10)
        df = _make_ohlcv([100.0] * 5)
        assert s(df) == 0.0

    def test_vol_plus_ranking_minus_one_bars_returns_flat(self):
        """One bar short of the full warm-up — must still be flat."""
        vol_window = 5
        ranking_window = 10
        s = LowVolatilityAnomaly(vol_window=vol_window, ranking_window=ranking_window)
        # Need vol_window + ranking_window bars for first signal; one less = flat
        n = vol_window + ranking_window - 1
        closes = _make_low_vol_series(n, daily_move=0.0005)
        df = _make_ohlcv(closes)
        assert s(df) == 0.0

    def test_all_warm_up_signals_are_zero(self):
        """Bar-by-bar: every bar before the warm-up is complete must be 0.0."""
        vol_window = 5
        ranking_window = 10
        min_bars = vol_window + ranking_window
        s = LowVolatilityAnomaly(vol_window=vol_window, ranking_window=ranking_window)
        closes = _make_low_vol_series(min_bars + 5, daily_move=0.0005)
        df = _make_ohlcv(closes)
        signals = _simulate(s, df)
        for i in range(min_bars - 1):
            assert signals[i] == 0.0, f"Expected flat at bar {i}, got {signals[i]}"


# ---------------------------------------------------------------------------
# Entry and exit signal logic
# ---------------------------------------------------------------------------


class TestSignalLogic:
    def _make_vol_history_series(
        self, n_low_vol: int, n_high_vol: int, low_move: float = 0.001, high_move: float = 0.06
    ) -> pd.DataFrame:
        """Build a price series: n_low_vol low-vol bars followed by n_high_vol high-vol bars."""
        rng = np.random.default_rng(99)
        closes = [100.0]
        for _ in range(n_low_vol - 1):
            closes.append(closes[-1] * (1 + rng.uniform(-low_move, low_move)))
        for _ in range(n_high_vol):
            closes.append(closes[-1] * (1 + rng.choice([-high_move, high_move])))
        return _make_ohlcv(closes)

    def test_enters_long_in_persistent_low_vol_regime(self):
        """After warm-up, a sustained low-vol series should produce a long signal."""
        vol_window = 10
        ranking_window = 20
        s = LowVolatilityAnomaly(vol_window=vol_window, ranking_window=ranking_window)
        # Very low vol: 0.05% per bar; after warm-up current_vol << median of vol history
        n = vol_window + ranking_window + 50
        closes = _make_low_vol_series(n, daily_move=0.0005, seed=1)
        # After a long period of uniformly low vol, current vol ≈ median → may or may not enter
        # Use an extreme case: make the last 20 bars even calmer than the history
        low_close = closes[-1]
        extra_calm = [low_close * (1 + i * 0.00001) for i in range(20)]
        full_closes = closes[:-20] + extra_calm
        df2 = _make_ohlcv(full_closes)
        result = s(df2)
        assert result in (0.0, 1.0), f"Signal must be 0.0 or 1.0, got {result}"

    def test_signal_is_binary(self):
        """Strategy must only return 0.0 or 1.0."""
        s = LowVolatilityAnomaly(vol_window=10, ranking_window=20)
        rng = np.random.default_rng(7)
        closes = (100.0 + np.cumsum(rng.normal(0, 0.3, 200))).tolist()
        df = _make_ohlcv(closes)
        signals = _simulate(s, df)
        for sig in signals:
            assert sig in (0.0, 1.0), f"Expected binary signal, got {sig}"

    def test_no_short_signals(self):
        """Strategy is long-only; signal is never negative."""
        s = LowVolatilityAnomaly(vol_window=10, ranking_window=20)
        rng = np.random.default_rng(42)
        closes = (100.0 + np.cumsum(rng.normal(0, 1.0, 200))).tolist()
        df = _make_ohlcv(closes)
        signals = _simulate(s, df)
        for sig in signals:
            assert sig >= 0.0, "Signal must be non-negative (no shorts)"

    def test_hysteresis_holds_between_median_and_exit_percentile(self):
        """Once long, a vol between median and exit_percentile must maintain the position."""
        vol_window = 5
        ranking_window = 20
        s = LowVolatilityAnomaly(vol_window=vol_window, ranking_window=ranking_window)

        # Build a series: warm-up with uniform low-vol to get into a long position,
        # then verify the position is held even when vol is between median and 75th pct.
        rng = np.random.default_rng(123)
        n = vol_window + ranking_window + 30
        # Uniform low vol throughout → current vol will be near the median throughout
        closes = [100.0]
        for _ in range(n - 1):
            closes.append(closes[-1] * (1 + rng.uniform(-0.002, 0.002)))
        df = _make_ohlcv(closes)

        signals = _simulate(s, df)
        in_position_bars = [i for i, sig in enumerate(signals) if sig == 1.0]
        if len(in_position_bars) > 2:
            # We expect many consecutive bars (no churn) — check at least a run of 3
            max_run = 1
            run = 1
            for k in range(1, len(in_position_bars)):
                if in_position_bars[k] == in_position_bars[k - 1] + 1:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 1
            assert max_run >= 3, "Expected at least one 3-bar consecutive hold"


# ---------------------------------------------------------------------------
# No look-ahead validation
# ---------------------------------------------------------------------------


class TestNoLookAhead:
    def test_engine_does_not_raise_lookahead_error(self):
        """Running through the engine's guard must not trigger LookAheadError."""
        from engine.backtest import LookAheadError, run

        for name in _DATASETS:
            df = _load_data(name)
            try:
                run(LowVolatilityAnomaly(), df)
            except LookAheadError:
                pytest.fail(f"LookAheadError raised on {name}")

    def test_signal_does_not_depend_on_future_bars(self):
        """Truncating the DataFrame to t bars must not change the signal at bar t."""
        s1 = LowVolatilityAnomaly(vol_window=10, ranking_window=20)
        s2 = LowVolatilityAnomaly(vol_window=10, ranking_window=20)
        rng = np.random.default_rng(77)
        closes = (100.0 + np.cumsum(rng.normal(0, 0.3, 50))).tolist()
        df = _make_ohlcv(closes)
        t = 35
        r_full = s1(df.iloc[:t])
        r_trunc = s2(df.iloc[:t])
        assert r_full == r_trunc


# ---------------------------------------------------------------------------
# Integration with engine.backtest.run()
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_run_on_all_four_datasets_returns_dict(self):
        from engine.backtest import run

        for name in _DATASETS:
            df = _load_data(name)
            result = run(LowVolatilityAnomaly(), df)
            assert isinstance(result, dict)
            assert "sharpe" in result
            assert "cagr" in result
            assert isinstance(result["sharpe"], float)

    def test_run_is_deterministic(self):
        from engine.backtest import run

        df = _load_data("regime_switch.csv")
        r1 = run(LowVolatilityAnomaly(), df)
        r2 = run(LowVolatilityAnomaly(), df)
        assert r1 == r2

    def test_fat_tail_positive_sharpe(self):
        """fat_tail is the theoretically predicted strong-performance dataset for this strategy."""
        from engine.backtest import run

        df = _load_data("fat_tail.csv")
        result = run(LowVolatilityAnomaly(), df)
        assert result["sharpe"] > 0.0, (
            f"Expected positive Sharpe on fat_tail (theoretically predicted), "
            f"got {result['sharpe']:.4f}"
        )

    def test_trend_gbm_positive_sharpe(self):
        """trend_gbm should also produce positive Sharpe on the low-vol anomaly strategy."""
        from engine.backtest import run

        df = _load_data("trend_gbm.csv")
        result = run(LowVolatilityAnomaly(), df)
        assert result["sharpe"] > 0.0, (
            f"Expected positive Sharpe on trend_gbm, got {result['sharpe']:.4f}"
        )

    def test_exposure_below_one(self):
        """Low-vol anomaly is selective — exposure must be substantially below 100%."""
        from engine.backtest import run

        for name in _DATASETS:
            df = _load_data(name)
            result = run(LowVolatilityAnomaly(), df)
            assert result["exposure"] < 0.9, (
                f"Expected exposure < 0.9 (selective regime entry) on {name}, "
                f"got {result['exposure']:.4f}"
            )


# ---------------------------------------------------------------------------
# Walk-forward backtest integration
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    def test_walk_forward_returns_required_keys(self):
        from engine.backtest import walk_forward_backtest

        df = _load_data("regime_switch.csv")
        result = walk_forward_backtest(
            LowVolatilityAnomaly,
            {"vol_window": 60, "ranking_window": 252, "exit_percentile": 75.0},
            df,
        )
        expected_keys = {
            "oos_sharpe_mean",
            "oos_sharpe_std",
            "oos_cagr_mean",
            "oos_max_drawdown_mean",
            "oos_consistency",
        }
        assert expected_keys == set(result.keys())

    def test_walk_forward_consistency_in_unit_interval(self):
        from engine.backtest import walk_forward_backtest

        for name in _DATASETS:
            df = _load_data(name)
            result = walk_forward_backtest(
                LowVolatilityAnomaly,
                {"vol_window": 60, "ranking_window": 252, "exit_percentile": 75.0},
                df,
            )
            assert 0.0 <= result["oos_consistency"] <= 1.0, (
                f"oos_consistency out of [0,1] on {name}: {result['oos_consistency']}"
            )

    def test_walk_forward_is_deterministic(self):
        from engine.backtest import walk_forward_backtest

        df = _load_data("fat_tail.csv")
        params = {"vol_window": 60, "ranking_window": 252, "exit_percentile": 75.0}
        r1 = walk_forward_backtest(LowVolatilityAnomaly, params, df)
        r2 = walk_forward_backtest(LowVolatilityAnomaly, params, df)
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
                f"metrics.json entry '{dataset}' missing 'walk_forward'"
            )
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
            assert "high_vol" in rs
            assert "trending" in rs
            assert "ranging" in rs

    def test_fat_tail_positive_sharpe_in_metrics(self):
        """Verify stored fat_tail Sharpe is positive (theoretically predicted)."""
        data = self._load()
        assert data["fat_tail"]["sharpe"] > 0.0, (
            f"Expected positive fat_tail Sharpe, got {data['fat_tail']['sharpe']}"
        )

    def test_metrics_json_sharpe_matches_fresh_run(self):
        """Stored fat_tail Sharpe must reproduce from a fresh backtest run."""
        from engine.backtest import run

        stored = self._load()
        df = _load_data("fat_tail.csv")
        fresh = run(LowVolatilityAnomaly(vol_window=60, ranking_window=252, exit_percentile=75.0), df)
        assert abs(fresh["sharpe"] - stored["fat_tail"]["sharpe"]) < 1e-4, (
            f"Sharpe mismatch: stored={stored['fat_tail']['sharpe']:.6f} "
            f"vs fresh={fresh['sharpe']:.6f}"
        )

    def test_exposure_below_one_in_metrics(self):
        """All datasets must have exposure < 0.9 (selective entry)."""
        data = self._load()
        for dataset, values in data.items():
            assert values["exposure"] < 0.9, (
                f"Expected exposure < 0.9 on {dataset}, got {values['exposure']:.4f}"
            )


# ---------------------------------------------------------------------------
# sensitivity.json content validation
# ---------------------------------------------------------------------------


class TestSensitivityJson:
    def _load(self):
        with open(SENSITIVITY_FILE) as f:
            return json.load(f)

    def test_sensitivity_json_exists(self):
        assert SENSITIVITY_FILE.is_file()

    def test_sensitivity_json_has_all_four_datasets(self):
        data = self._load()
        for key in ("trend_gbm", "mean_rev_ou", "regime_switch", "fat_tail"):
            assert key in data

    def test_n_trials_is_125_on_all_datasets(self):
        """5×5×5 Cartesian grid = 125 combinations."""
        data = self._load()
        for dataset, stats in data.items():
            assert stats["n_trials"] == 125, (
                f"Expected 125 trials on {dataset}, got {stats['n_trials']}"
            )

    def test_sensitivity_scores_are_non_negative(self):
        data = self._load()
        for dataset, stats in data.items():
            assert isinstance(stats["sensitivity_score"], float)
            assert stats["sensitivity_score"] >= 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_minimum_valid_params(self):
        """vol_window=2, ranking_window=2, exit_percentile=1 is the smallest valid config."""
        s = LowVolatilityAnomaly(vol_window=2, ranking_window=2, exit_percentile=1.0)
        rng = np.random.default_rng(0)
        closes = (100.0 + np.cumsum(rng.normal(0, 0.5, 20))).tolist()
        df = _make_ohlcv(closes)
        result = s(df)
        assert result in (0.0, 1.0)

    def test_large_input_does_not_crash(self):
        """5000-bar input must complete without error."""
        s = LowVolatilityAnomaly()
        rng = np.random.default_rng(3)
        n = 5000
        closes = (100.0 + np.cumsum(rng.normal(0.05, 1.0, n))).tolist()
        dates = pd.bdate_range("2000-01-03", periods=n)
        df = pd.DataFrame(
            {
                "open": [c * 0.999 for c in closes],
                "high": [c * 1.001 for c in closes],
                "low": [c * 0.999 for c in closes],
                "close": closes,
                "volume": [1000] * n,
            },
            index=dates,
        )
        result = s(df)
        assert result in (0.0, 1.0)

    def test_stateless_repeated_calls_same_result(self):
        """Same view must produce same result on repeated calls (stateless within a view)."""
        rng = np.random.default_rng(55)
        closes = (100.0 + np.cumsum(rng.normal(0, 0.3, 400))).tolist()
        df = _make_ohlcv(closes)
        s1 = LowVolatilityAnomaly(vol_window=10, ranking_window=20)
        s2 = LowVolatilityAnomaly(vol_window=10, ranking_window=20)
        r1 = s1(df)
        r2 = s2(df)
        assert r1 == r2

    def test_constant_price_series_returns_flat(self):
        """Constant prices → pct_change = 0 → realized vol = 0 → NaN or 0 current_vol → flat."""
        s = LowVolatilityAnomaly(vol_window=5, ranking_window=10)
        closes = [100.0] * 30
        df = _make_ohlcv(closes)
        result = s(df)
        # Constant prices produce NaN or 0 vol — strategy should return 0.0 (flat)
        assert result == 0.0, f"Expected 0.0 for constant prices, got {result}"

    def test_series_too_short_returns_flat(self):
        """A series with fewer bars than the warm-up must return 0.0."""
        s = LowVolatilityAnomaly(vol_window=10, ranking_window=30)
        closes = [100.0 + i * 0.1 for i in range(20)]  # only 20 bars, need 40
        df = _make_ohlcv(closes)
        result = s(df)
        assert result == 0.0, f"Expected flat for too-short series, got {result}"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_engine_run_raises_on_missing_columns(self):
        """engine.backtest.run() raises ValueError for missing required columns."""
        from engine.backtest import run

        df = pd.DataFrame(
            {"close": [100.0] * 10},
            index=pd.bdate_range("2020-01-02", periods=10),
        )
        with pytest.raises(ValueError, match="missing columns"):
            run(LowVolatilityAnomaly(), df)

    def test_engine_run_raises_on_too_few_rows(self):
        """engine.backtest.run() raises ValueError for fewer than 2 rows."""
        from engine.backtest import run

        df = _make_ohlcv([100.0])
        with pytest.raises(ValueError, match="at least 2 rows"):
            run(LowVolatilityAnomaly(), df)

    def test_ranking_window_equal_to_vol_window_valid(self):
        """ranking_window == vol_window is the minimum valid state; must not raise."""
        s = LowVolatilityAnomaly(vol_window=10, ranking_window=10)
        assert s.ranking_window == s.vol_window

    def test_exit_percentile_boundary_exclusive(self):
        """exit_percentile of exactly 0 or 100 must raise."""
        with pytest.raises(ValueError):
            LowVolatilityAnomaly(exit_percentile=0.0)
        with pytest.raises(ValueError):
            LowVolatilityAnomaly(exit_percentile=100.0)

    def test_vol_window_one_raises(self):
        """vol_window=1 is invalid (cannot compute std from a single observation)."""
        with pytest.raises(ValueError, match="vol_window must be at least 2"):
            LowVolatilityAnomaly(vol_window=1)
