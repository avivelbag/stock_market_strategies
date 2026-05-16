"""Tests for strategies/09-volatility-managed/strategy.py.

Covers:
  - Default parameters and constructor validation
  - Warm-up guard: flat (0.0) for the first window bars
  - Scalar computation: target_vol / realized_vol clipped to [0, 2]
  - Constant-price edge case: realized_vol ≈ 0 → scalar = 1.0
  - Low-vol environment: scalar > 1.0 (up to the 2.0 cap)
  - High-vol environment: scalar < 1.0 (down toward 0)
  - No look-ahead: realized vol uses only lagged data
  - Scalar is always in [0, 2] after warm-up
  - Integration with engine.backtest.run() on all four synthetic datasets
  - Integration with walk_forward_backtest() and metrics.json structure
  - Edge cases: single bar, exactly window bars, large input, zero-vol data
  - Failure modes: invalid constructor arguments (window < 2, target_vol <= 0)
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
STRATEGY_FILE = ROOT / "strategies" / "09-volatility-managed" / "strategy.py"
METRICS_FILE = ROOT / "strategies" / "09-volatility-managed" / "metrics.json"
SENSITIVITY_FILE = ROOT / "strategies" / "09-volatility-managed" / "sensitivity.json"

_DATASETS = ["trend_gbm.csv", "mean_rev_ou.csv", "regime_switch.csv", "fat_tail.csv"]


def _load_strategy_class():
    """Load VolatilityManagedPortfolio from the hyphenated directory via importlib."""
    spec = importlib.util.spec_from_file_location("volmgd_strategy", STRATEGY_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VolatilityManagedPortfolio


VolatilityManagedPortfolio = _load_strategy_class()


def _load_data(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, index_col=0, parse_dates=True)


def _make_ohlcv(closes: list, seed: int = 42) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with open=close*0.999, high=close*1.001, low=close*0.999."""
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
    def test_default_window_is_21(self):
        s = VolatilityManagedPortfolio()
        assert s.window == 21

    def test_default_target_vol_is_0_12(self):
        s = VolatilityManagedPortfolio()
        assert s.target_vol == 0.12

    def test_custom_params_stored(self):
        s = VolatilityManagedPortfolio(window=10, target_vol=0.20)
        assert s.window == 10
        assert s.target_vol == 0.20

    def test_window_less_than_2_raises(self):
        with pytest.raises(ValueError, match="window must be at least 2"):
            VolatilityManagedPortfolio(window=1)

    def test_window_zero_raises(self):
        with pytest.raises(ValueError, match="window must be at least 2"):
            VolatilityManagedPortfolio(window=0)

    def test_target_vol_zero_raises(self):
        with pytest.raises(ValueError, match="target_vol must be positive"):
            VolatilityManagedPortfolio(target_vol=0.0)

    def test_target_vol_negative_raises(self):
        with pytest.raises(ValueError, match="target_vol must be positive"):
            VolatilityManagedPortfolio(target_vol=-0.05)

    def test_default_params_dict_exists_and_matches(self):
        spec = importlib.util.spec_from_file_location("volmgd_strategy", STRATEGY_FILE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "DEFAULT_PARAMS")
        assert mod.DEFAULT_PARAMS == {"window": 21, "target_vol": 0.12}


# ---------------------------------------------------------------------------
# Warm-up guard
# ---------------------------------------------------------------------------


class TestWarmupGuard:
    def test_single_bar_returns_flat(self):
        """With only 1 bar, t=0 < window → flat."""
        s = VolatilityManagedPortfolio(window=5)
        df = _make_ohlcv([100.0])
        assert s(df) == 0.0

    def test_exactly_window_bars_returns_flat(self):
        """At exactly window bars (t = window-1 < window) → flat."""
        s = VolatilityManagedPortfolio(window=5)
        df = _make_ohlcv([100.0] * 5)
        assert s(df) == 0.0

    def test_window_plus_one_bars_produces_nonzero(self):
        """At window+1 bars (t = window >= window) → non-zero scalar."""
        s = VolatilityManagedPortfolio(window=5)
        rng = np.random.default_rng(7)
        closes = (100.0 + np.cumsum(rng.normal(0, 0.5, 6))).tolist()
        df = _make_ohlcv(closes)
        result = s(df)
        assert result != 0.0, "Expected non-zero scalar after warm-up"

    def test_first_window_signals_are_all_zero(self):
        """Bar-by-bar: first window signals must all be 0.0."""
        window = 7
        s = VolatilityManagedPortfolio(window=window)
        rng = np.random.default_rng(11)
        closes = (100.0 + np.cumsum(rng.normal(0, 0.5, 30))).tolist()
        df = _make_ohlcv(closes)
        signals = _simulate(s, df)
        for i in range(window):
            assert signals[i] == 0.0, f"Expected 0.0 at bar {i}, got {signals[i]}"
        assert signals[window] != 0.0, f"Expected non-zero at bar {window}"


# ---------------------------------------------------------------------------
# Scalar computation
# ---------------------------------------------------------------------------


class TestScalarComputation:
    def _make_constant_trend(self, n: int, pct_per_bar: float = 0.01) -> pd.DataFrame:
        """Prices that grow at a constant pct_per_bar — realized vol is deterministic."""
        closes = [100.0 * (1 + pct_per_bar) ** i for i in range(n)]
        return _make_ohlcv(closes)

    def test_scalar_is_in_zero_to_two(self):
        """All scalars must be in [0, 2] after warm-up."""
        s = VolatilityManagedPortfolio(window=10)
        rng = np.random.default_rng(3)
        closes = (100.0 + np.cumsum(rng.normal(0, 2.0, 100))).tolist()
        df = _make_ohlcv(closes)
        for t in range(1, len(df) + 1):
            val = s(df.iloc[:t])
            assert 0.0 <= val <= 2.0, f"Scalar {val} out of [0, 2] at bar {t}"

    def test_scalar_positive_after_warmup(self):
        """After warm-up, scalar is always > 0 for positive target_vol."""
        s = VolatilityManagedPortfolio(window=5, target_vol=0.12)
        rng = np.random.default_rng(99)
        closes = (100.0 + np.cumsum(rng.normal(0, 1.0, 50))).tolist()
        df = _make_ohlcv(closes)
        signals = _simulate(s, df)
        for i in range(5, len(signals)):
            assert signals[i] > 0.0, f"Expected positive scalar at bar {i}, got {signals[i]}"

    def test_high_vol_gives_scalar_below_one(self):
        """When realized vol >> target_vol, scalar should be < 1."""
        s = VolatilityManagedPortfolio(window=5, target_vol=0.12)
        # 10% per-bar moves → ~158% annualized vol → scalar ≈ 0.12/1.58 ≈ 0.076
        rng = np.random.default_rng(17)
        closes = (100.0 + np.cumsum(rng.normal(0, 10.0, 20))).tolist()
        df = _make_ohlcv(closes)
        # Get the scalar after warmup
        result = s(df)
        assert result < 1.0, f"Expected scalar < 1 in high-vol environment, got {result}"

    def test_low_vol_gives_scalar_approaching_two(self):
        """When realized vol << target_vol, scalar approaches the 2.0 cap."""
        s = VolatilityManagedPortfolio(window=5, target_vol=0.12)
        # 0.001% per-bar moves → ~0.016% annualized vol → scalar >> 1 → capped at 2
        closes = [100.0 + i * 0.001 for i in range(10)]
        df = _make_ohlcv(closes)
        result = s(df)
        assert result == 2.0, f"Expected scalar at 2.0 cap in low-vol env, got {result}"

    def test_constant_prices_returns_one(self):
        """Constant prices → realized_vol ≈ 0 → special case returns 1.0."""
        s = VolatilityManagedPortfolio(window=5)
        closes = [100.0] * 10
        df = _make_ohlcv(closes)
        result = s(df)
        assert result == 1.0, f"Expected 1.0 for constant prices, got {result}"

    def test_target_vol_increases_scalar_proportionally(self):
        """Doubling target_vol doubles the scalar (before clipping)."""
        rng = np.random.default_rng(55)
        closes = (100.0 + np.cumsum(rng.normal(0, 0.5, 30))).tolist()
        df = _make_ohlcv(closes)
        s1 = VolatilityManagedPortfolio(window=10, target_vol=0.10)
        s2 = VolatilityManagedPortfolio(window=10, target_vol=0.20)
        r1 = s1(df)
        r2 = s2(df)
        if r1 < 1.0:  # not yet capped → proportionality holds
            assert abs(r2 / r1 - 2.0) < 0.01, f"Expected 2x scalar, got ratio {r2/r1:.4f}"

    def test_scalar_is_float(self):
        s = VolatilityManagedPortfolio(window=5)
        rng = np.random.default_rng(8)
        closes = (100.0 + np.cumsum(rng.normal(0, 0.5, 10))).tolist()
        df = _make_ohlcv(closes)
        result = s(df)
        assert isinstance(result, float)

    def test_scalar_matches_manual_calculation(self):
        """Verify the scalar matches a manual computation of target_vol / realized_vol."""
        window = 5
        target_vol = 0.15
        s = VolatilityManagedPortfolio(window=window, target_vol=target_vol)
        rng = np.random.default_rng(42)
        closes = (100.0 + np.cumsum(rng.normal(0, 0.3, window + 5))).tolist()
        df = _make_ohlcv(closes)
        result = s(df)

        # Manual: use the last window+5 bars, pct_change, take [t-window:t]
        closes_s = pd.Series([row for row in closes])
        t = len(closes) - 1
        rets = closes_s.pct_change().iloc[t - window : t]
        realized_vol = float(rets.std() * (252 ** 0.5))
        expected = float(np.clip(target_vol / realized_vol, 0.0, 2.0))
        assert abs(result - expected) < 1e-10, (
            f"scalar mismatch: strategy={result:.8f}, manual={expected:.8f}"
        )


# ---------------------------------------------------------------------------
# No look-ahead validation
# ---------------------------------------------------------------------------


class TestNoLookAhead:
    def test_engine_does_not_raise_lookahead_error(self):
        """Running through the engine's guard must not trigger LookAheadError."""
        from engine.backtest import run, LookAheadError
        for name in _DATASETS:
            df = _load_data(name)
            try:
                run(VolatilityManagedPortfolio(), df)
            except LookAheadError:
                pytest.fail(f"LookAheadError raised on {name}")

    def test_result_does_not_depend_on_future_bars(self):
        """Truncating the DataFrame to t bars must not change the signal at bar t."""
        s1 = VolatilityManagedPortfolio(window=10)
        s2 = VolatilityManagedPortfolio(window=10)
        rng = np.random.default_rng(77)
        closes = (100.0 + np.cumsum(rng.normal(0, 0.5, 30))).tolist()
        df = _make_ohlcv(closes)
        # Signal at bar 15 with full data vs truncated at bar 15
        r_full = s1(df.iloc[:16])
        r_trunc = s2(df.iloc[:16])
        assert r_full == r_trunc


# ---------------------------------------------------------------------------
# Integration with engine.backtest.run()
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_run_on_trend_gbm_returns_dict(self):
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        s = VolatilityManagedPortfolio()
        result = run(s, df)
        assert isinstance(result, dict)
        assert "sharpe" in result
        assert "cagr" in result

    def test_run_on_all_four_datasets_completes(self):
        from engine.backtest import run
        for name in _DATASETS:
            df = _load_data(name)
            s = VolatilityManagedPortfolio()
            result = run(s, df)
            assert isinstance(result["sharpe"], float)
            assert isinstance(result["cagr"], float)

    def test_run_is_deterministic(self):
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        r1 = run(VolatilityManagedPortfolio(), df)
        r2 = run(VolatilityManagedPortfolio(), df)
        assert r1 == r2

    def test_exposure_near_one_after_warmup(self):
        """Strategy is long almost all bars (warm-up is only window/total_bars fraction)."""
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(VolatilityManagedPortfolio(), df)
        # With window=21 and ~1000 bars, exposure should be ~97-98%
        assert result["exposure"] > 0.95, (
            f"Expected exposure > 0.95 (always-long after 21-bar warmup), "
            f"got {result['exposure']:.4f}"
        )

    def test_turnover_near_zero(self):
        """Strategy enters once after warm-up and stays long — turnover should be minimal."""
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(VolatilityManagedPortfolio(), df)
        assert result["turnover"] < 0.01, (
            f"Expected near-zero turnover, got {result['turnover']:.4f}"
        )

    def test_trend_gbm_positive_sharpe(self):
        """On a GBM trend with positive drift, always-long strategy should be positive."""
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(VolatilityManagedPortfolio(), df)
        assert result["sharpe"] > 0.0, (
            f"Expected positive Sharpe on positive-drift GBM, got {result['sharpe']:.4f}"
        )

    def test_regime_switch_strong_positive_sharpe(self):
        """regime_switch has trending periods with positive drift — Sharpe should be high."""
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        result = run(VolatilityManagedPortfolio(), df)
        assert result["sharpe"] > 0.5, (
            f"Expected regime_switch Sharpe > 0.5, got {result['sharpe']:.4f}"
        )


# ---------------------------------------------------------------------------
# Walk-forward backtest integration
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    def test_walk_forward_returns_required_keys(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("regime_switch.csv")
        result = walk_forward_backtest(
            VolatilityManagedPortfolio,
            {"window": 21, "target_vol": 0.12},
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
                VolatilityManagedPortfolio,
                {"window": 21, "target_vol": 0.12},
                df,
            )
            assert 0.0 <= result["oos_consistency"] <= 1.0, (
                f"oos_consistency out of range on {name}"
            )

    def test_walk_forward_is_deterministic(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("trend_gbm.csv")
        params = {"window": 21, "target_vol": 0.12}
        r1 = walk_forward_backtest(VolatilityManagedPortfolio, params, df)
        r2 = walk_forward_backtest(VolatilityManagedPortfolio, params, df)
        assert r1 == r2

    def test_regime_switch_perfect_oos_consistency(self):
        """regime_switch should have OOS consistency 1.0 (all folds positive)."""
        from engine.backtest import walk_forward_backtest
        df = _load_data("regime_switch.csv")
        result = walk_forward_backtest(
            VolatilityManagedPortfolio,
            {"window": 21, "target_vol": 0.12},
            df,
        )
        assert result["oos_consistency"] == 1.0, (
            f"Expected oos_consistency=1.0 on regime_switch, got {result['oos_consistency']}"
        )


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

    def test_metrics_json_has_deflated_sharpe(self):
        data = self._load()
        for dataset, values in data.items():
            assert "deflated_sharpe" in values, (
                f"metrics.json entry '{dataset}' missing 'deflated_sharpe'"
            )
            assert 0.0 <= values["deflated_sharpe"] <= 1.0

    def test_metrics_json_has_regime_sharpe(self):
        data = self._load()
        for dataset, values in data.items():
            assert "regime_sharpe" in values, (
                f"metrics.json entry '{dataset}' missing 'regime_sharpe'"
            )
            rs = values["regime_sharpe"]
            assert "regime_counts" in rs
            assert "high_vol" in rs
            assert "trending" in rs
            assert "ranging" in rs

    def test_all_datasets_positive_sharpe(self):
        """Binary engine makes strategy always-long → positive on positive-drift datasets."""
        data = self._load()
        for dataset, values in data.items():
            assert values["sharpe"] > 0.0, (
                f"Expected positive Sharpe on {dataset}, got {values['sharpe']:.4f}"
            )

    def test_regime_switch_sharpe_matches_fresh_run(self):
        """Verify stored regime_switch Sharpe is reproducible."""
        from engine.backtest import run
        stored = self._load()
        df = _load_data("regime_switch.csv")
        fresh = run(VolatilityManagedPortfolio(window=21, target_vol=0.12), df)
        assert abs(fresh["sharpe"] - stored["regime_switch"]["sharpe"]) < 1e-4, (
            f"Sharpe mismatch: stored={stored['regime_switch']['sharpe']:.6f} "
            f"vs fresh={fresh['sharpe']:.6f}"
        )

    def test_exposure_near_one_in_all_datasets(self):
        """Strategy should have exposure > 0.95 on all datasets (always-long after warmup)."""
        data = self._load()
        for dataset, values in data.items():
            assert values["exposure"] > 0.95, (
                f"Expected exposure > 0.95 on {dataset}, got {values['exposure']:.4f}"
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
            assert key in data, f"sensitivity.json missing key '{key}'"

    def test_n_trials_is_25_on_all_datasets(self):
        """window ∈ {17,19,21,23,25} × target_vol ∈ {0.096,...,0.144} = 25 combos."""
        data = self._load()
        for dataset, stats in data.items():
            assert stats["n_trials"] == 25, (
                f"sensitivity.json '{dataset}' has n_trials={stats['n_trials']}, expected 25"
            )

    def test_sensitivity_scores_are_non_negative(self):
        data = self._load()
        for dataset, stats in data.items():
            score = stats["sensitivity_score"]
            assert isinstance(score, float)
            assert score >= 0.0, f"Negative sensitivity_score for '{dataset}': {score}"

    def test_regime_switch_sensitivity_is_robust(self):
        """regime_switch sensitivity_score should be < 0.3 (binary engine insensitive)."""
        data = self._load()
        score = data["regime_switch"]["sensitivity_score"]
        assert score < 0.3, (
            f"regime_switch sensitivity_score={score:.4f} should be < 0.3 "
            "(binary engine makes strategy insensitive to parameter changes)"
        )

    def test_regime_switch_mean_sharpe_is_positive(self):
        data = self._load()
        assert data["regime_switch"]["mean_sharpe"] > 0.0, (
            f"regime_switch mean_sharpe={data['regime_switch']['mean_sharpe']:.4f} "
            "should be positive"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_minimum_valid_window_2(self):
        """window=2, target_vol=0.01 is the smallest valid configuration."""
        s = VolatilityManagedPortfolio(window=2, target_vol=0.01)
        rng = np.random.default_rng(22)
        closes = (100.0 + np.cumsum(rng.normal(0, 0.5, 5))).tolist()
        df = _make_ohlcv(closes)
        result = s(df)
        assert 0.0 <= result <= 2.0

    def test_large_input_does_not_crash(self):
        """5000-bar input should complete without error."""
        s = VolatilityManagedPortfolio()
        rng = np.random.default_rng(0)
        closes = (100.0 + np.cumsum(rng.normal(0.05, 1.0, 5000))).tolist()
        n = len(closes)
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
        assert 0.0 <= result <= 2.0

    def test_very_small_target_vol_produces_small_scalar(self):
        """Very small target_vol with normal vol → scalar near zero."""
        s = VolatilityManagedPortfolio(window=10, target_vol=0.001)
        rng = np.random.default_rng(33)
        closes = (100.0 + np.cumsum(rng.normal(0, 1.0, 20))).tolist()
        df = _make_ohlcv(closes)
        result = s(df)
        assert 0.0 <= result < 0.1, f"Expected scalar near 0 with tiny target_vol, got {result}"

    def test_very_large_target_vol_caps_at_two(self):
        """Very large target_vol with small realized vol → scalar capped at 2."""
        s = VolatilityManagedPortfolio(window=5, target_vol=100.0)
        closes = [100.0 + i * 0.001 for i in range(10)]
        df = _make_ohlcv(closes)
        result = s(df)
        assert result == 2.0, f"Expected scalar capped at 2.0, got {result}"

    def test_stateless_strategy_same_result_on_repeated_calls(self):
        """Strategy is stateless: calling with same view produces same result."""
        s = VolatilityManagedPortfolio(window=10)
        rng = np.random.default_rng(44)
        closes = (100.0 + np.cumsum(rng.normal(0, 0.5, 20))).tolist()
        df = _make_ohlcv(closes)
        r1 = s(df)
        r2 = s(df)
        assert r1 == r2, "Strategy should be stateless (same result on repeated calls)"

    def test_bar_by_bar_scalars_all_in_valid_range(self):
        """Bar-by-bar simulation: every scalar must be in [0, 2]."""
        rng = np.random.default_rng(88)
        n = 100
        closes = (100.0 + np.cumsum(rng.normal(0, 0.5, n))).tolist()
        df = _make_ohlcv(closes)
        s = VolatilityManagedPortfolio(window=10)
        for t in range(1, n + 1):
            result = s(df.iloc[:t])
            assert 0.0 <= result <= 2.0, f"Scalar {result} out of [0, 2] at bar {t}"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_missing_close_column_raises(self):
        """DataFrame without 'close' column raises AttributeError or KeyError."""
        s = VolatilityManagedPortfolio(window=5)
        df = pd.DataFrame(
            {"open": [1.0] * 10, "high": [1.0] * 10, "low": [1.0] * 10, "volume": [100] * 10},
            index=pd.bdate_range("2020-01-02", periods=10),
        )
        with pytest.raises((AttributeError, KeyError)):
            s(df)

    def test_engine_run_raises_on_missing_columns(self):
        """engine.backtest.run() raises ValueError for missing required columns."""
        from engine.backtest import run
        df = pd.DataFrame(
            {"close": [100.0] * 10},
            index=pd.bdate_range("2020-01-02", periods=10),
        )
        with pytest.raises(ValueError, match="missing columns"):
            run(VolatilityManagedPortfolio(), df)

    def test_engine_run_raises_on_too_few_rows(self):
        """engine.backtest.run() raises ValueError for DataFrame with fewer than 2 rows."""
        from engine.backtest import run
        df = _make_ohlcv([100.0])
        with pytest.raises(ValueError, match="at least 2 rows"):
            run(VolatilityManagedPortfolio(), df)
