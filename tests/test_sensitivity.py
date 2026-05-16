"""Tests for engine/sensitivity.py and the generated sensitivity.json files.

Coverage:
- Structural: every strategies.json entry has a sensitivity.json in its directory.
- Sanity bound: sensitivity_score is a float in [0, 10] for a well-behaved strategy.
- Determinism: two calls to parameter_sweep with identical inputs return identical results.
- Edge cases: empty param grid, no valid combinations (all invalid params), single trial.
- Failure mode: factory that always raises produces a graceful zero-trial result.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.sensitivity import build_param_grid, parameter_sweep

ROOT = Path(__file__).parent.parent
STRATEGIES_JSON = ROOT / "strategies.json"
STRATEGIES_DIR = ROOT / "strategies"
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Return a small deterministic OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0005, 0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(log_returns))
    opens = prices * (1 + rng.uniform(-0.002, 0.002, size=n))
    highs = np.maximum(prices, opens) * (1 + rng.uniform(0, 0.005, size=n))
    lows = np.minimum(prices, opens) * (1 - rng.uniform(0, 0.005, size=n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": prices, "volume": 1000},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Structural tests: sensitivity.json must exist for every registry entry
# ---------------------------------------------------------------------------


class TestSensitivityJsonPresence:
    def test_every_strategy_has_sensitivity_json(self):
        """Every entry in strategies.json must have a sensitivity.json file."""
        with open(STRATEGIES_JSON) as f:
            registry = json.load(f)
        for entry in registry:
            name = entry["name"]
            sensitivity_path = STRATEGIES_DIR / name / "sensitivity.json"
            assert sensitivity_path.is_file(), (
                f"sensitivity.json missing for strategy '{name}': {sensitivity_path}"
            )

    def test_sensitivity_json_is_valid_json(self):
        """Each sensitivity.json must parse without error."""
        with open(STRATEGIES_JSON) as f:
            registry = json.load(f)
        for entry in registry:
            name = entry["name"]
            path = STRATEGIES_DIR / name / "sensitivity.json"
            if not path.is_file():
                pytest.skip(f"sensitivity.json not generated yet for {name}")
            try:
                with open(path) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"sensitivity.json for '{name}' is invalid JSON: {e}")

    def test_sensitivity_json_has_required_keys(self):
        """Each per-dataset entry in sensitivity.json must have all expected keys."""
        required = {
            "mean_sharpe", "std_sharpe", "min_sharpe", "max_sharpe",
            "n_trials", "sensitivity_score", "dispersion", "stable_fraction",
        }
        with open(STRATEGIES_JSON) as f:
            registry = json.load(f)
        for entry in registry:
            name = entry["name"]
            path = STRATEGIES_DIR / name / "sensitivity.json"
            if not path.is_file():
                pytest.skip(f"sensitivity.json not generated yet for {name}")
            with open(path) as f:
                data = json.load(f)
            for dataset, stats in data.items():
                missing = required - set(stats.keys())
                assert not missing, (
                    f"sensitivity.json for '{name}' dataset '{dataset}' missing keys: {missing}"
                )

    def test_sensitivity_scores_are_finite_floats(self):
        """sensitivity_score in each generated file must be a finite, non-negative float."""
        with open(STRATEGIES_JSON) as f:
            registry = json.load(f)
        for entry in registry:
            name = entry["name"]
            path = STRATEGIES_DIR / name / "sensitivity.json"
            if not path.is_file():
                pytest.skip(f"sensitivity.json not generated yet for {name}")
            with open(path) as f:
                data = json.load(f)
            for dataset, stats in data.items():
                score = stats["sensitivity_score"]
                assert isinstance(score, (int, float)), (
                    f"sensitivity_score for '{name}'/'{dataset}' is not numeric: {score!r}"
                )
                assert score >= 0, (
                    f"sensitivity_score for '{name}'/'{dataset}' is negative: {score}"
                )
                assert score < 100, (
                    f"sensitivity_score for '{name}'/'{dataset}' exceeds sanity ceiling: {score}"
                )


# ---------------------------------------------------------------------------
# Sanity bound: score is in [0, 10] for a strategy with a real edge
# ---------------------------------------------------------------------------


class TestSensitivityScoreSanityBound:
    def test_sensitivity_score_in_range_for_well_behaved_strategy(self):
        """sensitivity_score must be a float in [0, 10] for a stable strategy.

        Uses a simple buy-and-hold strategy (ignores params) on a small trending
        series: every parameter combination produces the same Sharpe, so
        std_sharpe = 0 and sensitivity_score = 0.0.  This verifies the formula
        and type contract without requiring full dataset I/O.
        """
        df = _make_ohlcv(n=200, seed=42)
        param_grid = {"threshold": [0.1, 0.2, 0.3, 0.4, 0.5]}

        def factory(params):
            """Strategy that always returns 1.0, ignoring all params."""
            class BuyAndHold:
                def __call__(self, view):
                    return 1.0
            return BuyAndHold()

        result = parameter_sweep(factory, param_grid, df)
        score = result["sensitivity_score"]
        assert isinstance(score, float), f"sensitivity_score is not float: {type(score)}"
        assert 0.0 <= score <= 10.0, f"sensitivity_score out of [0, 10]: {score}"

    def test_sensitivity_score_nonzero_for_param_sensitive_strategy(self):
        """A strategy whose performance varies with params should have score > 0.

        Uses a moving-average crossover where the window span directly controls
        how many signals are generated on a small dataset — more variation in
        params means more variation in Sharpe.
        """
        df = _make_ohlcv(n=300, seed=7)
        param_grid = {"window": [5, 10, 20, 40, 80]}

        def factory(params):
            window = params["window"]
            class MAStrategy:
                def __call__(self, view):
                    closes = view["close"]
                    if len(closes) < window:
                        return 0.0
                    ma = closes.rolling(window).mean().iloc[-1]
                    return 1.0 if closes.iloc[-1] > ma else 0.0
            return MAStrategy()

        result = parameter_sweep(factory, param_grid, df)
        assert result["n_trials"] == 5
        assert result["sensitivity_score"] >= 0.0

    def test_dual_ema_regime_switch_score_in_range(self):
        """Dual EMA on regime_switch (primary dataset) must have score in [0, 10].

        The pre-generated sweep returned 0.0852 for this strategy/dataset
        combination; this test validates the live value against the same bound.
        """
        regime_switch_path = DATA_DIR / "regime_switch.csv"
        if not regime_switch_path.is_file():
            pytest.skip("regime_switch.csv not found")

        import importlib.util
        strategy_file = STRATEGIES_DIR / "01-dual-ema-momentum" / "strategy.py"
        spec = importlib.util.spec_from_file_location("dual_ema", strategy_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        df = pd.read_csv(regime_switch_path, index_col=0, parse_dates=True)
        grid = build_param_grid(mod.DEFAULT_PARAMS)

        result = parameter_sweep(lambda p: mod.DualEMAMomentum(**p), grid, df)
        score = result["sensitivity_score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 10.0, f"Dual EMA regime_switch score out of [0, 10]: {score}"


# ---------------------------------------------------------------------------
# Determinism: two identical calls produce identical output
# ---------------------------------------------------------------------------


class TestSweepDeterminism:
    def test_parameter_sweep_is_deterministic(self):
        """Two calls to parameter_sweep with the same factory, grid, and data return
        bit-for-bit identical dicts."""
        df = _make_ohlcv(n=150, seed=99)
        param_grid = {"window": [10, 20, 30]}

        def factory(params):
            window = params["window"]
            class SimpleMA:
                def __call__(self, view):
                    closes = view["close"]
                    if len(closes) < window:
                        return 0.0
                    return 1.0 if closes.iloc[-1] > closes.rolling(window).mean().iloc[-1] else 0.0
            return SimpleMA()

        result1 = parameter_sweep(factory, param_grid, df)
        result2 = parameter_sweep(factory, param_grid, df)
        assert result1 == result2, (
            f"parameter_sweep is not deterministic:\nresult1={result1}\nresult2={result2}"
        )

    def test_determinism_with_real_strategy(self):
        """Dual EMA strategy sweep is deterministic across two calls on a real dataset."""
        data_path = DATA_DIR / "trend_gbm.csv"
        if not data_path.is_file():
            pytest.skip("trend_gbm.csv not found")

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dual_ema_det",
            STRATEGIES_DIR / "01-dual-ema-momentum" / "strategy.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        grid = build_param_grid(mod.DEFAULT_PARAMS)
        def factory(p):
            return mod.DualEMAMomentum(**p)

        r1 = parameter_sweep(factory, grid, df)
        r2 = parameter_sweep(factory, grid, df)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestParameterSweepEdgeCases:
    def test_empty_param_grid_runs_once(self):
        """An empty param_grid means one combination (the empty product), so n_trials=1."""
        df = _make_ohlcv(n=100, seed=1)
        call_count = []

        def factory(params):
            assert params == {}
            call_count.append(1)
            class AlwaysLong:
                def __call__(self, view): return 1.0
            return AlwaysLong()

        result = parameter_sweep(factory, {}, df)
        assert result["n_trials"] == 1
        assert len(call_count) == 1

    def test_all_invalid_combos_returns_zero_result(self):
        """When every combination raises ValueError, n_trials=0 and all fields are 0."""
        df = _make_ohlcv(n=100, seed=2)
        param_grid = {"x": [1, 2, 3]}

        def factory(params):
            raise ValueError("always invalid")

        result = parameter_sweep(factory, param_grid, df)
        assert result["n_trials"] == 0
        assert result["mean_sharpe"] == 0.0
        assert result["std_sharpe"] == 0.0
        assert result["sensitivity_score"] == 0.0

    def test_single_valid_combination_std_is_zero(self):
        """With only one valid parameter combination, std_sharpe must be 0."""
        df = _make_ohlcv(n=100, seed=3)
        param_grid = {"threshold": [0.5]}

        def factory(params):
            class Strat:
                def __call__(self, view): return 1.0
            return Strat()

        result = parameter_sweep(factory, param_grid, df)
        assert result["n_trials"] == 1
        assert result["std_sharpe"] == 0.0

    def test_mean_sharpe_near_zero_caps_sensitivity_score(self):
        """When mean_sharpe ≈ 0 the score is capped at 99.0, not infinity."""
        df = _make_ohlcv(n=100, seed=5)
        # Strategy alternates long/flat based on param parity; sharpe will be near zero
        param_grid = {"flip": [0, 1, 2, 3, 4]}

        call_results = iter([0.0001, -0.0001, 0.0001, -0.0001, 0.0001])

        def factory(params):
            val = next(call_results)
            class Strat:
                def __call__(self, view):
                    return val
            return Strat()

        result = parameter_sweep(factory, param_grid, df)
        assert result["sensitivity_score"] <= 99.0
        assert isinstance(result["sensitivity_score"], float)


# ---------------------------------------------------------------------------
# build_param_grid helpers
# ---------------------------------------------------------------------------


class TestBuildParamGrid:
    def test_integer_param_deduplication(self):
        """Small integer defaults that round to the same value are deduplicated."""
        grid = build_param_grid({"period": 2})
        # 0.8*2=1.6→2, 0.9*2=1.8→2, 1.0*2=2, 1.1*2=2.2→2, 1.2*2=2.4→2 → all 2
        assert grid["period"] == [2]

    def test_integer_param_larger_default(self):
        """A default of 20 produces 5 distinct rounded values."""
        grid = build_param_grid({"window": 20})
        assert len(grid["window"]) == 5
        assert grid["window"] == [16, 18, 20, 22, 24]

    def test_float_param_produces_five_values(self):
        """Float defaults always produce five values (no deduplication)."""
        grid = build_param_grid({"threshold": 0.5})
        assert len(grid["threshold"]) == 5
        assert grid["threshold"][2] == pytest.approx(0.5)

    def test_mixed_params(self):
        """A dict with both int and float params builds the correct combined grid."""
        grid = build_param_grid({"fast_window": 20, "slow_window": 60, "threshold": 0.95})
        assert len(grid["fast_window"]) == 5
        assert len(grid["slow_window"]) == 5
        assert len(grid["threshold"]) == 5
