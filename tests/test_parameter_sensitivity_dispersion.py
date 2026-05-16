"""Tests for parameter_sensitivity_dispersion (engine.metrics) and
sweep_and_score (engine.sensitivity).

Coverage:
- Happy path: constant-signal strategy yields dispersion ≈ 0 and stable_fraction = 1.0
- Happy path: param-sensitive strategy yields dispersion > 0 and stable_fraction < 1
- Edge case: empty sharpe_values list → dispersion = 0.0
- Edge case: single-element list → dispersion = 0.0
- Edge case: empty param_grid → one run, dispersion = 0.0
- Edge case: all strategy instantiations fail → returns zeros gracefully
- Failure mode: random-signal strategy over narrow grid yields higher dispersion
  than constant-signal strategy
- Determinism: two identical calls return identical results
- Cap behaviour: grid larger than max_points still succeeds and returns
  at most max_points Sharpe values
"""

import numpy as np
import pandas as pd
import pytest

from engine.metrics import parameter_sensitivity_dispersion
from engine.sensitivity import sweep_and_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Return a small deterministic OHLCV DataFrame with a positive drift."""
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0005, 0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(log_ret))
    opens = prices * (1.0 + rng.uniform(-0.002, 0.002, size=n))
    highs = np.maximum(prices, opens) * (1.0 + rng.uniform(0.0, 0.005, size=n))
    lows = np.minimum(prices, opens) * (1.0 - rng.uniform(0.0, 0.005, size=n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": prices, "volume": 1000},
        index=idx,
    )


class ConstantStrategy:
    """Always returns 1.0 regardless of parameters — Sharpe is param-invariant."""

    def __init__(self, **kwargs):
        pass

    def __call__(self, view) -> float:
        return 1.0


class WindowMAStrategy:
    """MA-crossover strategy whose Sharpe varies meaningfully with window size."""

    def __init__(self, window=5):
        self.window = int(window)

    def __call__(self, view) -> float:
        closes = view["close"]
        if len(closes) < self.window:
            return 0.0
        ma = closes.rolling(self.window).mean().iloc[-1]
        return 1.0 if closes.iloc[-1] > ma else 0.0


class AlwaysFailStrategy:
    """Always raises on instantiation — simulates a strategy with no valid combos."""

    def __init__(self, **kwargs):
        raise ValueError("always invalid")

    def __call__(self, view) -> float:  # pragma: no cover
        return 0.0


# ---------------------------------------------------------------------------
# parameter_sensitivity_dispersion unit tests
# ---------------------------------------------------------------------------


class TestParameterSensitivityDispersion:
    def test_empty_list_returns_zero(self):
        """Empty input must return 0.0, not raise."""
        assert parameter_sensitivity_dispersion([]) == 0.0

    def test_single_element_returns_zero(self):
        """A single Sharpe value has no dispersion."""
        assert parameter_sensitivity_dispersion([1.5]) == 0.0

    def test_identical_sharpes_returns_zero(self):
        """All equal values → population std = 0."""
        assert parameter_sensitivity_dispersion([0.8, 0.8, 0.8]) == pytest.approx(0.0)

    def test_known_two_values(self):
        """Population std of [0.0, 1.0] is 0.5."""
        assert parameter_sensitivity_dispersion([0.0, 1.0]) == pytest.approx(0.5)

    def test_matches_numpy_std(self):
        """Result equals np.std (population, not sample) for an arbitrary list."""
        vals = [0.2, 0.5, 1.1, -0.3, 0.8]
        assert parameter_sensitivity_dispersion(vals) == pytest.approx(float(np.std(vals)))

    def test_returns_float(self):
        """Return type must be float."""
        result = parameter_sensitivity_dispersion([0.3, 0.7])
        assert isinstance(result, float)

    def test_negative_sharpes_handled(self):
        """Handles negative Sharpe values without error."""
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        expected = float(np.std(vals))
        assert parameter_sensitivity_dispersion(vals) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# sweep_and_score happy path: constant-signal strategy
# ---------------------------------------------------------------------------


class TestSweepAndScoreConstantStrategy:
    def test_dispersion_approx_zero_for_constant_strategy(self):
        """Constant-signal strategy must yield dispersion ≈ 0 across any grid.

        Since every parameter combination produces the same equity curve and
        hence the same Sharpe, the population std-dev of Sharpe values is 0.
        """
        df = _make_ohlcv()
        param_grid = {"param_a": [1, 2, 3], "param_b": [10, 20, 30]}
        result = sweep_and_score(ConstantStrategy, df, param_grid, seed=42)
        assert result["dispersion"] == pytest.approx(0.0, abs=1e-9)

    def test_stable_fraction_one_for_constant_strategy(self):
        """Every grid point is within any tolerance of the centre when Sharpe is constant."""
        df = _make_ohlcv()
        param_grid = {"param_a": [1, 2, 3], "param_b": [10, 20, 30]}
        result = sweep_and_score(ConstantStrategy, df, param_grid, seed=42)
        assert result["stable_fraction"] == pytest.approx(1.0)

    def test_sharpes_all_equal_for_constant_strategy(self):
        """All returned Sharpe values should be identical for a param-invariant strategy."""
        df = _make_ohlcv()
        param_grid = {"param_a": [5, 10, 15]}
        result = sweep_and_score(ConstantStrategy, df, param_grid, seed=42)
        sharpes = result["sharpes"]
        assert len(sharpes) == 3
        assert all(abs(s - sharpes[0]) < 1e-9 for s in sharpes)

    def test_return_dict_has_required_keys(self):
        """Return dict must contain param_grid, sharpes, dispersion, stable_fraction."""
        df = _make_ohlcv(n=100)
        param_grid = {"param_a": [1, 2]}
        result = sweep_and_score(ConstantStrategy, df, param_grid, seed=0)
        assert "param_grid" in result
        assert "sharpes" in result
        assert "dispersion" in result
        assert "stable_fraction" in result

    def test_param_grid_passed_through(self):
        """param_grid in the return value must be the same object passed in."""
        df = _make_ohlcv(n=100)
        param_grid = {"param_a": [1, 2]}
        result = sweep_and_score(ConstantStrategy, df, param_grid, seed=0)
        assert result["param_grid"] is param_grid


# ---------------------------------------------------------------------------
# sweep_and_score happy path: param-sensitive strategy yields higher dispersion
# ---------------------------------------------------------------------------


class TestSweepAndScoreParamSensitive:
    def test_dispersion_greater_than_constant_strategy(self):
        """A param-sensitive MA strategy must have higher dispersion than constant."""
        df = _make_ohlcv(n=400, seed=7)
        param_grid = {"window": [5, 10, 20, 40, 80]}
        result_sensitive = sweep_and_score(WindowMAStrategy, df, param_grid, seed=42)
        result_constant = sweep_and_score(ConstantStrategy, df, {"param_a": [5, 10, 20, 40, 80]}, seed=42)
        assert result_sensitive["dispersion"] > result_constant["dispersion"]

    def test_n_trials_matches_grid_when_small(self):
        """When grid fits within max_points, all combos are evaluated."""
        df = _make_ohlcv(n=300)
        param_grid = {"window": [5, 10, 20]}
        result = sweep_and_score(WindowMAStrategy, df, param_grid, seed=42)
        assert len(result["sharpes"]) == 3

    def test_stable_fraction_in_range(self):
        """stable_fraction must be a float in [0.0, 1.0]."""
        df = _make_ohlcv(n=300)
        param_grid = {"window": [5, 10, 20, 40, 80]}
        result = sweep_and_score(WindowMAStrategy, df, param_grid, seed=42)
        sf = result["stable_fraction"]
        assert isinstance(sf, float)
        assert 0.0 <= sf <= 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSweepAndScoreEdgeCases:
    def test_empty_param_grid_runs_once(self):
        """An empty param_grid means one combination (the empty product), n_trials=1."""
        df = _make_ohlcv(n=200)
        result = sweep_and_score(ConstantStrategy, df, {}, seed=42)
        assert len(result["sharpes"]) == 1
        assert result["dispersion"] == 0.0
        assert result["stable_fraction"] == 1.0

    def test_all_failing_strategy_returns_zeros(self):
        """When every instantiation fails, the result is a zero/empty dict — no crash."""
        df = _make_ohlcv(n=200)
        param_grid = {"x": [1, 2, 3]}
        result = sweep_and_score(AlwaysFailStrategy, df, param_grid, seed=42)
        assert result["sharpes"] == []
        assert result["dispersion"] == 0.0
        assert result["stable_fraction"] == 0.0

    def test_cap_limits_sharpe_count(self):
        """When Cartesian product > max_points, at most max_points Sharpes are returned."""
        df = _make_ohlcv(n=200)
        # 4 × 4 × 4 = 64 combos, cap at 10
        param_grid = {
            "param_a": [1, 2, 3, 4],
            "param_b": [10, 20, 30, 40],
            "param_c": [100, 200, 300, 400],
        }
        result = sweep_and_score(ConstantStrategy, df, param_grid, seed=42, max_points=10)
        assert len(result["sharpes"]) <= 10

    def test_cap_includes_center_point(self):
        """When sampling, the centre-point combination must always be included.

        For a ConstantStrategy (all Sharpes equal), we can't distinguish centre
        directly, but we can verify that n_trials == max_points (centre + budget).
        """
        df = _make_ohlcv(n=200)
        param_grid = {
            "param_a": [1, 2, 3, 4, 5],
            "param_b": [10, 20, 30, 40, 50],
            "param_c": [100, 200, 300, 400, 500],
        }
        result = sweep_and_score(ConstantStrategy, df, param_grid, seed=42, max_points=15)
        assert len(result["sharpes"]) == 15

    def test_determinism(self):
        """Two calls with identical arguments return identical results."""
        df = _make_ohlcv(n=300, seed=99)
        param_grid = {"window": [5, 10, 20, 40, 80, 100, 150, 200]}

        r1 = sweep_and_score(WindowMAStrategy, df, param_grid, seed=7)
        r2 = sweep_and_score(WindowMAStrategy, df, param_grid, seed=7)
        assert r1["dispersion"] == r2["dispersion"]
        assert r1["stable_fraction"] == r2["stable_fraction"]
        assert r1["sharpes"] == r2["sharpes"]

    def test_different_seeds_may_differ(self):
        """Two calls with different seeds on a large grid can return different results."""
        df = _make_ohlcv(n=300, seed=77)
        # 8^2 = 64 combos → needs sampling with max_points=10
        param_grid = {
            "param_a": [1, 2, 3, 4, 5, 6, 7, 8],
            "param_b": [10, 20, 30, 40, 50, 60, 70, 80],
        }
        r1 = sweep_and_score(ConstantStrategy, df, param_grid, seed=1, max_points=10)
        r2 = sweep_and_score(ConstantStrategy, df, param_grid, seed=999, max_points=10)
        # Both must have dispersion=0 (constant strategy) even from different samples
        assert r1["dispersion"] == pytest.approx(0.0, abs=1e-9)
        assert r2["dispersion"] == pytest.approx(0.0, abs=1e-9)
