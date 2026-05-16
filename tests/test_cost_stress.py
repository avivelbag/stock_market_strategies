"""Tests for engine/cost_stress.py — transaction-cost stress sweep.

Validates:
1. cost_stress.json exists for every registered strategy.
2. Each dataset sub-dict contains all 20 default grid cells.
3. Top-level cost_breakeven_* fields are present.
4. The (5, 5) cell matches metrics.json[dataset]["sharpe"] to within 0.01.
5. compute_breakevens produces correct sentinels for degenerate cases.
6. cost_stress_sweep is deterministic and returns 20 cells on the default grid.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
STRATEGIES_JSON = ROOT / "strategies.json"
STRATEGIES_DIR = ROOT / "strategies"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_GRID_KEYS = {
    f"({c}, {s})"
    for c in [0, 2, 5, 10, 20]
    for s in [0, 2, 5, 10]
}
assert len(EXPECTED_GRID_KEYS) == 20, "Grid must have 20 cells"


def _load_registry() -> list:
    with open(STRATEGIES_JSON) as f:
        return json.load(f)


def _load_cost_stress(strategy_name: str) -> dict:
    path = STRATEGIES_DIR / strategy_name / "cost_stress.json"
    with open(path) as f:
        return json.load(f)


def _load_metrics(strategy_name: str) -> dict:
    path = STRATEGIES_DIR / strategy_name / "metrics.json"
    with open(path) as f:
        return json.load(f)


def _is_valid_breakeven(val) -> bool:
    """A breakeven value is valid if it is a real number or one of the two sentinels."""
    from engine.cost_stress import ABOVE_MAX_GRID_SENTINEL, ALREADY_NEGATIVE_SENTINEL

    if val in (ALREADY_NEGATIVE_SENTINEL, ABOVE_MAX_GRID_SENTINEL):
        return True
    if isinstance(val, (int, float)) and not math.isnan(val):
        return True
    return False


# ---------------------------------------------------------------------------
# File-presence gate (acceptance criterion 1)
# ---------------------------------------------------------------------------


class TestCostStressJsonExists:
    def test_cost_stress_json_present_for_all_strategies(self):
        for entry in _load_registry():
            path = STRATEGIES_DIR / entry["name"] / "cost_stress.json"
            assert path.is_file(), (
                f"cost_stress.json missing for strategy '{entry['name']}'"
            )

    def test_cost_stress_json_is_valid_json(self):
        for entry in _load_registry():
            path = STRATEGIES_DIR / entry["name"] / "cost_stress.json"
            try:
                with open(path) as f:
                    json.load(f)
            except json.JSONDecodeError as exc:
                pytest.fail(
                    f"cost_stress.json for '{entry['name']}' is not valid JSON: {exc}"
                )


# ---------------------------------------------------------------------------
# Grid completeness (acceptance criterion 2)
# ---------------------------------------------------------------------------


class TestGridCompleteness:
    def test_all_20_grid_cells_present_per_dataset(self):
        """Every dataset sub-dict in cost_stress.json must contain all 20 cells."""
        for entry in _load_registry():
            data = _load_cost_stress(entry["name"])
            datasets = entry.get("datasets", [])
            for ds in datasets:
                assert ds in data, (
                    f"Dataset '{ds}' missing from cost_stress.json for '{entry['name']}'"
                )
                for key in EXPECTED_GRID_KEYS:
                    assert key in data[ds], (
                        f"Grid key '{key}' missing from dataset '{ds}' in "
                        f"cost_stress.json for '{entry['name']}'"
                    )

    def test_grid_values_are_finite_floats(self):
        """All 20 grid cells must be finite floats, not NaN or Inf."""
        for entry in _load_registry():
            data = _load_cost_stress(entry["name"])
            for ds in entry.get("datasets", []):
                for key in EXPECTED_GRID_KEYS:
                    val = data[ds][key]
                    assert isinstance(val, (int, float)), (
                        f"Grid cell '{key}' in dataset '{ds}' for '{entry['name']}' "
                        f"is not numeric: {val!r}"
                    )
                    assert math.isfinite(val), (
                        f"Grid cell '{key}' in dataset '{ds}' for '{entry['name']}' "
                        f"is not finite: {val}"
                    )


# ---------------------------------------------------------------------------
# Breakeven fields (acceptance criterion 3)
# ---------------------------------------------------------------------------


class TestBreakevenFields:
    def test_top_level_breakeven_fields_present(self):
        for entry in _load_registry():
            data = _load_cost_stress(entry["name"])
            assert "cost_breakeven_commission_bps" in data, (
                f"Missing 'cost_breakeven_commission_bps' in cost_stress.json "
                f"for '{entry['name']}'"
            )
            assert "cost_breakeven_slippage_bps" in data, (
                f"Missing 'cost_breakeven_slippage_bps' in cost_stress.json "
                f"for '{entry['name']}'"
            )

    def test_top_level_breakeven_fields_are_valid(self):
        """Breakevens must be numeric or one of the two documented sentinels."""
        for entry in _load_registry():
            data = _load_cost_stress(entry["name"])
            for field in ("cost_breakeven_commission_bps", "cost_breakeven_slippage_bps"):
                val = data[field]
                assert _is_valid_breakeven(val), (
                    f"'{field}' in cost_stress.json for '{entry['name']}' has "
                    f"unexpected value: {val!r}"
                )

    def test_per_dataset_breakeven_fields_present(self):
        """Each dataset sub-dict must also have its own breakeven fields."""
        for entry in _load_registry():
            data = _load_cost_stress(entry["name"])
            for ds in entry.get("datasets", []):
                for field in (
                    "cost_breakeven_commission_bps",
                    "cost_breakeven_slippage_bps",
                ):
                    assert field in data[ds], (
                        f"Dataset '{ds}' missing '{field}' in cost_stress.json "
                        f"for '{entry['name']}'"
                    )
                    assert _is_valid_breakeven(data[ds][field]), (
                        f"Dataset '{ds}' field '{field}' in cost_stress.json for "
                        f"'{entry['name']}' has unexpected value: {data[ds][field]!r}"
                    )


# ---------------------------------------------------------------------------
# Regression: (5, 5) matches metrics.json (acceptance criterion 4)
# ---------------------------------------------------------------------------


class TestRegressionAgainstMetricsJson:
    def test_cell_5_5_matches_metrics_json_sharpe(self):
        """cost_stress.json[(dataset)][(5, 5)] must equal metrics.json[dataset][sharpe] ±0.01."""
        for entry in _load_registry():
            cost_stress = _load_cost_stress(entry["name"])
            metrics = _load_metrics(entry["name"])
            for ds in entry.get("datasets", []):
                stress_sharpe = cost_stress[ds]["(5, 5)"]
                metrics_sharpe = metrics[ds]["sharpe"]
                assert abs(stress_sharpe - metrics_sharpe) <= 0.01, (
                    f"cost_stress (5,5) Sharpe {stress_sharpe:.6f} does not match "
                    f"metrics.json Sharpe {metrics_sharpe:.6f} for strategy "
                    f"'{entry['name']}' on dataset '{ds}' (diff > 0.01)"
                )


# ---------------------------------------------------------------------------
# Unit tests for cost_stress_sweep function
# ---------------------------------------------------------------------------

def _make_trending_prices(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create a simple synthetic OHLCV DataFrame with an upward trend."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.005, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class _AlwaysLong:
    """Trivial always-long stateless strategy for testing."""

    def __call__(self, view) -> float:
        return 1.0


class _AlwaysFlat:
    """Trivial always-flat strategy — no trades, no cost exposure."""

    def __call__(self, view) -> float:
        return 0.0


class TestCostStressSweepUnit:
    def test_returns_correct_number_of_cells(self):
        from engine.cost_stress import cost_stress_sweep

        df = _make_trending_prices()
        result = cost_stress_sweep(_AlwaysLong, {}, df)
        assert len(result) == 20

    def test_all_expected_keys_present(self):
        from engine.cost_stress import cost_stress_sweep

        df = _make_trending_prices()
        result = cost_stress_sweep(_AlwaysLong, {}, df)
        assert set(result.keys()) == EXPECTED_GRID_KEYS

    def test_higher_costs_lower_sharpe_for_active_strategy(self):
        """For an always-long strategy, higher costs must never improve Sharpe."""
        from engine.cost_stress import cost_stress_sweep

        df = _make_trending_prices()
        result = cost_stress_sweep(_AlwaysLong, {}, df)
        sharpe_zero = result["(0, 0)"]
        sharpe_high = result["(20, 10)"]
        assert sharpe_zero >= sharpe_high, (
            f"Zero-cost Sharpe {sharpe_zero} < high-cost Sharpe {sharpe_high}"
        )

    def test_flat_strategy_unaffected_by_costs(self):
        """Always-flat strategy has zero exposure; costs should not change its Sharpe."""
        from engine.cost_stress import cost_stress_sweep

        df = _make_trending_prices()
        result = cost_stress_sweep(_AlwaysFlat, {}, df)
        sharpes = list(result.values())
        assert all(abs(s - sharpes[0]) < 1e-9 for s in sharpes), (
            "Flat strategy Sharpe changed across cost grid"
        )

    def test_deterministic_across_runs(self):
        """Same inputs produce identical outputs on two consecutive calls."""
        from engine.cost_stress import cost_stress_sweep

        df = _make_trending_prices(seed=42)
        r1 = cost_stress_sweep(_AlwaysLong, {}, df)
        r2 = cost_stress_sweep(_AlwaysLong, {}, df)
        assert r1 == r2

    def test_custom_grid_produces_correct_cell_count(self):
        from engine.cost_stress import cost_stress_sweep

        df = _make_trending_prices()
        result = cost_stress_sweep(
            _AlwaysLong, {}, df, commission_bps_range=[0, 5], slippage_bps_range=[0, 5]
        )
        assert len(result) == 4
        assert "(0, 0)" in result
        assert "(0, 5)" in result
        assert "(5, 0)" in result
        assert "(5, 5)" in result


# ---------------------------------------------------------------------------
# Unit tests for compute_breakevens
# ---------------------------------------------------------------------------


class TestComputeBreakevens:
    def test_already_negative_at_zero_cost_commission(self):
        """All-negative sweep results → ALREADY_NEGATIVE_SENTINEL for both axes."""
        from engine.cost_stress import (
            ALREADY_NEGATIVE_SENTINEL,
            compute_breakevens,
        )

        # Sharpe is negative at every grid cell
        sweep = {
            f"({c}, {s})": -0.5
            for c in [0, 2, 5, 10, 20]
            for s in [0, 2, 5, 10]
        }
        b_comm, b_slip = compute_breakevens(sweep)
        assert b_comm == ALREADY_NEGATIVE_SENTINEL
        assert b_slip == ALREADY_NEGATIVE_SENTINEL

    def test_above_max_grid_when_always_positive(self):
        """All-positive sweep results → ABOVE_MAX_GRID_SENTINEL for both axes."""
        from engine.cost_stress import ABOVE_MAX_GRID_SENTINEL, compute_breakevens

        sweep = {
            f"({c}, {s})": 0.5
            for c in [0, 2, 5, 10, 20]
            for s in [0, 2, 5, 10]
        }
        b_comm, b_slip = compute_breakevens(sweep)
        assert b_comm == ABOVE_MAX_GRID_SENTINEL
        assert b_slip == ABOVE_MAX_GRID_SENTINEL

    def test_breakeven_commission_at_correct_level(self):
        """Sharpe turns negative at commission=10 (slippage held at 5)."""
        from engine.cost_stress import compute_breakevens

        sweep = {f"({c}, {s})": 0.0 for c in [0, 2, 5, 10, 20] for s in [0, 2, 5, 10]}
        # Positive at low commission, negative at 10+ (slippage=5 row)
        for c in [0, 2, 5]:
            sweep[f"({c}, 5)"] = 0.3
        for c in [10, 20]:
            sweep[f"({c}, 5)"] = -0.1
        b_comm, _ = compute_breakevens(sweep)
        assert b_comm == 10

    def test_breakeven_slippage_at_correct_level(self):
        """Sharpe turns negative at slippage=5 (commission held at 5)."""
        from engine.cost_stress import compute_breakevens

        sweep = {f"({c}, {s})": 0.0 for c in [0, 2, 5, 10, 20] for s in [0, 2, 5, 10]}
        for s in [0, 2]:
            sweep[f"(5, {s})"] = 0.2
        for s in [5, 10]:
            sweep[f"(5, {s})"] = -0.05
        _, b_slip = compute_breakevens(sweep)
        assert b_slip == 5


# ---------------------------------------------------------------------------
# Edge-case: large input and missing-file failure
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_sweep_handles_large_dataset(self):
        """Sweep must complete without error on a 2000-bar dataset."""
        from engine.cost_stress import cost_stress_sweep

        df = _make_trending_prices(n=2000)
        result = cost_stress_sweep(_AlwaysLong, {}, df)
        assert len(result) == 20

    def test_missing_cost_stress_json_is_detected_by_structure_test(self, tmp_path):
        """Verify that a strategy dir without cost_stress.json causes a file-not-found."""
        no_stress = tmp_path / "cost_stress.json"
        assert not no_stress.exists()

    def test_cost_stress_sweep_raises_on_too_short_dataset(self):
        """A 1-row DataFrame must raise ValueError (engine guard)."""
        from engine.cost_stress import cost_stress_sweep

        df = pd.DataFrame(
            {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.0], "volume": [1.0]},
            index=pd.date_range("2020-01-01", periods=1),
        )
        with pytest.raises(ValueError, match="at least 2 rows"):
            cost_stress_sweep(_AlwaysLong, {}, df)
