"""Tests for the PBO (Probability of Backtest Overfitting) implementation.

Covers:
- pbo() on a single-trial matrix (expected 0.0 — no selection took place)
- pbo() on a random-walk trials matrix (expected ≈ 0.5)
- pbo() returns 0.0 for degenerate / too-short inputs
- pbo() is deterministic across repeated calls
- pbo() is in [0, 1] on all inputs
- compute_all() includes "pbo" key when trials_matrix is provided
- compute_all() omits "pbo" key when trials_matrix is None
- all 9 strategies' metrics.json have a "pbo" key on every dataset
- strategies.json has a "pbo" field for every entry
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.antioverfitting import pbo
from engine import metrics as em

ROOT = Path(__file__).parent.parent
STRATEGIES_DIR = ROOT / "strategies"
DATASETS = ["trend_gbm", "mean_rev_ou", "regime_switch", "fat_tail"]

_ALL_STRATEGIES = [
    "01-dual-ema-momentum",
    "02-rsi-mean-reversion",
    "03-donchian-turtle-breakout",
    "04-52wk-high-proximity",
    "05-turn-of-month",
    "06-bollinger-mean-reversion",
    "07-absolute-momentum",
    "08-nr7-breakout",
    "09-volatility-managed",
]


def _rng_returns(seed: int, n_bars: int = 1000, n_trials: int = 25) -> np.ndarray:
    """Generate an (n_bars × n_trials) matrix of i.i.d. standard-normal returns."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_bars, n_trials))


def _make_equity(seed: int = 0, n: int = 300) -> pd.Series:
    """Minimal equity series for metrics tests — not realistic, just valid."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, n)
    values = 100.0 * np.cumprod(1.0 + rets)
    return pd.Series(values)


def _make_positions(n: int = 300) -> pd.Series:
    return pd.Series(np.ones(n))


# ---------------------------------------------------------------------------
# Core CSCV algorithm correctness
# ---------------------------------------------------------------------------


class TestPBOSingleTrial:
    def test_single_trial_returns_zero(self):
        """With exactly one trial there is no selection — PBO must be 0.0.

        A single-column matrix means k* is always the only trial; it trivially
        wins OOS rank (no competitors), so the IS-optimal never underperforms
        the median. pbo() must short-circuit to 0.0 for n_trials < 2.
        """
        matrix = _rng_returns(seed=0, n_bars=1000, n_trials=1)
        result = pbo(matrix)
        assert result == 0.0, f"Single-trial pbo must be 0.0, got {result}"

    def test_one_dominant_trial_low_pbo(self):
        """A trial that consistently dominates IS and OOS should yield near-zero PBO.

        Trial 0 is given a +5% daily drift advantage over all others.  Across
        all CSCV splits it will be IS-optimal and also OOS-optimal, so the
        underperform fraction should be very low (ideally 0.0 or very close).
        """
        rng = np.random.default_rng(42)
        n_bars, n_trials = 1000, 16
        matrix = rng.standard_normal((n_bars, n_trials)) * 0.01
        matrix[:, 0] += 0.05  # dominant trial with large positive drift
        result = pbo(matrix)
        assert result <= 0.1, (
            f"Dominant trial should yield low PBO (≤ 0.1), got {result:.4f}"
        )


class TestPBORandomWalk:
    def test_random_walk_matrix_near_half(self):
        """i.i.d. normal returns across all trials → IS selection is random → PBO ≈ 0.5.

        With pure noise across all N trials, the IS-best trial is selected by
        chance and its OOS rank is drawn from a uniform distribution. The
        expected PBO over many CSCV splits converges to 0.5. We allow a
        generous tolerance because n_trials=16 gives finite-sample variance.
        """
        matrix = _rng_returns(seed=7, n_bars=1000, n_trials=16)
        result = pbo(matrix)
        assert abs(result - 0.5) <= 0.15, (
            f"Random-walk matrix should yield PBO near 0.5, got {result:.4f}"
        )

    def test_random_walk_larger_matrix(self):
        """Larger trial count (n=32) → PBO should still be ≈ 0.5 for pure noise."""
        matrix = _rng_returns(seed=99, n_bars=1000, n_trials=32)
        result = pbo(matrix)
        assert abs(result - 0.5) <= 0.12, (
            f"Random-walk n_trials=32 should yield PBO near 0.5, got {result:.4f}"
        )


# ---------------------------------------------------------------------------
# Degenerate / edge-case inputs
# ---------------------------------------------------------------------------


class TestPBOEdgeCases:
    def test_too_few_bars_returns_zero(self):
        """Matrix with fewer rows than n_splits defaults to 0.0 (undefined)."""
        matrix = _rng_returns(seed=0, n_bars=8, n_trials=4)
        result = pbo(matrix, n_splits=16)
        assert result == 0.0, (
            f"Too-short matrix (n_bars=8 < n_splits=16) must return 0.0, got {result}"
        )

    def test_empty_matrix_returns_zero(self):
        """Zero-column matrix → n_trials < 2 → returns 0.0."""
        matrix = np.empty((1000, 0))
        result = pbo(matrix)
        assert result == 0.0, f"Empty matrix must return 0.0, got {result}"

    def test_1d_array_treated_as_single_trial(self):
        """A 1-D array is reshaped to (n_bars, 1) → single trial → 0.0."""
        arr = np.random.default_rng(0).standard_normal(1000)
        result = pbo(arr)
        assert result == 0.0, f"1-D input (single trial) must return 0.0, got {result}"

    def test_constant_returns_matrix_returns_zero(self):
        """All-zero returns matrix — degenerate Sharpe of 0 for every trial — must not crash."""
        matrix = np.zeros((500, 10))
        result = pbo(matrix)
        assert isinstance(result, float), "pbo must return a float even for degenerate input"
        assert 0.0 <= result <= 1.0, f"pbo must be in [0, 1], got {result}"


class TestPBOOutputRange:
    def test_output_always_in_unit_interval(self):
        """pbo() must return a value in [0.0, 1.0] regardless of input shape."""
        for seed in range(10):
            matrix = _rng_returns(seed=seed, n_bars=500, n_trials=8)
            result = pbo(matrix)
            assert 0.0 <= result <= 1.0, (
                f"pbo() out of [0,1] for seed={seed}: {result}"
            )

    def test_return_type_is_float(self):
        """pbo() must always return a Python float."""
        matrix = _rng_returns(seed=0, n_bars=500, n_trials=8)
        result = pbo(matrix)
        assert isinstance(result, float), f"Expected float, got {type(result)}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestPBODeterminism:
    def test_identical_output_on_repeated_calls(self):
        """pbo() is purely deterministic — same input must produce identical output."""
        matrix = _rng_returns(seed=42, n_bars=1000, n_trials=20)
        r1 = pbo(matrix)
        r2 = pbo(matrix)
        assert r1 == r2, f"pbo() is not deterministic: {r1} != {r2}"

    def test_identical_output_for_identical_matrices(self):
        """Two independently constructed identical matrices must yield the same PBO."""
        matrix_a = _rng_returns(seed=123, n_bars=800, n_trials=12)
        matrix_b = _rng_returns(seed=123, n_bars=800, n_trials=12)
        assert pbo(matrix_a) == pbo(matrix_b)


# ---------------------------------------------------------------------------
# Integration with compute_all()
# ---------------------------------------------------------------------------


class TestComputeAllWithPBO:
    def test_compute_all_includes_pbo_when_matrix_provided(self):
        """compute_all() must include 'pbo' key when trials_matrix is provided."""
        equity = _make_equity()
        positions = _make_positions()
        matrix = _rng_returns(seed=0, n_bars=len(equity) - 1, n_trials=8)
        result = em.compute_all(equity, positions, 0.0, matrix)
        assert "pbo" in result, "compute_all() must include 'pbo' when trials_matrix is given"

    def test_compute_all_omits_pbo_when_matrix_is_none(self):
        """compute_all() must NOT include 'pbo' key when trials_matrix is None."""
        equity = _make_equity()
        positions = _make_positions()
        result = em.compute_all(equity, positions, 0.0, None)
        assert "pbo" not in result, "compute_all() must not include 'pbo' without trials_matrix"

    def test_pbo_in_compute_all_is_in_unit_interval(self):
        """The 'pbo' value from compute_all() must lie in [0.0, 1.0]."""
        equity = _make_equity(seed=5)
        positions = _make_positions()
        matrix = _rng_returns(seed=5, n_bars=len(equity) - 1, n_trials=10)
        result = em.compute_all(equity, positions, 0.0, matrix)
        assert 0.0 <= result["pbo"] <= 1.0


# ---------------------------------------------------------------------------
# Backfilled metrics.json: all 9 strategies have "pbo" on all datasets
# ---------------------------------------------------------------------------


class TestMetricsJsonHasPBO:
    def _load(self, strategy: str) -> dict:
        path = STRATEGIES_DIR / strategy / "metrics.json"
        return json.loads(path.read_text())

    @pytest.mark.parametrize("strategy", _ALL_STRATEGIES)
    def test_strategy_has_pbo_on_all_datasets(self, strategy):
        """Every strategy's metrics.json must have a 'pbo' key on all 4 datasets."""
        m = self._load(strategy)
        for ds in DATASETS:
            assert "pbo" in m[ds], (
                f"metrics.json for {strategy}/{ds} is missing 'pbo'"
            )

    @pytest.mark.parametrize("strategy", _ALL_STRATEGIES)
    def test_pbo_values_are_in_unit_interval(self, strategy):
        """All backfilled PBO values must be in [0.0, 1.0]."""
        m = self._load(strategy)
        for ds in DATASETS:
            v = m[ds]["pbo"]
            assert isinstance(v, (int, float)), (
                f"{strategy}/{ds}: pbo must be numeric, got {type(v)}"
            )
            assert 0.0 <= v <= 1.0, (
                f"{strategy}/{ds}: pbo={v} is outside [0, 1]"
            )

    def test_existing_metrics_unchanged(self):
        """Adding 'pbo' must not alter any pre-existing metric values."""
        m = self._load("01-dual-ema-momentum")
        assert "cagr" in m["regime_switch"], "Pre-existing 'cagr' key missing"
        assert "deflated_sharpe" in m["regime_switch"], "Pre-existing 'deflated_sharpe' key missing"
        assert "sharpe" in m["regime_switch"], "Pre-existing 'sharpe' key missing"


# ---------------------------------------------------------------------------
# strategies.json has "pbo" field for every entry
# ---------------------------------------------------------------------------


class TestStrategiesJsonPBO:
    def _load_registry(self) -> list:
        return json.loads((ROOT / "strategies.json").read_text())

    def test_all_entries_have_pbo_field(self):
        """Every entry in strategies.json must have a 'pbo' key."""
        registry = self._load_registry()
        for entry in registry:
            assert "pbo" in entry, (
                f"strategies.json entry '{entry['name']}' is missing 'pbo' field"
            )

    def test_pbo_field_has_all_datasets(self):
        """The 'pbo' field for each entry must contain all 4 dataset keys."""
        registry = self._load_registry()
        for entry in registry:
            pbo_field = entry["pbo"]
            for ds in DATASETS:
                assert ds in pbo_field, (
                    f"strategies.json '{entry['name']}' pbo field missing dataset '{ds}'"
                )
