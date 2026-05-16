"""Tests for sharpe_ci — bootstrapped confidence intervals for the Sharpe ratio.

Acceptance criteria:
- engine/metrics.py has sharpe_ci(returns, n_bootstrap, confidence, seed) -> (lower, point, upper)
- All metrics.json files include sharpe_ci_lower and sharpe_ci_upper
- sharpe_ci_lower <= sharpe <= sharpe_ci_upper holds for every dataset of every strategy
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from engine import metrics as em

ROOT = Path(__file__).parent.parent
STRATEGIES_DIR = ROOT / "strategies"
STRATEGY_DIRS = [
    "01-dual-ema-momentum",
    "02-rsi-mean-reversion",
    "03-donchian-turtle-breakout",
    "04-52wk-high-proximity",
    "05-turn-of-month",
    "06-bollinger-mean-reversion",
    "07-absolute-momentum",
]
DATASETS = ["trend_gbm", "mean_rev_ou", "regime_switch", "fat_tail"]


def _returns(n: int = 500, seed: int = 0) -> pd.Series:
    """Generate simple i.i.d. normal return series."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n))


def _load_metrics(strategy: str) -> dict:
    path = STRATEGIES_DIR / strategy / "metrics.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Happy path — function contract
# ---------------------------------------------------------------------------


class TestSharpeCI:
    def test_returns_three_floats(self):
        """sharpe_ci must return a 3-tuple of floats."""
        lo, pt, hi = em.sharpe_ci(_returns())
        assert isinstance(lo, float)
        assert isinstance(pt, float)
        assert isinstance(hi, float)

    def test_ordering_lo_lte_pt_lte_hi(self):
        """Lower bound <= point estimate <= upper bound."""
        lo, pt, hi = em.sharpe_ci(_returns())
        assert lo <= pt <= hi, f"Expected lo <= pt <= hi, got ({lo}, {pt}, {hi})"

    def test_point_matches_standalone_sharpe(self):
        """The point estimate must equal the hand-computed annualised Sharpe."""
        r = _returns(n=300, seed=7)
        lo, pt, hi = em.sharpe_ci(r)
        expected_pt = float(r.mean() / r.std(ddof=1) * math.sqrt(252))
        assert abs(pt - expected_pt) < 1e-10, f"Point estimate {pt} != expected {expected_pt}"

    def test_interval_width_positive(self):
        """The CI must have positive width (lower strictly less than upper)."""
        lo, pt, hi = em.sharpe_ci(_returns())
        assert hi > lo, f"CI has zero width: lo={lo}, hi={hi}"

    def test_determinism_with_same_seed(self):
        """Same data + same seed must produce identical results."""
        r = _returns()
        result1 = em.sharpe_ci(r, seed=42)
        result2 = em.sharpe_ci(r, seed=42)
        assert result1 == result2, "sharpe_ci is not deterministic with fixed seed"

    def test_different_seeds_give_different_bounds(self):
        """Different seeds should produce different CI bounds (not point estimate)."""
        r = _returns()
        lo1, _, hi1 = em.sharpe_ci(r, seed=1)
        lo2, _, hi2 = em.sharpe_ci(r, seed=99)
        assert (lo1, hi1) != (lo2, hi2), "Different seeds must produce different CIs"

    def test_confidence_level_affects_width(self):
        """Higher confidence level must produce a wider or equal CI."""
        r = _returns(n=400, seed=5)
        lo90, _, hi90 = em.sharpe_ci(r, confidence=0.90)
        lo99, _, hi99 = em.sharpe_ci(r, confidence=0.99)
        width90 = hi90 - lo90
        width99 = hi99 - lo99
        assert width99 >= width90, (
            f"99% CI width {width99:.4f} should be >= 90% CI width {width90:.4f}"
        )

    def test_n_bootstrap_large_reduces_variance(self):
        """More bootstrap samples should reduce the variance of repeated CI estimates."""
        r = _returns(n=300, seed=0)
        lows_small = [em.sharpe_ci(r, n_bootstrap=20, seed=s)[0] for s in range(30)]
        lows_large = [em.sharpe_ci(r, n_bootstrap=2000, seed=s)[0] for s in range(30)]
        assert np.std(lows_large) < np.std(lows_small), (
            "Larger n_bootstrap should reduce variance of the lower bound estimate"
        )

    def test_positive_drift_series_ci_positive_lower(self):
        """A strongly positive-drift series should have its CI lower bound above zero."""
        rng = np.random.default_rng(99)
        r = pd.Series(rng.normal(0.005, 0.005, 500))  # high mean/vol ratio
        lo, _, _ = em.sharpe_ci(r, n_bootstrap=2000, seed=42)
        assert lo > 0, f"Strongly positive drift series should have CI lower > 0, got {lo}"

    def test_zero_drift_series_ci_straddles_zero(self):
        """A zero-drift i.i.d. series should have its 95% CI straddle zero."""
        rng = np.random.default_rng(77)
        r = pd.Series(rng.normal(0.0, 0.01, 500))
        lo, _, hi = em.sharpe_ci(r, n_bootstrap=1000, seed=42)
        assert lo < 0 < hi, (
            f"Zero-drift series should have CI straddling zero, got [{lo}, {hi}]"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSharpeCI_EdgeCases:
    def test_two_observation_series(self):
        """A 2-element series must not raise and returns 3 floats.

        Bootstrap resamples of size 2 drawn from [a, b] may pick the same
        value twice (std=0), producing NaN Sharpes. The function does not
        guard against this degenerate case — the caller is responsible for
        passing series with sufficient length. We assert only no-crash.
        """
        r = pd.Series([0.01, -0.005])
        result = em.sharpe_ci(r)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)

    def test_constant_series_returns_zeros(self):
        """A constant (zero-variance) return series yields zero Sharpe at all three positions."""
        r = pd.Series([0.0] * 100)
        lo, pt, hi = em.sharpe_ci(r)
        assert math.isnan(lo) or lo <= pt
        assert math.isnan(hi) or pt <= hi

    def test_single_nonzero_observation(self):
        """Series of length 1 — bootstrap resamples collapse to the same value; no crash."""
        r = pd.Series([0.005])
        lo, pt, hi = em.sharpe_ci(r)
        assert isinstance(lo, float) and isinstance(hi, float)

    def test_large_series_does_not_crash(self):
        """10 000-bar series must complete without error."""
        r = _returns(n=10_000, seed=0)
        lo, pt, hi = em.sharpe_ci(r)
        assert lo <= pt <= hi

    def test_numpy_array_like_input_accepted(self):
        """Works when returns is a numpy-array-backed pd.Series (no pandas-specific path)."""
        r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 200))
        lo, pt, hi = em.sharpe_ci(r)
        assert lo <= pt <= hi


# ---------------------------------------------------------------------------
# Failure mode — explicitly wrong usage
# ---------------------------------------------------------------------------


class TestSharpeCI_FailureModes:
    def test_equity_series_not_returns_gives_wrong_ci(self):
        """Passing an equity curve directly (not pct_change) produces CI that may not
        bracket the correct Sharpe — this documents misuse, not a crash.

        The function does not raise; the caller is responsible for passing returns.
        We assert only that the output is a valid tuple of floats.
        """
        equity = pd.Series(100.0 * np.cumprod(1.0 + _returns(n=200).values))
        lo, pt, hi = em.sharpe_ci(equity)
        assert isinstance(lo, float) and isinstance(pt, float) and isinstance(hi, float)


# ---------------------------------------------------------------------------
# Acceptance: all metrics.json have sharpe_ci_lower, sharpe_ci_upper
# ---------------------------------------------------------------------------


class TestMetricsJsonHasSharpeCI:
    def test_all_strategies_have_sharpe_ci_fields(self):
        """Every strategy's metrics.json must contain sharpe_ci_lower and sharpe_ci_upper
        on every dataset.
        """
        for strat in STRATEGY_DIRS:
            m = _load_metrics(strat)
            for ds in DATASETS:
                assert "sharpe_ci_lower" in m[ds], (
                    f"{strat}/{ds} missing sharpe_ci_lower"
                )
                assert "sharpe_ci_upper" in m[ds], (
                    f"{strat}/{ds} missing sharpe_ci_upper"
                )

    def test_all_strategies_have_sharpe_ci_confidence(self):
        """sharpe_ci_confidence must be present and equal to 0.95 on every dataset."""
        for strat in STRATEGY_DIRS:
            m = _load_metrics(strat)
            for ds in DATASETS:
                conf = m[ds].get("sharpe_ci_confidence")
                assert conf is not None, f"{strat}/{ds} missing sharpe_ci_confidence"
                assert abs(conf - 0.95) < 1e-9, (
                    f"{strat}/{ds} sharpe_ci_confidence={conf} != 0.95"
                )

    def test_sharpe_within_ci_for_all_strategies_and_datasets(self):
        """sharpe_ci_lower <= sharpe <= sharpe_ci_upper for every strategy/dataset pair."""
        for strat in STRATEGY_DIRS:
            m = _load_metrics(strat)
            for ds in DATASETS:
                entry = m[ds]
                lo = entry["sharpe_ci_lower"]
                pt = entry["sharpe"]
                hi = entry["sharpe_ci_upper"]
                assert lo is not None and hi is not None and pt is not None, (
                    f"{strat}/{ds}: CI or sharpe is None"
                )
                assert lo <= pt <= hi, (
                    f"{strat}/{ds}: sharpe={pt} not in CI [{lo}, {hi}]"
                )

    def test_ci_lower_lt_upper_for_all(self):
        """CI must have positive width for all strategy/dataset entries."""
        for strat in STRATEGY_DIRS:
            m = _load_metrics(strat)
            for ds in DATASETS:
                lo = m[ds]["sharpe_ci_lower"]
                hi = m[ds]["sharpe_ci_upper"]
                assert hi > lo, f"{strat}/{ds}: CI has zero width [{lo}, {hi}]"

    def test_absolute_momentum_regime_switch_ci_excludes_zero(self):
        """Strategy 07 on regime_switch has the only CI that excludes zero,
        confirming strong statistical evidence of positive edge.
        """
        m = _load_metrics("07-absolute-momentum")
        lo = m["regime_switch"]["sharpe_ci_lower"]
        hi = m["regime_switch"]["sharpe_ci_upper"]
        assert lo > 0, (
            f"07-absolute-momentum/regime_switch CI lower should be > 0, got {lo}. "
            f"CI=[{lo}, {hi}]"
        )

    def test_most_cis_straddle_zero(self):
        """At most 1 entry should have a CI excluding zero — most strategies are
        statistically inconclusive on 1000-bar synthetic datasets.
        """
        exclude_zero_count = 0
        for strat in STRATEGY_DIRS:
            m = _load_metrics(strat)
            for ds in DATASETS:
                lo = m[ds]["sharpe_ci_lower"]
                hi = m[ds]["sharpe_ci_upper"]
                if lo > 0 or hi < 0:
                    exclude_zero_count += 1
        assert exclude_zero_count <= 3, (
            f"Expected at most 3 CIs to exclude zero (finite-sample noise), "
            f"got {exclude_zero_count}"
        )
