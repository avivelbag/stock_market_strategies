"""Tests for ulcer_index, martin_ratio, and their integration into compute_all().

Covers the hand-constructed fixture from the suggestion, edge cases
(monotonic equity, single-point series, empty series), and the compute_all()
integration check.
"""

import math

import numpy as np
import pandas as pd
import pytest

from engine import metrics as em


# ---------------------------------------------------------------------------
# ulcer_index — happy path
# ---------------------------------------------------------------------------


class TestUlcerIndexHappyPath:
    def test_suggestion_fixture(self):
        """Hand-constructed case from the suggestion spec.

        equity = [100, 95, 90, 95, 100, 105]
        running_max = [100, 100, 100, 100, 100, 105]
        dd_pct = [0, -5, -10, -5, 0, 0]
        UI = sqrt((0 + 25 + 100 + 25 + 0 + 0) / 6) = sqrt(25) = 5.0
        """
        equity = pd.Series([100.0, 95.0, 90.0, 95.0, 100.0, 105.0])
        result = em.ulcer_index(equity)
        assert result == pytest.approx(5.0, abs=1e-4)

    def test_flat_drawdown_and_recovery(self):
        """A single drawdown to 80 and flat recovery gives a known RMS.

        equity = [100, 80, 80, 100]
        running_max = [100, 100, 100, 100]
        dd_pct = [0, -20, -20, 0]
        UI = sqrt((0 + 400 + 400 + 0) / 4) = sqrt(200) ≈ 14.1421
        """
        equity = pd.Series([100.0, 80.0, 80.0, 100.0])
        expected = math.sqrt(200.0)
        assert em.ulcer_index(equity) == pytest.approx(expected, abs=1e-4)

    def test_return_type_is_float(self):
        """ulcer_index must always return a Python float."""
        equity = pd.Series([100.0, 95.0, 90.0])
        result = em.ulcer_index(equity)
        assert isinstance(result, float)

    def test_non_negative(self):
        """UI is always >= 0 for any valid equity series."""
        rng = np.random.default_rng(42)
        equity = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, 500)))
        assert em.ulcer_index(equity) >= 0.0


# ---------------------------------------------------------------------------
# ulcer_index — edge cases
# ---------------------------------------------------------------------------


class TestUlcerIndexEdgeCases:
    def test_monotonically_rising_equity_gives_zero(self):
        """Equity that never draws down has UI == 0."""
        equity = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        assert em.ulcer_index(equity) == pytest.approx(0.0, abs=1e-10)

    def test_single_point_series_gives_zero(self):
        """A single-bar equity series has no drawdown history → UI == 0."""
        equity = pd.Series([100.0])
        assert em.ulcer_index(equity) == pytest.approx(0.0, abs=1e-10)

    def test_two_point_rising_series(self):
        """Two-bar rising series: bar 0 is the new max, bar 1 exceeds it → UI == 0."""
        equity = pd.Series([100.0, 110.0])
        assert em.ulcer_index(equity) == pytest.approx(0.0, abs=1e-10)

    def test_two_point_falling_series(self):
        """Two-bar falling series: bar 1 falls 10% below bar 0 peak.

        dd_pct = [0, -10]
        UI = sqrt((0 + 100) / 2) = sqrt(50) ≈ 7.0711
        """
        equity = pd.Series([100.0, 90.0])
        expected = math.sqrt(50.0)
        assert em.ulcer_index(equity) == pytest.approx(expected, abs=1e-4)

    def test_large_series_finite_result(self):
        """ulcer_index on a 1000-bar realistic equity series returns a finite float."""
        rng = np.random.default_rng(7)
        equity = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.012, 1000)))
        result = em.ulcer_index(equity)
        assert math.isfinite(result)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# ulcer_index — explicit failure mode
# ---------------------------------------------------------------------------


class TestUlcerIndexFailureModes:
    def test_correct_formula_not_simple_average(self):
        """Verify UI squares drawdowns before averaging — not a plain mean.

        For dd_pct = [0, -10], plain mean = 5, sqrt(5) ≈ 2.236.
        Squared mean: sqrt((0 + 100) / 2) = sqrt(50) ≈ 7.071.
        The two answers differ; the function must return the RMS, not the mean.
        """
        equity = pd.Series([100.0, 90.0])
        result = em.ulcer_index(equity)
        wrong_plain_mean = math.sqrt(5.0)
        assert result != pytest.approx(wrong_plain_mean, abs=1e-4), (
            "ulcer_index must square drawdowns before averaging (RMS), not average them"
        )
        assert result == pytest.approx(math.sqrt(50.0), abs=1e-4)


# ---------------------------------------------------------------------------
# martin_ratio — happy path
# ---------------------------------------------------------------------------


class TestMartinRatioHappyPath:
    def test_suggestion_fixture(self):
        """Martin Ratio for the suggestion fixture: annualized return / (UI / 100).

        equity = [100, 95, 90, 95, 100, 105], n=6, UI=5.0
        annual_return = (105/100)^(252/6) - 1 = 1.05^42 - 1
        martin_ratio = annual_return / 0.05
        """
        equity = pd.Series([100.0, 95.0, 90.0, 95.0, 100.0, 105.0])
        n = 6
        expected_annual = (105.0 / 100.0) ** (252.0 / n) - 1
        expected_martin = expected_annual / (5.0 / 100.0)
        result = em.martin_ratio(equity)
        assert result == pytest.approx(expected_martin, rel=1e-4)

    def test_positive_cagr_and_drawdown_gives_positive_martin(self):
        """Equity ending above start with a drawdown gives a positive Martin Ratio."""
        equity = pd.Series([100.0, 85.0, 90.0, 110.0])
        result = em.martin_ratio(equity)
        assert result > 0.0

    def test_negative_cagr_gives_negative_martin(self):
        """Equity ending below start with a drawdown gives a negative Martin Ratio."""
        equity = pd.Series([100.0, 85.0, 90.0, 80.0])
        result = em.martin_ratio(equity)
        assert result < 0.0

    def test_return_type_is_float(self):
        """martin_ratio must always return a Python float."""
        equity = pd.Series([100.0, 90.0, 110.0])
        result = em.martin_ratio(equity)
        assert isinstance(result, float)

    def test_periods_per_year_parameter(self):
        """periods_per_year parameter affects the annualization scaling."""
        equity = pd.Series([100.0, 95.0, 90.0, 95.0, 100.0, 105.0])
        mr_daily = em.martin_ratio(equity, periods_per_year=252)
        mr_weekly = em.martin_ratio(equity, periods_per_year=52)
        assert mr_daily != pytest.approx(mr_weekly, rel=1e-3), (
            "periods_per_year must affect the annualization — 252 and 52 should differ"
        )


# ---------------------------------------------------------------------------
# martin_ratio — edge cases
# ---------------------------------------------------------------------------


class TestMartinRatioEdgeCases:
    def test_zero_ulcer_index_returns_nan(self):
        """When equity never draws down (UI == 0), martin_ratio returns NaN."""
        equity = pd.Series([100.0, 101.0, 102.0, 103.0])
        result = em.martin_ratio(equity)
        assert math.isnan(result)

    def test_single_point_equity_returns_nan(self):
        """A single-bar equity has UI == 0 → martin_ratio returns NaN."""
        equity = pd.Series([100.0])
        result = em.martin_ratio(equity)
        assert math.isnan(result)

    def test_flat_terminal_value_with_drawdown_gives_zero_or_negative(self):
        """Equity returning to its start after a drawdown: CAGR == 0 → martin_ratio == 0."""
        equity = pd.Series([100.0, 80.0, 100.0])
        result = em.martin_ratio(equity)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_large_series_finite_result(self):
        """martin_ratio on a 1000-bar realistic equity series returns a finite float."""
        rng = np.random.default_rng(13)
        equity = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.012, 1000)))
        result = em.martin_ratio(equity)
        assert math.isfinite(result) or math.isnan(result)


# ---------------------------------------------------------------------------
# martin_ratio — explicit failure mode
# ---------------------------------------------------------------------------


class TestMartinRatioFailureModes:
    def test_nan_not_inf_when_ui_zero(self):
        """UI == 0 must produce NaN, not inf or ZeroDivisionError."""
        equity = pd.Series([100.0, 105.0, 110.0])
        result = em.martin_ratio(equity)
        assert math.isnan(result), "UI=0 must produce NaN, not inf or a numeric value"
        assert not math.isinf(result)


# ---------------------------------------------------------------------------
# compute_all() integration
# ---------------------------------------------------------------------------


class TestComputeAllIntegration:
    def _make_equity_and_positions(self, n: int = 200, seed: int = 0):
        rng = np.random.default_rng(seed)
        equity = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n)))
        positions = pd.Series(rng.choice([0.0, 1.0], size=n))
        return equity, positions

    def test_compute_all_contains_ulcer_index(self):
        """compute_all() must return a dict with 'ulcer_index' key."""
        equity, positions = self._make_equity_and_positions()
        result = em.compute_all(equity, positions)
        assert "ulcer_index" in result, "compute_all() must include 'ulcer_index'"

    def test_compute_all_contains_martin_ratio(self):
        """compute_all() must return a dict with 'martin_ratio' key."""
        equity, positions = self._make_equity_and_positions()
        result = em.compute_all(equity, positions)
        assert "martin_ratio" in result, "compute_all() must include 'martin_ratio'"

    def test_compute_all_ulcer_index_matches_direct_call(self):
        """compute_all()['ulcer_index'] must equal ulcer_index(equity) directly."""
        equity, positions = self._make_equity_and_positions(seed=42)
        result = em.compute_all(equity, positions)
        direct = em.ulcer_index(equity)
        assert result["ulcer_index"] == pytest.approx(direct, rel=1e-9)

    def test_compute_all_martin_ratio_matches_direct_call(self):
        """compute_all()['martin_ratio'] must equal martin_ratio(equity) directly."""
        equity, positions = self._make_equity_and_positions(seed=42)
        result = em.compute_all(equity, positions)
        direct = em.martin_ratio(equity)
        if math.isnan(direct):
            assert math.isnan(result["martin_ratio"])
        else:
            assert result["martin_ratio"] == pytest.approx(direct, rel=1e-9)

    def test_compute_all_returns_non_negative_ulcer_index(self):
        """compute_all()['ulcer_index'] must be non-negative."""
        equity, positions = self._make_equity_and_positions()
        result = em.compute_all(equity, positions)
        assert result["ulcer_index"] >= 0.0
