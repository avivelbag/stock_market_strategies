"""Tests for engine/backtest.py and engine/metrics.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.backtest import LookAheadError, run
import engine.metrics as metrics

DATA_DIR = Path(__file__).parent.parent / "data"


def _load(name: str) -> pd.DataFrame:
    """Load a committed synthetic CSV from data/."""
    return pd.read_csv(DATA_DIR / name, index_col=0, parse_dates=True)


def _always_long(view) -> float:
    """Trivial strategy: always hold a long position."""
    return 1.0


def _always_flat(view) -> float:
    """Trivial strategy: always stay flat."""
    return 0.0


def _lookahead_strategy(view) -> float:
    """Strategy that attempts to read one bar into the future via iloc."""
    return float(view.iloc[len(view)]["close"])


def _high_turnover(view) -> float:
    """Strategy that flips sign every bar (maximum turnover)."""
    t = len(view) - 1
    return 1.0 if t % 2 == 0 else -1.0


# ---------------------------------------------------------------------------
# Look-ahead guard
# ---------------------------------------------------------------------------


class TestLookAheadGuard:
    def test_lookahead_raises(self):
        df = _load("trend_gbm.csv")
        with pytest.raises(LookAheadError):
            run(_lookahead_strategy, df, {})

    def test_valid_strategy_does_not_raise(self):
        df = _load("trend_gbm.csv").iloc[:50]
        result = run(_always_long, df, {})
        assert isinstance(result, dict)

    def test_lookahead_error_message_is_clear(self):
        df = _load("trend_gbm.csv")
        with pytest.raises(LookAheadError, match="bar"):
            run(_lookahead_strategy, df, {})


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_identical_runs_give_identical_metrics(self):
        df = _load("trend_gbm.csv")
        cfg = {"commission_bps": 5, "slippage_bps": 5}
        m1 = run(_always_long, df, cfg)
        m2 = run(_always_long, df, cfg)
        assert m1 == m2

    def test_different_configs_give_different_metrics(self):
        df = _load("trend_gbm.csv")
        m_low_cost = run(_always_long, df, {"commission_bps": 0, "slippage_bps": 0})
        m_high_cost = run(_always_long, df, {"commission_bps": 20, "slippage_bps": 20})
        assert m_low_cost != m_high_cost


# ---------------------------------------------------------------------------
# Commission and slippage
# ---------------------------------------------------------------------------


class TestCostsApplied:
    def test_costs_reduce_cagr_for_high_turnover_strategy(self):
        df = _load("trend_gbm.csv")
        no_cost = run(
            _high_turnover, df, {"commission_bps": 0, "slippage_bps": 0, "allow_short": True}
        )
        with_cost = run(
            _high_turnover, df, {"commission_bps": 20, "slippage_bps": 20, "allow_short": True}
        )
        assert with_cost["cagr"] < no_cost["cagr"]

    def test_zero_cost_buy_hold_higher_cagr_than_with_cost(self):
        df = _load("trend_gbm.csv")
        # buy-and-hold has one entry trade; any non-zero cost should reduce final equity
        no_cost = run(_always_long, df, {"commission_bps": 0, "slippage_bps": 0})
        with_cost = run(_always_long, df, {"commission_bps": 10, "slippage_bps": 10})
        assert with_cost["cagr"] < no_cost["cagr"]

    def test_flat_strategy_has_no_turnover(self):
        df = _load("trend_gbm.csv")
        result = run(_always_flat, df, {})
        assert result["turnover"] == 0.0

    def test_flat_strategy_has_zero_exposure(self):
        df = _load("trend_gbm.csv")
        result = run(_always_flat, df, {})
        assert result["exposure"] == 0.0


# ---------------------------------------------------------------------------
# CAGR positive on trending series
# ---------------------------------------------------------------------------


class TestCAGROnTrendingSeries:
    def test_always_long_positive_cagr_on_gbm(self):
        df = _load("trend_gbm.csv")
        result = run(_always_long, df, {"commission_bps": 0, "slippage_bps": 0})
        assert result["cagr"] > 0.0

    def test_short_is_blocked_without_allow_short(self):
        df = _load("trend_gbm.csv")

        def always_short(view):
            return -1.0

        result_no_short = run(always_short, df, {"allow_short": False})
        # Position is forced to 0 when short disallowed; exposure should be 0
        assert result_no_short["exposure"] == 0.0

    def test_short_works_with_allow_short(self):
        df = _load("trend_gbm.csv")

        def always_short(view):
            return -1.0

        result_short = run(always_short, df, {"allow_short": True})
        assert result_short["exposure"] == 1.0


# ---------------------------------------------------------------------------
# Metrics math on known series
# ---------------------------------------------------------------------------


class TestMetricsMath:
    def test_cagr_exact_for_constant_daily_return(self):
        # 1% daily for exactly 252 days (253 equity points)
        n = 253
        equity = pd.Series(100.0 * (1.01 ** np.arange(n)))
        computed = metrics.cagr(equity)
        expected = 1.01**252 - 1
        assert abs(computed - expected) < 1e-6

    def test_max_drawdown_exact(self):
        # Peak 120 at bar 1, trough 80 at bar 4 → drawdown = (80-120)/120 = -1/3
        equity = pd.Series([100.0, 120.0, 90.0, 100.0, 80.0])
        computed = metrics.max_drawdown(equity)
        assert abs(computed - (-1.0 / 3.0)) < 1e-9

    def test_sharpe_positive_for_positive_mean_return(self):
        rng = np.random.default_rng(0)
        # 0.5% daily return with small noise → strong positive Sharpe
        returns = pd.Series(0.005 + rng.normal(0, 0.001, 252))
        equity = pd.Series(100.0 * (1 + returns).cumprod())
        assert metrics.sharpe(equity) > 1.0

    def test_sortino_zero_when_no_negative_returns(self):
        # Monotonically increasing equity → no negative returns → sortino = 0
        equity = pd.Series(np.linspace(100, 200, 253))
        assert metrics.sortino(equity) == 0.0

    def test_max_drawdown_zero_for_monotonic_series(self):
        equity = pd.Series(np.linspace(100, 200, 100))
        assert metrics.max_drawdown(equity) == 0.0

    def test_time_in_drawdown_zero_for_always_rising(self):
        equity = pd.Series(np.linspace(100, 200, 100))
        assert metrics.time_in_drawdown(equity) == 0.0

    def test_hit_rate_between_zero_and_one(self):
        df = _load("trend_gbm.csv")
        result = run(_always_long, df, {})
        assert 0.0 <= result["hit_rate"] <= 1.0

    def test_tail_ratio_positive_for_normal_returns(self):
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0.001, 0.01, 500))
        equity = pd.Series(100.0 * (1 + returns).cumprod())
        ratio = metrics.tail_ratio(equity)
        assert ratio > 0.0

    def test_calmar_zero_for_no_drawdown(self):
        equity = pd.Series(np.linspace(100, 200, 100))
        assert metrics.calmar(equity) == 0.0

    def test_exposure_one_for_always_long(self):
        df = _load("trend_gbm.csv")
        result = run(_always_long, df, {})
        # First bar has no prior signal; positions[0] = target set at bar 0
        assert result["exposure"] == 1.0

    def test_all_metrics_returned(self):
        df = _load("trend_gbm.csv").iloc[:100]
        result = run(_always_long, df, {})
        expected_keys = {
            "cagr", "volatility", "sharpe", "sortino", "calmar",
            "max_drawdown", "time_in_drawdown", "turnover", "hit_rate",
            "tail_ratio", "exposure",
        }
        assert expected_keys == set(result.keys())


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_columns_raises_value_error(self):
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="missing columns"):
            run(_always_long, df, {})

    def test_single_row_raises_value_error(self):
        df = _load("trend_gbm.csv").iloc[:1]
        with pytest.raises(ValueError, match="at least 2 rows"):
            run(_always_long, df, {})

    def test_minimal_two_row_df_succeeds(self):
        df = _load("trend_gbm.csv").iloc[:2]
        result = run(_always_long, df, {})
        assert isinstance(result, dict)
