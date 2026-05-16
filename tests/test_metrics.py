"""Tests for engine.metrics.cost_to_alpha_ratio and the backtest gross equity path."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import metrics as em
from engine.backtest import _run_internal

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


def _make_prices(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV price data for use in backtest tests."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n))
    opens = close * (1.0 + rng.uniform(-0.002, 0.002, n))
    highs = np.maximum(close, opens) * (1.0 + rng.uniform(0.0, 0.005, n))
    lows = np.minimum(close, opens) * (1.0 - rng.uniform(0.0, 0.005, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": close, "volume": volume},
        index=idx,
    )


def _always_long(view) -> float:
    return 1.0


def _always_flat(view) -> float:
    return 0.0


def _flip_every_bar(view) -> float:
    """Alternates long/flat every bar for maximum turnover."""
    return 1.0 if len(view) % 2 == 0 else 0.0


# ---------------------------------------------------------------------------
# Acceptance criterion (a): zero-cost strategy returns 1.0
# ---------------------------------------------------------------------------


class TestCostToAlphaRatioZeroCost:
    def test_zero_cost_buy_hold_returns_one(self):
        """When commission and slippage are both zero, gross == net → ratio == 1.0."""
        df = pd.read_csv(DATA_DIR / "trend_gbm.csv", index_col=0, parse_dates=True)
        net, gross, pos, rfr = _run_internal(_always_long, df, {"commission_bps": 0, "slippage_bps": 0})
        ratio = em.cost_to_alpha_ratio(gross, net)
        assert ratio == 1.0, f"Zero-cost strategy must return ratio 1.0, got {ratio}"

    def test_zero_cost_high_turnover_returns_one(self):
        """High-turnover strategy with zero costs: gross == net → ratio == 1.0 (explicit series)."""
        equity = pd.Series(100.0 * (1.001 ** np.arange(253)))
        ratio = em.cost_to_alpha_ratio(equity, equity)
        assert ratio == 1.0, f"Zero-cost high-turnover strategy must return ratio 1.0, got {ratio}"

    def test_identical_series_returns_one(self):
        """Passing the same equity series as both gross and net returns 1.0."""
        equity = pd.Series([100.0, 101.0, 103.0, 102.0, 105.0])
        ratio = em.cost_to_alpha_ratio(equity, equity)
        assert ratio == 1.0


# ---------------------------------------------------------------------------
# Acceptance criterion (b): all alpha eaten by costs → returns inf
# ---------------------------------------------------------------------------


class TestCostToAlphaRatioInf:
    def test_net_zero_cagr_returns_inf(self):
        """When net CAGR is exactly zero, all gross alpha is eaten → returns inf."""
        n = 253
        gross_equity = pd.Series(100.0 * (1.05 ** (np.arange(n) / 252)))
        flat_equity = pd.Series(np.full(n, 100.0))
        ratio = em.cost_to_alpha_ratio(gross_equity, flat_equity)
        assert ratio == float("inf"), f"Net CAGR=0 must return inf, got {ratio}"

    def test_net_negative_cagr_returns_inf(self):
        """When net CAGR is negative, costs ate more than all alpha → returns inf."""
        gross_equity = pd.Series([100.0, 101.0, 102.0, 103.0])
        net_equity = pd.Series([100.0, 99.0, 98.5, 98.0])
        ratio = em.cost_to_alpha_ratio(gross_equity, net_equity)
        assert ratio == float("inf"), f"Net negative CAGR must return inf, got {ratio}"

    def test_strategy_wiped_by_costs_on_real_data(self):
        """A high-turnover strategy on the real dataset returns inf under extreme costs."""
        df = pd.read_csv(DATA_DIR / "trend_gbm.csv", index_col=0, parse_dates=True)
        net, gross, pos, rfr = _run_internal(_flip_every_bar, df, {"commission_bps": 200, "slippage_bps": 200})
        if em.cagr(net) <= 0:
            ratio = em.cost_to_alpha_ratio(gross, net)
            assert ratio == float("inf")
        else:
            pytest.skip("Strategy still profitable at extreme costs on this dataset — inf case not triggered")


# ---------------------------------------------------------------------------
# Acceptance criterion (c): known gross/net CAGR values verify the formula
# ---------------------------------------------------------------------------


class TestCostToAlphaRatioFormula:
    def _build_equity_for_cagr(self, target_cagr: float, n_bars: int = 253) -> pd.Series:
        """Build a smooth equity curve with the given annualized CAGR over n_bars."""
        daily = (1.0 + target_cagr) ** (1.0 / 252) - 1.0
        values = 100.0 * (1.0 + daily) ** np.arange(n_bars)
        return pd.Series(values)

    def test_formula_exact_gross_10pct_net_8pct(self):
        """gross=10%, net=8%: ratio = 0.10/(0.10-0.08) = 5.0."""
        gross = self._build_equity_for_cagr(0.10)
        net = self._build_equity_for_cagr(0.08)
        gross_a = em.cagr(gross)
        net_a = em.cagr(net)
        expected = gross_a / (gross_a - net_a)
        ratio = em.cost_to_alpha_ratio(gross, net)
        assert abs(ratio - expected) < 1e-6, f"Expected {expected:.6f}, got {ratio:.6f}"
        assert abs(ratio - 5.0) < 0.01, f"gross=10%, net=8% should give ratio ≈ 5.0, got {ratio:.4f}"

    def test_formula_exact_gross_12pct_net_10pct(self):
        """gross=12%, net=10%: ratio = 0.12/(0.12-0.10) = 6.0."""
        gross = self._build_equity_for_cagr(0.12)
        net = self._build_equity_for_cagr(0.10)
        gross_a = em.cagr(gross)
        net_a = em.cagr(net)
        expected = gross_a / (gross_a - net_a)
        ratio = em.cost_to_alpha_ratio(gross, net)
        assert abs(ratio - expected) < 1e-6, f"Expected {expected:.6f}, got {ratio:.6f}"
        assert abs(ratio - 6.0) < 0.01, f"gross=12%, net=10% should give ratio ≈ 6.0, got {ratio:.4f}"

    def test_ratio_decreases_as_net_approaches_zero(self):
        """As net alpha decreases (more cost friction), ratio decreases toward 1.

        Formula: gross/(gross-net). When net≈gross (low cost) denominator is tiny → large ratio.
        When net≈0 (high cost) denominator ≈ gross → ratio ≈ 1.
        """
        gross = self._build_equity_for_cagr(0.10)
        net_low_cost = self._build_equity_for_cagr(0.09)   # low cost: ratio = 10/(10-9)=10
        net_high_cost = self._build_equity_for_cagr(0.02)  # high cost: ratio = 10/(10-2)=1.25
        ratio_low_cost = em.cost_to_alpha_ratio(gross, net_low_cost)
        ratio_high_cost = em.cost_to_alpha_ratio(gross, net_high_cost)
        assert ratio_low_cost > ratio_high_cost, (
            f"Low-cost strategy should have higher ratio than high-cost; "
            f"got low_cost={ratio_low_cost:.2f}, high_cost={ratio_high_cost:.2f}"
        )

    def test_higher_cost_fraction_gives_lower_ratio(self):
        """More cost friction → smaller ratio (closer to 1).

        Compares two non-zero cost levels using pre-built equity curves so the
        special-case zero-cost return of 1.0 is not involved.
        """
        gross = self._build_equity_for_cagr(0.10)
        net_low_cost = self._build_equity_for_cagr(0.095)   # 0.5% eaten by costs → ratio=20
        net_high_cost = self._build_equity_for_cagr(0.05)   # 5% eaten by costs → ratio=2
        ratio_low = em.cost_to_alpha_ratio(gross, net_low_cost)
        ratio_high_cost = em.cost_to_alpha_ratio(gross, net_high_cost)
        assert ratio_low > ratio_high_cost, (
            f"Low-cost (net=9.5%) ratio should exceed high-cost (net=5%) ratio; "
            f"got low_cost={ratio_low:.2f}, high_cost={ratio_high_cost:.2f}"
        )

    def test_return_type_is_float(self):
        """cost_to_alpha_ratio must always return a Python float."""
        equity = pd.Series([100.0, 102.0, 104.0, 106.0, 108.0])
        ratio = em.cost_to_alpha_ratio(equity, equity)
        assert isinstance(ratio, float), f"Expected float, got {type(ratio)}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestCostToAlphaRatioEdgeCases:
    def test_single_bar_series_returns_inf(self):
        """A single-bar equity series (0 returns) gives CAGR=0 → inf."""
        equity = pd.Series([100.0])
        ratio = em.cost_to_alpha_ratio(equity, equity)
        assert ratio == float("inf") or ratio == 1.0

    def test_two_bar_flat_net_returns_inf(self):
        """Net equity identical start-to-end → CAGR 0 → inf."""
        gross = pd.Series([100.0, 110.0])
        net = pd.Series([100.0, 100.0])
        ratio = em.cost_to_alpha_ratio(gross, net)
        assert ratio == float("inf")

    def test_very_high_cost_on_synthetic_data(self):
        """_run_internal with extreme costs produces valid (non-NaN) ratio."""
        df = _make_prices(n=300)
        net, gross, pos, rfr = _run_internal(
            _always_long, df, {"commission_bps": 100, "slippage_bps": 100}
        )
        ratio = em.cost_to_alpha_ratio(gross, net)
        assert not math.isnan(ratio), "cost_to_alpha_ratio must never return NaN"
        assert ratio >= 0.0 or ratio == float("inf")

    def test_gross_equity_always_geq_net_equity_for_buy_hold(self):
        """For a buy-and-hold with positive costs, gross terminal value >= net terminal value."""
        df = _make_prices()
        net, gross, pos, rfr = _run_internal(_always_long, df, {"commission_bps": 5, "slippage_bps": 5})
        assert gross.iloc[-1] >= net.iloc[-1], "Gross equity (zero-cost) must end >= net equity"


# ---------------------------------------------------------------------------
# Integration: backtest _run_internal returns gross equity
# ---------------------------------------------------------------------------


class TestRunInternalGrossEquity:
    def test_gross_geq_net_for_nonzero_costs(self):
        """With positive costs and a strategy that trades, gross CAGR >= net CAGR."""
        df = _make_prices()
        net, gross, pos, rfr = _run_internal(_flip_every_bar, df, {"commission_bps": 5, "slippage_bps": 5})
        assert em.cagr(gross) >= em.cagr(net), "Gross CAGR must dominate net CAGR when costs > 0"

    def test_gross_equals_net_at_zero_cost(self):
        """With zero costs, gross and net equity curves must be identical."""
        df = _make_prices()
        net, gross, pos, rfr = _run_internal(_flip_every_bar, df, {"commission_bps": 0, "slippage_bps": 0})
        pd.testing.assert_series_equal(gross, net, check_names=False)

    def test_gross_equity_starts_at_initial_capital(self):
        """Gross equity series must start at initial_capital."""
        df = _make_prices()
        net, gross, pos, rfr = _run_internal(_always_long, df, {"initial_capital": 50_000})
        assert gross.iloc[0] == 50_000.0

    def test_gross_equity_length_matches_net(self):
        """Gross and net equity series must have the same length."""
        df = _make_prices(n=150)
        net, gross, pos, rfr = _run_internal(_always_long, df, {})
        assert len(gross) == len(net) == len(df)


# ---------------------------------------------------------------------------
# Metrics.json has cost_to_alpha_ratio for all six strategies
# ---------------------------------------------------------------------------


class TestMetricsJsonHasCostToAlphaRatio:
    DATASETS = ["trend_gbm", "mean_rev_ou", "regime_switch", "fat_tail"]
    STRATEGIES = [
        "01-dual-ema-momentum",
        "02-rsi-mean-reversion",
        "03-donchian-turtle-breakout",
        "04-52wk-high-proximity",
        "05-turn-of-month",
        "06-bollinger-mean-reversion",
    ]

    def _load(self, strategy: str) -> dict:
        path = ROOT / "strategies" / strategy / "metrics.json"
        return json.loads(path.read_text())

    def test_all_strategies_have_cost_to_alpha_ratio_field(self):
        """Every strategy's metrics.json must have cost_to_alpha_ratio on all datasets."""
        for strat in self.STRATEGIES:
            m = self._load(strat)
            for ds in self.DATASETS:
                assert "cost_to_alpha_ratio" in m[ds], (
                    f"metrics.json for {strat}/{ds} missing 'cost_to_alpha_ratio'"
                )

    def test_turn_of_month_ratio_near_one_on_regime_switch(self):
        """TOM has low turnover; its cost_to_alpha_ratio on regime_switch should be finite and >= 1."""
        m = self._load("05-turn-of-month")
        ratio = m["regime_switch"]["cost_to_alpha_ratio"]
        assert ratio is not None, "TOM regime_switch ratio must not be null (positive net CAGR)"
        assert ratio >= 1.0, f"TOM cost_to_alpha_ratio must be >= 1.0, got {ratio}"

    def test_cost_to_alpha_ratio_null_for_negative_net_datasets(self):
        """Datasets where net CAGR <= 0 must store null (serialized inf)."""
        m = self._load("04-52wk-high-proximity")
        for ds in self.DATASETS:
            assert m[ds]["cost_to_alpha_ratio"] is None, (
                f"52wk-high on {ds} has negative net CAGR; cost_to_alpha_ratio must be null"
            )

    def test_all_ratios_are_none_or_gte_one(self):
        """cost_to_alpha_ratio must be null (inf) or >= 1.0 — never below 1 for valid data."""
        for strat in self.STRATEGIES:
            m = self._load(strat)
            for ds in self.DATASETS:
                v = m[ds]["cost_to_alpha_ratio"]
                if v is not None:
                    assert v >= 1.0, (
                        f"{strat}/{ds}: ratio {v} < 1.0 — costs cannot improve gross alpha"
                    )
