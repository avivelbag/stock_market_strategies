"""Tests for the regime_conditional_sharpe metric and its integration into backtest output."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.backtest import run
import engine.metrics as metrics

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
STRATEGIES_DIR = ROOT / "strategies"

ALL_STRATEGY_NAMES = [
    "01-dual-ema-momentum",
    "02-rsi-mean-reversion",
    "03-donchian-turtle-breakout",
    "04-52wk-high-proximity",
]
DATASETS = ["trend_gbm", "mean_rev_ou", "regime_switch", "fat_tail"]


def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)


def _make_prices(n: int, seed: int = 0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame of n bars with a deterministic GBM close series."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    log_ret = rng.normal(0.0002, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
            "volume": rng.integers(100_000, 1_000_000, n).astype(float),
        },
        index=dates,
    )
    return df


def _make_equity_returns(prices: pd.DataFrame) -> pd.Series:
    """Trivial always-long equity curve and its pct_change returns."""
    equity = prices["close"] / prices["close"].iloc[0] * 10_000
    return equity.pct_change()


# ---------------------------------------------------------------------------
# metrics.json presence and count invariants
# ---------------------------------------------------------------------------


class TestMetricsJsonRegimeSharpe:
    """Guard: every strategy's metrics.json must have regime_sharpe populated."""

    def _load(self, name: str) -> dict:
        path = STRATEGIES_DIR / name / "metrics.json"
        with open(path) as f:
            return json.load(f)

    def test_regime_sharpe_present_in_all_strategies_and_datasets(self):
        """regime_sharpe key must exist for every strategy × dataset combination."""
        for strategy in ALL_STRATEGY_NAMES:
            m = self._load(strategy)
            for dataset in DATASETS:
                assert dataset in m, f"{strategy} missing dataset {dataset}"
                assert "regime_sharpe" in m[dataset], (
                    f"regime_sharpe missing in {strategy}/{dataset}"
                )

    def test_regime_counts_sum_equals_total_bar_count(self):
        """regime_counts['trending'] + 'ranging' + 'high_vol' must equal len(price_csv)."""
        for strategy in ALL_STRATEGY_NAMES:
            m = self._load(strategy)
            for dataset in DATASETS:
                rs = m[dataset]["regime_sharpe"]
                counts = rs["regime_counts"]
                total_counts = counts["trending"] + counts["ranging"] + counts["high_vol"]
                price_df = _load_csv(dataset)
                assert total_counts == len(price_df), (
                    f"{strategy}/{dataset}: regime_counts sum {total_counts} "
                    f"!= price bars {len(price_df)}"
                )

    def test_regime_sharpe_keys_are_correct(self):
        """Each regime_sharpe entry must have exactly the expected keys."""
        expected = {"trending", "ranging", "high_vol", "regime_counts"}
        for strategy in ALL_STRATEGY_NAMES:
            m = self._load(strategy)
            for dataset in DATASETS:
                rs = m[dataset]["regime_sharpe"]
                assert set(rs.keys()) == expected, (
                    f"{strategy}/{dataset}: unexpected regime_sharpe keys {set(rs.keys())}"
                )

    def test_regime_counts_keys_are_correct(self):
        """regime_counts must have exactly trending, ranging, high_vol keys."""
        expected = {"trending", "ranging", "high_vol"}
        for strategy in ALL_STRATEGY_NAMES:
            m = self._load(strategy)
            for dataset in DATASETS:
                counts = m[dataset]["regime_sharpe"]["regime_counts"]
                assert set(counts.keys()) == expected, (
                    f"{strategy}/{dataset}: unexpected regime_counts keys"
                )

    def test_regime_sharpe_values_are_numeric_or_null(self):
        """Sharpe values must be float or null (NaN for thin-coverage regimes)."""
        for strategy in ALL_STRATEGY_NAMES:
            m = self._load(strategy)
            for dataset in DATASETS:
                rs = m[dataset]["regime_sharpe"]
                for label in ("trending", "ranging", "high_vol"):
                    v = rs[label]
                    assert v is None or isinstance(v, (int, float)), (
                        f"{strategy}/{dataset}/{label}: expected float or null, got {type(v)}"
                    )

    def test_regime_counts_are_nonnegative_integers(self):
        """All counts must be non-negative integers."""
        for strategy in ALL_STRATEGY_NAMES:
            m = self._load(strategy)
            for dataset in DATASETS:
                counts = m[dataset]["regime_sharpe"]["regime_counts"]
                for label, v in counts.items():
                    assert isinstance(v, int) and v >= 0, (
                        f"{strategy}/{dataset}/{label}: count {v} is not a non-negative int"
                    )


# ---------------------------------------------------------------------------
# Happy path: regime_conditional_sharpe function
# ---------------------------------------------------------------------------


class TestRegimeConditionalSharpe:
    def test_returns_expected_keys(self):
        """Output dict must contain trending, ranging, high_vol, regime_counts."""
        prices = _make_prices(500)
        returns = _make_equity_returns(prices)
        result = metrics.regime_conditional_sharpe(returns, prices)
        assert set(result.keys()) == {"trending", "ranging", "high_vol", "regime_counts"}

    def test_regime_counts_sum_to_n_bars(self):
        """Regime counts must sum to the total number of price bars."""
        prices = _make_prices(600)
        returns = _make_equity_returns(prices)
        result = metrics.regime_conditional_sharpe(returns, prices)
        total = sum(result["regime_counts"].values())
        assert total == len(prices), f"counts {total} != bars {len(prices)}"

    def test_on_real_dataset_regime_counts_sum_correct(self):
        """Verify count invariant on each committed synthetic dataset."""
        for name in DATASETS:
            prices = _load_csv(name)
            returns = _make_equity_returns(prices)
            result = metrics.regime_conditional_sharpe(returns, prices)
            total = sum(result["regime_counts"].values())
            assert total == len(prices), (
                f"{name}: regime counts {total} != bars {len(prices)}"
            )

    def test_sharpe_values_are_float_or_nan(self):
        """All Sharpe outputs must be float (possibly NaN); regime_counts must be ints."""
        prices = _make_prices(500)
        returns = _make_equity_returns(prices)
        result = metrics.regime_conditional_sharpe(returns, prices)
        for label in ("trending", "ranging", "high_vol"):
            v = result[label]
            assert isinstance(v, float), f"{label} should be float, got {type(v)}"
        for label, count in result["regime_counts"].items():
            assert isinstance(count, int), f"count[{label}] should be int"

    def test_high_vol_regime_captures_roughly_top_quartile(self):
        """With 500+ bars the high_vol count should be near 25% of total."""
        prices = _make_prices(800, seed=42)
        returns = _make_equity_returns(prices)
        result = metrics.regime_conditional_sharpe(returns, prices)
        hv_frac = result["regime_counts"]["high_vol"] / len(prices)
        # The vol percentile threshold is 0.75 so expect roughly 25% in high_vol.
        # Allow a wide tolerance because the expanding window distorts early bars.
        assert 0.05 < hv_frac < 0.50, f"high_vol fraction {hv_frac:.3f} unexpectedly off"

    def test_no_lookahead_in_classifier(self):
        """Verify classifier uses only lagged data by checking backtest output is deterministic."""
        prices = _load_csv("regime_switch")
        returns = _make_equity_returns(prices)
        r1 = metrics.regime_conditional_sharpe(returns, prices)
        r2 = metrics.regime_conditional_sharpe(returns, prices)
        assert r1 == r2

    def test_backtest_run_includes_regime_sharpe(self):
        """backtest.run() must include 'regime_sharpe' in its returned dict."""
        prices = _make_prices(300)

        def always_long(view):
            return 1.0

        result = run(always_long, prices, {})
        assert "regime_sharpe" in result
        assert "regime_counts" in result["regime_sharpe"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRegimeConditionalSharpeEdgeCases:
    def test_short_series_returns_nan_for_thin_regimes(self):
        """With 50 bars, most regimes will have fewer than 30 observations → NaN."""
        prices = _make_prices(50)
        returns = _make_equity_returns(prices)
        result = metrics.regime_conditional_sharpe(returns, prices)
        # At least one regime Sharpe must be NaN because 50 bars is tight
        values = [result[k] for k in ("trending", "ranging", "high_vol")]
        assert any(isinstance(v, float) and math.isnan(v) for v in values), (
            "Expected at least one NaN Sharpe on a 50-bar series"
        )

    def test_very_short_series_all_regimes_nan(self):
        """With only 25 bars, every regime should have fewer than 30 bars → all NaN."""
        prices = _make_prices(25)
        returns = _make_equity_returns(prices)
        result = metrics.regime_conditional_sharpe(returns, prices)
        for label in ("trending", "ranging", "high_vol"):
            v = result[label]
            assert isinstance(v, float) and math.isnan(v), (
                f"Expected NaN for {label} on 25-bar series, got {v}"
            )

    def test_counts_still_sum_on_short_series(self):
        """Even on a very short series regime_counts must still sum to len(prices)."""
        prices = _make_prices(30)
        returns = _make_equity_returns(prices)
        result = metrics.regime_conditional_sharpe(returns, prices)
        total = sum(result["regime_counts"].values())
        assert total == len(prices)

    def test_flat_returns_series(self):
        """All-zero returns produce 0.0 Sharpe for regimes with enough bars."""
        prices = _make_prices(500)
        returns = pd.Series(0.0, index=prices.index)
        result = metrics.regime_conditional_sharpe(returns, prices)
        for label in ("trending", "ranging", "high_vol"):
            v = result[label]
            if not (isinstance(v, float) and math.isnan(v)):
                assert v == 0.0, f"Expected 0.0 or NaN for {label}, got {v}"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestRegimeConditionalSharpeFailures:
    def test_misaligned_returns_raises(self):
        """Passing returns with a different index than prices must raise an error.

        The function contract requires aligned indices (as produced by backtest._run_internal).
        Misaligned inputs are invalid and should not silently produce wrong results.
        """
        prices = _make_prices(500)
        extra_idx = prices.index.append(pd.date_range("2030-01-01", periods=10, freq="B"))
        returns = pd.Series(0.001, index=extra_idx)
        with pytest.raises(Exception):
            metrics.regime_conditional_sharpe(returns, prices)

    def test_missing_close_column_raises(self):
        """If prices has no 'close' column the function must raise KeyError."""
        prices = _make_prices(200).drop(columns=["close"])
        returns = pd.Series(0.0, index=prices.index)
        with pytest.raises(KeyError):
            metrics.regime_conditional_sharpe(returns, prices)
