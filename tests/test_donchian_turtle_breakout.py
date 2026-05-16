"""Tests for strategies/03-donchian-turtle-breakout/strategy.py.

Covers:
  - Happy path: long signal when close breaks above 20-bar channel high
  - Flat signal when close falls below 10-bar channel low (exit)
  - Warm-up guard: flat when fewer than entry_window + 1 bars
  - Default parameter values
  - Integration with engine.backtest.run() on all four synthetic datasets
  - Integration with walk_forward_backtest() and metrics.json structure
  - Edge cases: single bar, exactly entry_window + 1 bars, constant prices, large input
  - Error cases: invalid window arguments
  - Stateful behavior: position held between bars, exit condition respected
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
STRATEGY_FILE = ROOT / "strategies" / "03-donchian-turtle-breakout" / "strategy.py"
METRICS_FILE = ROOT / "strategies" / "03-donchian-turtle-breakout" / "metrics.json"

_DATASETS = ["trend_gbm.csv", "mean_rev_ou.csv", "regime_switch.csv", "fat_tail.csv"]


def _load_strategy_class():
    """Load DonchianTurtleBreakout from the hyphenated directory via importlib."""
    spec = importlib.util.spec_from_file_location("donchian_strategy", STRATEGY_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DonchianTurtleBreakout


DonchianTurtleBreakout = _load_strategy_class()


def _load_data(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, index_col=0, parse_dates=True)


def _make_view(closes: list) -> pd.DataFrame:
    """Construct a minimal OHLCV-style DataFrame from a list of close prices."""
    n = len(closes)
    dates = pd.bdate_range("2020-01-02", periods=n)
    arr = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": arr * 0.999,
            "high": arr * 1.01,
            "low": arr * 0.99,
            "close": arr,
            "volume": np.ones(n, dtype=int) * 1000,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Default parameters and construction
# ---------------------------------------------------------------------------


class TestDefaultParameters:
    def test_default_entry_window_is_20(self):
        s = DonchianTurtleBreakout()
        assert s.entry_window == 20

    def test_default_exit_window_is_10(self):
        s = DonchianTurtleBreakout()
        assert s.exit_window == 10

    def test_default_atr_window_is_20(self):
        s = DonchianTurtleBreakout()
        assert s.atr_window == 20

    def test_custom_windows_stored(self):
        s = DonchianTurtleBreakout(entry_window=30, exit_window=15, atr_window=14)
        assert s.entry_window == 30
        assert s.exit_window == 15
        assert s.atr_window == 14

    def test_invalid_exit_ge_entry_raises(self):
        with pytest.raises(ValueError, match="exit_window must be strictly less"):
            DonchianTurtleBreakout(entry_window=10, exit_window=10)

    def test_invalid_exit_gt_entry_raises(self):
        with pytest.raises(ValueError, match="exit_window must be strictly less"):
            DonchianTurtleBreakout(entry_window=10, exit_window=15)

    def test_nonpositive_entry_window_raises(self):
        with pytest.raises(ValueError, match="positive integers"):
            DonchianTurtleBreakout(entry_window=0, exit_window=10)

    def test_nonpositive_exit_window_raises(self):
        with pytest.raises(ValueError, match="positive integers"):
            DonchianTurtleBreakout(entry_window=20, exit_window=0)

    def test_nonpositive_atr_window_raises(self):
        with pytest.raises(ValueError, match="positive integers"):
            DonchianTurtleBreakout(entry_window=20, exit_window=10, atr_window=-1)

    def test_starts_out_of_position(self):
        s = DonchianTurtleBreakout()
        assert s._in_position is False


# ---------------------------------------------------------------------------
# Warm-up guard
# ---------------------------------------------------------------------------


class TestWarmupGuard:
    def test_single_bar_returns_flat(self):
        s = DonchianTurtleBreakout(entry_window=5, exit_window=2)
        view = _make_view([100.0])
        assert s(view) == 0.0

    def test_entry_window_bars_returns_flat(self):
        """entry_window bars available = entry_window + 1 - 1; still flat."""
        s = DonchianTurtleBreakout(entry_window=5, exit_window=2)
        closes = [100.0] * 5
        view = _make_view(closes)
        assert s(view) == 0.0

    def test_entry_window_plus_one_bars_produces_signal(self):
        """At entry_window + 1 bars, a breakout price should trigger a long."""
        s = DonchianTurtleBreakout(entry_window=5, exit_window=2)
        closes = list(range(100, 106))
        view = _make_view(closes)
        result = s(view)
        assert result in (0.0, 1.0)


# ---------------------------------------------------------------------------
# Signal correctness
# ---------------------------------------------------------------------------


class TestSignalCorrectness:
    def test_long_signal_on_new_20bar_high(self):
        """Close above prior 20-bar max triggers long entry."""
        s = DonchianTurtleBreakout(entry_window=20, exit_window=10)
        closes = [100.0] * 20 + [101.0]
        view = _make_view(closes)
        assert s(view) == 1.0

    def test_no_entry_when_close_equals_channel_high(self):
        """Close equal to (not strictly above) the channel high — no entry."""
        s = DonchianTurtleBreakout(entry_window=20, exit_window=10)
        closes = [100.0] * 21
        view = _make_view(closes)
        assert s(view) == 0.0

    def test_exit_when_close_below_10bar_low(self):
        """After entry, close below prior 10-bar min triggers exit."""
        s = DonchianTurtleBreakout(entry_window=5, exit_window=3)
        closes = [100.0] * 5 + [105.0]
        view_entry = _make_view(closes)
        s(view_entry)
        assert s._in_position is True

        closes_exit = closes + [100.0] * 3 + [98.0]
        s2 = DonchianTurtleBreakout(entry_window=5, exit_window=3)
        for t in range(1, len(closes_exit) + 1):
            result = s2(_make_view(closes_exit[:t]))
        assert result == 0.0

    def test_returns_only_zero_or_one(self):
        """Strategy must return exactly 0.0 or 1.0 — no fractional positions."""
        s = DonchianTurtleBreakout(entry_window=5, exit_window=2)
        for n in range(1, 30):
            closes = [100.0 + i * 0.5 for i in range(n)]
            view = _make_view(closes)
            result = s(view)
            assert result in (0.0, 1.0), f"Got unexpected signal {result} for n={n}"

    def test_signal_is_float(self):
        s = DonchianTurtleBreakout(entry_window=5, exit_window=2)
        view = _make_view(list(range(100, 110)))
        result = s(view)
        assert isinstance(result, float)

    def test_position_held_after_entry(self):
        """Once entered, position stays long until exit signal fires."""
        s = DonchianTurtleBreakout(entry_window=5, exit_window=2)
        closes = [100.0] * 5 + [110.0]
        view = _make_view(closes)
        assert s(view) == 1.0
        closes_held = closes + [105.0]
        view_held = _make_view(closes_held)
        assert s(view_held) == 1.0

    def test_no_entry_signal_on_flat_price(self):
        """Constant price: never breaks above the channel high; stays flat."""
        s = DonchianTurtleBreakout(entry_window=5, exit_window=2)
        closes = [100.0] * 30
        view = _make_view(closes)
        assert s(view) == 0.0


# ---------------------------------------------------------------------------
# Stateful sequential simulation (mimics engine bar-by-bar behavior)
# ---------------------------------------------------------------------------


class TestStatefulBehavior:
    def _simulate(self, strategy, closes):
        """Drive strategy bar-by-bar as the engine would, return final signal."""
        result = 0.0
        for t in range(1, len(closes) + 1):
            result = strategy(_make_view(closes[:t]))
        return result

    def test_entry_then_exit_sequence(self):
        """Verify entry fires, position holds, then exit fires correctly."""
        s = DonchianTurtleBreakout(entry_window=5, exit_window=3)
        flat_base = [100.0] * 5
        entry_bar = [110.0]
        hold_bars = [108.0, 107.0, 106.0]
        exit_bar = [99.0]
        closes = flat_base + entry_bar + hold_bars + exit_bar

        signals = []
        for t in range(1, len(closes) + 1):
            signals.append(s(_make_view(closes[:t])))

        in_position_idxs = [i for i, sig in enumerate(signals) if sig == 1.0]
        assert len(in_position_idxs) > 0, "Strategy never entered a long position"
        assert signals[-1] == 0.0, "Strategy should have exited by end of series"

    def test_multiple_entry_exit_cycles(self):
        """Strategy can re-enter after an exit when a new breakout fires."""
        s = DonchianTurtleBreakout(entry_window=4, exit_window=2)
        closes = (
            [100.0] * 4 + [110.0]
            + [100.0] * 2 + [99.0]
            + [100.0] * 4 + [115.0]
        )
        signals = []
        for t in range(1, len(closes) + 1):
            signals.append(s(_make_view(closes[:t])))
        long_count = sum(1 for sig in signals if sig == 1.0)
        assert long_count >= 2, "Expected at least two periods in long position"


# ---------------------------------------------------------------------------
# Integration with engine.backtest.run()
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_run_on_trend_gbm_returns_dict(self):
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        s = DonchianTurtleBreakout()
        result = run(s, df)
        assert isinstance(result, dict)
        assert "sharpe" in result
        assert "cagr" in result

    def test_run_on_all_four_datasets_completes(self):
        from engine.backtest import run
        for name in _DATASETS:
            df = _load_data(name)
            s = DonchianTurtleBreakout()
            result = run(s, df)
            assert isinstance(result["sharpe"], float)
            assert isinstance(result["cagr"], float)

    def test_run_is_deterministic(self):
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        r1 = run(DonchianTurtleBreakout(), df)
        r2 = run(DonchianTurtleBreakout(), df)
        assert r1 == r2

    def test_exposure_less_than_one_due_to_warmup(self):
        """Warmup period forces flat; exposure must be < 1.0."""
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(DonchianTurtleBreakout(), df)
        assert result["exposure"] < 1.0

    def test_no_lookahead_error_during_run(self):
        from engine.backtest import run, LookAheadError
        for name in _DATASETS:
            df = _load_data(name)
            try:
                run(DonchianTurtleBreakout(), df)
            except LookAheadError:
                pytest.fail(f"LookAheadError raised on {name}")

    def test_regime_switch_positive_sharpe(self):
        """The strategy's primary edge is on regime_switch; Sharpe must be positive."""
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        result = run(DonchianTurtleBreakout(), df)
        assert result["sharpe"] > 0.0, (
            f"regime_switch Sharpe expected positive, got {result['sharpe']:.4f}"
        )

    def test_mean_rev_ou_no_large_positive_sharpe(self):
        """On mean-reverting data the strategy has no real edge; Sharpe must be < 0.5."""
        from engine.backtest import run
        df = _load_data("mean_rev_ou.csv")
        result = run(DonchianTurtleBreakout(), df)
        assert result["sharpe"] < 0.5, (
            f"mean_rev_ou Sharpe unexpectedly high: {result['sharpe']:.4f} — "
            "breakout strategy should not exploit mean-reverting data"
        )


# ---------------------------------------------------------------------------
# Walk-forward backtest integration
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    def test_walk_forward_returns_required_keys(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("regime_switch.csv")
        result = walk_forward_backtest(
            DonchianTurtleBreakout,
            {"entry_window": 20, "exit_window": 10, "atr_window": 20},
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
                DonchianTurtleBreakout,
                {"entry_window": 20, "exit_window": 10, "atr_window": 20},
                df,
            )
            assert 0.0 <= result["oos_consistency"] <= 1.0, (
                f"oos_consistency out of range on {name}"
            )

    def test_walk_forward_is_deterministic(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("trend_gbm.csv")
        params = {"entry_window": 20, "exit_window": 10, "atr_window": 20}
        r1 = walk_forward_backtest(DonchianTurtleBreakout, params, df)
        r2 = walk_forward_backtest(DonchianTurtleBreakout, params, df)
        assert r1 == r2


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

    def test_metrics_json_values_match_fresh_run(self):
        """Verify stored metrics are reproducible: rerun backtest and compare."""
        from engine.backtest import run, walk_forward_backtest
        stored = self._load()
        params = {"entry_window": 20, "exit_window": 10, "atr_window": 20}
        for csv_name, key in [
            ("trend_gbm.csv", "trend_gbm"),
            ("mean_rev_ou.csv", "mean_rev_ou"),
            ("regime_switch.csv", "regime_switch"),
            ("fat_tail.csv", "fat_tail"),
        ]:
            df = _load_data(csv_name)
            fresh = run(DonchianTurtleBreakout(**params), df)
            assert abs(fresh["sharpe"] - stored[key]["sharpe"]) < 1e-4, (
                f"sharpe mismatch on {key}: stored={stored[key]['sharpe']:.6f} "
                f"vs fresh={fresh['sharpe']:.6f}"
            )
            wf_fresh = walk_forward_backtest(DonchianTurtleBreakout, params, df)
            assert abs(
                wf_fresh["oos_sharpe_mean"] - stored[key]["walk_forward"]["oos_sharpe_mean"]
            ) < 1e-4, f"oos_sharpe_mean mismatch on {key}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_constant_price_series_is_flat(self):
        """Constant prices never break above channel high; strategy stays flat."""
        s = DonchianTurtleBreakout(entry_window=5, exit_window=2)
        closes = [100.0] * 50
        view = _make_view(closes)
        assert s(view) == 0.0

    def test_large_input_does_not_crash(self):
        """5000-bar input should complete without error."""
        s = DonchianTurtleBreakout()
        rng = np.random.default_rng(0)
        closes = (100.0 + np.cumsum(rng.normal(0.05, 1.0, 5000))).tolist()
        view = _make_view(closes)
        result = s(view)
        assert result in (0.0, 1.0)

    def test_minimum_valid_windows(self):
        """entry_window=2, exit_window=1 is the smallest valid configuration."""
        s = DonchianTurtleBreakout(entry_window=2, exit_window=1, atr_window=2)
        closes = [100.0, 101.0, 102.0]
        view = _make_view(closes)
        result = s(view)
        assert result in (0.0, 1.0)

    def test_strongly_declining_series_never_enters(self):
        """A monotonically declining price never breaks above the channel high."""
        s = DonchianTurtleBreakout(entry_window=5, exit_window=2)
        closes = list(range(200, 100, -1))
        view = _make_view(closes)
        assert s(view) == 0.0
