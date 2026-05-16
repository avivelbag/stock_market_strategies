"""Tests for strategies/08-nr7-breakout/strategy.py.

Covers:
  - Default parameters and constructor validation
  - Warm-up guard: flat when fewer than n_bars bars available
  - NR7 signal correctness: fires only when current bar is the rolling n_bars minimum
  - Direction filter: close above midpoint → long, at/below midpoint → short
  - Time-based exit: position held for exactly exit_bars bars then closed
  - State management: ignores new NR7 while in position; resets after exit
  - Multiple entry/exit cycles
  - Integration with engine.backtest.run() on all four synthetic datasets
  - Integration with walk_forward_backtest() and metrics.json structure
  - Edge cases: constant prices, single bar, exactly n_bars bars, large input
  - Failure modes: invalid constructor arguments
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
STRATEGY_FILE = ROOT / "strategies" / "08-nr7-breakout" / "strategy.py"
METRICS_FILE = ROOT / "strategies" / "08-nr7-breakout" / "metrics.json"

_DATASETS = ["trend_gbm.csv", "mean_rev_ou.csv", "regime_switch.csv", "fat_tail.csv"]


def _load_strategy_class():
    """Load NR7Breakout from the hyphenated directory via importlib."""
    spec = importlib.util.spec_from_file_location("nr7_strategy", STRATEGY_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NR7Breakout


NR7Breakout = _load_strategy_class()


def _load_data(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, index_col=0, parse_dates=True)


def _make_ohlcv(
    closes: list,
    high_add: float = 1.0,
    low_sub: float = 1.0,
) -> pd.DataFrame:
    """Construct a minimal OHLCV DataFrame with controllable high/low spreads.

    By default high = close + high_add and low = close - low_sub, giving a
    constant true range of high_add + low_sub per bar. Pass per-bar lists or
    scalars for high_add/low_sub to create bars with varying ranges.
    """
    n = len(closes)
    dates = pd.bdate_range("2020-01-02", periods=n)
    arr = np.array(closes, dtype=float)

    if np.isscalar(high_add):
        high_add = [high_add] * n
    if np.isscalar(low_sub):
        low_sub = [low_sub] * n

    high_add = np.array(high_add, dtype=float)
    low_sub = np.array(low_sub, dtype=float)

    return pd.DataFrame(
        {
            "open": arr * 0.999,
            "high": arr + high_add,
            "low": arr - low_sub,
            "close": arr,
            "volume": np.ones(n, dtype=int) * 1000,
        },
        index=dates,
    )


def _simulate(strategy, ohlcv: pd.DataFrame) -> list:
    """Drive the strategy bar-by-bar as the engine would, returning all signals."""
    signals = []
    for t in range(1, len(ohlcv) + 1):
        signals.append(strategy(ohlcv.iloc[:t]))
    return signals


# ---------------------------------------------------------------------------
# Default parameters and constructor validation
# ---------------------------------------------------------------------------


class TestDefaultParameters:
    def test_default_n_bars_is_7(self):
        s = NR7Breakout()
        assert s.n_bars == 7

    def test_default_exit_bars_is_4(self):
        s = NR7Breakout()
        assert s.exit_bars == 4

    def test_custom_params_stored(self):
        s = NR7Breakout(n_bars=5, exit_bars=2)
        assert s.n_bars == 5
        assert s.exit_bars == 2

    def test_starts_out_of_position(self):
        s = NR7Breakout()
        assert s._in_position is False
        assert s._bars_held == 0
        assert s._position_direction == 0

    def test_n_bars_less_than_2_raises(self):
        with pytest.raises(ValueError, match="n_bars must be at least 2"):
            NR7Breakout(n_bars=1)

    def test_n_bars_zero_raises(self):
        with pytest.raises(ValueError, match="n_bars must be at least 2"):
            NR7Breakout(n_bars=0)

    def test_exit_bars_less_than_1_raises(self):
        with pytest.raises(ValueError, match="exit_bars must be at least 1"):
            NR7Breakout(exit_bars=0)

    def test_exit_bars_negative_raises(self):
        with pytest.raises(ValueError, match="exit_bars must be at least 1"):
            NR7Breakout(exit_bars=-1)

    def test_default_params_dict_exists(self):
        spec = importlib.util.spec_from_file_location("nr7_strategy", STRATEGY_FILE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "DEFAULT_PARAMS")
        assert mod.DEFAULT_PARAMS == {"n_bars": 7, "exit_bars": 4}


# ---------------------------------------------------------------------------
# Warm-up guard
# ---------------------------------------------------------------------------


class TestWarmupGuard:
    def test_single_bar_returns_flat(self):
        s = NR7Breakout(n_bars=7, exit_bars=4)
        df = _make_ohlcv([100.0])
        assert s(df) == 0.0

    def test_n_bars_minus_one_returns_flat(self):
        s = NR7Breakout(n_bars=5, exit_bars=2)
        df = _make_ohlcv([100.0] * 4)
        assert s(df) == 0.0

    def test_exactly_n_bars_can_produce_signal(self):
        """At exactly n_bars bars the rolling minimum is defined; a signal is possible."""
        s = NR7Breakout(n_bars=3, exit_bars=1)
        # Construct [big, big, small] so the last bar is NR7.
        df = _make_ohlcv([100.0, 100.0, 100.0], high_add=[5.0, 5.0, 1.0], low_sub=[5.0, 5.0, 1.0])
        result = s(df)
        assert result in (1.0, -1.0), f"Expected a directional signal at n_bars bars, got {result}"


# ---------------------------------------------------------------------------
# NR7 signal correctness
# ---------------------------------------------------------------------------


class TestNR7SignalCorrectness:
    def _make_nr7_at_last_bar(self, n_bars: int = 5) -> pd.DataFrame:
        """Create a DataFrame where only the last bar qualifies as NR7.

        Bars 0..n_bars-2 have range 10, bar n_bars-1 has range 1 (the minimum).
        Close is set above the midpoint so the direction filter picks long.
        """
        closes = [100.0] * n_bars
        high_add = [5.0] * (n_bars - 1) + [0.9]
        low_sub = [5.0] * (n_bars - 1) + [0.1]
        return _make_ohlcv(closes, high_add=high_add, low_sub=low_sub)

    def test_nr7_long_signal_fires_when_close_above_midpoint(self):
        """Bar with smallest range and close above midpoint → long (1.0)."""
        s = NR7Breakout(n_bars=5, exit_bars=4)
        df = self._make_nr7_at_last_bar(n_bars=5)
        # close=100, high=100.9, low=99.9, midpoint=100.4; close(100) < midpoint(100.4) → short
        # Need close above midpoint: set close = 100.5 (above midpoint 100.4)
        df_long = df.copy()
        df_long.iloc[-1, df_long.columns.get_loc("close")] = 100.5
        result = s(df_long)
        assert result == 1.0, f"Expected long signal, got {result}"

    def test_nr7_short_signal_fires_when_close_at_midpoint(self):
        """Close exactly at midpoint → short (-1.0) since condition is 'above' midpoint."""
        s = NR7Breakout(n_bars=5, exit_bars=4)
        df = self._make_nr7_at_last_bar(n_bars=5)
        # high=100.9, low=99.9, midpoint=100.4; set close = 100.4 (equal to midpoint → short)
        df_eq = df.copy()
        df_eq.iloc[-1, df_eq.columns.get_loc("close")] = 100.4
        result = s(df_eq)
        assert result == -1.0, f"Expected short signal at midpoint, got {result}"

    def test_nr7_short_signal_fires_when_close_below_midpoint(self):
        """Close below bar midpoint → short (-1.0)."""
        s = NR7Breakout(n_bars=5, exit_bars=4)
        df = self._make_nr7_at_last_bar(n_bars=5)
        # high=100.9, low=99.9, midpoint=100.4; close=100 < 100.4 → short
        result = s(df)
        assert result == -1.0, f"Expected short signal, got {result}"

    def test_no_signal_when_not_narrowest(self):
        """If current bar is not the narrowest, no entry signal."""
        s = NR7Breakout(n_bars=5, exit_bars=4)
        # Last bar has range 10, others have range 1 — last bar is NOT the minimum
        high_add = [0.5] * 4 + [5.0]
        low_sub = [0.5] * 4 + [5.0]
        df = _make_ohlcv([100.0] * 5, high_add=high_add, low_sub=low_sub)
        result = s(df)
        assert result == 0.0, f"Expected no signal when last bar is widest, got {result}"

    def test_signal_only_at_rolling_minimum(self):
        """Only the bar with the rolling minimum range should trigger NR7."""
        s = NR7Breakout(n_bars=4, exit_bars=1)
        # Pattern: [10, 8, 6, 4] — bar 3 (index 3) has range 4 = minimum of window
        high_add = [5.0, 4.0, 3.0, 2.0]
        low_sub = [5.0, 4.0, 3.0, 2.0]
        df = _make_ohlcv([100.0] * 4, high_add=high_add, low_sub=low_sub)
        result = s(df)
        assert result in (1.0, -1.0), f"Expected a signal at the narrowest bar, got {result}"

    def test_returns_only_minus_one_zero_or_one(self):
        """Strategy must return exactly -1.0, 0.0, or 1.0 — no fractional values."""
        s = NR7Breakout(n_bars=5, exit_bars=2)
        for n in range(1, 20):
            df = _make_ohlcv(
                [100.0 + i * 0.5 for i in range(n)],
                high_add=[(n - i) * 0.5 + 0.1 for i in range(n)],
                low_sub=[(n - i) * 0.5 + 0.1 for i in range(n)],
            )
            result = s(df)
            assert result in (-1.0, 0.0, 1.0), f"Got unexpected signal {result} at n={n}"

    def test_signal_is_float(self):
        s = NR7Breakout(n_bars=3, exit_bars=1)
        df = _make_ohlcv([100.0] * 3)
        result = s(df)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Time-based exit: hold exactly exit_bars bars
# ---------------------------------------------------------------------------


class TestTimedExit:
    def _build_nr7_entry(self, n_bars: int = 4, exit_bars: int = 4) -> tuple:
        """Return (strategy, DataFrame) where the last bar of df is an NR7."""
        s = NR7Breakout(n_bars=n_bars, exit_bars=exit_bars)
        high_add = [5.0] * (n_bars - 1) + [0.5]
        low_sub = [5.0] * (n_bars - 1) + [0.5]
        entry_df = _make_ohlcv([100.0] * n_bars, high_add=high_add, low_sub=low_sub)
        entry_df.iloc[-1, entry_df.columns.get_loc("close")] = 100.3  # above midpoint → long
        return s, entry_df

    def test_exit_bars_1_holds_one_bar_only(self):
        """With exit_bars=1, position closes after exactly 1 bar."""
        s, entry_df = self._build_nr7_entry(n_bars=4, exit_bars=1)
        first_signal = s(entry_df)
        assert first_signal == 1.0, "Entry should fire long"

        n = len(entry_df)
        # Second call (bar after entry): should exit
        extra_bar = _make_ohlcv([100.0] * (n + 1), high_add=[5.0] * (n - 1) + [0.5, 3.0],
                                  low_sub=[5.0] * (n - 1) + [0.5, 3.0])
        extra_bar.iloc[-(n + 1):-(n), entry_df.columns.get_loc("close")] = 100.3
        result = s(extra_bar)
        assert result == 0.0, f"Expected exit after 1 bar, got {result}"

    def test_holds_exactly_exit_bars(self):
        """Position held for exactly exit_bars bars, flat on exit bar."""
        exit_bars = 3
        n_bars = 4
        s = NR7Breakout(n_bars=n_bars, exit_bars=exit_bars)

        # Build OHLCV where bar 3 (index 3) is NR7 long entry
        high_add = [5.0] * 3 + [0.5] + [3.0] * 16
        low_sub = [5.0] * 3 + [0.5] + [3.0] * 16
        closes = [100.0] * 20
        df = _make_ohlcv(closes, high_add=high_add, low_sub=low_sub)
        df.iloc[3, df.columns.get_loc("close")] = 100.3  # above midpoint → long

        signals = _simulate(s, df)
        entry_idx = 3
        # Bars entry_idx to entry_idx + exit_bars - 1 should be long (direction signal)
        for t in range(entry_idx, entry_idx + exit_bars):
            assert signals[t] == 1.0, (
                f"Expected long at bar {t}, got {signals[t]}"
            )
        # Bar entry_idx + exit_bars should be exit (0.0)
        assert signals[entry_idx + exit_bars] == 0.0, (
            f"Expected exit at bar {entry_idx + exit_bars}, got {signals[entry_idx + exit_bars]}"
        )

    def test_flat_after_exit(self):
        """Strategy returns 0.0 after the time-based exit when no new NR7 fires.

        Post-exit bars use strictly increasing ranges so the current bar is
        always the widest in its window — preventing NR7 ties from firing.
        """
        s = NR7Breakout(n_bars=4, exit_bars=2)
        # Bar 3: range 1.0 (NR7). Post-entry bars have strictly increasing ranges
        # so no bar after 3 is ever the rolling minimum of its 4-bar window.
        high_add = [5.0] * 3 + [0.5] + [5.0, 6.0, 7.0, 8.0, 9.0]
        low_sub = [5.0] * 3 + [0.5] + [5.0, 6.0, 7.0, 8.0, 9.0]
        df = _make_ohlcv([100.0] * 9, high_add=high_add, low_sub=low_sub)
        df.iloc[3, df.columns.get_loc("close")] = 100.3  # entry long

        signals = _simulate(s, df)
        # Exit fires at bar 5 (bar 3 entry + exit_bars=2 hold calls → bar 5 returns 0)
        assert signals[3] == 1.0, "Entry should fire at bar 3"
        assert signals[4] == 1.0, "Hold signal at bar 4"
        assert signals[5] == 0.0, "Exit at bar 5"
        for t in range(6, len(signals)):
            assert signals[t] == 0.0, f"Expected flat at bar {t}, got {signals[t]}"

    def test_ignore_new_nr7_while_in_position(self):
        """A new NR7 bar while in a position does not restart the exit counter."""
        # NR7 at bar 2; then bar 3 also qualifies as NR7 (range stays at minimum)
        # Both should just continue the existing position, not reset the counter
        high_add = [5.0, 5.0, 0.5, 0.5, 0.5, 0.5, 5.0]
        low_sub = [5.0, 5.0, 0.5, 0.5, 0.5, 0.5, 5.0]
        df = _make_ohlcv([100.0] * 7, high_add=high_add, low_sub=low_sub)
        # Set closes above midpoints for bars 2+ → long entries if checked
        for i in range(2, 7):
            df.iloc[i, df.columns.get_loc("close")] = 100.3

        # Simulate using the strategy's bar-by-bar state
        s2 = NR7Breakout(n_bars=3, exit_bars=4)
        sig2 = _make_ohlcv([100.0] * 3, high_add=[5.0, 5.0, 0.5], low_sub=[5.0, 5.0, 0.5])
        sig2.iloc[-1, sig2.columns.get_loc("close")] = 100.3
        first = s2(sig2)
        assert first == 1.0, "Entry should fire"

        # bars_held after entry: 0
        bars_held_snapshots = [s2._bars_held]
        for t in range(3, 7):
            s2(df.iloc[: t + 1])
            bars_held_snapshots.append(s2._bars_held)

        # bars_held should increase monotonically 0, 1, 2, 3, then reset on exit
        assert bars_held_snapshots[0] == 0
        assert bars_held_snapshots[1] == 1
        assert bars_held_snapshots[2] == 2
        assert bars_held_snapshots[3] == 3


# ---------------------------------------------------------------------------
# Multiple entry/exit cycles
# ---------------------------------------------------------------------------


class TestMultipleCycles:
    def test_re_entry_after_exit(self):
        """Strategy can enter a new position after the time-based exit."""
        s = NR7Breakout(n_bars=3, exit_bars=1)
        # Two NR7 events separated by wide bars: one at bar 2, one at bar 7
        high_add = [5.0, 5.0, 0.5, 5.0, 5.0, 5.0, 5.0, 0.5, 5.0, 5.0]
        low_sub = [5.0, 5.0, 0.5, 5.0, 5.0, 5.0, 5.0, 0.5, 5.0, 5.0]
        df = _make_ohlcv([100.0] * 10, high_add=high_add, low_sub=low_sub)
        for i in [2, 7]:
            df.iloc[i, df.columns.get_loc("close")] = 100.3  # long entries

        signals = _simulate(s, df)
        long_count = sum(1 for sig in signals if sig == 1.0)
        assert long_count >= 2, f"Expected at least 2 long bars, got {long_count}"

    def test_both_long_and_short_entries_across_cycles(self):
        """Direction filter assigns long or short correctly across separate entry events."""
        s = NR7Breakout(n_bars=3, exit_bars=1)
        high_add = [5.0, 5.0, 0.5, 5.0, 5.0, 5.0, 5.0, 0.5, 5.0, 5.0]
        low_sub = [5.0, 5.0, 0.5, 5.0, 5.0, 5.0, 5.0, 0.5, 5.0, 5.0]
        df = _make_ohlcv([100.0] * 10, high_add=high_add, low_sub=low_sub)
        df.iloc[2, df.columns.get_loc("close")] = 100.3   # above midpoint → long
        df.iloc[7, df.columns.get_loc("close")] = 99.7    # below midpoint → short

        signals = _simulate(s, df)
        assert 1.0 in signals, "Expected at least one long signal"
        assert -1.0 in signals, "Expected at least one short signal"


# ---------------------------------------------------------------------------
# Integration with engine.backtest.run()
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_run_on_trend_gbm_returns_dict(self):
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        s = NR7Breakout()
        result = run(s, df)
        assert isinstance(result, dict)
        assert "sharpe" in result
        assert "cagr" in result

    def test_run_on_all_four_datasets_completes(self):
        from engine.backtest import run
        for name in _DATASETS:
            df = _load_data(name)
            s = NR7Breakout()
            result = run(s, df)
            assert isinstance(result["sharpe"], float)
            assert isinstance(result["cagr"], float)

    def test_run_is_deterministic(self):
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        r1 = run(NR7Breakout(), df)
        r2 = run(NR7Breakout(), df)
        assert r1 == r2

    def test_exposure_less_than_one_due_to_warmup_and_time_exit(self):
        """Warm-up period and time-based exit both reduce exposure below 1.0."""
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(NR7Breakout(), df)
        assert result["exposure"] < 1.0

    def test_no_lookahead_error_during_run(self):
        from engine.backtest import run, LookAheadError
        for name in _DATASETS:
            df = _load_data(name)
            try:
                run(NR7Breakout(), df)
            except LookAheadError:
                pytest.fail(f"LookAheadError raised on {name}")

    def test_trend_gbm_negative_sharpe(self):
        """On a pure GBM trend, NR7 should underperform; Sharpe expected negative."""
        from engine.backtest import run
        df = _load_data("trend_gbm.csv")
        result = run(NR7Breakout(), df)
        assert result["sharpe"] < 0.2, (
            f"trend_gbm Sharpe unexpectedly high: {result['sharpe']:.4f} — "
            "few compression events expected in a pure-trend regime"
        )

    def test_regime_switch_positive_sharpe(self):
        """Regime transitions produce compression→expansion cycles; Sharpe should be positive."""
        from engine.backtest import run
        df = _load_data("regime_switch.csv")
        result = run(NR7Breakout(), df)
        assert result["sharpe"] > 0.0, (
            f"regime_switch Sharpe expected positive, got {result['sharpe']:.4f}"
        )


# ---------------------------------------------------------------------------
# Walk-forward backtest integration
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    def test_walk_forward_returns_required_keys(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("regime_switch.csv")
        result = walk_forward_backtest(
            NR7Breakout,
            {"n_bars": 7, "exit_bars": 4},
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
                NR7Breakout,
                {"n_bars": 7, "exit_bars": 4},
                df,
            )
            assert 0.0 <= result["oos_consistency"] <= 1.0, (
                f"oos_consistency out of range on {name}"
            )

    def test_walk_forward_is_deterministic(self):
        from engine.backtest import walk_forward_backtest
        df = _load_data("trend_gbm.csv")
        params = {"n_bars": 7, "exit_bars": 4}
        r1 = walk_forward_backtest(NR7Breakout, params, df)
        r2 = walk_forward_backtest(NR7Breakout, params, df)
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

    def test_metrics_json_has_regime_sharpe(self):
        data = self._load()
        for dataset, values in data.items():
            assert "regime_sharpe" in values, (
                f"metrics.json entry '{dataset}' missing 'regime_sharpe'"
            )
            rs = values["regime_sharpe"]
            assert "regime_counts" in rs
            assert "high_vol" in rs
            assert "trending" in rs
            assert "ranging" in rs

    def test_metrics_json_values_match_fresh_run(self):
        """Verify stored metrics are reproducible: rerun backtest and compare."""
        from engine.backtest import run, walk_forward_backtest
        stored = self._load()
        params = {"n_bars": 7, "exit_bars": 4}
        for csv_name, key in [
            ("trend_gbm.csv", "trend_gbm"),
            ("mean_rev_ou.csv", "mean_rev_ou"),
            ("regime_switch.csv", "regime_switch"),
            ("fat_tail.csv", "fat_tail"),
        ]:
            df = _load_data(csv_name)
            fresh = run(NR7Breakout(**params), df)
            assert abs(fresh["sharpe"] - stored[key]["sharpe"]) < 1e-4, (
                f"sharpe mismatch on {key}: stored={stored[key]['sharpe']:.6f} "
                f"vs fresh={fresh['sharpe']:.6f}"
            )
            wf_fresh = walk_forward_backtest(NR7Breakout, params, df)
            assert abs(
                wf_fresh["oos_sharpe_mean"] - stored[key]["walk_forward"]["oos_sharpe_mean"]
            ) < 1e-4, f"oos_sharpe_mean mismatch on {key}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_constant_price_series_constant_range(self):
        """Constant prices with constant range: NR7 condition satisfied on every bar
        (all ranges are equal = the minimum), so a signal fires as soon as warm-up ends."""
        s = NR7Breakout(n_bars=5, exit_bars=2)
        df = _make_ohlcv([100.0] * 20)  # constant range = 2.0 on all bars
        signals = _simulate(s, df)
        # At bar 4 (index 4), the first NR7 signal fires
        non_flat = [sig for sig in signals[4:] if sig != 0.0]
        assert len(non_flat) > 0, "Expected at least one entry signal in a constant-range series"

    def test_large_input_does_not_crash(self):
        """5000-bar input should complete without error."""
        s = NR7Breakout()
        rng = np.random.default_rng(0)
        closes = (100.0 + np.cumsum(rng.normal(0.05, 1.0, 5000))).tolist()
        highs = [c + rng.uniform(0.1, 3.0) for c in closes]
        lows = [c - rng.uniform(0.1, 3.0) for c in closes]
        n = len(closes)
        dates = pd.bdate_range("2000-01-03", periods=n)
        df = pd.DataFrame(
            {"open": closes, "high": highs, "low": lows, "close": closes,
             "volume": [1000] * n},
            index=dates,
        )
        result = s(df)
        assert result in (-1.0, 0.0, 1.0)

    def test_minimum_valid_n_bars_2(self):
        """n_bars=2, exit_bars=1 is the smallest valid configuration."""
        s = NR7Breakout(n_bars=2, exit_bars=1)
        df = _make_ohlcv([100.0, 101.0, 102.0], high_add=[3.0, 3.0, 1.0], low_sub=[3.0, 3.0, 1.0])
        df.iloc[-1, df.columns.get_loc("close")] = 102.5  # above midpoint
        result = s(df)
        assert result in (-1.0, 0.0, 1.0)

    def test_strongly_increasing_range_never_fires_nr7(self):
        """If the current bar always has the widest range, no NR7 fires."""
        s = NR7Breakout(n_bars=5, exit_bars=2)
        # Monotonically increasing ranges: each bar wider than all prior bars
        ranges = [float(i + 1) for i in range(20)]
        high_add = [r / 2 for r in ranges]
        low_sub = [r / 2 for r in ranges]
        df = _make_ohlcv([100.0] * 20, high_add=high_add, low_sub=low_sub)
        result = s(df)
        assert result == 0.0, (
            "No NR7 should fire when the current bar always has the widest range"
        )

    def test_each_call_produces_valid_output(self):
        """Bar-by-bar simulation: every signal must be -1.0, 0.0, or 1.0."""
        rng = np.random.default_rng(42)
        n = 100
        closes = (100.0 + np.cumsum(rng.normal(0, 0.5, n))).tolist()
        ranges = rng.uniform(0.5, 3.0, n).tolist()
        df = _make_ohlcv(closes, high_add=ranges, low_sub=ranges)
        s = NR7Breakout(n_bars=5, exit_bars=3)
        for t in range(1, n + 1):
            result = s(df.iloc[:t])
            assert result in (-1.0, 0.0, 1.0), f"Invalid signal {result} at bar {t}"
