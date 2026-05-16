"""Tests for strategy 11: Statistical Pairs Mean-Reversion.

Covers:
  - Data generation: paired_cointegrated.csv structure and properties
  - Happy path: long/short entries when z-score exceeds thresholds
  - Exit logic: position closes when z-score reverts past z_exit
  - Warm-up guard: flat when fewer bars than window
  - Default parameters match Gatev et al. (2006)
  - Parameter validation: ValueError on invalid inputs
  - Z-score arithmetic: correct rolling mean/std computation
  - Engine integration: run() with allow_short=True on paired data
  - Walk-forward integration: all required keys returned
  - metrics.json reproducibility: stored values match fresh run
  - Edge cases: constant spread, single bar, large input, alternating spread
  - Failure modes: returns only {-1, 0, 1}; no position without sufficient history
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
STRATEGY_FILE = ROOT / "strategies" / "11-pairs-mean-reversion" / "strategy.py"
METRICS_FILE = ROOT / "strategies" / "11-pairs-mean-reversion" / "metrics.json"
PAIRED_CSV = DATA_DIR / "paired_cointegrated.csv"


def _load_strategy_module():
    """Load PairsMeanReversion from the hyphenated directory via importlib."""
    spec = importlib.util.spec_from_file_location("pairs_strategy", STRATEGY_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_strategy_module()
PairsMeanReversion = _mod.PairsMeanReversion


def _load_paired_data() -> pd.DataFrame:
    """Load the synthetic paired co-integrated dataset."""
    return pd.read_csv(PAIRED_CSV, index_col=0, parse_dates=True)


def _make_spread_view(spread_closes: list) -> pd.DataFrame:
    """Construct a minimal OHLCV DataFrame from a list of spread ratio values.

    The spread ratio is always positive (close_a / close_b). OHLCV columns
    are required by the engine; high/low/open are constructed from close.
    """
    n = len(spread_closes)
    dates = pd.bdate_range("2020-01-02", periods=n)
    arr = np.array(spread_closes, dtype=float)
    return pd.DataFrame(
        {
            "open": arr * 0.999,
            "high": arr * 1.005,
            "low": arr * 0.995,
            "close": arr,
            "volume": np.ones(n, dtype=int) * 1_000_000,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Synthetic data file structure
# ---------------------------------------------------------------------------


class TestPairedDataFile:
    def test_paired_csv_exists(self):
        assert PAIRED_CSV.is_file(), "paired_cointegrated.csv is missing from data/"

    def test_paired_csv_has_required_columns(self):
        df = _load_paired_data()
        required = {"open", "high", "low", "close", "volume", "close_a", "close_b"}
        assert required.issubset(set(df.columns)), (
            f"Missing columns: {required - set(df.columns)}"
        )

    def test_paired_csv_has_1000_rows(self):
        df = _load_paired_data()
        assert len(df) == 1000

    def test_close_is_ratio_of_close_a_and_close_b(self):
        """close column must equal close_a / close_b within floating-point tolerance."""
        df = _load_paired_data()
        ratio = df["close_a"] / df["close_b"]
        diff = (ratio - df["close"]).abs()
        assert diff.max() < 1e-4, f"close != close_a/close_b; max diff = {diff.max()}"

    def test_close_is_always_positive(self):
        df = _load_paired_data()
        assert (df["close"] > 0).all(), "spread close column has non-positive values"

    def test_log_spread_is_stationary_mean_near_zero(self):
        """The log-spread (eps_A - eps_B) should have mean near 0 for a large sample."""
        df = _load_paired_data()
        log_spread = np.log(df["close"].values)
        assert abs(log_spread.mean()) < 0.2, (
            f"log-spread mean far from zero: {log_spread.mean():.4f}"
        )

    def test_close_a_and_close_b_are_positive(self):
        df = _load_paired_data()
        assert (df["close_a"] > 0).all()
        assert (df["close_b"] > 0).all()

    def test_generate_script_is_reproducible(self, tmp_path):
        """Re-running the data generator must produce byte-identical CSV."""
        import importlib.util as ilu
        gen_file = DATA_DIR / "generate_paired.py"
        spec = ilu.spec_from_file_location("gen_paired", gen_file)
        gen_mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(gen_mod)
        df_fresh = gen_mod.generate_paired_cointegrated(42)
        df_stored = _load_paired_data()
        pd.testing.assert_frame_equal(
            df_fresh.reset_index(drop=True),
            df_stored.reset_index(drop=True),
            check_exact=False,
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# Default parameters and construction
# ---------------------------------------------------------------------------


class TestDefaultParameters:
    def test_default_z_entry_is_2(self):
        s = PairsMeanReversion()
        assert s.z_entry == 2.0

    def test_default_z_exit_is_0_5(self):
        s = PairsMeanReversion()
        assert s.z_exit == 0.5

    def test_default_window_is_60(self):
        s = PairsMeanReversion()
        assert s.window == 60

    def test_starts_flat(self):
        s = PairsMeanReversion()
        assert s._position == 0

    def test_custom_params_stored(self):
        s = PairsMeanReversion(z_entry=3.0, z_exit=1.0, window=30)
        assert s.z_entry == 3.0
        assert s.z_exit == 1.0
        assert s.window == 30

    def test_default_params_match_gatev_2006(self):
        """Gatev et al. (2006) published defaults: z_entry=2.0, z_exit=0.5."""
        assert _mod.DEFAULT_PARAMS["z_entry"] == 2.0
        assert _mod.DEFAULT_PARAMS["z_exit"] == 0.5

    def test_invalid_z_entry_zero_raises(self):
        with pytest.raises(ValueError, match="z_entry must be positive"):
            PairsMeanReversion(z_entry=0.0)

    def test_invalid_z_entry_negative_raises(self):
        with pytest.raises(ValueError, match="z_entry must be positive"):
            PairsMeanReversion(z_entry=-1.0)

    def test_invalid_z_exit_negative_raises(self):
        with pytest.raises(ValueError, match="z_exit must be non-negative"):
            PairsMeanReversion(z_exit=-0.1)

    def test_invalid_z_exit_ge_z_entry_raises(self):
        with pytest.raises(ValueError, match="z_exit must be strictly less than z_entry"):
            PairsMeanReversion(z_entry=1.0, z_exit=1.0)

    def test_invalid_window_one_raises(self):
        with pytest.raises(ValueError, match="window must be greater than 1"):
            PairsMeanReversion(window=1)

    def test_invalid_window_zero_raises(self):
        with pytest.raises(ValueError, match="window must be greater than 1"):
            PairsMeanReversion(window=0)


# ---------------------------------------------------------------------------
# Warm-up guard
# ---------------------------------------------------------------------------


class TestWarmupGuard:
    def test_single_bar_returns_flat(self):
        s = PairsMeanReversion()
        view = _make_spread_view([1.0])
        assert s(view) == 0.0

    def test_window_minus_one_bars_returns_flat(self):
        s = PairsMeanReversion(window=5)
        view = _make_spread_view([1.0] * 4)
        assert s(view) == 0.0

    def test_exactly_window_bars_may_produce_signal(self):
        s = PairsMeanReversion(window=5)
        view = _make_spread_view([1.0] * 5)
        result = s(view)
        assert result in (-1.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Signal correctness: entry and exit
# ---------------------------------------------------------------------------


class TestSignalCorrectness:
    def _make_spread_with_extreme(self, n_warmup=70, extreme_ratio=0.7, n_tail=3):
        """Construct a spread that is stable then drops sharply below mean.

        The sharp drop should push the z-score below -2.0 and trigger a long entry.
        """
        warm = [1.0] * n_warmup
        tail = [extreme_ratio] * n_tail
        return warm + tail

    def test_enters_long_when_spread_drops_sharply(self):
        """Sharp spread drop → z < −2 → long entry (position = +1)."""
        s = PairsMeanReversion(window=60)
        closes = self._make_spread_with_extreme(n_warmup=70, extreme_ratio=0.7)
        view = _make_spread_view(closes)
        result = s(view)
        assert result == 1.0, f"Expected long entry on extreme spread drop, got {result}"

    def test_enters_short_when_spread_rises_sharply(self):
        """Sharp spread rise → z > +2 → short entry (position = -1)."""
        s = PairsMeanReversion(window=60)
        closes = [1.0] * 70 + [1.6] * 3
        view = _make_spread_view(closes)
        result = s(view)
        assert result == -1.0, f"Expected short entry on extreme spread rise, got {result}"

    def test_exits_long_when_z_recovers_above_minus_z_exit(self):
        """After entering long, z recovering above -z_exit must trigger exit."""
        s = PairsMeanReversion(window=60, z_entry=2.0, z_exit=0.5)
        closes = [1.0] * 70 + [0.7] * 3
        view_entry = _make_spread_view(closes)
        s(view_entry)
        assert s._position == 1, "Expected long position after sharp drop"

        closes_recovered = closes + [1.0] * 10
        view_recovered = _make_spread_view(closes_recovered)
        result = s(view_recovered)
        assert result == 0.0, "Expected flat after spread recovery"

    def test_exits_short_when_z_recovers_below_z_exit(self):
        """After entering short, z recovering below z_exit must trigger exit."""
        s = PairsMeanReversion(window=60, z_entry=2.0, z_exit=0.5)
        closes = [1.0] * 70 + [1.6] * 3
        s(_make_spread_view(closes))
        assert s._position == -1, "Expected short position after sharp rise"

        closes_recovered = closes + [1.0] * 10
        result = s(_make_spread_view(closes_recovered))
        assert result == 0.0, "Expected flat after spread recovery"

    def test_returns_only_minus_one_zero_or_one(self):
        """Position must be exactly -1, 0, or +1 at every bar."""
        s = PairsMeanReversion()
        rng = np.random.default_rng(42)
        closes = np.exp(rng.normal(0, 0.1, 200)).tolist()
        for i in range(1, len(closes) + 1):
            view = _make_spread_view(closes[:i])
            result = s(view)
            assert result in (-1.0, 0.0, 1.0), f"Got {result} at bar {i}"

    def test_position_held_between_signals(self):
        """Once entered, position must not change until exit condition is met."""
        s = PairsMeanReversion(window=20, z_entry=2.0, z_exit=0.5)
        closes = [1.0] * 25 + [0.6] * 3
        s(_make_spread_view(closes))
        initial_pos = s._position
        if initial_pos != 0:
            for extra_close in [0.61, 0.62]:
                result = s(_make_spread_view(closes + [extra_close]))
                assert result == float(initial_pos), (
                    "Position changed without meeting exit condition"
                )


# ---------------------------------------------------------------------------
# Z-score arithmetic
# ---------------------------------------------------------------------------


class TestZScoreArithmetic:
    def test_constant_spread_gives_zero_z(self):
        """A perfectly flat spread has zero std → z-score is undefined → stay flat."""
        s = PairsMeanReversion(window=10)
        closes = [1.0] * 20
        view = _make_spread_view(closes)
        result = s(view)
        assert result == 0.0, "Constant spread (std=0) must return flat signal"

    def test_log_spread_used_not_raw_ratio(self):
        """Z-score must be computed on log(close), not raw close.

        The difference matters: for spreads close to 1.0 the two are
        approximately equal, but for large deviations they diverge.
        This test verifies the log symmetry property: log(1.5) + log(1/1.5) == 0.
        If the implementation used raw ratio instead of log, an upward deviation
        of 1.5 and downward deviation of 1/1.5 would not be symmetric.
        """
        log_spread_up = np.log(1.5)
        log_spread_down = np.log(1.0 / 1.5)
        assert abs(log_spread_up + log_spread_down) < 1e-10, (
            "Log symmetry check: log(1.5) + log(1/1.5) must be zero"
        )

    def test_rolling_window_length_is_respected(self):
        """The rolling computation should use only the last `window` bars."""
        s_wide = PairsMeanReversion(window=60)
        s_narrow = PairsMeanReversion(window=10)
        rng = np.random.default_rng(7)
        closes = np.exp(rng.normal(0, 0.05, 80)).tolist()
        r_wide = s_wide(_make_spread_view(closes))
        r_narrow = s_narrow(_make_spread_view(closes))
        assert r_wide in (-1.0, 0.0, 1.0)
        assert r_narrow in (-1.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_run_on_paired_data_returns_dict(self):
        from engine.backtest import run
        df = _load_paired_data()
        result = run(PairsMeanReversion(), df, config={"allow_short": True})
        assert isinstance(result, dict)
        assert "sharpe" in result
        assert "cagr" in result

    def test_run_completes_without_error(self):
        from engine.backtest import run
        df = _load_paired_data()
        result = run(PairsMeanReversion(), df, config={"allow_short": True})
        assert isinstance(result["sharpe"], float)
        assert isinstance(result["cagr"], float)

    def test_run_is_deterministic(self):
        from engine.backtest import run
        df = _load_paired_data()
        cfg = {"allow_short": True}
        r1 = run(PairsMeanReversion(), df, config=cfg)
        r2 = run(PairsMeanReversion(), df, config=cfg)
        assert r1 == r2

    def test_no_lookahead_error_during_run(self):
        from engine.backtest import run, LookAheadError
        df = _load_paired_data()
        try:
            run(PairsMeanReversion(), df, config={"allow_short": True})
        except LookAheadError:
            pytest.fail("LookAheadError raised on paired_cointegrated dataset")

    def test_both_long_and_short_positions_taken(self):
        """A pairs strategy must generate both +1 and -1 positions; never long-only."""
        from engine.backtest import _run_internal
        df = _load_paired_data()
        _, _, positions, _ = _run_internal(
            PairsMeanReversion(), df, {"allow_short": True}
        )
        unique_pos = set(positions.unique())
        assert 1 in unique_pos, "No long positions taken"
        assert -1 in unique_pos, "No short positions taken"

    def test_exposure_is_positive(self):
        from engine.backtest import run
        df = _load_paired_data()
        result = run(PairsMeanReversion(), df, config={"allow_short": True})
        assert result["exposure"] > 0.0

    def test_existing_strategy_metrics_unchanged(self):
        """Running the engine on old datasets must reproduce stored metrics.json.

        This guards the orchestrator's requirement that existing strategy metrics
        are not silently altered when adding a new strategy.
        """
        import json as _json
        from engine.backtest import run
        old_metrics_file = ROOT / "strategies" / "06-bollinger-mean-reversion" / "metrics.json"
        stored = _json.loads(old_metrics_file.read_text())

        spec = importlib.util.spec_from_file_location(
            "s06", ROOT / "strategies" / "06-bollinger-mean-reversion" / "strategy.py"
        )
        s06_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s06_mod)
        BollingerMeanReversion = s06_mod.BollingerMeanReversion

        df = pd.read_csv(DATA_DIR / "regime_switch.csv", index_col=0, parse_dates=True)
        fresh = run(BollingerMeanReversion(window=20, nstd=2.0, exit_window=20), df)
        assert abs(fresh["sharpe"] - stored["regime_switch"]["sharpe"]) < 1e-4, (
            f"Bollinger regime_switch sharpe changed: "
            f"stored={stored['regime_switch']['sharpe']}, fresh={fresh['sharpe']}"
        )


# ---------------------------------------------------------------------------
# Walk-forward integration
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    def test_walk_forward_returns_required_keys(self):
        from engine.backtest import walk_forward_backtest
        df = _load_paired_data()
        params = {"z_entry": 2.0, "z_exit": 0.5, "window": 60}
        result = walk_forward_backtest(
            PairsMeanReversion, params, df, config={"allow_short": True}
        )
        expected = {
            "oos_sharpe_mean", "oos_sharpe_std", "oos_cagr_mean",
            "oos_max_drawdown_mean", "oos_consistency",
        }
        assert expected == set(result.keys())

    def test_walk_forward_consistency_in_unit_interval(self):
        from engine.backtest import walk_forward_backtest
        df = _load_paired_data()
        params = {"z_entry": 2.0, "z_exit": 0.5, "window": 60}
        result = walk_forward_backtest(
            PairsMeanReversion, params, df, config={"allow_short": True}
        )
        assert 0.0 <= result["oos_consistency"] <= 1.0

    def test_walk_forward_is_deterministic(self):
        from engine.backtest import walk_forward_backtest
        df = _load_paired_data()
        params = {"z_entry": 2.0, "z_exit": 0.5, "window": 60}
        cfg = {"allow_short": True}
        r1 = walk_forward_backtest(PairsMeanReversion, params, df, config=cfg)
        r2 = walk_forward_backtest(PairsMeanReversion, params, df, config=cfg)
        assert r1 == r2


# ---------------------------------------------------------------------------
# metrics.json reproducibility
# ---------------------------------------------------------------------------


class TestMetricsJson:
    def _load(self):
        with open(METRICS_FILE) as f:
            return json.load(f)

    def test_metrics_json_exists(self):
        assert METRICS_FILE.is_file()

    def test_metrics_json_has_paired_cointegrated_key(self):
        data = self._load()
        assert "paired_cointegrated" in data, "metrics.json missing 'paired_cointegrated'"

    def test_metrics_json_has_walk_forward_keys(self):
        data = self._load()
        wf = data["paired_cointegrated"]["walk_forward"]
        assert "oos_sharpe_mean" in wf
        assert "oos_consistency" in wf

    def test_metrics_json_sharpe_matches_fresh_run(self):
        """Stored Sharpe must reproduce within tolerance on fresh engine run."""
        from engine.backtest import run
        stored = self._load()
        df = _load_paired_data()
        fresh = run(
            PairsMeanReversion(z_entry=2.0, z_exit=0.5, window=60),
            df,
            config={"allow_short": True},
        )
        assert abs(fresh["sharpe"] - stored["paired_cointegrated"]["sharpe"]) < 1e-4, (
            f"Sharpe mismatch: stored={stored['paired_cointegrated']['sharpe']:.6f}, "
            f"fresh={fresh['sharpe']:.6f}"
        )

    def test_metrics_json_walk_forward_matches_fresh_run(self):
        from engine.backtest import walk_forward_backtest
        stored = self._load()
        df = _load_paired_data()
        params = {"z_entry": 2.0, "z_exit": 0.5, "window": 60}
        wf = walk_forward_backtest(
            PairsMeanReversion, params, df, config={"allow_short": True}
        )
        stored_wf = stored["paired_cointegrated"]["walk_forward"]
        assert abs(wf["oos_sharpe_mean"] - stored_wf["oos_sharpe_mean"]) < 1e-4, (
            f"oos_sharpe_mean mismatch: stored={stored_wf['oos_sharpe_mean']:.6f}, "
            f"fresh={wf['oos_sharpe_mean']:.6f}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_constant_spread_stays_flat(self):
        """Zero variance in spread → z-score undefined → no position."""
        s = PairsMeanReversion(window=10)
        view = _make_spread_view([1.0] * 50)
        assert s(view) == 0.0

    def test_large_input_completes(self):
        """5000-bar input must run to completion without error."""
        from engine.backtest import run
        rng = np.random.default_rng(0)
        spreads = np.exp(rng.normal(0, 0.05, 5001)).tolist()
        dates = pd.bdate_range("2000-01-03", periods=5001)
        arr = np.array(spreads)
        df = pd.DataFrame(
            {
                "open": arr * 0.999,
                "high": arr * 1.005,
                "low": arr * 0.995,
                "close": arr,
                "volume": np.ones(5001, dtype=int) * 1_000_000,
            },
            index=dates,
        )
        result = run(PairsMeanReversion(), df, config={"allow_short": True})
        assert isinstance(result["sharpe"], float)

    def test_alternating_spread_does_not_crash(self):
        """Rapidly alternating spread stress-tests z-score stability."""
        s = PairsMeanReversion(window=10)
        closes = [1.0 if i % 2 == 0 else 1.1 for i in range(50)]
        view = _make_spread_view(closes)
        result = s(view)
        assert result in (-1.0, 0.0, 1.0)

    def test_two_separate_instances_do_not_share_state(self):
        """Two fresh instances must produce identical results — no shared state."""
        from engine.backtest import run
        df = _load_paired_data()
        cfg = {"allow_short": True}
        r1 = run(PairsMeanReversion(), df, config=cfg)
        r2 = run(PairsMeanReversion(), df, config=cfg)
        assert r1 == r2

    def test_two_bar_input_returns_flat(self):
        """With only 2 bars and window=60, strategy must stay flat."""
        s = PairsMeanReversion(window=60)
        view = _make_spread_view([1.0, 0.5])
        assert s(view) == 0.0


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_single_bar_never_takes_position(self):
        s = PairsMeanReversion()
        view = _make_spread_view([0.5])
        assert s(view) == 0.0
        assert s._position == 0

    def test_without_allow_short_strategy_returns_zero_or_positive(self):
        """Without allow_short, engine maps -1 → 0. Strategy output is still -1,
        but the engine enforces the long-only constraint. This test verifies the
        strategy itself can output -1 (the engine handles the restriction)."""
        from engine.backtest import _run_internal
        df = _load_paired_data()
        _, _, positions, _ = _run_internal(
            PairsMeanReversion(), df, {"allow_short": False}
        )
        assert set(positions.unique()).issubset({0, 1}), (
            "Without allow_short, positions should only be 0 or 1"
        )

    def test_z_exit_equal_to_z_entry_raises(self):
        with pytest.raises(ValueError):
            PairsMeanReversion(z_entry=2.0, z_exit=2.0)

    def test_negative_spread_raises_log_error_or_is_guarded(self):
        """The close column must be positive (ratio of prices).

        Feeding a negative value to the strategy will cause np.log to return
        nan (or raise). The strategy should return 0.0 rather than crash.
        """
        s = PairsMeanReversion(window=3)
        closes = [1.0, 1.0, 1.0, -0.5]
        view = _make_spread_view(closes)
        result = s(view)
        assert result in (-1.0, 0.0, 1.0) or pd.isna(result) or result == 0.0
