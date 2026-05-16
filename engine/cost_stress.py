"""Transaction-cost stress sweep: identify cost-fragile strategies.

For each strategy, runs the full backtest over a 5×4 grid of commission ×
slippage values. Returns per-cell Sharpe ratios and two breakeven scalars:
the smallest cost at which the strategy first loses its edge. Strategies that
are already unprofitable at zero cost receive the "already_negative_at_zero_cost"
sentinel; strategies that remain profitable across the full grid receive ">max_grid".

CLI usage (from repo root):
    python -m engine.cost_stress
"""

import importlib.util
import inspect
import itertools
import json
import sys
from pathlib import Path
from typing import Union

import pandas as pd

from engine.backtest import run

DEFAULT_COMMISSION_BPS: list = [0, 2, 5, 10, 20]
DEFAULT_SLIPPAGE_BPS: list = [0, 2, 5, 10]
BASE_COMMISSION_BPS: int = 5
BASE_SLIPPAGE_BPS: int = 5

ALREADY_NEGATIVE_SENTINEL: str = "already_negative_at_zero_cost"
ABOVE_MAX_GRID_SENTINEL: str = ">max_grid"


def cost_stress_sweep(
    strategy_cls,
    params: dict,
    dataset: pd.DataFrame,
    commission_bps_range: list = None,
    slippage_bps_range: list = None,
    config: dict = None,
) -> dict:
    """Run a full backtest over a grid of commission × slippage cost assumptions.

    Instantiates a fresh strategy instance for each grid cell to prevent state
    leakage across cells in stateful strategies (RSI, Donchian, 52-Week High).
    All other config parameters are held constant; only commission_bps and
    slippage_bps vary.

    Args:
        strategy_cls: Strategy class that accepts **params on construction.
        params: Constructor keyword arguments passed to strategy_cls per cell.
        dataset: OHLCV DataFrame with DatetimeIndex. Must have ≥2 rows.
        commission_bps_range: One-way commission levels in basis points.
            Defaults to [0, 2, 5, 10, 20].
        slippage_bps_range: Half-spread slippage levels in basis points.
            Defaults to [0, 2, 5, 10].
        config: Additional backtest config overrides. commission_bps and
            slippage_bps in this dict are overwritten per grid cell.

    Returns:
        Dict mapping "(commission_bps, slippage_bps)" string keys to Sharpe
        ratios (float, rounded to 6 decimal places). Default grid: 20 entries.
    """
    if commission_bps_range is None:
        commission_bps_range = DEFAULT_COMMISSION_BPS
    if slippage_bps_range is None:
        slippage_bps_range = DEFAULT_SLIPPAGE_BPS

    base_config = dict(config or {})
    results = {}

    for c_bps, s_bps in itertools.product(commission_bps_range, slippage_bps_range):
        cfg = {**base_config, "commission_bps": c_bps, "slippage_bps": s_bps}
        strategy = strategy_cls(**params)
        sharpe = run(strategy, dataset, cfg)["sharpe"]
        results[f"({c_bps}, {s_bps})"] = round(float(sharpe), 6)

    return results


def _first_negative_or_sentinel(
    sweep_results: dict,
    sweep_range: list,
    held_value: int,
    sweep_axis: str,
) -> Union[int, str]:
    """Find the first cost level at which Sharpe goes negative.

    Checks costs in ascending order. The held_value is the fixed cost on the
    orthogonal axis (BASE_SLIPPAGE_BPS when sweeping commission, vice versa).

    Returns ALREADY_NEGATIVE_SENTINEL when Sharpe is already negative at the
    lowest cost in sweep_range (inclusive of zero cost when 0 is in the range).
    Returns ABOVE_MAX_GRID_SENTINEL when Sharpe stays non-negative through the
    entire sweep_range.

    Args:
        sweep_results: Output dict from cost_stress_sweep.
        sweep_range: Sorted ascending list of cost levels to iterate.
        held_value: Fixed cost on the other axis for this sweep.
        sweep_axis: "commission" (sweep commission, hold slippage) or
            "slippage" (sweep slippage, hold commission).

    Returns:
        First cost level where Sharpe < 0, or a sentinel string.
    """

    def _key(level: int) -> str:
        if sweep_axis == "commission":
            return f"({level}, {held_value})"
        return f"({held_value}, {level})"

    sorted_range = sorted(sweep_range)

    if sweep_results.get(_key(sorted_range[0]), 0.0) < 0:
        return ALREADY_NEGATIVE_SENTINEL

    for level in sorted_range:
        if sweep_results.get(_key(level), 0.0) < 0:
            return level

    return ABOVE_MAX_GRID_SENTINEL


def compute_breakevens(
    sweep_results: dict,
    commission_bps_range: list = None,
    slippage_bps_range: list = None,
) -> tuple:
    """Compute cost breakeven thresholds for both cost axes.

    Breakeven commission: smallest commission (with slippage held at
    BASE_SLIPPAGE_BPS=5) at which Sharpe first becomes negative.
    Breakeven slippage: smallest slippage (commission held at
    BASE_COMMISSION_BPS=5) at which Sharpe first becomes negative.

    Edge cases produce sentinel strings rather than numeric values:
    - "already_negative_at_zero_cost": strategy loses money even with no costs.
    - ">max_grid": strategy stays profitable across the full grid.

    Args:
        sweep_results: Output of cost_stress_sweep (20 cells for default grid).
        commission_bps_range: Commission values used in the sweep (sorted internally).
        slippage_bps_range: Slippage values used in the sweep (sorted internally).

    Returns:
        (breakeven_commission, breakeven_slippage) — each is an int or sentinel.
    """
    if commission_bps_range is None:
        commission_bps_range = DEFAULT_COMMISSION_BPS
    if slippage_bps_range is None:
        slippage_bps_range = DEFAULT_SLIPPAGE_BPS

    b_comm = _first_negative_or_sentinel(
        sweep_results, commission_bps_range, BASE_SLIPPAGE_BPS, "commission"
    )
    b_slip = _first_negative_or_sentinel(
        sweep_results, slippage_bps_range, BASE_COMMISSION_BPS, "slippage"
    )
    return b_comm, b_slip


def _aggregate_breakeven(breakevens: list) -> Union[int, str]:
    """Aggregate per-dataset breakeven values into a single worst-case summary.

    Priority (most fragile to least):
    1. "already_negative_at_zero_cost" — any dataset has this → return it.
    2. Minimum numeric breakeven — take the most fragile dataset.
    3. ">max_grid" — all datasets retain positive Sharpe across full grid.

    Args:
        breakevens: List of breakeven values (ints or sentinel strings) from
            multiple datasets for the same strategy.

    Returns:
        Single breakeven value representing the worst case across datasets.
    """
    if any(b == ALREADY_NEGATIVE_SENTINEL for b in breakevens):
        return ALREADY_NEGATIVE_SENTINEL
    numeric = [b for b in breakevens if isinstance(b, (int, float))]
    if numeric:
        return min(numeric)
    return ABOVE_MAX_GRID_SENTINEL


def _load_strategy_module(strategy_dir: Path, module_name: str):
    """Load a strategy Python module from its file path via importlib.

    Args:
        strategy_dir: Directory containing strategy.py.
        module_name: Name to assign the loaded module (must be a valid identifier).

    Returns:
        Loaded module object with attributes accessible via dot notation.
    """
    spec = importlib.util.spec_from_file_location(
        module_name, strategy_dir / "strategy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_strategy_class(mod):
    """Extract the primary strategy class from a dynamically loaded module.

    Identifies classes defined in the module itself (not imported from
    dependencies) by checking that obj.__module__ matches the module's __name__.
    Returns the first such class found alphabetically.

    Args:
        mod: Module loaded via _load_strategy_module.

    Returns:
        Strategy class.

    Raises:
        AttributeError: If no class defined in this module is found.
    """
    for _, obj in sorted(inspect.getmembers(mod, inspect.isclass)):
        if obj.__module__ == mod.__name__:
            return obj
    raise AttributeError(f"No strategy class defined in module '{mod.__name__}'")


def run_all_strategies(
    strategies_json_path: Path,
    strategies_dir: Path,
    data_dir: Path,
    commission_bps_range: list = None,
    slippage_bps_range: list = None,
) -> None:
    """Run cost stress sweep for all strategies and write cost_stress.json files.

    Reads strategies.json to discover registered strategies, loads each
    strategy module, sweeps over all configured datasets, then writes
    cost_stress.json to each strategy's directory.

    Output JSON structure:
        {
          "<dataset_name>": {
            "(c_bps, s_bps)": <sharpe>,  # 20 grid cells
            ...,
            "cost_breakeven_commission_bps": <int or sentinel>,
            "cost_breakeven_slippage_bps": <int or sentinel>
          },
          ...,
          "cost_breakeven_commission_bps": <worst-case across datasets>,
          "cost_breakeven_slippage_bps": <worst-case across datasets>
        }

    Args:
        strategies_json_path: Path to strategies.json registry file.
        strategies_dir: Root directory containing <strategy-name>/ subdirectories.
        data_dir: Directory containing <dataset_name>.csv files.
        commission_bps_range: Commission bps values to sweep.
        slippage_bps_range: Slippage bps values to sweep.
    """
    if commission_bps_range is None:
        commission_bps_range = DEFAULT_COMMISSION_BPS
    if slippage_bps_range is None:
        slippage_bps_range = DEFAULT_SLIPPAGE_BPS

    with open(strategies_json_path) as f:
        registry = json.load(f)

    for entry in registry:
        strategy_name = entry["name"]
        strategy_dir = strategies_dir / strategy_name
        safe_name = f"strat_{strategy_name.replace('-', '_')}"
        mod = _load_strategy_module(strategy_dir, safe_name)
        strategy_cls = _get_strategy_class(mod)
        params = mod.DEFAULT_PARAMS

        all_comm_breakevens = []
        all_slip_breakevens = []
        output = {}

        for dataset_name in entry.get("datasets", []):
            df = pd.read_csv(
                data_dir / f"{dataset_name}.csv", index_col=0, parse_dates=True
            )
            sweep = cost_stress_sweep(
                strategy_cls,
                params,
                df,
                commission_bps_range=commission_bps_range,
                slippage_bps_range=slippage_bps_range,
            )
            b_comm, b_slip = compute_breakevens(
                sweep, commission_bps_range, slippage_bps_range
            )
            dataset_result = dict(sweep)
            dataset_result["cost_breakeven_commission_bps"] = b_comm
            dataset_result["cost_breakeven_slippage_bps"] = b_slip
            output[dataset_name] = dataset_result

            all_comm_breakevens.append(b_comm)
            all_slip_breakevens.append(b_slip)

        output["cost_breakeven_commission_bps"] = _aggregate_breakeven(
            all_comm_breakevens
        )
        output["cost_breakeven_slippage_bps"] = _aggregate_breakeven(
            all_slip_breakevens
        )

        out_path = strategy_dir / "cost_stress.json"
        out_path.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Written: {out_path}")


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    run_all_strategies(
        strategies_json_path=ROOT / "strategies.json",
        strategies_dir=ROOT / "strategies",
        data_dir=ROOT / "data",
    )
    print("Cost stress sweep complete.")
    sys.exit(0)
