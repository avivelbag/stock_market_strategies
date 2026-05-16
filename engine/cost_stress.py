"""Transaction-cost stress sweep over a commission × slippage grid."""

import importlib.util
import inspect
import itertools
import json
import sys
from pathlib import Path

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
    commission_bps_range: list | None = None,
    slippage_bps_range: list | None = None,
    config: dict | None = None,
) -> dict:
    """Run the backtest over a grid of commission × slippage assumptions."""
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
) -> int | str:
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
    commission_bps_range: list | None = None,
    slippage_bps_range: list | None = None,
) -> tuple:
    """Return (breakeven_commission_bps, breakeven_slippage_bps) for a sweep result."""
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


def _aggregate_breakeven(breakevens: list) -> int | str:
    if any(b == ALREADY_NEGATIVE_SENTINEL for b in breakevens):
        return ALREADY_NEGATIVE_SENTINEL
    numeric = [b for b in breakevens if isinstance(b, (int, float))]
    if numeric:
        return min(numeric)
    return ABOVE_MAX_GRID_SENTINEL


def _load_strategy_module(strategy_dir: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, strategy_dir / "strategy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_strategy_class(mod):
    for _, obj in sorted(inspect.getmembers(mod, inspect.isclass)):
        if obj.__module__ == mod.__name__:
            return obj
    raise AttributeError(f"No strategy class defined in module '{mod.__name__}'")


def run_all_strategies(
    strategies_json_path: Path,
    strategies_dir: Path,
    data_dir: Path,
    commission_bps_range: list | None = None,
    slippage_bps_range: list | None = None,
) -> None:
    """Sweep all registered strategies and write cost_stress.json to each strategy dir."""
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
