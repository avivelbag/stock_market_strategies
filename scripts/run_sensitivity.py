#!/usr/bin/env python3
"""Generate sensitivity.json for every strategy listed in strategies.json.

Run from the repo root:
    python3 scripts/run_sensitivity.py

For each strategy this script:
  1. Loads the strategy module and reads its DEFAULT_PARAMS dict.
  2. Builds a ±20% parameter grid (5 values per param, deduplicated for integers).
  3. Runs parameter_sweep on all four synthetic datasets.
  4. Writes strategies/<name>/sensitivity.json with the per-dataset results.

The output format is::

    {
      "trend_gbm": {"mean_sharpe": ..., "std_sharpe": ..., ..., "sensitivity_score": ...},
      "mean_rev_ou": {...},
      "regime_switch": {...},
      "fat_tail": {...}
    }
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from engine.sensitivity import build_param_grid, parameter_sweep  # noqa: E402

DATA_DIR = ROOT / "data"
STRATEGIES_DIR = ROOT / "strategies"
STRATEGIES_JSON = ROOT / "strategies.json"
DATASETS = ["trend_gbm", "mean_rev_ou", "regime_switch", "fat_tail"]


def _load_strategy_module(strategy_dir: Path, module_name: str):
    """Load a strategy module from its directory by path."""
    strategy_file = strategy_dir / "strategy.py"
    spec = importlib.util.spec_from_file_location(module_name, strategy_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_data(name: str) -> pd.DataFrame:
    """Load a synthetic dataset CSV into a DataFrame with DatetimeIndex."""
    return pd.read_csv(DATA_DIR / f"{name}.csv", index_col=0, parse_dates=True)


def _find_strategy_class(mod):
    """Return the first class defined in the module that has a __call__ method.

    Assumes the strategy module exports exactly one strategy class (the
    convention followed by all strategies in this repository).
    """
    import inspect
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ == mod.__name__ and callable(obj):
            return obj
    raise AttributeError(f"No strategy class found in module {mod.__name__}")


def run_strategy_sensitivity(strategy_dir: Path, module_name: str) -> dict:
    """Run the full sensitivity sweep for one strategy across all datasets.

    Args:
        strategy_dir: Path to the strategy directory (contains strategy.py).
        module_name: Unique module name used for importlib (avoids collisions).

    Returns:
        Dict mapping dataset name to the parameter_sweep result dict.
    """
    mod = _load_strategy_module(strategy_dir, module_name)
    default_params = mod.DEFAULT_PARAMS
    strategy_cls = _find_strategy_class(mod)
    param_grid = build_param_grid(default_params)

    def factory(params):
        return strategy_cls(**params)

    result = {}
    for dataset in DATASETS:
        df = _load_data(dataset)
        sweep = parameter_sweep(factory, param_grid, df)
        result[dataset] = {k: round(v, 6) if isinstance(v, float) else v for k, v in sweep.items()}

    return result


def main():
    with open(STRATEGIES_JSON) as f:
        registry = json.load(f)

    for entry in registry:
        name = entry["name"]
        strategy_dir = STRATEGIES_DIR / name
        if not strategy_dir.is_dir():
            print(f"Skipping {name}: directory not found")
            continue

        print(f"Running sensitivity sweep for {name} ...")
        module_name = name.replace("-", "_")
        sensitivity = run_strategy_sensitivity(strategy_dir, module_name)
        out = strategy_dir / "sensitivity.json"
        out.write_text(json.dumps(sensitivity, indent=2) + "\n")
        print(f"  Written: {out}")
        for dataset, stats in sensitivity.items():
            score = stats.get("sensitivity_score", "?")
            n = stats.get("n_trials", "?")
            print(f"    {dataset}: score={score:.4f}, n_trials={n}")

    print("Done.")


if __name__ == "__main__":
    main()
