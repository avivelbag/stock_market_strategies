#!/usr/bin/env python3
"""Generate sensitivity.json for every strategy listed in strategies.json.

Run from the repo root:
    python3 scripts/run_sensitivity.py

For each strategy this script:
  1. Loads the strategy module and reads its DEFAULT_PARAMS dict.
  2. Builds a ±20% parameter grid (5 values per param, deduplicated for integers).
  3. Calls sweep_and_score on all four synthetic datasets (seed=42, max_points=25).
  4. Writes strategies/<name>/sensitivity.json with per-dataset results including
     the new dispersion and stable_fraction fields.

Output format per dataset::

    {
      "mean_sharpe": ...,
      "std_sharpe": ...,
      "min_sharpe": ...,
      "max_sharpe": ...,
      "n_trials": ...,
      "sensitivity_score": ...,
      "dispersion": ...,
      "stable_fraction": ...
    }

dispersion equals std_sharpe (both are the population std-dev of the Sharpe
distribution across grid points).  stable_fraction is the fraction of grid
points whose Sharpe falls within 0.2 of the centre-point (default-parameter)
Sharpe.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from engine.sensitivity import build_param_grid, sweep_and_score  # noqa: E402

DATA_DIR = ROOT / "data"
STRATEGIES_DIR = ROOT / "strategies"
STRATEGIES_JSON = ROOT / "strategies.json"
DATASETS = ["trend_gbm", "mean_rev_ou", "regime_switch", "fat_tail"]
_SEED = 42


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
    """Return the first class defined in the module that has a __call__ method."""
    import inspect
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ == mod.__name__ and callable(obj):
            return obj
    raise AttributeError(f"No strategy class found in module {mod.__name__}")


def _stats_from_sharpes(sharpes: list) -> dict:
    """Compute legacy parameter_sweep-compatible stats from a list of Sharpes."""
    if not sharpes:
        return {
            "mean_sharpe": 0.0,
            "std_sharpe": 0.0,
            "min_sharpe": 0.0,
            "max_sharpe": 0.0,
            "n_trials": 0,
            "sensitivity_score": 0.0,
        }
    mean_sharpe = float(np.mean(sharpes))
    std_sharpe = float(np.std(sharpes))
    if abs(mean_sharpe) < 1e-10:
        sensitivity_score = 99.0
    else:
        sensitivity_score = min(std_sharpe / abs(mean_sharpe), 99.0)
    return {
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "min_sharpe": float(np.min(sharpes)),
        "max_sharpe": float(np.max(sharpes)),
        "n_trials": len(sharpes),
        "sensitivity_score": sensitivity_score,
    }


def run_strategy_sensitivity(strategy_dir: Path, module_name: str) -> dict:
    """Run the full sensitivity sweep for one strategy across all datasets.

    Args:
        strategy_dir: Path to the strategy directory (contains strategy.py).
        module_name: Unique module name used for importlib (avoids collisions).

    Returns:
        Dict mapping dataset name to the combined stats dict (legacy fields plus
        dispersion and stable_fraction).
    """
    mod = _load_strategy_module(strategy_dir, module_name)
    default_params = mod.DEFAULT_PARAMS
    strategy_cls = _find_strategy_class(mod)
    param_grid = build_param_grid(default_params)

    result = {}
    for dataset in DATASETS:
        df = _load_data(dataset)
        scored = sweep_and_score(strategy_cls, df, param_grid, seed=_SEED)
        stats = _stats_from_sharpes(scored["sharpes"])
        stats["dispersion"] = round(scored["dispersion"], 6)
        stats["stable_fraction"] = round(scored["stable_fraction"], 6)
        result[dataset] = {k: round(v, 6) if isinstance(v, float) else v for k, v in stats.items()}

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
            disp = stats.get("dispersion", "?")
            sf = stats.get("stable_fraction", "?")
            print(f"    {dataset}: score={score:.4f}, n={n}, dispersion={disp:.4f}, stable_fraction={sf:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
