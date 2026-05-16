# Stock Strategies — Backtest Library

A reproducible Python backtest framework for developing, comparing, and ranking
algorithmic trading strategies against synthetic price datasets.

## Repository layout

```
engine/             Python backtest engine and metrics
  backtest.py       Core runner: run(strategy_fn, prices_df, config) → metrics dict
  metrics.py        Standalone metric functions (CAGR, Sharpe, Sortino, …)
data/               Committed synthetic OHLCV CSVs + generator script
  generate_synthetic.py   Regenerate CSVs from seeds (auditable provenance)
  README.md         Seed and statistical properties for each dataset
  trend_gbm.csv     GBM, positive drift
  mean_rev_ou.csv   Ornstein-Uhlenbeck mean-reversion
  regime_switch.csv Alternating trend / mean-reversion regimes
  fat_tail.csv      Student-t(3) fat-tailed innovations
strategies/         One directory per strategy (created by future swarm cycles)
strategies.json     Registry of implemented strategies (empty array initially)
tests/              pytest suite
  test_engine.py    Engine correctness, look-ahead guard, reproducibility
  test_structure.py Structural gate: strategies.json schema and file presence
RANKING.md          Ranking criteria and leaderboard
```

## Running tests

```bash
python3 -m pytest tests/ -x --tb=short -q
```

All tests must pass before submitting a strategy. The merge harness re-runs the
suite against `main` after every merge.

## Running the backtest entrypoint

```bash
python -m engine.backtest
```

When no strategy module is specified, it prints usage and exits cleanly. To run
a strategy:

```bash
python -m engine.backtest my_strategy.my_function data/trend_gbm.csv
```

## Return convention

Signals computed from data up to close[t] are filled at open[t+1]. Each bar's
return is **open[t+1]→close[t+1]** (intraday only). Consequences:

- **Overnight gap returns are not captured.** The close[t]→open[t+1] move is
  silently dropped for every bar. Strategies with a significant edge in overnight
  gaps (e.g., earnings holds, pre-market momentum) will appear weaker than their
  true P&L.
- **Multi-bar holds compound intraday legs only.** A three-day hold earns three
  separate open→close returns, not the end-to-end close-to-close return.

This is a deliberate design choice to keep the fill model simple and auditable.
All benchmark comparisons must use this same convention; cross-convention
comparisons are not meaningful.

## Implementing a strategy

A strategy is a Python callable with this signature:

```python
def my_strategy(view) -> float:
    """Return positive (long), negative (short), or zero (flat)."""
    close = view['close']
    # view is sliced to the current bar — cannot access future data
    return 1.0 if close.iloc[-1] > close.mean() else 0.0
```

Pass it to the engine:

```python
from engine.backtest import run
import pandas as pd

prices = pd.read_csv("data/trend_gbm.csv", index_col=0, parse_dates=True)
config = {"commission_bps": 5, "slippage_bps": 5, "allow_short": False}
metrics = run(my_strategy, prices, config)
print(metrics)
```

## Adding a strategy to the registry

1. Create `strategies/<name>/` with `README.md`, `metrics.json`, and `strategy.ts`.
2. Append an entry to `strategies.json`: `{"name": "<name>"}`.
3. Run the full test suite to verify the structural gate passes.
