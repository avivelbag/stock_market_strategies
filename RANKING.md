# Strategy Ranking

## Ranking criteria

Strategies are scored across five dimensions. Each criterion carries equal weight
unless otherwise noted for a specific cycle.

| # | Criterion | What it measures |
|---|-----------|-----------------|
| 1 | **Thesis** | Does the strategy exploit a coherent, documented market inefficiency? Is the economic rationale clear and falsifiable? Strategies backed by plausible first-principles or peer-reviewed research score higher. |
| 2 | **Robustness** | Does performance hold across multiple synthetic regimes (trending, mean-reverting, fat-tailed) and parameter perturbations? Strategies that require fine-tuned parameters to survive score lower. |
| 3 | **Simplicity** | Fewer parameters, shorter signal lookback, and fewer lines of TypeScript. Occam's razor: a simpler strategy that performs similarly is always preferred — it is less likely to be overfit. |
| 4 | **Walk-forward** | Out-of-sample evaluation via `walk_forward_backtest` (n_splits=5 anchored folds, fixed params — no per-fold re-fitting). Sub-criteria: **oos_sharpe_mean** (primary) and **oos_consistency** (fraction of folds with positive Sharpe). Strategies with `oos_consistency < 0.6` are annotated as "walk-forward inconsistent". In-sample metrics are ignored for ranking. |
| 5 | **Raw performance** | Sharpe, CAGR, and max drawdown on the full backtest window. Used as a tiebreaker when criteria 1–4 are close. |

---

## Leaderboard

| Rank | Strategy | Thesis | Robustness | Simplicity | Walk-fwd Sharpe | Raw CAGR |
|------|----------|--------|------------|------------|-----------------|----------|
| —    | *(none yet)* | — | — | — | — | — |
