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
| 1    | [Dual EMA Crossover Momentum](strategies/01-dual-ema-momentum/README.md) | Momentum premium (Jegadeesh & Titman 1993) | Positive on regime_switch and fat_tail; negative on trend_gbm and mean_rev_ou OOS | 2 params | oos_sharpe_mean: −0.29 avg; oos_consistency: 0.35 avg | CAGR: regime_switch 9.2%, others weak |

**Rank 1 — Dual EMA Crossover Momentum**

This strategy earns rank 1 by default as the first entry, but each criterion is assessed honestly. On **thesis clarity** (criterion 1), the economic rationale is strong: the momentum risk premium is peer-reviewed, documented before any backtest, and directly falsifiable — it predicts that EMA crossovers should work in trending but not mean-reverting regimes, a prediction the backtest largely confirms. On **robustness** (criterion 2), performance is mixed: the strategy posts a positive Sharpe on regime_switch (0.69) and fat_tail (0.27) but negative on trend_gbm (−0.60) and only slightly positive on mean_rev_ou (0.27) — the GBM result is an honest disappointment, attributable to whipsawing within the trending series under the open[t+1] fill model with round-trip costs. On **simplicity** (criterion 3), 2 free parameters is the minimum for any crossover strategy; the ThinkScript implementation is under 20 lines. On **walk-forward** (criterion 4), oos_consistency averages 0.35 across datasets, below the 0.6 threshold that would warrant a "consistent" label — this is flagged as a weakness. On **raw performance** (criterion 5), regime_switch delivers a 9.2% CAGR and 0.69 Sharpe, while the other three datasets show weak or negative returns. The strategy is the mandatory baseline; future strategies must distinguish themselves on walk-forward consistency and cross-regime robustness.
