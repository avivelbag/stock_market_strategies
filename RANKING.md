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
| 1    | [Dual EMA Crossover Momentum](strategies/01-dual-ema-momentum/README.md) | Momentum premium (Jegadeesh & Titman 1993) | In-sample positive on regime_switch (0.69), fat_tail (0.27), mean_rev_ou (0.27); negative on trend_gbm (−0.60). OOS walk-forward negative on 3 of 4 datasets; avg oos_consistency 0.35 | 2 params | oos_sharpe_mean: −0.29 avg; oos_consistency: 0.35 avg | CAGR: regime_switch 9.2%, others weak |
| 2    | [RSI Mean-Reversion (Connors RSI-2)](strategies/02-rsi-mean-reversion/README.md) | Behavioral overreaction/reversal (De Bondt & Thaler 1985; Jegadeesh 1990) | Positive on 3 of 4 datasets; negative only on trend_gbm | 3 params | oos_sharpe_mean: 0.30 avg; oos_consistency: 0.70 avg | CAGR: regime_switch 11.3%, fat_tail 8.6% |

**Rank 1 — Dual EMA Crossover Momentum**

This strategy earns rank 1 by default as the first entry, but each criterion is assessed honestly. On **thesis clarity** (criterion 1), the economic rationale is strong: the momentum risk premium is peer-reviewed, documented before any backtest, and directly falsifiable — it predicts that EMA crossovers should work in trending but not mean-reverting regimes, a prediction the backtest largely confirms. On **robustness** (criterion 2), in-sample performance is mixed: positive Sharpe on regime_switch (0.69), fat_tail (0.27), and mean_rev_ou (0.27), but negative on trend_gbm (−0.60). Crucially, in-sample positivity does not translate OOS: the walk-forward oos_sharpe_mean is negative on 3 of 4 datasets (−0.90 trend_gbm, −0.02 mean_rev_ou, −0.49 regime_switch; only fat_tail is positive at 0.24). This in-sample/OOS divergence is the primary robustness weakness. On **simplicity** (criterion 3), 2 free parameters is the minimum for any crossover strategy; the ThinkScript implementation is under 20 lines. On **walk-forward** (criterion 4), oos_consistency averages 0.35 across datasets, well below the 0.6 threshold — this is flagged as a weakness. On **raw performance** (criterion 5), regime_switch delivers a 9.2% CAGR and 0.69 Sharpe in-sample, while the other three datasets show weak or negative in-sample returns. The strategy is the mandatory baseline; future strategies must distinguish themselves on walk-forward consistency and cross-regime robustness.

**Rank 2 — RSI Mean-Reversion (Connors RSI-2)**

[**thesis**] The behavioral overreaction thesis (De Bondt & Thaler 1985; Jegadeesh 1990 short-horizon reversals) is peer-reviewed and falsifiable: RSI-2 should underperform on pure trending datasets and outperform where dislocations revert. The backtest confirms this directionally — Sharpe is negative on trend_gbm (−0.17) and positive on regime_switch (0.99) and fat_tail (0.73). Thesis criterion is met at the same level as 01-dual-ema-momentum. [**robustness**] The strategy is positive on three of four datasets versus two of four for 01-dual-ema-momentum, making it more cross-regime robust overall. The edge on mean_rev_ou is weak (Sharpe 0.14) — the OU process generates oscillations too small to exceed the RSI(2) < 10 threshold frequently enough to overcome round-trip costs. [**simplicity**] Three free parameters versus two for 01-dual-ema-momentum, a minor penalty; all three are published Connors defaults, not grid-searched. The ThinkScript is under 25 lines. [**walk-forward**] oos_consistency is 1.0 on regime_switch (five for five folds positive, oos_sharpe_mean 0.74), 0.6 on mean_rev_ou, fat_tail, and trend_gbm. The average oos_consistency across datasets is 0.70, comfortably above the 0.6 walk-forward consistency threshold, compared to 0.35 for 01-dual-ema-momentum. This is the primary differentiator that moves 02-rsi-mean-reversion ahead of 01-dual-ema-momentum in the walk-forward criterion. [**raw performance**] The regime_switch CAGR is 11.3% (vs 9.2% for 01-dual-ema-momentum), Sharpe 0.99 vs 0.69. The strategy ranks second by convention as a new entrant, but outscores the baseline on criteria 2, 4, and 5; the baseline wins only on criterion 3 (one fewer parameter).

---

## Deflated Sharpe Ratio (DSR) — anti-overfitting metric

Starting from this cycle, each strategy's `metrics.json` includes a `deflated_sharpe` field computed
via `engine.antioverfitting.run_with_dsr` with `n_trials=1`. The DSR (Bailey & López de Prado 2014)
is the probability that the observed Sharpe ratio is above zero after correcting for finite-sample bias
and non-normality in the return distribution. With `n_trials=1`, no multiple-testing penalty is applied —
this is the conservative lower bound appropriate for a single prior-specified strategy where no parameter
search was performed.

**What DSR adds to the ranking criteria.** The standard Sharpe ratio answers "how large is the risk-adjusted
return?" but is silent on whether that return could have arisen by chance in a single-sample backtest.
DSR answers the complementary question: "given the length, skewness, and kurtosis of the return series,
what is the probability the observed edge is real?" A strategy with a high annualised Sharpe on a short
or fat-tailed series may still carry a low DSR, signalling that the observed Sharpe is consistent with
a lucky draw rather than a genuine edge.

**Effect on current ranking.** DSR does not change the ordering of the two existing strategies.
Both strategies use published, prior-set parameters with no grid search, making `n_trials=1` the correct
input. On regime_switch — the strongest dataset for both — RSI Mean-Reversion has a higher DSR (0.975)
than Dual EMA (0.914), consistent with its higher in-sample Sharpe on a longer effective signal window.
On trend_gbm, both strategies have low DSR (0.115 for EMA, 0.371 for RSI), correctly reflecting that
the observed Sharpe is close to or below zero and therefore carries little statistical confidence.
The DSR values reinforce the existing ranking: RSI Mean-Reversion dominates on the datasets where it
has a real edge, while Dual EMA's regime_switch in-sample edge does not survive OOS evaluation.
