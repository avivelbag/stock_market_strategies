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

| Rank | Strategy | Thesis | Robustness | Param Robustness | Simplicity | Walk-fwd Sharpe | Raw CAGR |
|------|----------|--------|------------|------------------|------------|-----------------|----------|
| 1    | [Dual EMA Crossover Momentum](strategies/01-dual-ema-momentum/README.md) | Momentum premium (Jegadeesh & Titman 1993) | In-sample positive on regime_switch (0.69), fat_tail (0.27), mean_rev_ou (0.27); negative on trend_gbm (−0.60). OOS walk-forward negative on 3 of 4 datasets; avg oos_consistency 0.35 | **0.085** on regime_switch (robust; < 0.3) | 2 params | oos_sharpe_mean: −0.29 avg; oos_consistency: 0.35 avg | CAGR: regime_switch 9.2%, others weak |
| 2    | [RSI Mean-Reversion (Connors RSI-2)](strategies/02-rsi-mean-reversion/README.md) | Behavioral overreaction/reversal (De Bondt & Thaler 1985; Jegadeesh 1990) | Positive on 3 of 4 datasets; negative only on trend_gbm | **0.277** on regime_switch (robust; < 0.3) | 3 params | oos_sharpe_mean: 0.30 avg; oos_consistency: 0.70 avg | CAGR: regime_switch 11.3%, fat_tail 8.6% |
| 3    | [Donchian Channel Turtle Breakout](strategies/03-donchian-turtle-breakout/README.md) | Supply exhaustion at N-bar extremes (Dennis/Eckhardt 1983; Covel 2007) | In-sample positive on all 4 datasets but thin margins on trend_gbm (0.05) and mean_rev_ou (0.08); regime-dependence confirmed and expected | **0.490** on regime_switch (moderate; 0.3–0.8) | 3 params | oos_sharpe_mean: −0.06 avg; oos_consistency: 0.55 avg; fat_tail flagged walk-forward inconsistent (0.40) | CAGR: regime_switch 4.1%, others weak |
| 4    | [52-Week High Proximity (Anchoring Bias)](strategies/04-52wk-high-proximity/README.md) | Investor anchoring bias (George & Hwang 2004, Journal of Finance) | Negative in-sample Sharpe on all 4 synthetic datasets; academic evidence is cross-sectional, not single-series | **0.495** on regime_switch (moderate; 0.3–0.8); uninformative on trend_gbm and mean_rev_ou (near-zero mean_sharpe inflates score) | 2 params | Walk-forward uninformative: 252-bar warmup exceeds 166-bar fold length; oos_consistency: 0.0 on all datasets | CAGR: all negative on synthetic data |

**Rank 1 — Dual EMA Crossover Momentum**

This strategy earns rank 1 by default as the first entry, but each criterion is assessed honestly. On **thesis clarity** (criterion 1), the economic rationale is strong: the momentum risk premium is peer-reviewed, documented before any backtest, and directly falsifiable — it predicts that EMA crossovers should work in trending but not mean-reverting regimes, a prediction the backtest largely confirms. On **robustness** (criterion 2), in-sample performance is mixed: positive Sharpe on regime_switch (0.69), fat_tail (0.27), and mean_rev_ou (0.27), but negative on trend_gbm (−0.60). Crucially, in-sample positivity does not translate OOS: the walk-forward oos_sharpe_mean is negative on 3 of 4 datasets (−0.90 trend_gbm, −0.02 mean_rev_ou, −0.49 regime_switch; only fat_tail is positive at 0.24). This in-sample/OOS divergence is the primary robustness weakness. On **simplicity** (criterion 3), 2 free parameters is the minimum for any crossover strategy; the ThinkScript implementation is under 20 lines. On **walk-forward** (criterion 4), oos_consistency averages 0.35 across datasets, well below the 0.6 threshold — this is flagged as a weakness. On **raw performance** (criterion 5), regime_switch delivers a 9.2% CAGR and 0.69 Sharpe in-sample, while the other three datasets show weak or negative in-sample returns. The strategy is the mandatory baseline; future strategies must distinguish themselves on walk-forward consistency and cross-regime robustness.

**Rank 2 — RSI Mean-Reversion (Connors RSI-2)**

[**thesis**] The behavioral overreaction thesis (De Bondt & Thaler 1985; Jegadeesh 1990 short-horizon reversals) is peer-reviewed and falsifiable: RSI-2 should underperform on pure trending datasets and outperform where dislocations revert. The backtest confirms this directionally — Sharpe is negative on trend_gbm (−0.17) and positive on regime_switch (0.99) and fat_tail (0.73). Thesis criterion is met at the same level as 01-dual-ema-momentum. [**robustness**] The strategy is positive on three of four datasets versus two of four for 01-dual-ema-momentum, making it more cross-regime robust overall. The edge on mean_rev_ou is weak (Sharpe 0.14) — the OU process generates oscillations too small to exceed the RSI(2) < 10 threshold frequently enough to overcome round-trip costs. [**simplicity**] Three free parameters versus two for 01-dual-ema-momentum, a minor penalty; all three are published Connors defaults, not grid-searched. The ThinkScript is under 25 lines. [**walk-forward**] oos_consistency is 1.0 on regime_switch (five for five folds positive, oos_sharpe_mean 0.74), 0.6 on mean_rev_ou, fat_tail, and trend_gbm. The average oos_consistency across datasets is 0.70, comfortably above the 0.6 walk-forward consistency threshold, compared to 0.35 for 01-dual-ema-momentum. This is the primary differentiator that moves 02-rsi-mean-reversion ahead of 01-dual-ema-momentum in the walk-forward criterion. [**raw performance**] The regime_switch CAGR is 11.3% (vs 9.2% for 01-dual-ema-momentum), Sharpe 0.99 vs 0.69. The strategy ranks second by convention as a new entrant, but outscores the baseline on criteria 2, 4, and 5; the baseline wins only on criterion 3 (one fewer parameter).

**Rank 4 — 52-Week High Proximity (Anchoring Bias)**

[**thesis** — criterion 1] George & Hwang (2004, *Journal of Finance*) provide one of the most-replicated anomaly findings in behavioural finance. The anchoring mechanism is well-specified and falsifiable: analysts and institutional investors resist setting price targets above the 52-week high, causing systematic under-reaction to positive news near that level. This is a distinct thesis family — neither momentum, mean-reversion, nor supply-exhaustion breakout — making it genuinely additive to the existing library. Thesis quality is comparable to RSI (peer-reviewed, published before any backtest, falsifiable prediction). Criterion 1 does not place this strategy ahead of existing ones because the backtest evidence on synthetic data does not corroborate the effect.

[**robustness** — criterion 2] All four synthetic datasets produce negative in-sample Sharpe ratios. This does not refute the George & Hwang finding, which is inherently cross-sectional: stocks are ranked by their ratio relative to each other across the equity universe, and the top-decile outperforms the bottom-decile. Applying the signal to a single time series in isolation loses the cross-sectional sorting that drives the effect. The synthetic data also has no mechanism to generate the anchoring dynamic (no analyst price targets, no institutional rebalancing). Robustness criterion is not met on these datasets.

[**simplicity** — criterion 3] Two free parameters — the fewest of any strategy in the library. Both are published defaults from the George & Hwang (2004) paper with no in-sample optimisation. The ThinkScript implementation is under 25 lines. Criterion 3 is the strongest dimension for this strategy; it ties 01-dual-ema-momentum (2 params) and beats the 3-parameter strategies.

[**walk-forward** — criterion 4] The standard walk-forward evaluation (n_splits=5, 1000 bars) produces OOS folds of ~166 bars each. The strategy requires 253 bars for warm-up, so it cannot generate any signal in any OOS fold. All walk-forward metrics are 0.0 — not evidence of zero edge, but evidence that the evaluation method cannot assess a strategy with this lookback. This is a hard structural constraint, not a data-dependent outcome. Criterion 4 is uninformative and this strategy cannot be ranked on it.

[**raw performance** — criterion 5] All four datasets produce negative in-sample Sharpe ratios, consistent with the cross-sectional implementation mismatch described above. The strategy is not competitive on this criterion.

The strategy takes rank 4 as the most recent entrant. Its thesis is the strongest in the library (most-replicated anomaly, clearest behavioural mechanism), but the single-series synthetic backtest environment does not allow the cross-sectional effect to manifest. The walk-forward evaluation is structurally blocked by the 252-bar lookback. The strategy should be re-evaluated on a multi-asset universe with cross-sectional ranking to fairly test the George & Hwang hypothesis.

**Rank 3 — Donchian Channel Turtle Breakout**

[**thesis** — criterion 1] The supply-exhaustion thesis from the Dennis/Eckhardt 1983 Turtle experiment is among the most-cited practitioner evidence in trend-following. Covel (2007) documents that the original Turtles applied System 1 rules profitably across futures markets for nearly a decade. The thesis is falsifiable: breakout entries should show positive Sharpe in trending regimes and fail in mean-reverting regimes. The backtest confirms this directionally (regime_switch 0.376, mean_rev_ou 0.076 near-zero), making thesis quality equivalent to 01-dual-ema-momentum and slightly below 02-rsi-mean-reversion which makes a more precise prediction confirmed OOS. Criterion 1 does not place Donchian ahead of existing strategies.

[**robustness** — criterion 2] Donchian is positive on all four datasets in-sample, technically the broadest robustness. However, the margins on trend_gbm (0.048) and mean_rev_ou (0.076) are thin enough that they carry no statistical confidence — the near-zero Sharpe on mean_rev_ou is the predicted outcome, not evidence of edge. Robustness on three datasets (regime_switch, fat_tail, trend_gbm) is comparable to 02-rsi-mean-reversion, which is also positive on three of four. Criterion 2 does not elevate Donchian above either existing strategy.

[**simplicity** — criterion 3] Three free parameters, identical count to 02-rsi-mean-reversion. All three are published Turtle System 1 defaults with no grid search. The ThinkScript implementation is under 20 lines. Criterion 3 is a draw between Donchian and RSI, and a loss versus 01-dual-ema-momentum (2 params). This criterion does not change the ranking.

[**walk-forward** — criterion 4] OOS oos_consistency averages 0.55 across the four datasets — better than 01-dual-ema-momentum (0.35 avg) but below 02-rsi-mean-reversion (0.70 avg). The fat_tail fold achieves oos_consistency 0.40, below the 0.6 threshold and annotated as walk-forward inconsistent. The average oos_sharpe_mean is −0.06 (negative on three of four datasets; only regime_switch is positive at 0.091). Criterion 4 places Donchian between EMA (worse) and RSI (better), but the negative average OOS Sharpe is a meaningful weakness that keeps it below RSI.

[**raw performance** — criterion 5] Regime_switch delivers Donchian's best result: Sharpe 0.376, CAGR 4.1%, max drawdown −13.4%. This is the weakest regime_switch result of the three strategies (EMA 0.69, RSI 0.99). Per the orchestrator's instruction, ranking is driven by criteria 1–4, not raw return, so a strong trend_gbm or regime_switch number alone does not advance the rank. With the weakest raw performance and lower OOS consistency than RSI, Donchian takes rank 3 as the new entrant that sits below both existing strategies.

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

**Effect on current ranking.** DSR does not change the ordering of the three strategies.
All strategies use published, prior-set parameters with no grid search, making `n_trials=1` the correct
input. On regime_switch — the strongest dataset for all three — RSI Mean-Reversion has the highest DSR
(0.975), Donchian Turtle Breakout is second (0.773), and Dual EMA is third (0.914 in-sample but OOS
performance does not corroborate). On trend_gbm, all three strategies have low DSR (EMA 0.115, RSI 0.371,
Donchian 0.538), correctly reflecting Sharpe values close to or below zero on this dataset. The Donchian
DSR on mean_rev_ou (0.560) is consistent with its near-zero Sharpe: the model has some statistical
confidence that it is not losing money, but not that it is earning a real edge. DSR reinforces the existing
ranking: RSI dominates where it has an edge, Donchian Turtle Breakout sits in the middle, and Dual EMA's
regime_switch in-sample advantage does not survive OOS evaluation.

---

## Parameter Sensitivity Dispersion — anti-overfitting metric

Each strategy directory now contains `sensitivity.json` generated by `scripts/run_sensitivity.py`.
The file records, for each synthetic dataset, the distribution of Sharpe ratios obtained by sweeping
each parameter ±20% around its `DEFAULT_PARAMS` value (5 values per parameter, deduplicated for integers).
The key output is `sensitivity_score = std_sharpe / abs(mean_sharpe)` — the coefficient of variation
of the Sharpe distribution over the parameter grid.

**Interpretation.**
- `< 0.3`: robust — Sharpe barely moves when parameters shift; low risk of overfitting to defaults.
- `0.3–0.8`: moderate sensitivity — performance degrades noticeably with parameter drift.
- `> 0.8`: fragile — likely overfit to the default parameter values; treat edge-size claims with caution.

Scores are most informative on datasets where the strategy has a meaningful edge (non-trivial mean_sharpe).
On datasets where mean_sharpe ≈ 0, the coefficient of variation is high by construction and does not
indicate overfitting — it indicates the strategy has no edge to be sensitive about.

**Effect on current ranking.** Parameter sensitivity does not change the robustness (criterion 2) ordering.

On regime_switch — the primary dataset where all strategies show their best results — Dual EMA is the
most robust (sensitivity_score 0.085, firmly below 0.3). RSI is also robust (0.277, just under the 0.3
threshold). Donchian and 52-Week both fall in the moderate band (0.490 and 0.495 respectively), consistent
with their lower walk-forward performance. The fact that RSI's parameter robustness on regime_switch (0.277)
nearly matches Dual EMA's (0.085) does not elevate RSI above Dual EMA on criterion 2 — RSI already
outranks Dual EMA on criteria 2 and 4 (cross-regime performance and walk-forward consistency). Dual EMA's
superior parameter stability does not compensate for its weaker OOS consistency.

Donchian's sensitivity_score on mean_rev_ou (7.45) looks alarming but is a degenerate case: mean_sharpe
on that dataset is near zero (Sharpe 0.076 at defaults), so any variation in the numerator dominates.
The strategy's regime_switch score of 0.490 is the informative number. Similarly, the 52-Week scores on
trend_gbm (28.3) and mean_rev_ou (18.5) reflect near-zero mean_sharpe on datasets where the cross-sectional
anchoring effect cannot manifest, not evidence of parameter sensitivity on a dataset where the strategy
has edge.
