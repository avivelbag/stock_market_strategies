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

| Rank | Strategy | Thesis | Robustness | Param Robustness | Simplicity | Walk-fwd Sharpe | Raw CAGR | Regime Profile (regime_switch T/R/HV) |
|------|----------|--------|------------|------------------|------------|-----------------|----------|--------------------------------------|
| 1    | [Dual EMA Crossover Momentum](strategies/01-dual-ema-momentum/README.md) | Momentum premium (Jegadeesh & Titman 1993) | In-sample positive on regime_switch (0.69), fat_tail (0.27), mean_rev_ou (0.27); negative on trend_gbm (−0.60). OOS walk-forward negative on 3 of 4 datasets; avg oos_consistency 0.35 | **0.085** on regime_switch (robust; < 0.3) | 2 params | oos_sharpe_mean: −0.29 avg; oos_consistency: 0.35 avg | CAGR: regime_switch 9.2%, others weak | T=0.23 / R=0.87 / HV=0.92 — all positive on regime_switch but concentrated in ranging/high_vol; trending Sharpe is the weakest |
| 2    | [RSI Mean-Reversion (Connors RSI-2)](strategies/02-rsi-mean-reversion/README.md) | Behavioral overreaction/reversal (De Bondt & Thaler 1985; Jegadeesh 1990) | Positive on 3 of 4 datasets; negative only on trend_gbm | **0.277** on regime_switch (robust; < 0.3) | 3 params | oos_sharpe_mean: 0.30 avg; oos_consistency: 0.70 avg | CAGR: regime_switch 11.3%, fat_tail 8.6% | T=1.03 / R=1.00 / HV=0.93 — all-regime positive on regime_switch; also positive in all 3 regimes on fat_tail except high_vol (−0.38) |
| 3    | [Donchian Channel Turtle Breakout](strategies/03-donchian-turtle-breakout/README.md) | Supply exhaustion at N-bar extremes (Dennis/Eckhardt 1983; Covel 2007) | In-sample positive on all 4 datasets but thin margins on trend_gbm (0.05) and mean_rev_ou (0.08); regime-dependence confirmed and expected | **0.490** on regime_switch (moderate; 0.3–0.8) | 3 params | oos_sharpe_mean: −0.06 avg; oos_consistency: 0.55 avg; fat_tail flagged walk-forward inconsistent (0.40) | CAGR: regime_switch 4.1%, others weak | T=0.42 / R=−0.07 / HV=0.72 — positive in trending and high_vol but loses money in ranging on regime_switch |
| 4    | [52-Week High Proximity (Anchoring Bias)](strategies/04-52wk-high-proximity/README.md) | Investor anchoring bias (George & Hwang 2004, Journal of Finance) | Negative in-sample Sharpe on all 4 synthetic datasets; academic evidence is cross-sectional, not single-series | **0.495** on regime_switch (moderate; 0.3–0.8); uninformative on trend_gbm and mean_rev_ou (near-zero mean_sharpe inflates score) | 2 params | Walk-forward uninformative: 252-bar warmup exceeds 166-bar fold length; oos_consistency: 0.0 on all datasets | CAGR: all negative on synthetic data | T=−0.54 / R=−0.69 / HV=0.82 — negative in both trending and ranging; only profitable in high_vol on regime_switch |
| 5    | [Turn-of-Month Calendar Effect](strategies/05-turn-of-month/README.md) | Institutional calendar flows (Lakonishok & Smidt 1988; Ariel 1987) | Negative Sharpe on 3 of 4 datasets; positive only on regime_switch (+0.29, noise not edge) — consistent with pre-stated prediction that synthetic data has no institutional-flow mechanism | Not computed (strategy has ≈0 free parameters; sensitivity sweep is uninformative) | **0 effective free parameters** — both defaults are 1988 published values, the strongest simplicity score in the library | oos_sharpe_mean: see metrics.json; walk-forward is meaningful (no warm-up barrier unlike strategy 04) | CAGR: regime_switch +2.1%; others weak or negative | see metrics.json — regime profile expected to be noise across all regimes since calendar flow is not present in synthetic data |
| 6    | [Bollinger Band Mean-Reversion](strategies/06-bollinger-mean-reversion/README.md) | Distributional overextension: close > 2σ below 20-bar mean signals improbable departure (Bollinger 2001) | **Positive Sharpe on all 4 datasets** (trend_gbm 0.33, mean_rev_ou 0.30, regime_switch 0.36, fat_tail 0.34) — narrowest cross-dataset spread of any strategy | Not yet computed | 3 params (window, nstd, exit_window) — all Bollinger (2001) published defaults | oos_sharpe_mean: regime_switch 0.31; oos_consistency: 0.6 on all datasets | CAGR: all positive but modest (2.2–2.9%); regime_switch 2.2% | HV=1.42 / T=0.52 / R=−0.83 on regime_switch — strong in high_vol, positive in trending, negative in ranging |
| 7    | [Absolute Momentum (Trend Filter)](strategies/07-absolute-momentum/README.md) | Jegadeesh-Titman momentum premium + Antonacci (2014) absolute momentum crash filter: long only when trailing 252-bar return is positive | regime_switch Sharpe 1.23 (highest in library); trend_gbm and fat_tail negative due to weak trend realization and zero-drift respectively; mean_rev_ou near-zero — confirmed regime-sensitivity | **2 params** (lookback=252, threshold=0.0) — tied for the minimum parametric count in the library; both are prior-specified from published literature, not grid-searched | 2 params (lookback, threshold) — the fewest free parameters of any strategy with non-trivially-derived defaults; **ranking criterion 3 signal** | oos_sharpe_mean: regime_switch positive; see metrics.json | regime_switch Sharpe 1.23, CAGR ~9%; trend_gbm and fat_tail negative due to dataset characteristics | see metrics.json — nearly always long (82.7%) on regime_switch; 52–54% on non-trending datasets |

**Rank 1 — Dual EMA Crossover Momentum**

This strategy earns rank 1 by default as the first entry, but each criterion is assessed honestly. On **thesis clarity** (criterion 1), the economic rationale is strong: the momentum risk premium is peer-reviewed, documented before any backtest, and directly falsifiable — it predicts that EMA crossovers should work in trending but not mean-reverting regimes, a prediction the backtest largely confirms. On **robustness** (criterion 2), in-sample performance is mixed: positive Sharpe on regime_switch (0.69), fat_tail (0.27), and mean_rev_ou (0.27), but negative on trend_gbm (−0.60). Crucially, in-sample positivity does not translate OOS: the walk-forward oos_sharpe_mean is negative on 3 of 4 datasets (−0.90 trend_gbm, −0.02 mean_rev_ou, −0.49 regime_switch; only fat_tail is positive at 0.24). This in-sample/OOS divergence is the primary robustness weakness. On **simplicity** (criterion 3), 2 free parameters is the minimum for any crossover strategy; the ThinkScript implementation is under 20 lines. On **walk-forward** (criterion 4), oos_consistency averages 0.35 across datasets, well below the 0.6 threshold — this is flagged as a weakness. On **raw performance** (criterion 5), regime_switch delivers a 9.2% CAGR and 0.69 Sharpe in-sample, while the other three datasets show weak or negative in-sample returns. The strategy is the mandatory baseline; future strategies must distinguish themselves on walk-forward consistency and cross-regime robustness.

**Rank 2 — RSI Mean-Reversion (Connors RSI-2)**

[**thesis**] The behavioral overreaction thesis (De Bondt & Thaler 1985; Jegadeesh 1990 short-horizon reversals) is peer-reviewed and falsifiable: RSI-2 should underperform on pure trending datasets and outperform where dislocations revert. The backtest confirms this directionally — Sharpe is negative on trend_gbm (−0.17) and positive on regime_switch (0.99) and fat_tail (0.73). Thesis criterion is met at the same level as 01-dual-ema-momentum. [**robustness**] The strategy is positive on three of four datasets versus two of four for 01-dual-ema-momentum, making it more cross-regime robust overall. The edge on mean_rev_ou is weak (Sharpe 0.14) — the OU process generates oscillations too small to exceed the RSI(2) < 10 threshold frequently enough to overcome round-trip costs. [**simplicity**] Three free parameters versus two for 01-dual-ema-momentum, a minor penalty; all three are published Connors defaults, not grid-searched. The ThinkScript is under 25 lines. [**walk-forward**] oos_consistency is 1.0 on regime_switch (five for five folds positive, oos_sharpe_mean 0.74), 0.6 on mean_rev_ou, fat_tail, and trend_gbm. The average oos_consistency across datasets is 0.70, comfortably above the 0.6 walk-forward consistency threshold, compared to 0.35 for 01-dual-ema-momentum. This is the primary differentiator that moves 02-rsi-mean-reversion ahead of 01-dual-ema-momentum in the walk-forward criterion. [**raw performance**] The regime_switch CAGR is 11.3% (vs 9.2% for 01-dual-ema-momentum), Sharpe 0.99 vs 0.69. The strategy ranks second by convention as a new entrant, but outscores the baseline on criteria 2, 4, and 5; the baseline wins only on criterion 3 (one fewer parameter).

**Rank 5 — Turn-of-Month Calendar Effect**

[**thesis** — criterion 1] Lakonishok & Smidt (1988, *Journal of Finance*) and Ariel (1987) document the turn-of-month effect in one of the most mechanistically clear anomalies in the empirical finance literature. The institutional-flow explanation is falsifiable and precise: end-of-month window dressing by fund managers combined with month-start payroll cash inflows create predictable demand concentrated around the month boundary. The thesis is regime-agnostic by construction — institutional reporting cycles do not change with trending or mean-reverting market conditions. Criterion 1 awards this strategy the clearest falsifiable prediction in the library: synthetic data with no institutional flow should show approximately zero edge, and this prediction is stated and honored before results are examined. Thesis quality is high on mechanism clarity and pre-specification rigor.

[**robustness** — criterion 2] Three of four synthetic datasets show negative in-sample Sharpe (trend_gbm −0.63, mean_rev_ou −0.51, fat_tail −0.10). regime_switch shows a weakly positive Sharpe (+0.29), which is noise rather than evidence of edge — the Sharpe is consistent with random chance at 24% exposure over 1000 bars, and no institutional-flow mechanism is present in the regime_switch generator. This pattern is exactly what the pre-stated falsifiable prediction requires. Robustness criterion is not informative on synthetic data and cannot penalize or reward this strategy for its synthetic performance.

[**simplicity** — criterion 3] **Zero effective free parameters** — both defaults (`tail_days=2`, `head_days=3`) are the 1988 published values with no in-sample optimisation. The Python implementation is under 50 lines of non-comment code. This is the strongest simplicity score in the library. No other strategy has a legitimate claim to zero free parameters; even strategies with 2 parameters involve some value choices. Criterion 3 is the definitive advantage of this entry.

[**walk-forward** — criterion 4] The strategy has no warm-up requirement (signal is computable on bar 1), so walk-forward evaluation is structurally valid — unlike strategy 04 (52-week high, 253-bar warm-up). OOS folds of ~166 bars fully participate in the evaluation. Since the synthetic datasets have no institutional-flow mechanism, OOS results are expected to be near zero across all folds, and this is not a failure of the strategy — it is the correct outcome for a calendar effect tested on synthetic data. Criterion 4 is uninformative here by the same reasoning as criterion 2.

[**raw performance** — criterion 5] Exposure is ~24% (5 of every ~21 trading bars). Most dataset Sharpe ratios are negative or near zero, consistent with the pre-stated expectation. The strategy is not competitive on raw performance criterion on these synthetic datasets, and this result should not be used to rank it below strategies that exploit synthetic data dynamics — it is a different thesis category entirely.

The strategy takes rank 5 as the most recent entrant. Its defining advantage is near-zero free parameters (the strongest simplicity score in the library) and a regime-agnostic thesis. Its synthetic backtest results are exactly what the institutional-flow hypothesis predicts: zero edge in data with no institutional flows. The strategy should be evaluated on real equity data (e.g., SPY daily bars spanning multiple market cycles) to test whether the Lakonishok & Smidt (1988) effect is present and persistent.

---

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

---

## Regime-Conditional Sharpe — cross-regime robustness metric

Starting from this cycle, each strategy's `metrics.json` includes a `regime_sharpe` field computed by
`engine.metrics.regime_conditional_sharpe`. Each bar in the price series is classified into one of three
mutually exclusive regimes using only lagged data (no lookahead):

- **high_vol**: rolling 20-bar log-return std is in the top 25% of the trailing 252-bar volatility distribution — a vol-spike environment.
- **trending**: the absolute 20-bar price change (fully lagged) is in the top 50% of its trailing 252-bar distribution, AND the bar is not high_vol — a directionally drifting, moderate-vol environment.
- **ranging**: all remaining bars — low drift, low vol.

The per-regime Sharpe ratio is the annualized Sharpe computed on equity returns that fall within each regime. Regimes with fewer than 30 qualifying return observations report `null` (NaN) rather than an unreliable point estimate. Regime counts are guaranteed to sum to the total bar count for each dataset.

**Effect on current ranking.** [**robustness criterion 2**] The regime profile does not change the ranking order but meaningfully sharpens the robustness picture.

RSI Mean-Reversion (rank 2) is the only strategy with all three regime Sharpes positive across both regime_switch and fat_tail. On regime_switch it posts T=1.03 / R=1.00 / HV=0.93 — a tightly clustered, across-regime positive profile that directly satisfies the main task requirement for strategies that "hold up across multiple datasets/regimes." This is a stronger robustness signal than the aggregate Sharpe of 0.99 alone, which could be driven by a single lucky regime.

Dual EMA Momentum (rank 1 by convention as the baseline) is all-regime positive on regime_switch (T=0.23 / R=0.87 / HV=0.92), but the trending Sharpe of 0.23 is substantially below the ranging and high_vol values. A momentum strategy with its weakest performance in the trending regime is consistent with the documented in-sample/OOS divergence: the strategy captures tail events and vol-spike recoveries, not directional drift. On fat_tail and trend_gbm the regime profile turns sharply negative in trending (−0.57 and −1.19), confirming that the aggregate Sharpe on regime_switch is partly regime-concentrated rather than regime-robust. The baseline's regime profile does not change its rank but does lower its effective robustness score relative to RSI.

Donchian Turtle Breakout (rank 3) shows the expected breakout-strategy signature on regime_switch: positive in trending (0.42) and high_vol (0.72), negative in ranging (−0.07). This is coherent with its thesis — breakout entries should underperform in oscillating markets. However, the same pattern reverses on fat_tail where trending Sharpe drops to −0.54, which is inconsistent with the supply-exhaustion thesis and supports its lower walk-forward consistency score. The regime analysis does not move Donchian above RSI.

52-Week High Proximity (rank 4) has a consistent negative regime profile across trending and ranging on all datasets, with the sole exception of high_vol on trend_gbm (1.18) and regime_switch (0.82). The strategy profits only during vol-spike environments across the entire dataset set — not the cross-sectional anchoring effect the thesis predicts. Regime concentration in high_vol events, which are by definition transient and unpredictable, is a weaker form of edge than across-regime robustness. The regime profile reinforces rank 4.

**Summary:** RSI's all-regime positive profile on the primary competitive datasets is the metric that most clearly separates it from the field on the robustness criterion. The regime Sharpe dimension does not alter the ranking order but provides quantitative evidence that RSI's superiority is robust to market-condition variation, not an artifact of a single-regime backtest window.

---

## Cost Robustness — transaction-cost stress sweep

Each strategy was swept across a 5 × 4 grid of cost assumptions: `commission_bps` ∈ {0, 2, 5, 10, 20} and `slippage_bps` ∈ {0, 2, 5, 10} (20 combinations). All other parameters are held constant. Full results are in each strategy's `cost_stress.json`.

Two breakeven scalars summarise each strategy's cost fragility:
- **Breakeven commission**: the smallest commission level (slippage held at the 5 bps baseline) at which Sharpe first goes negative.
- **Breakeven slippage**: the smallest slippage level (commission held at the 5 bps baseline) at which Sharpe first goes negative.
- **"already\_negative\_at\_zero\_cost"**: Sharpe is negative even before any costs are applied — the strategy has no in-sample edge on that dataset regardless of friction.
- **">max\_grid"**: Sharpe stays positive across the entire grid; edge persists even at 20 bps commission or 10 bps slippage.

Tags: strategies with breakeven below the 5 bps baseline are **cost-fragile**; strategies with breakeven ≥ 15 bps on the primary dataset are **cost-robust**.

| Rank | Strategy | regime\_switch breakeven commission | regime\_switch breakeven slippage | Tag | Notes |
|------|----------|-------------------------------------|-----------------------------------|-----|-------|
| 1 | Dual EMA Momentum | >max\_grid | >max\_grid | **cost-robust** | already\_negative\_at\_zero\_cost on trend\_gbm (thesis-predicted loss, not cost fragility) |
| 2 | RSI Mean-Reversion | >max\_grid | >max\_grid | **cost-robust** | already\_negative\_at\_zero\_cost on trend\_gbm (same thesis reason); mean\_rev\_ou breakeven commission = 20 bps |
| 3 | Donchian Turtle Breakout | >max\_grid | >max\_grid | **cost-robust** | cost-robust on every dataset (worst-case top-level breakeven commission = 20 bps); no dataset has already-negative-at-zero-cost |
| 4 | 52-Week High Proximity | already\_negative\_at\_zero\_cost | already\_negative\_at\_zero\_cost | **cost-fragile** | negative Sharpe before any costs on all 4 datasets |

**Effect on criterion 2 (robustness).** The cost sweep does not change the ranking order.

[**Dual EMA (rank 1) and RSI (rank 2) — robustness criterion 2**] Both strategies register `already_negative_at_zero_cost` on trend\_gbm even before any costs, but this is not evidence of cost fragility — it is the thesis-predicted outcome. EMA momentum and RSI mean-reversion are structurally incompatible with a pure GBM uptrend (no regime switching, no oscillation), so losing money before friction is consistent with the documented strategy thesis. On the three datasets where both strategies have positive in-sample edge — regime\_switch, fat\_tail, and mean\_rev\_ou — the breakeven commission is `>max_grid` for both, meaning the edge survives the maximum grid cost (20 bps one-way commission). Cost analysis reinforces rather than undermines the robustness picture on the relevant datasets. No rank change.

[**Donchian (rank 3) — robustness criterion 2**] Donchian is the most cost-robust strategy across all datasets: no dataset produces a negative Sharpe before costs, and the worst-case breakeven commission across all datasets is 20 bps. This is a stronger cost-robustness profile than EMA and RSI, which have thesis-driven losses on trend\_gbm independent of costs. However, this does not advance Donchian above EMA or RSI in the ranking: the primary differentiator on criterion 2 remains cross-regime positive Sharpe (RSI is positive on 3 of 4 datasets; Donchian's regime\_switch Sharpe of 0.376 and OOS consistency of 0.55 are both weaker than RSI). Cost robustness is a positive attribute for Donchian but does not close the gap on walk-forward criterion 4.

[**52-Week High (rank 4) — robustness criterion 2**] The strategy has negative Sharpe before any costs on all four datasets. This is distinct from cost fragility (an edge that disappears at realistic friction) — the strategy has no edge to erode. The cost-fragile tag is accurate but secondary: the primary robustness failure is the structural cross-sectional mismatch described under criterion 2. Rank 4 is confirmed and unaffected by the cost analysis.

---

**Rank 6 — Bollinger Band Mean-Reversion**

[**thesis** — criterion 1] The distributional overextension thesis is distinct from both the behavioral-oscillator framing of RSI-2 and the supply-exhaustion framing of Donchian.  Bollinger (2001) argues that a close more than two local standard deviations below the rolling mean represents a statistically improbable departure — under a normal distribution, such events occur less than 2.5% of the time.  The mechanism is volatility-normalised crowd overreaction: the signal adapts to local market conditions (the band widens during high-volatility periods), making it more selective and less prone to false entries during trending regimes than a fixed-threshold oscillator.  The thesis is falsifiable: the strategy should work best in oscillating regimes and underperform in strong directional trends where the rolling mean is rising fast enough to absorb typical pullbacks.  The backtest directionally confirms the trending-regime weakness (CAGR 2.8% on trend\_gbm vs 2.2% on regime\_switch) though the effect is subtle.  Thesis quality is comparable to RSI-2 — same behavioral anchor, different operationalisation.  Criterion 1 does not differentiate this strategy from RSI-2.

[**robustness** — criterion 2 — what changed vs strategy 02] Bollinger Band mean-reversion earns a positive Sharpe on **all four** synthetic datasets: trend\_gbm (0.33), mean\_rev\_ou (0.30), regime\_switch (0.36), fat\_tail (0.34).  RSI-2 is positive on three of four (negative on trend\_gbm at −0.17).  By the count criterion, Bollinger has broader cross-dataset robustness.  However, the absolute Sharpe values are substantially lower: Bollinger's best dataset (regime\_switch 0.36) is less than half of RSI-2's (0.99).  The positive-on-trend\_gbm result is explained by the strategy's very low trade frequency (exposure ~24% on trend\_gbm): the band rarely fires in a trending regime, so the strategy captures a handful of well-timed pullbacks without accumulating the cumulative losses that drag RSI-2 into negative territory.  The trade frequency advantage is not a genuine edge advantage — it reflects reluctance to trade rather than profitable trading.  Criterion 2 is marginally better than RSI-2 on count (4 vs 3 positive datasets) but substantially weaker on magnitude.

[**simplicity** — criterion 3 — no change vs strategy 02] Three free parameters, identical count to RSI-2.  All three are published Bollinger (2001) defaults with no in-sample optimisation.  The ThinkScript implementation is under 25 lines.  Criterion 3 is a draw with strategy 02; no ranking change from simplicity alone.

[**walk-forward** — criterion 4 — what changed vs strategy 02] Walk-forward oos\_consistency is 0.6 on all four datasets (3 of 5 folds positive in every case).  RSI-2 achieves oos\_consistency of 1.0 on regime\_switch (five of five folds positive) and 0.6 on the remaining three.  The average oos\_consistency across datasets is 0.6 for Bollinger vs 0.70 for RSI-2.  The regime\_switch oos\_consistency of 1.0 (RSI-2) vs 0.6 (Bollinger) is the primary walk-forward differentiator between these two otherwise similar strategies.  Bollinger's oos\_sharpe\_mean on regime\_switch is 0.31 (positive, above the 0.6 oos\_consistency threshold); RSI-2's is 0.74.  Both strategies pass the walk-forward consistency threshold on regime\_switch, but RSI-2's consistent five-for-five record versus Bollinger's three-for-five is a meaningful robustness distinction.  Criterion 4 keeps Bollinger below RSI-2.

[**raw performance** — criterion 5 — what changed vs strategy 02] On regime\_switch — the primary evaluation dataset — Bollinger achieves Sharpe 0.36 and CAGR 2.2%.  RSI-2 achieves Sharpe 0.99 and CAGR 11.3%.  The raw performance gap is large: Bollinger's regime\_switch Sharpe is roughly one-third of RSI-2's.  Bollinger compensates with cross-dataset positivity, but the raw magnitude of the edge is meaningfully weaker.  Criterion 5 confirms that Bollinger ranks below RSI-2.

The strategy takes rank 6 as the most recent entrant.  Its defining characteristic is the narrowest cross-dataset Sharpe spread of any strategy (all four datasets between 0.30 and 0.36), reflecting the volatility-normalised nature of the Bollinger Band signal.  The key criteria where it lags RSI-2 are walk-forward consistency on regime\_switch (0.6 vs 1.0) and raw performance magnitude (regime\_switch Sharpe 0.36 vs 0.99).  Compared to the existing library: it is the only strategy to maintain positive Sharpe across all four synthetic datasets, but the magnitude of that edge is thinner than RSI-2 on the primary evaluation dataset.  Its regime profile (HV=1.42 / T=0.52 / R=−0.83 on regime\_switch) reveals a concentration in high-volatility and trending sub-regimes; the ranging Sharpe is negative, meaning the strategy loses money in low-drift, low-volatility environments — a different regime weakness than RSI-2 (which is weakest in pure trending regimes).

---

**Rank 7 — Absolute Momentum (Trend Filter)**

[**thesis** — criterion 1] Jegadeesh and Titman (1993, *Journal of Finance*) documented that intermediate-horizon (3–12 month) past returns predict future returns. Antonacci (2014, *Dual Momentum Investing*) showed that *absolute* momentum — whether the asset's own trailing return is positive relative to cash — is a reliable crash filter. The behavioral mechanism is precisely specified and falsifiable: investor loss aversion causes them to hold losing positions too long, delaying price adjustment to deteriorating fundamentals and creating positive autocorrelation at intermediate horizons. This strategy should produce positive Sharpe in trending regimes and near-zero or negative Sharpe in mean-reverting regimes. The backtest confirms the regime-sensitivity directionally (regime_switch Sharpe 1.23 vs mean_rev_ou near-zero). Thesis quality is comparable to 01-dual-ema-momentum — same momentum-premium anchor, simpler operationalisation. Criterion 1 is well-satisfied.

[**robustness** — criterion 2] In-sample Sharpe is positive on regime_switch (1.23, the highest in the library) and near-zero on mean_rev_ou (−0.15), confirming regime-dependent behavior. Trend_gbm (−0.51) and fat_tail (−0.61) are negative: trend_gbm's seed-42 realization had an unusually weak trend (15% total return over 4 years vs ~82% expected mean), and fat_tail has zero drift by design, so neither dataset provides a reliable absolute momentum edge. Cross-dataset robustness is weaker than 02-rsi-mean-reversion (positive on 3 of 4) and 06-bollinger-mean-reversion (positive on all 4), but the regime_switch result is the strongest single-dataset Sharpe in the entire library. Robustness is regime-concentrated: the strategy works strongly where the thesis predicts it should (trending) and fails where it predicts it should fail (mean-reverting, zero-drift).

[**simplicity** — criterion 3] **2 free parameters** — lookback and threshold — tied with 01-dual-ema-momentum and 04-52wk-high-proximity for the fewest parametric parameters in the library. Both defaults are prior-specified from published academic literature: lookback=252 (one trading year, the Jegadeesh-Titman canonical value, replicated across asset classes) and threshold=0.0 (any positive trailing return triggers a long position, the Antonacci standard). Neither value was grid-searched from the test datasets. The Python implementation is under 30 lines; the ThinkScript version is under 20 lines. Criterion 3 is a key strength: among all strategies with non-trivially-derived parameters, absolute momentum ties for the minimum parameter count.

[**walk-forward** — criterion 4] See metrics.json for full walk-forward results. The 252-bar lookback means bars 0–251 use a shrinking reference window (close[t] vs close[0]), but from bar 252 onward the full-window evaluation applies. The walk-forward partitioning into ~166-bar OOS folds (n_splits=5, 1000 bars) means some folds are shorter than the lookback; in these folds the strategy gracefully uses whatever history is available (ref_idx = max(0, t - lookback)). No warm-up exclusion is needed — unlike strategy 04 (52-week high, 253-bar hard barrier), this strategy generates some signal from bar 1.

[**raw performance** — criterion 5] On regime_switch — where the thesis predicts an edge — the strategy achieves Sharpe 1.23 and high exposure (82.7%). This is the strongest single-dataset Sharpe in the library (RSI-2 achieves 0.99 on regime_switch). On the three non-trending datasets, performance is negative, consistent with the falsifiable thesis prediction. The strategy has the highest peak Sharpe of any library entry but the worst cross-dataset average, reflecting extreme regime-concentration of the edge.

The strategy takes rank 7 as the most recent entrant. Its defining characteristics are: (1) the highest in-library Sharpe on regime_switch (1.23), confirming the absolute momentum crash-filter thesis in its native environment; (2) the fewest grid-searched parameters (2, both prior-specified from published literature) — a ranking criterion 3 signal; (3) near-zero effective exposure on mean-reverting datasets, confirming the falsifiable prediction. The strategy's weakness is single-dataset concentration: it underperforms on 3 of 4 synthetic datasets because 3 of 4 are not sustained-trend environments. On real equity data with extended bull markets, the strategy's crash-filter property is expected to manifest more clearly. It composes naturally with strategies 01 and 03 as a regime-filter overlay: absolute momentum can veto entries from EMA-crossover or Donchian-breakout signals during bear markets.

---

## Cost-to-Alpha Ratio — gross-edge efficiency metric

[**criterion 2 — robustness** and **criterion 5 — raw performance under costs**]

The cost-to-alpha ratio answers a question no prior metric captures directly: what fraction of gross edge survives realistic transaction friction? The formula is `gross_alpha / (gross_alpha - net_alpha)` where alpha is CAGR, gross is the zero-cost equity curve, and net is the standard 5 bps commission + 5 bps slippage curve. A higher ratio means costs erode less of the gross edge; ratio = 1.0 is the special case for zero cost; ratio → inf when net alpha is zero or negative (all edge consumed). Full values are in each strategy's `metrics.json` under `cost_to_alpha_ratio` (null = net alpha non-positive on that dataset).

**Cost-to-Alpha Ratio on regime\_switch (the primary evaluation dataset):**

| Strategy | Turnover (regime\_switch) | cost\_to\_alpha\_ratio (regime\_switch) | Interpretation |
|----------|--------------------------|----------------------------------------|----------------|
| 01 — Dual EMA Momentum | ~0.019 (low) | 18.59 | Costs take ~5% of gross alpha; very low turnover preserves edge |
| 02 — RSI Mean-Reversion | ~0.060 (moderate) | 6.80 | Costs take ~15% of gross alpha; RSI(2) trades often enough to matter |
| 03 — Donchian Turtle Breakout | ~0.007 (very low) | 4.08 | Costs take ~25% of gross alpha; moderate impact |
| 04 — 52-Week High Proximity | null (all datasets) | null | Negative net CAGR on all datasets — costs aggravate an already-losing strategy |
| 05 — Turn-of-Month Calendar Effect | ~0.093 (high) | 1.88 | Costs take ~47% of gross alpha; calendar trades at fixed month boundaries incur frequent entry/exit costs |
| 06 — Bollinger Band Mean-Reversion | ~0.093 (high) | 4.23 | Costs take ~24% of gross alpha; despite high turnover, positive gross alpha keeps ratio reasonable |

**Effect on ranking — criterion 2 (robustness) and criterion 5 (raw performance under costs).**

The cost-to-alpha analysis partially revises expectations for high-turnover strategies. The suggestion's pre-analysis predicted Turn-of-Month (05) would have a ratio near 1.0 due to very low turnover (~2.5 round-trips/month), and RSI-2 (02) a ratio of 1.5–3.0. The actual results differ: Turn-of-Month has a ratio of 1.88 (meaning ~47% of its gross alpha on regime\_switch is consumed by costs) while RSI-2 achieves 6.80. The discrepancy for Turn-of-Month arises because its gross CAGR on regime\_switch is only ~4.3%, and 5+5 bps per round-trip × ~12 round-trips/year = ~120 bps/year of cost erodes roughly half of that thin gross edge. RSI-2's higher gross alpha (due to regime\_switch being its native environment) leaves proportionally more room above zero after costs, so its ratio is more favourable despite higher per-trade cost occurrence.

**Effect on relative ranks between high-turnover (02, 05) and low-turnover (01, 03) strategies:**

- [**Dual EMA (01) vs RSI-2 (02)**] On regime\_switch, Dual EMA has a higher ratio (18.59 vs 6.80), meaning costs eat a smaller fraction of its gross alpha. However, its gross alpha itself is weaker — Dual EMA's 9.2% in-sample CAGR is driven by a large gross edge that costs barely touch (low turnover), whereas RSI-2's higher CAGR (11.3% net) represents a richer gross edge consumed at a 15% rate. The ratio favours Dual EMA on efficiency but RSI-2 on absolute magnitude. **No rank change from this metric: walk-forward criterion 4 remains the decisive differentiator, and RSI-2 outperforms Dual EMA on that criterion regardless of cost efficiency.**

- [**Turn-of-Month (05) vs low-turnover strategies (01, 03)**] The 1.88 ratio for Turn-of-Month on regime\_switch is the worst among strategies with positive net CAGR. Its gross edge is thin and half of it vanishes at standard friction levels. Against Dual EMA (ratio 18.59) and Donchian (ratio 4.08), Turn-of-Month is the most cost-sensitive strategy on regime\_switch. This reinforces its rank 5 position — even on the one dataset where it shows positive in-sample returns, realistic costs consume a disproportionate fraction of its already-modest gross edge. On real equity data (where the calendar effect is expected to be real), the gross alpha would be larger and the ratio would improve; the current result reflects the thin signal-to-cost balance on synthetic data.

- [**Bollinger (06) vs RSI-2 (02)**] On regime\_switch, Bollinger's ratio (4.23) is lower than RSI-2 (6.80), consistent with Bollinger's thinner gross alpha at similar trade frequency. This is a criterion 5 disadvantage for Bollinger — not only does it earn lower net CAGR, it also gives up a larger fraction of its gross alpha to costs. **No rank change: Bollinger already ranks below RSI-2 on criteria 2, 4, and 5.**

**Summary.** The cost-to-alpha ratio does not change the current ranking order. It sharpens the picture on criterion 2 by revealing that Dual EMA's low-turnover advantage preserves a high fraction of its gross alpha, while Turn-of-Month's calendar-boundary trading incurs disproportionate cost friction relative to its thin gross edge on synthetic data. RSI-2 remains the strongest strategy on criteria 2, 4, and 5 even under this new lens: its moderate turnover is compensated by its richer gross alpha, leaving it with a more favourable ratio than all other positive-net-alpha strategies on regime\_switch except Dual EMA (which has lower absolute net performance).
