# Strategy 11 — Statistical Pairs Mean-Reversion

## Thesis

Two assets are co-integrated when a linear combination of their log prices is
stationary — i.e., a shared non-stationary factor cancels in the spread, leaving
an equilibrium relationship that cannot drift indefinitely. When the spread deviates
beyond its rolling mean by more than `z_entry` standard deviations, co-integration
theory implies reversion to the equilibrium, providing a structural basis for the
trade.

This is a portfolio-level thesis, distinct in mechanism from all ten single-asset
strategies in the library. The edge is not a behavioral overreaction in one time
series, nor a distributional signal about one asset's z-score — it is the breakdown
and re-establishment of an economically-grounded cross-asset equilibrium. The
theoretical underpinning follows the error-correction mechanism of Engle & Granger
(1987): if prices share a common stochastic trend, the co-integration residual is
stationary and its expected value is zero regardless of where the individual prices
are.

**Primary citation:** Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006).
"Pairs trading: Performance of a relative-value arbitrage rule." *Review of
Financial Studies*, 19(3), 797–827. This paper established empirical profitability
of the z-score entry/exit framework across 1962–2002 US equities and remains the
benchmark reference in the pairs-trading literature.

## Parameters

| Parameter | Default | Source |
|-----------|---------|--------|
| `z_entry` | 2.0     | Gatev et al. (2006): entry at 2 historical standard deviations |
| `z_exit`  | 0.5     | Gatev et al. (2006): exit as spread reverts toward mean |
| `window`  | 60      | Rolling window for mean and std of the log-spread |

All defaults are from the Gatev et al. (2006) paper and were not grid-searched from
the synthetic dataset.

## Signal logic

The strategy is fed a spread instrument whose `close` column equals `close_A / close_B`.
The log of this ratio equals `log_A − log_B`, the log-price spread.

```
log_spread[t] = log(close[t])  # = log_A[t] − log_B[t]
z[t] = (log_spread[t] − rolling_mean(log_spread, window)) / rolling_std(log_spread, window)

if z < −z_entry:  enter long (long-A / short-B)
if z > +z_entry:  enter short (short-A / long-B)

exit long  when z > −z_exit   (spread recovering toward mean)
exit short when z <  z_exit   (spread recovering toward mean)
```

The engine's `allow_short=True` config is required; the strategy returns +1 (long
spread), -1 (short spread), or 0 (flat).

## Contrast with single-asset counterparts

| Strategy | Signal type | Equilibrium basis | Two-leg? |
|----------|-------------|-------------------|----------|
| 02 RSI Mean-Reversion | Behavioral overreaction (momentum oscillator) | None — momentum signal only | No |
| 06 Bollinger Mean-Reversion | Distributional overextension (z-score) | None — statistical regularity only | No |
| **11 Pairs (this strategy)** | Co-integration residual (structural equilibrium) | Co-integration vector — economically grounded | Yes |

Both 02 and 06 exploit mean-reversion, but in a single time series with no theoretical
equilibrium level. This strategy's equilibrium is structural: if co-integration holds,
the spread *must* revert. Regime breaks (structural changes, delistings, sector
divergences) directly invalidate the thesis — a risk that 02 and 06 do not face.

## Regime predictions

| Regime | Prediction | Reason |
|--------|-----------|--------|
| Ranging / mean-reverting | **Helps** | Spread oscillates frequently around equilibrium; many entry/exit cycles |
| High-volatility | **Helps (short-horizon)** | Larger spread dislocations → larger mean-reversion profits |
| Trending | **Hurts** | A persistent trend in one asset widens the spread indefinitely; co-integration may be temporarily overwhelmed |
| Regime break / structural change | **Strongly hurts** | Co-integration relationship may collapse; the spread no longer reverts |

Backtest results on `paired_cointegrated.csv` are consistent with these predictions:
Sharpe is positive in ranging (+0.99) and high-volatility (+0.91) sub-regimes, and
negative in the trending sub-regime (−0.22). The trending weakness is the expected
outcome, not a defect.

## Backtest results (paired_cointegrated dataset)

| Metric | Value |
|--------|-------|
| CAGR | 4.0% |
| Sharpe | 0.47 |
| Sortino | 0.49 |
| Max drawdown | −8.2% |
| Exposure | 41.4% |
| OOS Sharpe mean (walk-forward) | 0.75 |
| OOS consistency | 0.80 |
| Deflated Sharpe | 0.82 |

Walk-forward OOS consistency of 0.80 (4 of 5 folds positive) is the highest in the
library alongside strategy 09. The DSR of 0.82 indicates high confidence that the
edge is genuine after correcting for finite-sample bias.

## Dataset note

This strategy is evaluated on `data/paired_cointegrated.csv` — a synthetic dataset
generated specifically for pairs testing. The standard four single-asset datasets
(trend_gbm, mean_rev_ou, regime_switch, fat_tail) are single-series instruments and
cannot meaningfully test a spread strategy. The paired dataset is generated by
`data/generate_paired.py` (seed=42, 1000 bars) and shares the generation conventions
of the other synthetic datasets.

## ThinkorSwim parity statement

**Single-symbol limitation (noted divergence, not a defect):** ThinkorSwim cannot
execute two instruments simultaneously from a single strategy script. The `strategy.ts`
trades a single-leg proxy where the "spread" is approximated as the ratio of the
current close to a historical reference close on the same symbol, applying the same
z-score entry/exit logic. This divergence is deliberate. The Python engine trades the
true two-leg spread and is the ranking source of truth. The TOS script is useful only
for directional signal validation on one leg.

**Parameter parity:** `strategy.ts` declares its tunables as `input z_entry = 2.0;`,
`input z_exit = 0.5;`, and `input window = 60;` — identical snake_case names to
`strategy.py`. No name mapping is required.
