# Strategy 12: Volatility-Conditioned RSI Mean-Reversion

## Mechanism (stated before results)

De Bondt & Thaler (1985) and Jegadeesh (1990) document that short-horizon
reversals are driven by investor overreaction. The key premise of this strategy
is that **overreaction is amplified when uncertainty is high**: in volatile
market regimes, prices diverge further from fundamental value, creating larger
and more reliable mean-reversion opportunities. Strategy 02 (RSI-2) enters
mean-reversion trades regardless of the volatility environment, trading small
and large dislocations indiscriminately.

This strategy adds a volatility-regime gate: RSI(2) entry signals are only
executed when the current realized volatility is in the **top quartile** of its
trailing 252-bar distribution. Once in a position, the exit is RSI-only — the
vol filter does NOT apply to exits. This avoids the pathological case of being
forced out of a mean-reversion trade simply because vol normalizes, even if
price has not yet reverted.

The falsifiable hypothesis: the RSI-2 mean-reversion edge is **concentrated in
high-volatility regimes** where overreaction is most extreme. The vol filter
should therefore improve Sharpe (by culling low-quality setups) at the cost of
reduced trade count.

## Prior predictions (before examining metrics)

- **fat_tail**: Volatility clustering is present in fat-tailed processes (ARCH
  effects). The filter should allow only the highest-overreaction episodes —
  expected Sharpe improvement vs. strategy 02.
- **regime_switch**: The high-vol sub-regime is explicitly simulated; the filter
  should selectively activate in that regime, improving signal quality.
- **mean_rev_ou**: The OU process has relatively stable variance; the vol
  percentile rank will fluctuate less; the filter will cut trades but
  improvement is uncertain.
- **trend_gbm**: A trending series. RSI-2 fades trends regardless — the vol
  filter may not help and may introduce additional short-side exposure during
  trending high-vol periods.
- **Trade count**: The vol filter should roughly halve trade count vs. strategy
  02, improving the cost-to-alpha ratio.

## Parameters

Eight parameters, with full provenance:

| Parameter | Default | Source | Notes |
|-----------|---------|--------|-------|
| `rsi_window` | 2 | Connors (2009) | Prior-specified; not grid-searched |
| `rsi_entry_long` | 10 | Connors (2009) | Prior-specified; oversold threshold |
| `rsi_exit_long` | 70 | Connors (2009) | Prior-specified; exit before extreme overbought |
| `rsi_entry_short` | 90 | Connors (2009) | Prior-specified; overbought entry for short |
| `rsi_exit_short` | 30 | Connors (2009) | Prior-specified; cover short before extreme oversold |
| `vol_window` | 21 | Moreira & Muir (2017) | Prior-specified; one calendar month lookback |
| `vol_lookback` | 252 | Standard practice | Prior-specified; one-year percentile window |
| `vol_threshold` | 0.75 | **Single free choice** | 75th percentile = Engle (2004) canonical high-vol definition |

The RSI parameters (5) are identical to strategy 02 — zero additional RSI
degrees of freedom are introduced. The `vol_window` (21) is the Moreira & Muir
(2017) published default also used in strategy 09. The `vol_lookback` (252) is
the standard one-year lookback for percentile normalization. The **only freely
chosen parameter** is `vol_threshold=0.75`: the 75th percentile is the
canonical "high volatility" definition from the volatility clustering literature
(Engle 2004 ARCH survey). Sensitivity over [0.5, 0.9] should be reported, as
this is the single parameter for which alternative values were available.

## Actual results

**Trade count**: The vol filter does roughly halve turnover vs. strategy 02
(about 40–58% of strategy 02's per-dataset turnover), consistent with the
pre-stated prediction.

**Sharpe**: The combined long/short strategy shows **negative Sharpe on all
four datasets**, contradicting the prediction for fat_tail and regime_switch.

| Dataset | Strategy 02 Sharpe | Strategy 12 Sharpe | Strategy 02 Turnover | Strategy 12 Turnover | Strategy 12 Exposure |
|---------|-------------------|--------------------|---------------------|---------------------|---------------------|
| trend_gbm | −0.165 | −0.728 | 0.060 | 0.036 | 0.117 |
| mean_rev_ou | +0.145 | −0.427 | 0.061 | 0.038 | 0.125 |
| regime_switch | +0.986 | −0.173 | 0.069 | 0.058 | 0.170 |
| fat_tail | +0.728 | −0.406 | 0.078 | 0.038 | 0.137 |

**Why the prediction failed — the short side:** Running strategy 12 with
`allow_short=False` (long-only, vol-gated) produces **positive Sharpe on all
four datasets** (trend_gbm: +0.40, mean_rev_ou: +0.85, regime_switch: +0.47,
fat_tail: +0.18). The vol-conditioned long entries are profitable — the
hypothesis is confirmed for the long side. The failure is entirely attributable
to the short side (entering short when RSI > 90 AND high_vol).

The short side loses money for a structural reason: high-vol periods on these
synthetic datasets often coincide with directional momentum rather than pure
overreaction. When RSI-2 spikes above 90 during a high-vol trending episode,
the asset is moving in a direction that has momentum behind it. The mean-
reversion trade (short at extreme overbought) is fighting momentum, which
dominates over any overreaction component. Strategy 09 (volatility-managed)
similarly documents that high-vol periods coincide with momentum in the
binary-engine approximation. The regime_switch dataset, which has explicit
trending sub-regimes, makes this particularly severe: high-vol trending periods
generate RSI > 90 signals that do not revert on the short side.

**Hit rate comparison**: Strategy 02 achieves a hit rate of ~0.24 across all
datasets. Strategy 12 shows hit rates of 0.05–0.08, indicating the vol-
conditioned entries (especially shorts) are winning on only 5–8% of active
bars. The vol filter did not improve the quality of individual setups — it
selected high-vol periods that are systematically harder for short mean-
reversion.

## TOS implementation note

ThinkorSwim has no built-in `PercentRank` function for arbitrary indicator
series, and the naive `Sum(volRaw < volRaw[0], n)` approach does **not** work
in thinkScript. Inside `Sum(expr, n)`, thinkScript evaluates `expr` at offsets
0 through n−1, shifting **all** references including `[0]`. So
`volRaw < volRaw[0]` at iteration offset `i` becomes `volRaw[i] < volRaw[i]`,
which is always false. The result is a vol filter that never fires and silently
disables all entries — a fundamental thinkScript limitation, not a data issue.

`strategy.ts` instead uses a **min-max range normalization**:

```thinkscript
def volHigh = Highest(volRaw, volLookback);
def volLow  = Lowest(volRaw, volLookback);
def volRank = if volHigh != volLow
              then (volRaw - volLow) / (volHigh - volLow)
              else 0.5;
def highVol = volRank >= volThreshold;
```

`volRank` = 0.0 when current vol is at its `volLookback`-bar minimum; 1.0 when
at its maximum. With `volThreshold=0.75`, `highVol` fires when realized vol is
in the top 25% of its historical **range** — conceptually similar to
top-quartile of the distribution but not identical. The two measures agree when
vol is uniformly distributed over the window; they diverge when vol is skewed
(e.g., a single spike dominates the maximum, compressing all other bars toward
zero range-rank while their count-based percentile rank varies normally).

**Divergence from Python**: The Python implementation uses
`pd.Series.rolling(vol_lookback).rank(pct=True)` — a count-based percentile
rank (fraction of historical values ≤ current value). The TOS min-max rank is a
monotone transformation of vol magnitude, not a fractional count. The filter
activates in broadly similar high-vol regimes in practice, but will differ on
specific bars when the vol distribution is skewed. This divergence is inherent
to the language constraint and is documented here rather than hidden.

## Warm-up

The strategy requires at minimum `vol_window + vol_lookback` bars (21 + 252 =
273 bars) before a valid vol percentile can be computed and any entry can fire.
The warm-up is handled automatically by the NaN propagation: during the
warm-up period `high_vol = False`, so no entries fire and the position remains
flat.

## Relationship to other library strategies

- **02-rsi-mean-reversion**: Same RSI parameters; adds vol-regime gate. Long
  side improves over strategy 02 when vol-filtered (higher Sharpe per bar
  active). Combined long/short is inferior.
- **09-volatility-managed**: Shares `vol_window=21` from Moreira & Muir (2017).
  Strategy 09 scales position size continuously; strategy 12 gates entry
  discretely. Strategy 09 also documented that high-vol periods coincide with
  positive momentum in the binary-engine approximation.
- **10-low-volatility-anomaly**: Inverse thesis — strategy 10 holds during
  LOW-vol periods; strategy 12 enters during HIGH-vol periods.
