# Volatility-Managed Portfolio — Strategy 09

## Citation

Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*, 72(4), 1611–1644.

## Economic Thesis

### Variance Risk Premium and Risk-Aversion Dynamics

The central finding of Moreira & Muir (2017) is that equity expected returns are
**negatively correlated with concurrent realized variance**. When the variance of
equity returns spikes, risk-averse investors demand a higher premium to maintain
their exposure — but they also reduce their holdings *at the moment* the premium
is highest. The result is that expected returns are elevated precisely during
periods when most investors cannot stomach holding the risk.

A strategy that mechanically scales equity exposure inversely to realized variance
exploits this dynamic: it holds full (or leveraged) size when markets are calm
and the variance risk premium is thin, and reduces exposure during volatile periods
when the premium is elevated — locking in the improvement in the risk/reward ratio
through disciplined position sizing rather than through market timing.

The mechanism is not about predicting the direction of the next return. It is about
holding the correct *amount* of equity given the current risk environment. This is
explicitly a **risk-sizing strategy**, not a signal-timing strategy.

### Why Expected Returns Fall with Contemporaneous Variance

The behavioral and structural underpinnings:

1. **Risk aversion**: Investors have concave utility. When variance rises, they
   tolerate less risk per unit of expected return, so equilibrium requires a higher
   expected return to clear the market. But because investors reduce their holdings
   during high-variance periods (to manage portfolio risk), demand falls precisely
   when supply is effectively higher — creating the negative correlation.

2. **Leverage constraints**: Many institutional investors face VaR or volatility
   limits. When realized variance spikes, they are forced to de-risk regardless of
   expected returns, creating an involuntary reduction in demand.

3. **The premium**: A patient investor without hard leverage constraints can act
   as the counterparty — absorbing risk during high-variance periods in exchange
   for the elevated premium, then scaling back when variance is low and the premium
   is thin.

### Relationship to Other Library Strategies

| Strategy | Mechanism | Signal type | This strategy's contrast |
|----------|-----------|-------------|--------------------------|
| 07 — Absolute Momentum | Long only when 252-day trailing return > 0 | Direction filter (crash filter) | Vol-Managed sizes continuously; Abs Mom exits entirely. Both are long-only and compose naturally as a combined overlay. |
| 08 — NR7 Breakout | Enters after volatility compression (NR7 bar) and holds 4 bars | Timing signal | NR7 *buys* after quiet periods; this strategy *reduces size* during volatile periods. Opposite volatility-response intuitions. |

## Parameters

Both parameters are prior-specified from Moreira & Muir (2017) and were not derived
from the synthetic test datasets. No grid search was performed.

| Parameter | Default | Source | Rationale |
|-----------|---------|--------|-----------|
| `window` | 21 | Moreira & Muir (2017) | One calendar month of trading days — the paper's baseline realized-variance lookback |
| `target_vol` | 0.12 | Moreira & Muir (2017) | 12% annualized target volatility — the paper's baseline scaling denominator |

**Note on parameter count**: The suggestion described this strategy as having "one
free parameter." This is incorrect — there are two free parameters: `window` and
`target_vol`. Both appear as prior-specified defaults from the published paper and
both are represented identically in `strategy.py` and `strategy.ts`.

## Signal Mechanics

At each bar t, the position scalar is:

```
scalar = clip(target_vol / realized_vol(t-1, window), 0, 2)
```

where `realized_vol` is the annualized standard deviation of the prior `window`
daily returns (lagged: excludes the current bar's return).

- `scalar > 1`: low-volatility environment — leverage up (up to 2x maximum)
- `scalar = 1`: realized vol equals target — hold full exposure
- `scalar < 1`: high-volatility environment — reduce exposure
- `scalar = 0`: realized vol is effectively zero (constant-price edge case) — special case returns 1.0

## Engine Limitation: Binary Signal Approximation

**Important**: the backtest engine used by this library maps any positive return
to a full long position (1x) and zero to flat. Since `scalar` is always positive
for `target_vol > 0` and `realized_vol > 0`, the engine treats this strategy as
**always long after the 21-bar warm-up period**. The fractional exposure scaling
(e.g., 0.5× during high volatility) is not captured by the binary engine.

The metrics in this file therefore reflect a **long-after-warm-up binary
approximation**, not the true risk-managed performance documented in Moreira &
Muir (2017). The true variance-risk-premium benefit requires a portfolio engine
that supports non-binary position weights.

As a consequence:

- Metrics across all four datasets reflect the underlying dataset drift, not
  the risk-sizing signal.
- The sensitivity analysis shows near-zero sensitivity to both `window` and
  `target_vol` — correct, because changing these parameters does not change
  the binary (always-long) signal in this engine.
- This is the library's first **risk-sizing strategy**, introducing a new
  category distinct from signal-timing strategies.

## Backtest Results Summary

Standard cost model: 5 bps commission + 5 bps slippage per side.

| Dataset | Sharpe | CAGR | Max DD | Exposure | Walk-fwd Sharpe | Walk-fwd Consistency |
|---------|--------|------|--------|----------|-----------------|----------------------|
| trend_gbm | +0.235 | +2.7% | −29.6% | 97.9% | +0.260 | 0.80 |
| mean_rev_ou | +0.239 | +2.8% | −25.8% | 97.9% | +0.243 | 0.60 |
| regime_switch | +1.217 | +22.3% | −17.6% | 97.9% | +0.640 | 1.00 |
| fat_tail | +0.390 | +5.5% | −26.3% | 97.9% | +0.317 | 0.60 |

Regime-conditional Sharpe on regime_switch: HV=+2.05 / T=+0.68 / R=+0.88

## Behavior by Regime

Because the strategy is binary-long after warm-up in this engine, performance
reflects the underlying dataset's drift and volatility structure:

- **trend_gbm** (GBM with μ=0.15/yr positive drift): Sharpe 0.235 reflects
  capture of the positive drift less round-trip costs (~0.001 turnover, so costs
  are negligible at ~1 bp/year). The high max drawdown (−29.6%) reflects the
  strategy's inability to reduce size during volatile periods — the key loss from
  the binary engine restriction.

- **mean_rev_ou** (OU mean-reverting process): Sharpe 0.239, similar to trend_gbm.
  The OU process oscillates around a fixed mean; being constantly long captures
  any positive drift in the OU process. Regime profile HV=0.63/T=0.35/R=−0.09
  shows the ranging regime is slightly negative — consistent with mean-reversion
  eroding a constant-long position.

- **regime_switch** (alternating GBM and OU): Sharpe 1.217 with OOS consistency
  1.00 (five of five folds profitable OOS). The regime_switch dataset's trending
  periods generate large directional moves captured by the constant-long exposure.
  The 95% bootstrapped CI [0.252, 2.187] **excludes zero** — the strongest
  statistical evidence of edge on this dataset in the library, tied with Absolute
  Momentum. All three regime sub-Sharpes are positive (HV=2.05 / T=0.68 / R=0.88).

- **fat_tail** (zero-drift t(3) innovations): Sharpe 0.390, positive despite zero
  drift by construction. The positive result reflects the specific realization of
  this seed-fixed synthetic dataset; the expected value over many seeds would be
  near zero. The CI [−0.531, 1.346] straddles zero, confirming this result does
  not constitute statistical evidence of edge.

## What Would Falsify the True Strategy

1. **No vol-scaling benefit**: If a fractional-position backtest shows that the
   scaled strategy (with true fractional position sizing) does not outperform the
   unscaled constant-long strategy on Sharpe or Calmar, the variance-risk-premium
   thesis fails on this data.

2. **Negative vol-return correlation absent**: If realized variance and subsequent
   expected returns are uncorrelated in the test data (as expected for pure-GBM
   data with constant volatility), the strategy's scaling adds no value. Pure GBM
   has constant drift and no variance autocorrelation, making vol-management
   irrelevant — consistent with thin results on trend_gbm (0.235 Sharpe vs
   essentially buy-and-hold performance).

## ThinkorSwim Parity Statement

Python `strategy.py` and `strategy.ts` are in parity on the following:

| Rule | Python | ThinkorSwim |
|------|--------|-------------|
| Daily return | `close.pct_change()` = close[t]/close[t−1] − 1 | `(close / close[1]) - 1` |
| Realized vol | `std(rets) * sqrt(252)` | `StdDev(dailyRet, window) * Sqrt(252)` |
| Parameters | `window=21`, `target_vol=0.12` | `input window = 21`, `input target_vol = 0.12` |
| Scalar clipping | `np.clip(scalar, 0, 2)` | `Min(target_vol / realizedVol, 2.0)` |
| Entry logic | positive scalar → long | `warmedUp and scalar > 0` → BUY_AUTO |
| Fill price | `open[t+1]` | `open[-1]` (TOS notation) |

**Documented divergence (one-bar offset)**: Python uses `iloc[t-window:t]` which
excludes the current bar's return; ThinkScript `StdDev(dailyRet, window)` includes
the current bar's return. This is a one-bar offset in the vol window (1 of 21 bars)
and does not affect signal direction. Both implementations use return standard
deviation, **not** ATR or TrueRange.

**Documented divergence (position cap)**: Python `np.clip(scalar, 0, 2)` expresses
the 2x cap as a scalar. ThinkScript cannot express this as a single `AddOrder`
because portfolio-level fractional weights are not supported; instead, two
conditional `AddOrder` blocks implement the normal (scalar < 2) and double-leverage
(scalar >= 2) cases. This portfolio-execution divergence is structural and unavoidable.
