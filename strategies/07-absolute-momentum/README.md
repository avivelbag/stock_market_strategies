# Absolute Momentum (Trend Filter) — Strategy 07

## Behavioral Thesis

### The Jegadeesh-Titman Momentum Premium

Jegadeesh and Titman (1993, *Journal of Finance*) documented that stocks with positive
intermediate-horizon (3–12 month) past returns continue to outperform those with negative
past returns for several subsequent months. The mechanism: investors underreact to
slow-moving fundamental changes because they anchor to prior beliefs and require repeated
confirmatory signals before adjusting expectations. This gradual price adjustment creates
positive autocorrelation at intermediate horizons — past winners keep winning, past losers
keep losing, until the signal is fully incorporated.

### Antonacci's Absolute Momentum

Gary Antonacci (2014, *Dual Momentum Investing*) extended the momentum insight to what he
calls "absolute momentum": instead of comparing an asset's return to peers (relative or
cross-sectional momentum), compare the asset's own trailing return to cash (zero). If the
asset's trailing return is positive — it is beating cash — go long. If negative — it has
underperformed cash over the lookback window — exit to cash.

This distinction matters: absolute momentum acts as a **crash filter**. In bear markets,
prices decline over extended periods, causing the trailing return to turn negative. The
strategy exits before the drawdown reaches its full depth — sacrificing some late-bull-market
gains to avoid the bulk of the bear-market losses. The trade-off is whipsaw: in choppy,
sideways markets the signal may oscillate around zero, generating excess turnover without
a directional trend to exploit.

### Why Loss Aversion Creates the Signal

The causal mechanism is grounded in behavioral finance. Loss aversion (Kahneman & Tversky
1979) causes investors to hold losing positions too long — they avoid realizing losses,
which delays the price adjustment that should occur when fundamentals deteriorate. As a
result, bad news is incorporated into prices slowly: a stock that has already declined over
the past year is more likely to continue declining than to immediately reverse, because the
full investor capitulation has not yet occurred. The absolute momentum signal exploits this
delayed capitulation.

At very long horizons (3–5 years) this reverses — the overreaction eventually corrects,
producing the long-horizon reversal documented by De Bondt and Thaler (1985). The 252-bar
window is specifically the intermediate horizon where the underreaction / momentum effect
dominates, before the long-run reversal takes over.

## Relationship to Prior Strategies

**01-dual-ema-momentum**: Both exploit momentum at intermediate horizons. The key difference
is the signal construction. Dual EMA uses a fast/slow exponential-moving-average crossover,
which is a weighted trailing return that emphasizes recent price moves more heavily. Absolute
momentum uses a simple point-to-point trailing return from exactly `lookback` bars ago to
today, which is more sensitive to what happened precisely one year ago. Absolute momentum is
the simpler signal (one comparison vs. two exponential smoothings).

**03-donchian-turtle-breakout**: Both are trend-following and long-only. Donchian fires when
the price makes a new 20-bar high (channel breakout, a supply-exhaustion signal). Absolute
momentum fires when the trailing 252-bar return is positive (a trend-continuation signal at
a much longer horizon). Donchian is faster and more reactive; absolute momentum is slower and
more selective. The Donchian signal can trigger on short-lived volatility spikes; absolute
momentum requires a sustained trend to fire.

Absolute momentum can be used as a **veto overlay** on top of either strategy: only take a
Dual EMA or Donchian entry signal when the absolute momentum filter is also positive (i.e.,
the one-year trend is up). This composability is a distinctive property — it acts as a regime
filter that prevents entries during bear markets.

## What Breaks the Edge

**Short lookbacks (< 20 bars)**: At very short horizons, daily noise dominates. A 5-bar or
10-bar trailing return is essentially a random variable — the signal-to-noise ratio is too
low to produce a reliable edge. The Jegadeesh-Titman result requires at least a 3-month
lookback to manifest.

**Very long lookbacks (> 500 bars)**: A 2–3 year trailing return reacts too slowly to regime
changes. In a market that transitions from bull to bear within 6 months, a 500-bar lookback
will remain positive long after the bear market has begun, providing no crash-filter benefit.

**Ranging / choppy markets**: In markets with no directional trend over the lookback window,
the trailing return oscillates near zero. Every time it crosses the threshold in either
direction, the strategy switches position, generating transaction costs without capturing a
directional move. High-frequency regime changes create excess turnover that erodes any gross
edge. This is the primary failure mode on mean-reverting datasets.

**High threshold values**: Requiring a large positive trailing return (e.g., threshold = 0.10
= 10%) makes the strategy very selective — it will almost never trigger in anything but
strong bull markets. This increases inactivity without improving signal quality.

## Parameters and Justification

| Parameter | Default | Source | Range |
|-----------|---------|--------|-------|
| `lookback` | 252 bars | Jegadeesh & Titman (1993); Antonacci (2014) — one trading year, the canonical intermediate-horizon sweet spot, replicated across equity, bond, currency, and commodity markets | 20–500 |
| `threshold` | 0.0 | Antonacci (2014) — "beat cash" = any positive return; industry standard, not optimized | −0.10 to 0.10 |

Both defaults are prior-specified from published academic and practitioner research. Neither
value was selected by grid search on the synthetic test datasets. This makes the strategy the
entry in the library with the fewest grid-searched parameters: **zero free parameters**
(the 2 parameters have canonical, pre-specified values).

## Backtest Behavior by Regime

- **trend_gbm** (GBM with μ=0.15/yr positive drift): The strategy should be nearly always
  long — a sustained positive-drift process will maintain a positive trailing return over
  most 252-bar windows. Flat periods occur only near the start (when reference bar is bar 0)
  or after rare extended drawdowns.

- **mean_rev_ou** (Ornstein-Uhlenbeck mean-reverting process): The strategy should be
  frequently flat — prices oscillate around a fixed mean, so the 252-bar trailing return
  will frequently be near zero or negative. This is exactly the regime where the strategy's
  crash-filter behavior saves no drawdown, because there is no trend to follow.

- **regime_switch** (alternating GBM and OU regimes): The strategy should be long during
  the GBM trending sub-periods and increasingly flat during the OU mean-reverting sub-periods
  as the trailing return catches up to the regime change.

- **fat_tail** (zero-drift t(3) innovations): With zero expected drift, the trailing return
  will be near zero on average. Fat tails mean occasional large moves that push the trailing
  return clearly positive or clearly negative, creating intermittent long periods.

## Implementation Notes

The signal is **stateless**: position at bar t depends only on `close[t]` and
`close[max(0, t - lookback)]`. No internal state is maintained between bars. This differs
from strategies 02, 06 (Bollinger) and 04 (52-week high) which track `_in_position`. The
consequence: position can flip every bar in principle — but in practice the 252-bar trailing
return changes slowly, producing low turnover.

**Warm-up**: When fewer than `lookback` bars of history are available, `ref_idx = max(0, t - lookback) = 0`,
so the reference price is `close[0]`. The strategy computes the return from the first available
bar to the current bar — this is not a warm-up exclusion but a graceful degradation. On a
1000-bar dataset, bars 0–252 use a shrinking lookback window; bars 252+ use the full year.

**ThinkorSwim note**: The ThinkScript implementation uses `close[lookback]` which accesses
the close `lookback` bars ago — equivalent to the Python `close[t - lookback]`. ThinkScript
handles the insufficient-history case internally (no trades are placed before bar `lookback`).
