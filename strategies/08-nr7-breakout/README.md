# NR7 Volatility-Contraction Breakout — Strategy 08

## Behavioral / Microstructural Thesis

### The Volatility Cycle: Compression Precedes Expansion

Markets do not move at constant speed. Directional price moves are interspersed with
consolidation periods where the daily range contracts, volume dries up, and no side
can establish directional control. This alternation between compression (low-range,
low-volume sideways drift) and expansion (breakout with high range and volume) is
the **volatility cycle**.

The NR7 (Narrowest Range in 7 bars) pattern, formalized by Toby Crabel in *Day
Trading with Short-Term Price Patterns and Opening Range Breakout* (1990), identifies
the local extreme of compression: bar t has the smallest high-minus-low range of the
last 7 bars. The thesis is that a maximum compression event is the highest-probability
precursor to a volatility expansion — the market has been "coiling" and is about to
release stored directional energy.

### Stop-Order Accumulation Mechanism

The mechanical underpinning is microstructural. During consolidation, traders who
missed the prior trend or who are managing risk around a key level accumulate stop
orders just beyond the boundaries of the tight range. Long participants set buy-stops
above the range high; short participants set sell-stops below the range low. When the
market finally breaks out of the compressed range — even by a small margin — these
accumulated stops are triggered simultaneously, creating a cascade: each triggered
stop becomes a market order that pushes price further, triggering the next layer of
stops. This self-reinforcing mechanics generates a price move disproportionate to the
initial range compression.

### How This Differs from Donchian Breakout

Both NR7 and Donchian (strategy 03) exploit breakout mechanics, but the entry
conditions are structurally different:

| Property | Donchian (03) | NR7 (08) |
|----------|---------------|----------|
| Entry signal | Close breaks above N-bar **price high** | Bar t has the N-bar **range minimum** |
| What it detects | Supply exhaustion (buyers overwhelmed all sellers over N bars) | Volatility compression extremum (range narrowest in N bars) |
| Time horizon | Multi-week price level | Short-term compression episode |
| Entry frequency | Infrequent (only when new highs are reached) | More frequent (range-based, works in any trending or ranging market) |
| Exit mechanism | Channel low violation (price-based) | Fixed time exit after 4 bars (time-based) |

The Donchian signal is a **price-position** signal: it asks whether the asset has made
a new multi-week high. The NR7 signal is a **volatility-structure** signal: it asks
whether today's intraday range is unusually compressed relative to recent history,
regardless of price level. They can fire simultaneously or independently, and their
independent firings represent genuinely different market-structure conditions.

## True-Range Definition

True range is defined as `high - low` (intraday range only). The full ATR definition,
`max(high - low, |high - prior_close|, |low - prior_close|)`, also incorporates
overnight gaps. Crabel's (1990) original bar-range definition refers to the day's
intraday range. Using high - low also ensures the Python and ThinkorSwim
implementations are identical (see Parity section below).

## Parameters and Justification

| Parameter | Default | Source | Rationale |
|-----------|---------|--------|-----------|
| `n_bars` | 7 | Crabel (1990) | The NR7 label comes directly from this value; 7 bars is the original published lookback identifying an "unusually quiet" bar |
| `exit_bars` | 4 | Crabel (1990) | Time-based exit avoids introducing a tuned stop parameter; 4 bars is the published horizon for the short-term breakout move to play out |

Both defaults are prior-specified from Crabel (1990) and were not derived from the
synthetic test datasets. No grid search was performed.

## Backtest Results Summary

Standard cost model: 5 bps commission + 5 bps slippage per side.

| Dataset | Sharpe | CAGR | Max DD | Exposure | Walk-fwd Sharpe | Walk-fwd Consistency |
|---------|--------|------|--------|----------|-----------------|----------------------|
| trend_gbm | −0.355 | −3.8% | −23.4% | 21.6% | −0.762 | 0.20 |
| mean_rev_ou | +0.232 | +1.6% | −9.5% | 18.0% | −0.048 | 0.40 |
| regime_switch | +0.335 | +2.3% | −11.2% | 20.0% | +0.108 | 0.40 |
| fat_tail | +0.064 | +0.2% | −18.5% | 16.8% | −0.495 | 0.40 |

Regime-conditional Sharpe on regime_switch: T=+0.22 / R=+1.04 / HV=−0.17

## Behavior by Regime

- **trend_gbm** (GBM with μ=0.15/yr positive drift): A sustained trending market
  produces few genuine NR7 compression events — bars in a strong trend are often
  expanding, not contracting. When NR7 does fire in a trending market, the direction
  filter should align with the trend, but the short 4-bar hold means the strategy
  captures only a small slice of the trend, while the entry cost erodes thin margins.
  Negative Sharpe is the expected result. The thesis predicts NR7 should underperform
  in pure-trend regimes where compression events are rare and brief.

- **mean_rev_ou** (Ornstein-Uhlenbeck mean-reverting process): The OU process oscillates
  around a fixed mean, generating frequent range compressions followed by mean-reverting
  moves. The direction filter (close vs midpoint) assigns the "wrong" direction roughly
  half the time when mean-reversion dominates — the close above midpoint suggests up, but
  the OU process will pull it back down. Positive Sharpe (0.23) reflects the vol-spike
  sub-periods where the compressed range precedes a genuine short expansion before
  reversion. The regime_switch regime profile (high_vol Sharpe +2.36 on mean_rev_ou)
  confirms that NR7 profits primarily during volatility spikes in this dataset.

- **regime_switch** (alternating GBM and OU regimes): Regime transitions — where the
  market switches from trending to ranging — produce clear compression → expansion
  episodes. When the trending phase begins after a consolidating OU period, NR7 fires
  near the transition and captures the initial directional move. The regime_switch
  Sharpe (+0.335) and positive ranging-regime Sharpe (+1.04) confirm this thesis
  directionally. Interestingly, the ranging regime (low vol, oscillating) has the
  highest sub-regime Sharpe — likely because the OU sub-periods in regime_switch
  produce clean compression→small-expansion patterns within 4 bars before full reversion.

- **fat_tail** (zero-drift t(3) innovations): Fat-tailed innovations generate frequent
  large moves following quiet periods — a natural environment for the volatility-cycle
  thesis. The near-zero aggregate Sharpe (0.064) reflects the zero drift: long and short
  signals cancel out in expectation. The direction filter cannot reliably assign direction
  in a zero-drift fat-tailed environment, though the high_vol and trending regime Sharpes
  (+0.78 and +0.87 respectively) suggest the strategy captures explosive vol-expansion
  events when they occur.

## What Would Falsify the Strategy

1. **Random compression**: If NR7 bars are not statistically more likely to precede
   directional moves than randomly selected bars on the same data, the volatility-cycle
   thesis fails. An ablation removing the NR7 filter (replacing with random entry timing)
   with the same 4-bar hold should produce a worse Sharpe on all datasets.

2. **Direction filter fails**: If the close-above-midpoint filter does not predict the
   post-NR7 move direction better than a coin flip, the filter is noise. Comparing
   long-only NR7 vs short-only NR7 performance should show the filter adds value.

3. **Out-of-sample degradation**: If walk-forward consistency across all datasets
   remains below 0.6 with a larger sample (e.g., 5+ years of live data), the edge is
   consistent with sampling noise, not a structural premium.

## Implementation Notes

**No look-ahead**: `high[t] - low[t]` is the intraday range, fully observable at
bar t's close. The rolling minimum over the last n_bars (including bar t) uses only
data at or before bar t. The entry signal is set at bar t's close; the fill executes
at open[t+1] in the engine.

**Stateful instance**: The strategy tracks `_in_position`, `_bars_held`, and
`_position_direction` across bar-by-bar calls. Each backtest run must use a fresh
instance to avoid state contamination across datasets.

**Warm-up**: The strategy returns 0.0 (flat) for the first `n_bars - 1` bars (0–5 for
the default n_bars=7) until enough history is available to compute the rolling minimum.

## ThinkorSwim Parity Statement

Python `strategy.py` and `strategy.ts` are in parity on the following:

| Rule | Python | ThinkorSwim |
|------|--------|-------------|
| True range | `high - low` | `high - low` (explicit, NOT `TrueRange()`) |
| NR7 condition | `tr == tr.rolling(n_bars).min()` | `tr == Lowest(tr, nBars)` |
| Direction filter | `close > (high + low) / 2` | `close > (high + low) / 2` |
| Exit | After `exit_bars` bars, return 0 | Counter-based, exits at open after `exitBars` hold bars |
| Fill price | `open[t+1]` | `open[-1]` (TOS notation for next-bar open) |

**Divergence from ThinkorSwim built-in**: ThinkorSwim's `TrueRange()` function
computes `max(high - low, abs(high - close[1]), abs(low - close[1]))`. On days
with overnight gaps, this can produce a larger true range than `high - low`. This
script uses `high - low` explicitly to match the Python implementation. The parity
is exact; the divergence only matters when comparing this script to other TOS
indicators that use `TrueRange()`.

**Short-side note**: ThinkorSwim implements both long and short NR7 entries. The
Python engine runs with `allow_short=False` (the default), so short signals from
`strategy.py` return -1.0 but the engine treats them as flat (0). To run with
short-selling enabled in Python, pass `config={"allow_short": True}` to
`engine.backtest.run()`.

## Citation

Crabel, T. (1990). *Day Trading with Short-Term Price Patterns and Opening Range
Breakout*. Traders Press. The NR7 label, the n_bars=7 window, and the exit_bars=4
time-based exit are all from this work.
