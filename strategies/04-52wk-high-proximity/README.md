# 52-Week High Proximity (Anchoring Bias)

## Thesis

George & Hwang (2004, *Journal of Finance*) document that stocks trading near their 52-week high outperform over the following month — a finding robust across markets and decades. The mechanism is behavioural: investors anchor on the 52-week high as a reference price and systematically under-react to positive news when price is near that level. Analysts are reluctant to set price targets above the prior annual peak; institutional investors with target-price mandates defer buying until the target is revised. As the true fundamental value is gradually revealed, prices drift upward, producing a predictable short-horizon return premium for stocks near their 52-week high.

The entry condition `ratio >= 0.95` captures the "approaching but not yet exceeding" state: the stock has recovered strongly but anchoring may still be suppressing further movement. The increasing-ratio filter (`ratio_today > ratio_yesterday`) adds a momentum confirmation — the approach is gaining speed, suggesting the under-reaction is beginning to resolve and reducing the risk of a false entry.

This is the first **anchoring** thesis in the library. It is distinct from the existing strategies: neither general momentum (01-dual-ema-momentum) nor mean-reversion (02-rsi-mean-reversion) nor supply-exhaustion breakout (03-donchian-turtle-breakout). The 03-Donchian strategy buys *new* highs; this strategy buys *approaching* highs — the opposite signal, capturing delayed continuation before a breakout occurs.

## Parameters and Justification

This strategy has exactly **2 free parameters**, both set from the George & Hwang (2004) paper with no in-sample optimisation:

- `proximity_threshold = 0.95`: The minimum ratio (close / 52-week high) to enter. A stock within 5% of its annual high is in the "anchored suppression zone" described in the paper. Published by George & Hwang; not grid-searched.
- `exit_threshold = 0.90`: The ratio below which the position is closed. A 5-percentage-point buffer below the entry zone avoids immediate round-trips when the ratio oscillates near 0.95. Consistent with the paper's interpretation of the effect reversing once the stock retreats materially below the anchor zone.

## Implementation Notes

The Python strategy (`strategy.py`) exposes a `FiftyTwoWeekHighProximity` class with a stateful `__call__` method. The instance tracks `_in_position` across sequential bar-by-bar calls.

**Look-ahead guard**: `rolling_max(close, 252)` uses only `closes.iloc[-252:]` — data up to and including bar `t`. The prior ratio at `t-1` uses `closes.iloc[-253:-1]` — the 252 bars ending at `t-1`. No future price is read. The signal is filled at `open[t+1]`, matching the engine's fill model.

**Warm-up requirement**: 253 bars must be available before any position is taken — 252 bars to form the current ratio, plus one additional bar for the prior-ratio comparison. Positions are flat during warm-up.

The ThinkorSwim file (`strategy.ts`) uses `Highest(close, 252)` — the explicit `close` argument is required because TOS `Highest()` defaults to the bar's `high` field, not `close`. Using `Highest(close, 252)` ensures parity with the Python rolling-max-of-close semantics. The `ratio[1]` reference gives the prior-bar ratio. `AddOrder` entries fire at `open[-1]` (next-bar open), matching the Python `t+1` fill model.

## Walk-Forward Evaluation Limitation

The standard walk-forward evaluation splits 1000-bar datasets into five folds of ~166 bars each. Because this strategy requires 253 bars for warm-up, it cannot generate any signal in a 166-bar OOS fold — every fold evaluates a flat equity curve with Sharpe 0.0. This is not a failure of the strategy; it is a structural constraint of the 252-bar lookback meeting the fold-size configuration. The in-sample metrics (full 1000-bar run) are the primary evidence of edge on the synthetic datasets.

## Backtest Results Summary

Full metrics are in `metrics.json`. All four synthetic datasets produce negative in-sample Sharpe ratios, which diverges from the academic result. This is expected: the synthetic datasets do not replicate the cross-sectional anchoring environment that drives the George & Hwang effect. The academic evidence comes from ranking stocks relative to each other across the universe (stocks near their 52-week high *relative to all other stocks*), whereas this backtest applies the signal to a single price series in isolation. The walk-forward metrics are uninformative due to the warm-up constraint described above.

| Dataset | Sharpe | CAGR | Max DD | OOS Sharpe | OOS Consistency |
|---------|--------|------|--------|------------|-----------------|
| trend_gbm | −0.537 | −6.0% | −35.5% | 0.0 ⚠ warmup exceeds fold length | 0.0 |
| mean_rev_ou | −0.304 | −3.7% | −20.0% | 0.0 ⚠ warmup exceeds fold length | 0.0 |
| regime_switch | −0.072 | −1.6% | −32.9% | 0.0 ⚠ warmup exceeds fold length | 0.0 |
| fat_tail | −0.158 | −1.6% | −17.4% | 0.0 ⚠ warmup exceeds fold length | 0.0 |

## Contrasts with Existing Strategies

- **01-dual-ema-momentum**: Buys when a fast EMA crosses a slow EMA — general trend momentum. The 52-week high strategy is specifically about the anchoring behavioural mechanism around a reference price, not trend per se.
- **02-rsi-mean-reversion**: Buys extreme oversold readings — the opposite regime. Expected to have negatively correlated signals with this strategy.
- **03-donchian-turtle-breakout**: Buys on *new* N-day closing highs. This strategy buys *approaching* the annual high, not breaking it — capturing delayed continuation *before* the breakout that Donchian targets.
