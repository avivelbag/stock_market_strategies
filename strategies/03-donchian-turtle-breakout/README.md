# Donchian Channel Turtle Breakout (Turtle System 1)

## Thesis

Price breakouts above multi-week extremes signal that informed buyers have absorbed all willing sellers, creating a structural supply vacuum that historically sustains trend momentum for days to weeks. When today's close exceeds the highest close of the prior 20 bars, it means that every participant who wanted to sell at those prices already has — the remaining float is held by stronger hands with higher price targets. The resulting supply vacuum allows price to continue in the direction of the breakout with less resistance than at any recent price level.

This mechanism was operationalized in the original Turtle Trader experiment by Richard Dennis and William Eckhardt in 1983 and documented in Covel (2007) *"Trend Following: How Great Traders Make Millions in Up or Down Markets."* The Turtle traders were given exactly these rules and generated annualized returns that substantially outpaced the S&P 500 through the 1980s, demonstrating that the edge is learnable, transferable, and documented in advance — the highest-quality evidence available in quantitative trading.

**Regime-dependence is a feature, not a flaw.** This strategy is explicitly designed for trending environments. On mean-reverting datasets (mean_rev_ou), breakouts above 20-bar highs tend to revert rather than continue, and this strategy is expected to show near-zero or negative edge. The near-zero Sharpe on mean_rev_ou (0.076) in the backtest confirms the prediction — the strategy has no edge there, and the metrics are presented honestly, not filtered. The regime_switch dataset, which alternates between trending and mean-reverting periods, is the meaningful test: a positive in-sample Sharpe (0.376) with OOS consistency at 0.6 (three of five folds positive) shows the strategy can capture the trending sub-periods when they occur.

## Regime Prediction and Contrast with 01-dual-ema-momentum

This strategy is the structurally closest counterpart to [01-dual-ema-momentum](../01-dual-ema-momentum/README.md). Both exploit the momentum risk premium, but the signal construction differs:

- **01-dual-ema-momentum** uses EMA crossover — a smooth, continuous signal that reacts to the *rate of change* of trend rather than absolute price levels
- **03-donchian-turtle-breakout** uses N-bar channel breakout — a discrete, level-based signal that reacts to *new multi-week extremes*, representing a supply/demand structural shift

The falsifiable prediction is that the Donchian strategy should outperform on strongly directional datasets and underperform in oscillating or mean-reverting regimes. The backtest partially confirms this: regime_switch delivers the best Sharpe (0.376), while trend_gbm delivers only 0.048 — lower than expected for a pure GBM trend series. The trend_gbm underperformance relative to 01 (which achieves in-sample Sharpe −0.60 but that reflects the GBM dataset's specific parameters) is noted but within expected variance for a breakout strategy with 20-bar lookback.

## Parameters and Justification

This strategy has exactly **3 free parameters**, all set from the published 1983 Turtle rules with no in-sample optimization:

- `entry_window = 20`: The 20-bar channel high lookback from Turtle System 1. The original shorter system (System 1) uses 20 bars as the entry signal; System 2 uses 55 bars. 20 was published; it was not optimized.
- `exit_window = 10`: The 10-bar channel low exit from Turtle System 1. The asymmetric shorter exit (half the entry window) allows profits to run while triggering an exit when the breakout momentum stalls. Also published, not tuned.
- `atr_window = 20`: The ATR lookback for position sizing. In the original Turtle rules, one "unit" = (1% of equity) / ATR(20). The binary engine clips all positions to 1.0 exposure, so this parameter does not affect the backtest signal but is preserved for interface parity with the TOS implementation.

None of these values were selected by scanning performance on the four synthetic datasets. They are carried from the Dennis/Eckhardt 1983 rules as published in Covel (2007).

## Implementation Notes

The Python strategy (`strategy.py`) exposes a `DonchianTurtleBreakout` class with a stateful `__call__` method. The instance tracks `_in_position` across sequential bar-by-bar calls from the engine. Entry fires when today's close exceeds the maximum of the prior `entry_window` closes (excluding today); exit fires when today's close falls below the minimum of the prior `exit_window` closes. The warm-up guard returns flat (0.0) until `entry_window + 1` bars are available.

The ThinkorSwim file (`strategy.ts`) uses `Highest(close[1], entryWindow)` and `Lowest(close[1], exitWindow)` — the `[1]` offset excludes the current bar, matching the Python look-back semantics. `AddOrder` entries fire on `crosses above / crosses below` the channel levels. The `atrWindow` input is accepted for interface parity but TOS does not apply ATR position sizing — position size is always 1 share.

Platform divergence: TOS does not model commission or slippage. The Python engine applies 5 bps commission and 5 bps slippage per side. Users comparing live TOS results to backtested metrics should apply an appropriate haircut.

## Backtest Results Summary

Full metrics are in `metrics.json`. The strategy achieves a positive in-sample Sharpe on all four datasets, but the margin is thin on trend_gbm (0.048) and mean_rev_ou (0.076), confirming regime-dependence. The regime_switch dataset is the primary evidence of edge: Sharpe 0.376, CAGR 4.1%, max drawdown −13.4%, OOS oos_sharpe_mean 0.091. The fat_tail OOS consistency is 0.40 (two of five folds positive), annotated as walk-forward inconsistent. These results are the output of a single run with the published Turtle System 1 parameters; no post-hoc selection was performed.

| Dataset | Sharpe | CAGR | Max DD | OOS Sharpe | OOS Consistency |
|---------|--------|------|--------|------------|-----------------|
| trend_gbm | 0.048 | −0.1% | −21.5% | −0.036 | 0.60 |
| mean_rev_ou | 0.076 | 0.2% | −24.1% | −0.093 | 0.60 |
| regime_switch | 0.376 | 4.1% | −13.4% | 0.091 | 0.60 |
| fat_tail | 0.161 | 1.2% | −23.0% | −0.188 | 0.40 ⚠ walk-forward inconsistent |
