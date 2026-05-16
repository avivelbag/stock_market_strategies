# RSI Mean-Reversion (Connors RSI-2)

## Behavioral and Microstructure Mechanism

Short-term price extremes systematically proxy for crowd overreaction. De Bondt & Thaler (1985) documented that portfolios of prior losers significantly outperform portfolios of prior winners over 3–5 year horizons, attributing this to investor overreaction to news. Jegadeesh (1990) refined the result for short horizons — at 1-month lookbacks, reversal rather than continuation is the dominant pattern. The microstructure explanation is anchoring bias combined with disposition effect: retail participants anchor to recent prices and cut winners too early while holding losers, creating a predictable inventory imbalance that institutional liquidity providers exploit by fading extremes. RSI(2) with tight thresholds (10/90) operationalizes this by identifying when two-day price momentum has reached a statistical extreme relative to recent history, signaling that the crowded trade has exhausted itself and a short-term reversal is likely.

## Regime Prediction and Contrast with 01-dual-ema-momentum

This strategy is the structural counterpart to [01-dual-ema-momentum](../01-dual-ema-momentum/README.md). Where the EMA crossover captures the *slow diffusion* of information into prices (momentum), RSI-2 captures the *overreaction* that precedes a correction. The falsifiable prediction is that RSI-2 should underperform on pure trending datasets (trend_gbm) where there is no overreaction to fade, while outperforming on mean-reverting datasets (mean_rev_ou) and fat-tailed datasets where large dislocations create the extremes the strategy depends on. The empirical results confirm the second half of this prediction: RSI-2 achieves Sharpe 0.99 on regime_switch and 0.73 on fat_tail versus 0.69 and 0.27 for 01-dual-ema-momentum. On trend_gbm, both strategies are negative (RSI-2 at -0.17, EMA at -0.60) — RSI-2 loses less because it trades infrequently. The one counter-intuitive result is mean_rev_ou: EMA-momentum edges RSI-2 (0.27 vs 0.14), likely because the OU process generates small, frequent oscillations where RSI(2) < 10 is uncommon and the round-trip cost erodes a thin edge. The regime_switch result is the strongest differentiator: RSI-2 achieves oos_consistency of 1.0 (all five walk-forward folds positive) versus 0.2 for EMA-momentum on the same dataset.

## Parameters and Justification

This strategy has exactly **3 free parameters**, all set from prior published work with no in-sample optimization:

- `rsi_period = 2`: Ultra-short RSI window from Connors & Alvarez (2009) "Short Term Trading Strategies That Work". The 2-day lookback isolates single-bar overreaction events rather than multi-week trend deviations that a longer RSI would capture.
- `oversold = 10`: Entry threshold below which RSI(2) must fall to trigger a long position. The 10-level was published by Connors as capturing statistically significant oversold extremes at the 2-day period. Values below 5 produce too few trades; values above 20 generate false positives in trending markets.
- `overbought = 90`: Exit threshold above which the long position is closed. Symmetric counterpart to the oversold level; empirically validated in the same source.

None of these values were selected by scanning performance on the four synthetic datasets. They are carried directly from peer-reviewed practitioner literature, which is the primary defense against in-sample overfitting.

## Implementation Notes

The Python strategy (`strategy.py`) exposes a `RSIMeanReversion` class with a stateful `__call__` method. Because the entry and exit rules are not symmetric at every bar (the strategy holds a position until an exit signal fires, not just while RSI is below the oversold level), the instance tracks `_in_position` across sequential bar-by-bar calls. RSI is computed using Wilder's exponential smoothing, implemented as `ewm(alpha=1/period, adjust=False)`, consistent with the standard definition. The warm-up guard returns flat (0.0) until `rsi_period + 1` bars are available.

The ThinkorSwim file (`strategy.ts`) uses the native `RSI(Length = rsiPeriod)` function and `AddOrder` entries/exits matching the Python binary logic. The only platform difference is that TOS does not model slippage explicitly — the Python engine applies 5 bps commission and 5 bps slippage per side. This difference is called out so users comparing live TOS results to backtested metrics apply an appropriate haircut.

## Backtest Results Summary

Full metrics are in `metrics.json`. The strategy earns a positive Sharpe on three of four datasets and achieves oos_consistency of 1.0 on regime_switch (all five walk-forward folds positive), making it substantially more walk-forward consistent than 01-dual-ema-momentum on that dataset (oos_consistency 0.2). The trend_gbm result is negative (Sharpe -0.17) as predicted — a pure geometric Brownian motion with positive drift offers no mean-reversion edge. These results are presented without selection or filtering; they are the output of a single run with the Connors default parameters.
