# Dual EMA Crossover Momentum

## Economic Mechanism

The momentum risk premium is one of the most replicated anomalies in empirical finance. Jegadeesh & Titman (1993) documented that equities which outperformed over the past 3–12 months continued to outperform over the following 3–12 months, generating abnormal returns that could not be explained by contemporaneous risk factors. The prevailing explanation is the slow diffusion of information: fundamental news (earnings revisions, sector rotation, macro regime shifts) takes time to be fully incorporated into prices because many market participants update their beliefs gradually and institutional flows have execution delays. The dual EMA crossover operationalizes this: the fast EMA tracks recent price momentum while the slow EMA represents the entrenched trend. A crossover-up signals that short-term momentum has overtaken the long-term baseline, implying the information diffusion process has begun but not yet completed.

## Regimes That Favor or Break the Edge

This strategy performs best in persistent trending regimes where price autocorrelation is positive over the signal lookback (20–60 days). Conversely, the strategy is harmed by mean-reverting regimes (e.g., Ornstein-Uhlenbeck processes) where a crossover-up is quickly reversed — the strategy enters at the local peak and exits after a loss. Fat-tailed, low-drift regimes present a mixed picture: the EMA crossover filters out noise but is slow to react to sudden large moves. In alternating trend/mean-reversion regimes, performance depends on the timing of regime boundaries relative to the EMA windows. The honest picture from the synthetic backtest is that the strategy earns a positive Sharpe on the regime-switch dataset (which has 200-bar trending segments) and the fat-tail dataset, but struggles on the pure GBM series in this particular draw — the open[t+1] fill and round-trip costs penalize a strategy that whipsaws during flat periods within the trending series.

## Parameters

This strategy has exactly **2 free parameters**:

- `fast_window = 20` (EMA span for short-term trend): chosen to capture approximately one calendar month of trading days, the lower bound of the Jegadeesh & Titman momentum horizon. Set before any backtest was run, anchored to the typical institutional rebalancing cycle.
- `slow_window = 60` (EMA span for long-term trend): three calendar months, representing the midpoint of the 3–12 month momentum horizon. The 3× ratio between fast and slow is a conventional choice that ensures the two EMAs measure materially different timescales without introducing excessive lag.

Neither value was selected by optimizing in-sample performance on any dataset. They are prior beliefs about the relevant momentum timescale.

## Implementation

The Python strategy (`strategy.py`) exposes a `DualEMAMomentum` class with a `__call__` method that receives a guarded price view up to bar `t` and returns `1.0` (long) when `EMA(20) > EMA(60)`, or `0.0` (flat) otherwise. The strategy is warm-up-aware: it returns flat until at least 60 bars are available, preventing the slow EMA from operating on an insufficient history. Signals are filled at `open[t+1]`, capturing the intraday `open[t+1]→close[t+1]` return rather than the overnight gap.

The ThinkScript file (`strategy.ts`) implements equivalent logic for Thinkorswim: `AddOrder(OrderType.BUY_AUTO, ...)` fires on a crossover-up event and `AddOrder(OrderType.SELL_TO_CLOSE, ...)` fires on a crossover-down, matching the Python binary long/flat signal exactly.

## Backtest Results Summary

Full metrics are in `metrics.json`. Walk-forward OOS evaluation (5 anchored folds, fixed parameters, no per-fold re-fitting) shows `oos_consistency` ranging from 0.2 (trend_gbm, regime_switch) to 0.6 (fat_tail), indicating that the strategy does not hold a robust edge across all fold windows in these synthetic datasets. The regime_switch dataset produces the strongest in-sample Sharpe (0.69) due to the 200-bar trending segments. These results are presented without selection or filtering — they are the honest output of a single run with the default parameters.
