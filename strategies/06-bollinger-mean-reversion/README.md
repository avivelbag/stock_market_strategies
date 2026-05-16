# Bollinger Band Mean-Reversion

## Statistical Overextension Thesis

Bollinger Bands measure how far the current price has deviated from its recent mean in volatility-normalised units — a z-score computed from a rolling standard deviation.  When price closes below the lower band (`mean - 2σ`), it has moved more than two local standard deviations from its average.  Under the assumption that daily log-returns are approximately independent over short windows, such an observation has a theoretical probability of less than 2.5%; empirically it signals crowd overreaction and predicts a near-term reversion to the mean.

The distributional overextension mechanism differs from momentum explanations: the signal is not that momentum has exhausted itself (RSI) or that a supply boundary has been breached (Donchian).  It is that price has reached a statistically improbable departure from its own recent history, measured on a scale that adapts automatically to local volatility.  Bollinger (2001) codified this signal and it has been a standard practitioner tool for over two decades; the default parameters (window=20, nstd=2.0) are the published original values, not grid-searched from any backtest.

## Difference from RSI-2 (strategy 02)

| Dimension | BB Mean-Reversion (this strategy) | RSI-2 Mean-Reversion (strategy 02) |
|-----------|-----------------------------------|-------------------------------------|
| Signal type | Distributional (z-score from rolling mean) | Momentum-oscillator (relative rank of close) |
| Entry condition | Continuous: close below `mean − 2σ` | Threshold: RSI(2) < 10 |
| Entry frequency | Rare on trending data (large `σ` absorbs drift) | Also rare, but for different reason: RSI-2 < 10 requires two consecutive down-days |
| Scale normalisation | Yes — wide bands in volatile regimes, narrow in calm | No — same 0–100 scale regardless of volatility |
| Bounded? | No — z-score can be arbitrarily negative | Yes — RSI is always in [0, 100] |
| Exit | Close above middle band (same SMA) | Close above RSI(2) = 90 |

Both exploit the behavioral overreaction thesis (De Bondt & Thaler 1985; Jegadeesh 1990 short-horizon reversals) but operationalize it differently.  BB entry depth is a continuous, scale-normalised quantity; RSI-2 is a nonlinear, bounded oscillator.  Practically: in high-volatility regimes, the rolling `σ` expands and fewer closes breach the lower band, making BB more selective during stress — the opposite of RSI-2, which fires on rank-based price declines regardless of how large the individual moves are.

## What Breaks This Strategy

**Trending regimes** are the primary failure mode.  In a sustained uptrend, price rarely closes below the lower band because the rolling mean is continuously rising and `σ` is moderate.  When an occasional brief pullback does breach the lower band, the exit (close above the middle band) comes quickly but captures a smaller fraction of the eventual recovery — the exit fires as soon as close crosses the rising SMA, which may be well below the peak.  Net effect: low trade frequency, small per-trade gain, high cost burden relative to profit.  The backtest confirms: trend\_gbm produces the lowest CAGR (2.8%) of the four datasets.

**Fat-tailed, single-direction shocks** (e.g., a crash with no subsequent recovery within the backtest window) can hold the position in drawdown indefinitely because the middle band only rises after price recovers.  The max\_drawdown of −19.5% on trend\_gbm reflects a case where a long entry was triggered by a temporary pullback in an overall uptrend, and the exit took many bars.

## Parameters and Justification

This strategy has exactly **3 free parameters**:

- `window = 20`: Rolling lookback for both the mean and standard deviation.  The 20-bar window (approximately one calendar month) is the Bollinger (2001) published default.  It balances responsiveness (short enough to track recent volatility) against noise (long enough that a single outlier bar does not dominate `σ`).  Values from 10 to 50 produce qualitatively similar behaviour; sensitivity is low on regime\_switch (the primary evaluation dataset).
- `nstd = 2.0`: Standard deviation multiplier defining the lower band.  At 2.0, fewer than 5% of observations breach the band under a normal distribution.  Bollinger (2001) specified 2.0 as the default; values below 1.5 produce excessive trades; values above 2.5 produce too few signals to overcome fixed costs.
- `exit_window = 20`: Lookback for the middle-band exit.  Set equal to `window` by default — the exit SMA and entry SMA share the same lookback, avoiding an asymmetric parameterisation.

None of these values were selected by scanning performance on the four synthetic datasets.  They are the original Bollinger (2001) defaults applied directly.

## Implementation Notes

The Python strategy (`strategy.py`) uses a stateful `BollingerMeanReversion` class with a `_in_position` flag, following the same pattern as `02-rsi-mean-reversion`.  The rolling mean and standard deviation are computed with `ddof=1` (sample std), consistent with pandas defaults and the Bollinger convention.  A warm-up guard returns flat (0.0) until `window` bars are available.

When `rolling_std == 0` (constant price series), the lower band equals the mean and the entry condition `close < lower_band` can never be met; the guard returns 0.0 immediately, preventing a divide-by-zero or false signal.

The ThinkorSwim file (`strategy.ts`) uses the native `Average(close, window)` for the rolling mean and `StdDev(close, window)` for the standard deviation.  TOS computes `StdDev` with the population formula (n denominator) while Python's `pandas.Series.rolling.std` uses the sample formula (n−1 denominator).  For `window=20` the difference is `√(20/19) ≈ 1.026`, meaning TOS bands are ~2.6% narrower than the Python bands — an immaterial practical difference but documented here for precision.  The addOrder entries use `open[-1]` (next-bar open), matching the Python engine's `t→t+1` fill model.

**Platform divergence summary:**
1. TOS `StdDev` uses n denominator; Python uses n−1.  Effect: TOS lower band is marginally wider (not narrower — correction: TOS bands are slightly narrower because `std_population < std_sample`).
2. TOS does not model explicit commission or slippage.  The Python engine applies 5 bps commission + 5 bps slippage per side.  Apply a ~10 bps per-trade haircut when comparing live TOS results to backtested metrics.
3. TOS uses `crosses below`/`crosses above` which fire on a bar-to-bar transition; the Python strategy fires on the state of the current bar.  For daily bars this is equivalent.

## Backtest Results Summary

Full metrics are in `metrics.json`.  The strategy earns a positive Sharpe across all four datasets: trend\_gbm (0.33), mean\_rev\_ou (0.30), regime\_switch (0.36), fat\_tail (0.34).  This cross-regime consistency comes at the cost of lower per-dataset Sharpe compared to RSI-2, which reaches 0.99 on regime\_switch.

The regime\_switch walk-forward oos\_consistency is 0.6 (3 of 5 folds positive).  This is below RSI-2's 1.0 on the same dataset — the primary differentiator that keeps strategy 06 ranked below strategy 02.  However, the flat, cross-dataset Sharpe profile (all four datasets within 0.30–0.36) is a distinct robustness signature: no other strategy achieves positive Sharpe on all four datasets with this narrow a spread.

The strategy cites [02-rsi-mean-reversion](../02-rsi-mean-reversion/README.md) as the closest analogue and uses the regime\_switch oos\_consistency of 1.0 (strategy 02) vs 0.6 (this strategy) as the primary ranking differentiator.
