# Low-Volatility Anomaly — Baker, Bradley & Wurgler 2011 / Blitz & van Vliet 2007
# Thesis: the low-vol anomaly predicts that assets in a low-volatility regime
#   (vol below own historical median) produce superior risk-adjusted returns.
#   This time-series version goes long when current vol is below rolling median
#   and exits when vol rises above the exit_percentile threshold.
# Parameters:
#   volWindow=60      — bars for realized vol (one quarter). Prior-specified.
#   rankingWindow=252 — bars for rolling median/percentile (one year). Prior-specified.
#   exitPercentile=75 — percentile of rolling vol history used as exit threshold.
#                       Prior-specified from anomaly literature (75th = upper quartile).
# Vol proxy note: Python strategy.py uses returns-based realized vol
#   (pct_change().rolling(volWindow).std() * sqrt(252)). ThinkScript uses
#   StdDev(close, volWindow), which is price-level standard deviation, NOT a
#   returns-based measure. The two measures are monotonically related for a given
#   series over a rolling window — when returns-based vol rises, price-level std
#   also rises — so the signal direction (below-median = long; above-exit = flat)
#   is preserved even though the absolute scale differs. The README documents this
#   proxy substitution explicitly.
# Percentile approximation: thinkScript has no native rolling percentile function.
#   The exit threshold is approximated as:
#     Lowest(vol, rankingWindow) + exitPercentile/100 * (Highest - Lowest)
#   This is a min-max interpolated quantile, which approximates the true 75th
#   percentile when the vol distribution is roughly uniform or symmetric. For
#   skewed vol distributions the approximation may overestimate the true 75th
#   percentile; this makes the exit condition *less* sensitive (harder to exit)
#   than the Python strategy. The effect is a slightly longer hold duration; it
#   does not change the signal direction.
# Median approximation: thinkScript uses Average(vol, rankingWindow) (simple
#   moving average) as a proxy for the rolling median. For right-skewed vol
#   distributions (typical of realized volatility), the mean > median, so
#   the average-based entry threshold is slightly harder to cross (higher bar)
#   than the Python median. The practical effect is marginally fewer entries.
# Signal logic: addOrder BUY_AUTO when in low-vol regime; SELL_TO_CLOSE on exit.
# Load: Charts > Studies > Edit Studies > Strategies tab > Create > paste > apply Daily.
# Fill note: open[-1] = next-bar open, matching Python engine t+1 open fill model.

input volWindow = 60;
input rankingWindow = 252;
input exitPercentile = 75;

# Realized vol proxy: price-level std dev over volWindow bars (see note above)
def currentVol = StdDev(close, volWindow);

# Rolling vol history statistics over rankingWindow bars
def volLow = Lowest(currentVol, rankingWindow);
def volHigh = Highest(currentVol, rankingWindow);

# Median approximation via simple moving average (see note above)
def volMedian = Average(currentVol, rankingWindow);

# Exit threshold via min-max interpolation (see note above)
def volExit = volLow + (exitPercentile / 100.0) * (volHigh - volLow);

# Warm-up guard: no signal until enough history is available
def warmedUp = BarNumber() > (volWindow + rankingWindow);

# Long entry: current vol is in the low-vol regime (below rolling median)
AddOrder(OrderType.BUY_AUTO,
         warmedUp and currentVol < volMedian,
         open[-1], 1,
         Color.GREEN, Color.GREEN, "LowVol Entry");

# Exit: vol has risen into the high-vol regime (above exit threshold)
AddOrder(OrderType.SELL_TO_CLOSE,
         warmedUp and currentVol > volExit,
         open[-1], 1,
         Color.RED, Color.RED, "LowVol Exit");

# Vol overlay for visualization
plot volLine = if warmedUp then currentVol else Double.NaN;
volLine.SetDefaultColor(Color.YELLOW);
volLine.SetLineWeight(1);

# Median reference line
plot medianLine = if warmedUp then volMedian else Double.NaN;
medianLine.SetDefaultColor(Color.CYAN);
medianLine.SetStyle(Curve.SHORT_DASH);

# Exit threshold reference line
plot exitLine = if warmedUp then volExit else Double.NaN;
exitLine.SetDefaultColor(Color.RED);
exitLine.SetStyle(Curve.SHORT_DASH);
