# Volatility-Conditioned RSI Mean-Reversion Strategy (Strategy 12)
# Thesis: RSI-2 overreaction reversals are concentrated in high-vol regimes.
# Enters long/short RSI-2 signals ONLY when realized vol is in the top quartile
# of its trailing 252-bar distribution.
#
# TOS approximation note: ThinkorSwim has no PercentRank function for arbitrary
# indicator series, and the Sum() approach Sum(volRaw < volRaw[0], n) does NOT
# work in thinkScript — inside Sum, every reference (including [0]) is shifted
# by the iteration offset, so volRaw[0] evaluates to volRaw at each offset
# (i.e., the comparison is always volRaw[i] < volRaw[i] = false). This is a
# fundamental thinkScript limitation: there is no way to "freeze" the current
# bar's value inside a Sum.
#
# WORKAROUND USED HERE: min-max range normalization via Highest/Lowest.
#   volRank = (volRaw - Lowest(volRaw, n)) / (Highest(volRaw, n) - Lowest(volRaw, n))
# This gives 0.0 when vol is at its window minimum, 1.0 at its window maximum.
# At volThreshold=0.75, highVol fires when vol is in the top 25% of its
# historical RANGE — conceptually similar to top-quartile of DISTRIBUTION but
# not identical (the distribution approximation differs when vol is skewed).
#
# DIVERGENCE FROM PYTHON: the Python implementation uses
# pd.Series.rolling(vol_lookback).rank(pct=True), which is a count-based
# percentile rank (fraction of historical values <= current value). The TOS
# min-max rank is a monotone transform of the true percentile rank but will
# differ numerically. The filter activates in similar regimes in practice but
# may fire on a slightly different set of bars. This is not a defect — it is
# an inherent limitation of approximating a count-based rank in a language
# without native rolling rank support.
#
# Signal fill model: open[-1] = next-bar open, matching the Python engine's
# open[t+1] fill model (signal at t filled at open[t+1]).
#
# Load: Charts > Studies > Edit Studies > Strategies > Create > paste > apply

input volWindow = 21;
input volLookback = 252;
input volThreshold = 0.75;
input rsiWindow = 2;
input rsiEntryLong = 10;
input rsiExitLong = 70;
input rsiEntryShort = 90;
input rsiExitShort = 30;

def dailyReturn = close / close[1] - 1;
def volRaw = StdDev(dailyReturn, volWindow) * Sqrt(252);

# Min-max range normalization: 0.0 = window minimum, 1.0 = window maximum.
# This is a working approximation for the Python count-based percentile rank.
# See header note for why Sum(volRaw < volRaw[0], n) cannot be used in thinkScript.
def volHigh = Highest(volRaw, volLookback);
def volLow = Lowest(volRaw, volLookback);
def volRank = if volHigh != volLow then (volRaw - volLow) / (volHigh - volLow) else 0.5;
def highVol = volRank >= volThreshold;

def rsiValue = RSI(Length = rsiWindow);

def enterLong = highVol and rsiValue crosses below rsiEntryLong;
def exitLong = rsiValue crosses above rsiExitLong;
def enterShort = highVol and rsiValue crosses above rsiEntryShort;
def exitShort = rsiValue crosses below rsiExitShort;

addOrder(OrderType.BUY_AUTO, enterLong, open[-1], 1, Color.GREEN, Color.GREEN, "VolRSI Buy");
addOrder(OrderType.SELL_TO_CLOSE, exitLong, open[-1], 1, Color.RED, Color.RED, "VolRSI Sell");
addOrder(OrderType.SELL_SHORT, enterShort, open[-1], 1, Color.ORANGE, Color.ORANGE, "VolRSI Short");
addOrder(OrderType.BUY_TO_COVER, exitShort, open[-1], 1, Color.CYAN, Color.CYAN, "VolRSI Cover");

plot rsiPlot = rsiValue;
plot entryLongLine = rsiEntryLong;
plot exitLongLine = rsiExitLong;
plot entryShortLine = rsiEntryShort;
plot exitShortLine = rsiExitShort;
plot volRankPlot = volRank * 100;
plot volThresholdLine = volThreshold * 100;

rsiPlot.SetDefaultColor(Color.CYAN);
entryLongLine.SetDefaultColor(Color.GREEN);
exitLongLine.SetDefaultColor(Color.YELLOW);
entryShortLine.SetDefaultColor(Color.RED);
exitShortLine.SetDefaultColor(Color.ORANGE);
volRankPlot.SetDefaultColor(Color.BLUE);
volThresholdLine.SetDefaultColor(Color.MAGENTA);
