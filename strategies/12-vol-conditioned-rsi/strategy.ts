# Volatility-Conditioned RSI Mean-Reversion Strategy (Strategy 12)
# Thesis: RSI-2 overreaction reversals are concentrated in high-vol regimes.
# Enters long/short RSI-2 signals ONLY when realized vol is in the top quartile
# of its trailing 252-bar distribution.
#
# TOS approximation note: ThinkorSwim has no PercentRank function for arbitrary
# indicator series. The percentile rank of current vol within its trailing window
# is approximated as:
#   volRank = Sum(volRaw < volRaw[0], volLookback) / volLookback
# This counts how many of the past volLookback bars had vol below the current
# bar's vol. Small numerical differences from the Python rolling rank (which uses
# inclusive equality and average-method ties) are expected and not a defect.
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

# Manual percentile rank: fraction of past volLookback bars with vol < current vol
def volRank = Sum(volRaw < volRaw[0], volLookback) / volLookback;
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
