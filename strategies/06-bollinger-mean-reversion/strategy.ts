# Bollinger Band Mean-Reversion Strategy
# Thesis: close below lower Bollinger Band signals distributional overextension;
#         exit when close crosses back above the middle band (SMA).
# Parameters: window=20, nStd=2.0 (Bollinger 2001 defaults)
# Load: Charts > Studies > Edit Studies > Strategies > Create > paste > apply
# Signal logic: addOrder BUY_AUTO when close crosses below lower band; SELL_TO_CLOSE when close crosses above SMA
# TOS fill note: open[-1] = next-bar open, matching the Python engine's open[t+1] fill model
# Platform divergence: TOS does not model explicit slippage; Python engine applies 5 bps commission + 5 bps slippage per side.
#   Apply a ~10 bps per-trade haircut when comparing live TOS P&L to backtested metrics.

input window = 20;
input nStd = 2.0;

def middleBand = Average(close, window);
def bandWidth = StdDev(close, window);
def lowerBand = middleBand - nStd * bandWidth;

def enterLong = close crosses below lowerBand;
def exitLong = close crosses above middleBand;

AddOrder(OrderType.BUY_AUTO, enterLong, open[-1], 1, Color.GREEN, Color.GREEN, "BB Buy");
AddOrder(OrderType.SELL_TO_CLOSE, exitLong, open[-1], 1, Color.RED, Color.RED, "BB Sell");

plot middleBandPlot = middleBand;
plot lowerBandPlot = lowerBand;
middleBandPlot.SetDefaultColor(Color.YELLOW);
lowerBandPlot.SetDefaultColor(Color.RED);
