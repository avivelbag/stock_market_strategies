# Absolute Momentum (Trend Filter) Strategy
# Thesis: hold long when the asset's own trailing lookback-bar return is positive;
#         exit to cash otherwise (Antonacci 2014 absolute momentum / crash filter).
# Parameters: lookback=252 (one trading year, Jegadeesh-Titman canonical value),
#             threshold=0.0 (any positive trailing return = long)
# Load: Charts > Studies > Edit Studies > Strategies > Create > paste > apply
# Signal logic: addOrder BUY_AUTO when trailing return crosses above threshold;
#               addOrder SELL_TO_CLOSE when trailing return drops to threshold or below
# TOS fill note: open[-1] = next-bar open, matching the Python engine's open[t+1] fill model
# Platform divergence: TOS does not model explicit slippage; Python engine applies 5 bps commission + 5 bps slippage per side.
#   Apply a ~10 bps per-trade haircut when comparing live TOS P&L to backtested metrics.
# TOS StdDev note: trailing return uses close[lookback] which is native to thinkScript aggregation.

input lookback = 252;
input threshold = 0.0;

def trailingReturn = (close - close[lookback]) / close[lookback];
def isLong = trailingReturn > threshold;

AddOrder(OrderType.BUY_AUTO, isLong and !isLong[1], open[-1], 1, Color.GREEN, Color.GREEN, "AbsMom Buy");
AddOrder(OrderType.SELL_TO_CLOSE, !isLong and isLong[1], open[-1], 1, Color.RED, Color.RED, "AbsMom Sell");

plot trailingReturnLine = trailingReturn;
plot thresholdLine = threshold;
trailingReturnLine.SetDefaultColor(Color.CYAN);
thresholdLine.SetDefaultColor(Color.YELLOW);
