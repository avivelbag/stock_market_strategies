# Donchian Channel Turtle Breakout Strategy (Turtle System 1, Dennis/Eckhardt 1983)
# Thesis: close breaking above N-bar channel high signals supply exhaustion and trend continuation
# Parameters: entryWindow=20, exitWindow=10, atrWindow=20 (all published Turtle System 1 defaults)
# Load: Charts > Studies > Edit Studies > Strategies tab > Create > paste > apply to SPY Daily.
# Signal logic: addOrder BUY_AUTO when close breaks above 20-bar channel high; SELL_TO_CLOSE below 10-bar low
# TOS fill note: open[-1] = next-bar open, matching the Python engine's open[t+1] fill model
# Parity: Python and TOS use identical entry/exit logic. TOS does not model slippage;
#   the Python engine applies 5 bps commission + 5 bps slippage per side.
# Platform divergence: TOS Highest/Lowest include bar[1] offset to exclude the current bar,
#   matching the Python look-back of prior N closes. atrWindow is accepted as an input for
#   interface parity but TOS ATR sizing is not applied — position size is always 1 share.

input entryWindow = 20;
input exitWindow = 10;
input atrWindow = 20;

def channelHigh = Highest(close[1], entryWindow);
def channelLow = Lowest(close[1], exitWindow);

def enterLong = close crosses above channelHigh;
def exitLong = close crosses below channelLow;

AddOrder(OrderType.BUY_AUTO, enterLong, open[-1], 1, Color.GREEN, Color.GREEN, "Donchian Buy");
AddOrder(OrderType.SELL_TO_CLOSE, exitLong, open[-1], 1, Color.RED, Color.RED, "Donchian Sell");

plot channelHighLine = channelHigh;
plot channelLowLine = channelLow;
channelHighLine.SetDefaultColor(Color.CYAN);
channelLowLine.SetDefaultColor(Color.YELLOW);
