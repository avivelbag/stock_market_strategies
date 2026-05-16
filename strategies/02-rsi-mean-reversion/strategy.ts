# RSI Mean-Reversion Strategy (Connors RSI-2 variant)
# Thesis: RSI extremes proxy for crowd overreaction; trade the mean reversion
# Parameters: rsiPeriod=2, oversoldLevel=10, overboughtLevel=90
# Load: Charts > Studies > Edit Studies > Strategies > Create > paste > apply
# Signal logic: addOrder BUY_AUTO when RSI crosses below oversold; SELL_TO_CLOSE when above overbought
# TOS fill note: open[-1] = next-bar open, matching the Python engine's open[t+1] fill model

input rsiPeriod = 2;
input oversoldLevel = 10;
input overboughtLevel = 90;

def rsiValue = RSI(Length = rsiPeriod);

def enterLong = rsiValue crosses below oversoldLevel;
def exitLong = rsiValue crosses above overboughtLevel;

AddOrder(OrderType.BUY_AUTO, enterLong, open[-1], 1, Color.GREEN, Color.GREEN, "RSI Buy");
AddOrder(OrderType.SELL_TO_CLOSE, exitLong, open[-1], 1, Color.RED, Color.RED, "RSI Sell");

plot rsiPlot = rsiValue;
plot oversoldLine = oversoldLevel;
plot overboughtLine = overboughtLevel;
rsiPlot.SetDefaultColor(Color.CYAN);
oversoldLine.SetDefaultColor(Color.GREEN);
overboughtLine.SetDefaultColor(Color.RED);
