# 52-Week High Proximity Strategy (George & Hwang 2004, anchoring bias)
# Thesis: stocks approaching their 52-week high outperform due to investor anchoring.
#   Participants under-react to positive news near the 52-week high, causing delayed
#   price discovery that resolves as the stock drifts toward and through the anchor.
# Parameters: proximityThreshold=0.95, exitThreshold=0.90
# Load: Charts > Studies > Edit Studies > Strategies tab > Create > paste > apply to SPY Daily.
# Signal logic: addOrder BUY_AUTO when ratio >= 0.95 and ratio > ratio[1]; SELL_TO_CLOSE when ratio < 0.90
# TOS fill note: open[-1] = next-bar open, matching the Python engine's open[t+1] fill model.
# Parity: Python and TOS use identical entry/exit logic. TOS does not model slippage;
#   the Python engine applies 5 bps commission + 5 bps slippage per side.
# Platform divergence: TOS Highest() defaults to the bar's high field, not close.
#   Use Highest(close, 252) explicitly to match the Python rolling-max-of-close semantics.
#   Position size is always 1 share in TOS; the Python engine uses full 1.0 exposure.

input proximityThreshold = 0.95;
input exitThreshold = 0.90;

def high52 = Highest(close, 252);
def ratio = close / high52;

def entryCondition = ratio >= proximityThreshold and ratio > ratio[1];
def exitCondition = ratio < exitThreshold;

AddOrder(OrderType.BUY_AUTO, entryCondition, open[-1], 1, Color.GREEN, Color.GREEN, "52wk Proximity Buy");
AddOrder(OrderType.SELL_TO_CLOSE, exitCondition, open[-1], 1, Color.RED, Color.RED, "52wk Proximity Sell");

plot ratioLine = ratio;
plot proximityLevel = proximityThreshold;
plot exitLevel = exitThreshold;
ratioLine.SetDefaultColor(Color.CYAN);
proximityLevel.SetDefaultColor(Color.GREEN);
exitLevel.SetDefaultColor(Color.RED);
