# Dual EMA Crossover Momentum Strategy
# Thesis: long when short-term EMA crosses above long-term EMA (momentum premium)
# Parameters: fast_window=20 (fastLength), slow_window=60 (slowLength)
# Load: Charts > Studies > Edit Studies > Strategies > Create > paste > apply
# Signal logic: addOrder BUY_AUTO on crossover-up, SELL_TO_CLOSE on crossover-down

input fastLength = 20;
input slowLength = 60;

def fastEMA = ExpAverage(close, fastLength);
def slowEMA = ExpAverage(close, slowLength);

def crossUp = fastEMA crosses above slowEMA;
def crossDown = fastEMA crosses below slowEMA;

AddOrder(OrderType.BUY_AUTO, crossUp, open[-1], 1, Color.GREEN, Color.GREEN, "EMA Buy");
AddOrder(OrderType.SELL_TO_CLOSE, crossDown, open[-1], 1, Color.RED, Color.RED, "EMA Sell");

plot fastLine = fastEMA;
plot slowLine = slowEMA;
fastLine.SetDefaultColor(Color.CYAN);
slowLine.SetDefaultColor(Color.YELLOW);
