# NR7 Volatility-Contraction Breakout Strategy (Crabel 1990)
# Thesis: bar with the narrowest range in 7 days marks peak volatility compression;
#   accumulated stop-orders beyond the range edges trigger a self-reinforcing breakout.
# Parameters: nBars=7, exitBars=4 (both Crabel 1990 published defaults)
# True-range definition: high - low. This script uses high - low directly rather than
#   the built-in TrueRange() function. ThinkorSwim's TrueRange() adds prior-close gaps:
#   TrueRange() = max(high - low, abs(high - close[1]), abs(low - close[1])).
#   Python strategy.py also uses high - low, so both implementations are identical.
#   On gap-open days the two definitions diverge; high - low is the more conservative
#   measure and matches Crabel's original bar-range definition exactly.
# Parity note: Python engine runs with allow_short=False (default), so short signals
#   are treated as flat. ThinkorSwim implements both long and short sides as written.
# Signal logic: addOrder BUY_AUTO on NR7 long; SELL_TO_CLOSE after exitBars bars
# Load: Charts > Studies > Edit Studies > Strategies tab > Create > paste > apply Daily.
# Fill note: open[-1] = next-bar open, matching Python engine t+1 open fill model.

input nBars = 7;
input exitBars = 4;

# True range = high - low (NOT TrueRange() — see header comment)
def tr = high - low;
def rollingMinTR = Lowest(tr, nBars);
def isNR7 = tr == rollingMinTR;
def midpoint = (high + low) / 2;

# Direction filter: close above midpoint → long candidate; at or below → short candidate
def nr7Long = isNR7 and close > midpoint;
def nr7Short = isNR7 and close <= midpoint;

# Time-based exit counter.
# -1 = no position. 0 = entry bar. 1..exitBars-2 = hold bars. Transitions to -1 at exitBars-1.
# New NR7 while in position is ignored (hold condition takes priority via ordering of branches).
def longCounter;
longCounter = if longCounter[1] >= 0 and longCounter[1] < exitBars - 1
                  then longCounter[1] + 1
              else if nr7Long and longCounter[1] < 0 and shortCounter[1] < 0
                  then 0
              else -1;

def shortCounter;
shortCounter = if shortCounter[1] >= 0 and shortCounter[1] < exitBars - 1
                   then shortCounter[1] + 1
               else if nr7Short and longCounter[1] < 0 and shortCounter[1] < 0
                   then 0
               else -1;

# Entry fires on the bar the counter is set to 0 (transitioning from -1)
def enterLong  = longCounter  == 0 and longCounter[1]  < 0;
def enterShort = shortCounter == 0 and shortCounter[1] < 0;

# Exit fires on the bar the counter drops back to -1 (after holding exitBars bars)
def exitLong  = longCounter  == -1 and longCounter[1]  >= 0;
def exitShort = shortCounter == -1 and shortCounter[1] >= 0;

AddOrder(OrderType.BUY_AUTO,     enterLong,  open[-1], 1, Color.GREEN, Color.GREEN, "NR7 Long Entry");
AddOrder(OrderType.SELL_TO_CLOSE, exitLong,  open[-1], 1, Color.RED,   Color.RED,   "NR7 Long Exit");
AddOrder(OrderType.SELL_SHORT,   enterShort, open[-1], 1, Color.RED,   Color.RED,   "NR7 Short Entry");
AddOrder(OrderType.BUY_TO_CLOSE,  exitShort, open[-1], 1, Color.GREEN, Color.GREEN, "NR7 Short Exit");

# Visualization: highlight NR7 bars with an arrow at the low
plot nr7Mark = if isNR7 then low else Double.NaN;
nr7Mark.SetDefaultColor(Color.YELLOW);
nr7Mark.SetPaintingStrategy(PaintingStrategy.ARROW_UP);

plot trLine = tr;
trLine.SetDefaultColor(Color.GRAY);
trLine.SetLineWeight(1);
