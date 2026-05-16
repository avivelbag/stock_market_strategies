# Turn-of-Month Calendar Effect Strategy (Lakonishok & Smidt 1988 / Ariel 1987)
# Thesis: institutional window-dressing and month-start cash inflows create
#   predictable demand in the last tailDays and first headDays trading bars
#   of each calendar month.
# Parameters: tailDays=2, headDays=3 — published Lakonishok & Smidt (1988) defaults.
# Load: Charts > Studies > Edit Studies > Strategies tab > Create > paste > apply to SPY Daily.
# Signal logic: addOrder BUY_AUTO when TOM window starts; SELL_TO_CLOSE when it ends.
# TOS fill note: open[-1] = next-bar open, matching the Python engine's open[t+1] fill model.
# Parity (HEAD): monthChanged = GetMonth() != GetMonth()[1] detects the first trading
#   bar of each month with no look-ahead, bar-for-bar matching strategy.py's index grouping.
# Parity (TAIL): uses forward month references GetMonth()[-k] (k=1..5) to detect the
#   last tailDays trading bars of each month. This is one-to-five-bar look-ahead in TOS
#   (standard practice for TOS backtesting). strategy.py uses BMonthEnd calendar arithmetic
#   for the same computation; for synthetic data (no holidays) both approaches are bar-for-bar
#   identical. For real equity data with market holidays, a <=1 bar divergence may occur on
#   months where a holiday falls on the final trading day.
# Platform divergence: TOS does not model slippage; the Python engine applies 5 bps
#   commission + 5 bps slippage per side. Position size is 1 share in TOS; Python uses
#   full 1.0 exposure. TOS ThinkScript offsets must be integer literals, so tailDays
#   is supported via explicit if-else branches for values 1 through 5.

input tailDays = 2;
input headDays = 3;

# HEAD: first headDays trading bars of the month (no look-ahead).
# GetMonth() != GetMonth()[1] is True on the first bar of a new month.
def monthChanged = GetMonth() != GetMonth()[1];
def headCount = if monthChanged then 1 else headCount[1] + 1;
def inHead = headCount <= headDays;

# TAIL: last tailDays trading bars of the month.
# GetMonth()[-k] is the month k bars into the future (look-ahead, standard TOS practice).
# inTail is True when the month changes within the next tailDays bars (inclusive).
def m = GetMonth();
def inTail = if tailDays >= 5 then m != GetMonth()[-5]
             else if tailDays >= 4 then m != GetMonth()[-4]
             else if tailDays >= 3 then m != GetMonth()[-3]
             else if tailDays >= 2 then m != GetMonth()[-2]
             else m != GetMonth()[-1];

def inTOM = inHead or inTail;

# Entry fires on the first bar of the TOM window; exit fires on the first bar outside it.
AddOrder(OrderType.BUY_AUTO, inTOM and !inTOM[1], open[-1], 1, Color.GREEN, Color.GREEN, "TOM Entry");
AddOrder(OrderType.SELL_TO_CLOSE, !inTOM and inTOM[1], open[-1], 1, Color.RED, Color.RED, "TOM Exit");

plot tomSignal = if inTOM then 1 else 0;
tomSignal.SetDefaultColor(Color.CYAN);
tomSignal.SetPaintingStrategy(PaintingStrategy.HISTOGRAM);
