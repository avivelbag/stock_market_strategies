// Pairs Mean-Reversion — ThinkorSwim single-leg proxy
//
// PARITY STATEMENT / NOTED DIVERGENCE:
// The Python strategy trades two legs simultaneously (long-A / short-B or
// short-A / long-B) as a dollar-neutral spread. ThinkorSwim (TOS) cannot
// execute two instruments from a single strategy script. This port instead
// trades a *single-leg proxy*: it approximates the spread as the ratio of
// the current bar's close to a historical reference close (close[window-1]),
// applying the same z-score entry/exit logic to this intra-symbol price
// ratio. This divergence is deliberate and expected: the TOS script serves
// only as a directional signal validator, not as a full pairs implementation.
// The Python engine is the ranking source of truth.
//
// Parameters (Gatev, Goetzmann & Rouwenhorst 2006 defaults):
//   Z_ENTRY = 2.0  — standard deviations to trigger entry
//   Z_EXIT  = 0.5  — standard deviations to trigger exit
//   WINDOW  = 60   — rolling look-back for spread mean and std

declare Z_ENTRY = 2.0;
declare Z_EXIT = 0.5;
declare WINDOW = 60;

def spreadProxy = close / close[WINDOW - 1];
def logSpread = Log(spreadProxy);

def rollingMean = Average(logSpread, WINDOW);
def rollingStd = StdDev(logSpread, WINDOW);
def zScore = if rollingStd > 0 then (logSpread - rollingMean) / rollingStd else 0;

def inLong = CompoundValue(1,
    if !inLong[1] && zScore < -Z_ENTRY then 1
    else if inLong[1] && zScore > -Z_EXIT then 0
    else inLong[1], 0);

def inShort = CompoundValue(1,
    if !inShort[1] && zScore > Z_ENTRY then 1
    else if inShort[1] && zScore < Z_EXIT then 0
    else inShort[1], 0);

addOrder(OrderType.BUY_AUTO, inLong && !inLong[1], open[-1], 1, CustomColor.GREEN, CustomColor.GREEN, "Long Entry");
addOrder(OrderType.SELL_TO_CLOSE, !inLong && inLong[1], open[-1], 1, CustomColor.GREEN, CustomColor.GREEN, "Long Exit");
addOrder(OrderType.SELL_SHORT_AUTO, inShort && !inShort[1], open[-1], 1, CustomColor.RED, CustomColor.RED, "Short Entry");
addOrder(OrderType.BUY_TO_CLOSE, !inShort && inShort[1], open[-1], 1, CustomColor.RED, CustomColor.RED, "Short Exit");
