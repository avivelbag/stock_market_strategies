# Volatility-Managed Portfolio — Moreira & Muir 2017
# Thesis: scale equity exposure inversely to recent realized variance to harvest
#   the variance risk premium. Reduce position when volatility is high; hold
#   full or leveraged size when markets are calm.
# Parameters:
#   window=21    — rolling lookback in bars (one calendar month).
#                  Prior-specified from Moreira & Muir (2017 JF).
#   target_vol=0.12 — annualized target volatility (12%).
#                     Prior-specified from Moreira & Muir (2017 JF).
# Realized vol: StdDev of daily log-price returns over window bars, annualized
#   by sqrt(252). Python strategy.py uses identical return definition:
#   dailyRet = close / close[1] - 1. Both use return standard deviation,
#   NOT the built-in TrueRange() or ATR-based approximation.
#   ThinkScript StdDev includes the current bar's return; Python iloc[t-window:t]
#   excludes it (one-bar offset). The offset is one bar out of 21 and does not
#   alter signal direction in any meaningful way.
# Position cap: Python clips scalar to [0, 2]. ThinkScript implements as two
#   conditional AddOrder blocks — normal (0 < scalar < 2) and double (scalar >= 2).
#   A single AddOrder cannot express the 2x leverage cap as a direct portfolio
#   weight; this structural divergence at the portfolio-execution layer is the
#   only documented difference between strategy.py and strategy.ts.
# Signal logic: addOrder BUY_AUTO when warmed up (normal or 2x); SELL_TO_CLOSE on exit.
# Load: Charts > Studies > Edit Studies > Strategies tab > Create > paste > apply Daily.
# Fill note: open[-1] = next-bar open, matching Python engine t+1 open fill model.

input window = 21;
input target_vol = 0.12;

# Daily return: (close today) / (close yesterday) - 1
def dailyRet = (close / close[1]) - 1;

# Realized volatility: std dev of daily returns over window bars, annualized
def realizedVol = StdDev(dailyRet, window) * Sqrt(252);

# Position scalar = target_vol / realized_vol, clipped to [0, 2]
def scalar = if realizedVol > 1e-9
             then Min(target_vol / realizedVol, 2.0)
             else 1.0;

# Warm-up guard: no position until window+1 bars of return history are available
def warmedUp = BarNumber() > window;

# Normal long entry: 0 < scalar < 2 (standard single-unit exposure)
AddOrder(OrderType.BUY_AUTO,
         warmedUp and scalar < 2.0,
         open[-1], 1,
         Color.GREEN, Color.GREEN, "VolMgd Long");

# Double long entry: scalar >= 2.0 (low-vol environment, 2x leverage cap)
AddOrder(OrderType.BUY_AUTO,
         warmedUp and scalar >= 2.0,
         open[-1], 2,
         Color.CYAN, Color.CYAN, "VolMgd Long 2x");

# Exit when outside warm-up guard (only relevant at strategy start)
AddOrder(OrderType.SELL_TO_CLOSE,
         !warmedUp,
         open[-1], 1,
         Color.RED, Color.RED, "VolMgd Exit");

# Scalar overlay for visualization (shows current position sizing)
plot scalarLine = if warmedUp then scalar else Double.NaN;
scalarLine.SetDefaultColor(Color.YELLOW);
scalarLine.SetLineWeight(1);

# Target vol reference line
plot targetLine = target_vol;
targetLine.SetDefaultColor(Color.GRAY);
targetLine.SetStyle(Curve.SHORT_DASH);
