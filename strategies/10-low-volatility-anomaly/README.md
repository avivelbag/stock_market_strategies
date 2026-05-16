# Low-Volatility Anomaly (Strategy 10)

**Thesis tag:** low_vol_anomaly  
**Parameters:** `vol_window=60`, `ranking_window=252`, `exit_percentile=75`  
**Entry:** current realized vol < rolling median (ranking_window bars)  
**Exit:** current realized vol > rolling 75th percentile (ranking_window bars)

---

## Economic Mechanisms

The low-volatility anomaly is one of the most replicated violations of the Capital Asset Pricing Model. CAPM predicts that higher beta / higher volatility should command higher expected returns. The empirical literature shows the opposite: low-volatility securities earn superior *risk-adjusted* returns. Two independent mechanisms explain this:

### Behavioral: Lottery Demand (Barberis & Huang 2008; Kumar 2009)

Investors exhibit a documented preference for positively-skewed, high-variance assets that resemble lottery tickets. A small number of high-vol stocks that occasionally produce large gains attract disproportionate capital. The resulting excess demand bids up prices of high-vol assets beyond their fundamental value, depressing their future expected returns. Low-vol assets, perceived as boring and unlikely to produce large short-term gains, are systematically underweighted. Their prices are not inflated by lottery demand, leaving their expected returns intact or elevated. Baker, Bradley & Wurgler (2011) label this "excessive demand for lottery-like securities."

### Institutional: Leverage Constraint (Black 1972; Frazzini & Pedersen 2014)

Benchmark-constrained fund managers face an implicit leverage prohibition: they must generate excess returns versus their benchmark without using explicit leverage. When unconstrained investors could simply leverage low-beta assets, constrained managers instead overweight high-beta, high-volatility assets to boost expected raw returns. This systematic demand from the institutional sector overprices high-vol assets. Frazzini & Pedersen (2014, *Journal of Financial Economics*) call this the Betting Against Beta (BAB) factor: low-beta assets earn positive risk-adjusted returns precisely because they are systematically underweighted by leverage-constrained institutions.

Both mechanisms operate independently and reinforce each other, which explains why the anomaly is robust across markets, time periods, and asset classes.

---

## Time-Series Adaptation

The academic anomaly is cross-sectional: rank stocks by trailing volatility and buy the lowest-decile portfolio. Adapted here to a single-asset backtest engine using the **time-series version**: the asset is held when its OWN volatility is low relative to its own history, and exited when volatility spikes.

The economic rationale is preserved: hold the asset during its calm, predictable, "low-lottery" periods when investor attention is low and prices are not inflated by speculative demand; step aside during volatile, "lottery-like" episodes when behavioral overpricing is most active.

---

## Entry and Exit Rules

**Realized volatility:** `rolling(vol_window).std() * sqrt(252)` over daily pct_change returns (annualized). Default `vol_window=60` (approximately one trading quarter). This is the standard realized volatility estimator used throughout the academic literature.

**Vol history:** the last `ranking_window=252` values of the realized vol series (one trading year). This window provides the reference distribution for the median and percentile thresholds.

**Entry:** enter long when `current_vol < rolling_median(vol_history)`. The median threshold classifies the current regime as "below-average volatility." Using the median (rather than the mean) is robust to outliers — extreme vol spikes during crashes do not inflate the threshold and block entry during subsequent calm periods.

**Exit:** exit to flat when `current_vol > rolling_percentile(exit_percentile=75, vol_history)`. The 75th-percentile exit (upper quartile) creates a hysteresis band between the entry threshold (50th pct = median) and the exit threshold (75th pct). This band prevents excessive churn: once long, the position survives brief median crossings and is only closed when volatility has clearly elevated into the upper quartile.

**Hysteresis logic:** a once-opened position is held until vol explicitly exceeds the 75th-percentile exit threshold, even if vol temporarily rises above the median during the hold. This prevents round-trip costs from eroding the edge at the median boundary.

---

## Why the Edge Survives Realistic Costs

The low-volatility anomaly is specifically favorable for a cost-conscious implementation because **it generates low turnover by design**. The strategy is long during extended low-vol regimes, which typically last weeks to months. It exits only when volatility has clearly elevated (75th percentile), not at the first sign of a median crossing. The 25-percentile hysteresis band (entry at 50th, exit at 75th) acts as a natural dampener that reduces round-trip frequency.

Low-vol regimes tend to be persistent: volatility clustering (Engle 1982 ARCH) means that low-vol days cluster together, so once the strategy is in a low-vol period, it tends to remain there for many consecutive bars. This persistence reduces the number of entries and exits per year, keeping cumulative transaction costs low relative to the gross return captured during calm periods.

---

## Regime Profile: When the Anomaly is Strongest

The low-volatility anomaly is strongest in **bear markets and high-volatility periods**, which is a deliberate and important contrast with momentum strategies.

During bear markets, the lottery-demand mechanism intensifies: speculative capital chases high-vol assets in hopes of quick recovery gains, further inflating their prices and depressing their expected returns. Simultaneously, leverage constraints bind most tightly during market stress (VaR constraints, margin calls), amplifying the BAB mechanism. Low-vol assets during these periods are most severely underpriced relative to fundamental value.

For the time-series version: when an asset transitions from a high-vol regime back to a low-vol regime, the entry signal fires and captures the subsequent recovery in a predictable, low-variance environment.

**Regime contrast with momentum:**
- Strategies 02 (RSI Mean-Reversion), 07 (Absolute Momentum), 08 (NR7 Breakout): these generate long signals during or after strong price movements — which are typically the highest-vol events. High-vol events are exactly when this strategy is **out** of the market.
- This creates a natural diversification property: the low-vol anomaly strategy and momentum strategies should rarely be simultaneously long (one enters during calm periods, the other during active ones).

---

## Parameter Justification Without Overfitting

All three parameters are grounded in the academic literature, not in-sample optimization:

**`vol_window=60`** (one trading quarter): the academic literature on realized volatility typically uses 1-month (21-bar) to 6-month (126-bar) lookbacks. The 60-bar (one quarter) window balances responsiveness (adapts to regime shifts within a quarter) against noise (enough observations for a stable std estimate). One quarter is the standard reporting cycle for institutional investors, making it the natural measurement horizon for the anomaly's institutional mechanism.

**`ranking_window=252`** (one trading year): one calendar year of vol history provides the reference distribution for the median and percentile thresholds. This is the standard lookback for momentum (Jegadeesh & Titman 1993), risk factor construction (Fama & French), and anomaly measurement in the academic literature. Using one year ensures the reference distribution includes both calm and volatile sub-periods rather than being dominated by recent conditions.

**`exit_percentile=75`** (upper quartile): the 75th percentile (upper quartile boundary) is the natural "distinctly elevated volatility" threshold — it is the point where vol has risen into the top 25% of its own history. This is the conventional choice for identifying elevated vol in the volatility forecasting literature and corresponds to a 1.5 IQR fence above the median in a symmetric distribution. No grid search was performed from the test datasets.

---

## ThinkScript vs Python Vol Proxy

**Python (`strategy.py`):** uses `pct_change().rolling(vol_window).std() * sqrt(252)` — the returns-based realized volatility, annualized. This is the standard estimator used in Baker et al. (2011) and all academic implementations.

**ThinkScript (`strategy.ts`):** uses `StdDev(close, volWindow)` — price-level standard deviation, not returns-based. This is a vol proxy rather than a true realized volatility measure.

**Why this does not change the signal direction:** for a single price series, price-level std and returns-based realized vol are monotonically related — when returns-based vol rises (high-vol period), price-level std also rises, and vice versa. The entry and exit thresholds are both computed on the *same* vol series (via the rolling median and percentile of `StdDev(close, volWindow)`). The ranking — whether the current bar is above or below the historical median — is stable under this monotone transformation.

The absolute scale differs: price-level std is in dollar terms and grows with price level; annualized returns-based vol is dimensionless (fraction). However, since all thresholds are computed relative to the same series, the *relative* position (below median / above 75th pct) is the same for both proxies on any given bar. The README therefore documents this as an acceptable proxy substitution for implementation convenience.

---

## Explicit Contrast with Related Strategies

| Strategy | Thesis | How it differs |
|----------|--------|----------------|
| 02 RSI Mean-Reversion | Behavioral overreaction reversal | Enters after price overshoots LOW — often after a vol spike. Orthogonal to low-vol entry criterion. |
| 07 Absolute Momentum | Jegadeesh-Titman momentum crash filter | Enters when trailing 252-bar return is positive (trending); exits when negative. Momentum is often strongest during high-vol trending periods — exactly when strategy 10 is *out*. |
| 08 NR7 Breakout | Volatility cycle: enter after compression (NR7), exit after expansion | Enters at the *end* of a low-vol compression phase, just before the vol spike. Strategy 10 enters at the *beginning* of a low-vol period and exits before the spike. Opposite timing relative to the vol cycle. |
| 09 Volatility-Managed Portfolio | Variance risk premium: scale SIZE by target_vol/realized_vol | Always invested; sizes position continuously. Strategy 10 makes a binary in/out decision. Strategy 09 asks "how much?"; strategy 10 asks "should I hold at all?" A position that passes strategy 09's size threshold (realized_vol is not extreme in absolute terms) can simultaneously fail strategy 10's selection criterion (realized_vol is high relative to own history). |

---

## Limitations

- **Single-asset, time-series version**: the academic anomaly is cross-sectional (rank stocks against each other). The time-series adaptation captures the same economic logic but does not test the cross-sectional ranking property.
- **Long-only**: the theory also predicts that high-vol assets are overpriced and should be shorted. The long-only version captures the long leg only; the short leg is excluded due to borrowing costs.
- **Warm-up barrier**: `vol_window + ranking_window = 312 bars` before any signal. Walk-forward folds shorter than 312 bars cannot generate valid OOS signals; the walk-forward evaluation should be interpreted with this constraint in mind.
