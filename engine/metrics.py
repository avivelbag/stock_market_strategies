"""Metrics for evaluating backtest equity curves.

All functions operate on a pd.Series equity curve or auxiliary position series
and return a scalar. Keep all math here — backtest.py calls these; strategies never do.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
from scipy.stats import norm as _norm

ANNUALIZE = 252  # trading days per year
_EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant used in DSR Emax formula


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate.

    Blind spot: ignores path — identical CAGR can hide extreme volatility or
    lucky timing over a short window.

    Args:
        equity: Portfolio value series (must have at least 2 points, all positive).

    Returns:
        Annualized geometric return as a decimal (e.g. 0.12 = 12%).
    """
    n_returns = len(equity) - 1
    if n_returns <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (ANNUALIZE / n_returns) - 1)


def volatility(equity: pd.Series) -> float:
    """Annualized standard deviation of daily returns.

    Blind spot: assumes i.i.d. normal returns; understates tail risk and
    ignores autocorrelation (common in trend strategies).

    Args:
        equity: Portfolio value series.

    Returns:
        Annualized volatility as a decimal.
    """
    returns = equity.pct_change().dropna()
    return float(returns.std() * np.sqrt(ANNUALIZE))


def sharpe(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio.

    Blind spot: penalizes upside and downside volatility equally, so a
    strategy with large positive outliers is penalized.

    Args:
        equity: Portfolio value series.
        risk_free_rate: Annualized risk-free rate as a decimal.

    Returns:
        Sharpe ratio (annualized). Returns 0.0 if return std is zero.
    """
    returns = equity.pct_change().dropna()
    rf_daily = risk_free_rate / ANNUALIZE
    excess = returns - rf_daily
    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(ANNUALIZE))


def sortino(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sortino ratio (uses downside deviation only).

    Blind spot: sparse drawdowns give a falsely high value; strategies with
    rare but catastrophic losses may still show high Sortino.

    Args:
        equity: Portfolio value series.
        risk_free_rate: Annualized risk-free rate as a decimal.

    Returns:
        Sortino ratio (annualized). Returns 0.0 if no negative-excess days.
    """
    returns = equity.pct_change().dropna()
    rf_daily = risk_free_rate / ANNUALIZE
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = downside.std()
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(ANNUALIZE))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough percentage decline in the equity curve.

    Blind spot: single worst episode; doesn't capture frequency or duration
    of drawdowns — use time_in_drawdown for that.

    Args:
        equity: Portfolio value series.

    Returns:
        Maximum drawdown as a negative decimal (e.g. -0.20 means 20% drawdown).
    """
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def calmar(equity: pd.Series) -> float:
    """CAGR divided by absolute max drawdown.

    Blind spot: one catastrophic loss ruins the score even for an otherwise
    consistently profitable strategy.

    Args:
        equity: Portfolio value series.

    Returns:
        Calmar ratio. Returns 0.0 if max drawdown is zero.
    """
    mdd = abs(max_drawdown(equity))
    if mdd == 0.0:
        return 0.0
    return float(cagr(equity) / mdd)


def time_in_drawdown(equity: pd.Series) -> float:
    """Fraction of bars where equity is below its running peak.

    Blind spot: treats a 1% and 50% drawdown identically — use
    max_drawdown for severity.

    Args:
        equity: Portfolio value series.

    Returns:
        Fraction of time in drawdown (0.0 to 1.0).
    """
    peak = equity.cummax()
    return float((equity < peak).mean())


def turnover(positions: pd.Series) -> float:
    """Mean absolute daily change in position fraction.

    Blind spot: ignores trade size direction; a flip from +1 to -1 scores
    twice a plain entry from 0 to +1 — both may have very different costs.

    Args:
        positions: Series of position values (e.g. -1, 0, or 1).

    Returns:
        Mean absolute daily position change, seeded from a flat (0) prior so
        the opening entry trade is counted (cost is already deducted in equity).
    """
    prev = positions.shift(1).fillna(0)
    return float((positions - prev).abs().mean())


def hit_rate(equity: pd.Series) -> float:
    """Fraction of trading days with a positive return.

    Blind spot: a few large wins can dominate profit even with a low hit rate
    (e.g. trend-following typically has hit_rate < 0.5 but still profits).

    Args:
        equity: Portfolio value series.

    Returns:
        Hit rate (0.0 to 1.0).
    """
    returns = equity.pct_change().dropna()
    return float((returns > 0).mean())


def tail_ratio(equity: pd.Series) -> float:
    """Ratio of 95th-percentile gain to absolute 5th-percentile loss.

    Blind spot: ignores the distribution shape between tails and is
    sensitive to extreme outliers.

    Args:
        equity: Portfolio value series.

    Returns:
        Tail ratio. Returns 0.0 if the 5th-percentile return is zero.
    """
    returns = equity.pct_change().dropna()
    p95 = float(np.percentile(returns, 95))
    p5 = float(abs(np.percentile(returns, 5)))
    if p5 == 0.0:
        return 0.0
    return float(p95 / p5)


def exposure(positions: pd.Series) -> float:
    """Average fraction of time with a non-zero position.

    Blind spot: a fully-hedged (long + short) book shows 100% exposure
    but is market-neutral.

    Args:
        positions: Series of position values (e.g. -1, 0, or 1).

    Returns:
        Exposure fraction (0.0 to 1.0).
    """
    return float((positions != 0).mean())


def walk_forward_consistency(oos_sharpes: list) -> float:
    """Fraction of OOS walk-forward windows with positive Sharpe ratio.

    Args:
        oos_sharpes: List of OOS Sharpe ratios, one per fold.

    Returns:
        Fraction in [0.0, 1.0]. Returns 0.0 for empty input.
    """
    if not oos_sharpes:
        return 0.0
    return float(np.mean([s > 0 for s in oos_sharpes]))


def sharpe_distribution_stats(equity: pd.Series) -> tuple:
    """Return (skewness, kurtosis) of the daily return distribution.

    These values are required inputs for deflated_sharpe(). The DSR formula
    (Bailey & López de Prado 2014) applies a Cornish-Fisher correction for
    non-normality: skewed or fat-tailed returns inflate the apparent Sharpe ratio,
    and these moments are used to deflate it back to a calibrated t-statistic.

    The kurtosis returned is Pearson's kurtosis (normal distribution = 3.0), not
    excess kurtosis (Fisher's, normal = 0.0). This matches the convention in the
    Bailey & López de Prado formula where the denominator term is (kurtosis - 1)/4,
    which equals 0.5 for a normal distribution.

    Args:
        equity: Portfolio value series.

    Returns:
        (skewness, kurtosis) — both floats. Returns (0.0, 3.0) for series with
        fewer than 4 return observations (normal-distribution defaults).
    """
    returns = equity.pct_change().dropna()
    if len(returns) < 4:
        return 0.0, 3.0
    skewness = float(_scipy_stats.skew(returns))
    kurtosis = float(_scipy_stats.kurtosis(returns, fisher=False))
    return skewness, kurtosis


def deflated_sharpe(
    equity: pd.Series,
    n_trials: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    The standard Sharpe ratio is silent on how many parameter combinations or
    strategies were evaluated before selecting the winner. A strategy that appears
    to have a positive Sharpe may have been cherry-picked from hundreds of trials,
    making it indistinguishable from luck. The DSR closes this blind spot by
    computing the probability that the observed Sharpe exceeds zero after applying
    a family-wise correction for multiple testing, non-normality, and finite sample
    bias. Reference: Bailey, D. H. & López de Prado, M. (2014). "The Deflated Sharpe
    Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    The Journal of Portfolio Management, 40(5), 94–107.

    Formula:
        SR*  = SR_hat × √((T − 1) / (1 − γ₃·SR_hat + ((γ₄ − 1)/4)·SR_hat²))
        Emax = (1 − γ) × √(2·ln(n_trials))   [Ledoit-Wolf approximation; 0 when n_trials=1]
        DSR  = Φ(SR* − Emax)

    where SR_hat is the per-bar (non-annualised) Sharpe, T is the number of return
    bars, γ₃ is skewness, γ₄ is Pearson's kurtosis (normal = 3), γ is the
    Euler-Mascheroni constant (≈ 0.5772), and Φ is the standard normal CDF.

    IMPORTANT: SR_hat is computed per-bar (not annualised) so that T and SR_hat
    are on the same timescale. This differs from the annualised Sharpe in
    compute_all() — do not substitute annualised values here.

    Args:
        equity: Portfolio value series (must have at least 2 points).
        n_trials: Number of strategies or parameter sets evaluated before selecting
            this one. Use 1 for a single, prior-specified strategy (conservative
            lower bound with no multiple-testing correction). Values > 1 apply an
            increasing penalty via the Ledoit-Wolf expected-maximum correction.
        skewness: Skewness of the return distribution. Obtain via
            sharpe_distribution_stats(equity)[0].
        kurtosis: Pearson's kurtosis of the return distribution (normal = 3.0).
            Obtain via sharpe_distribution_stats(equity)[1].

    Returns:
        DSR in [0, 1]. Values near 1 indicate the edge is likely real even after
        multiple-testing correction; values near 0 indicate the observed Sharpe
        is consistent with random selection from n_trials independent trials.
    """
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    T = len(returns)
    std = returns.std()
    if std == 0.0:
        return 0.5
    sr_hat = float(returns.mean() / std)

    denom = 1.0 - skewness * sr_hat + ((kurtosis - 1.0) / 4.0) * sr_hat ** 2
    if denom <= 0.0:
        denom = 1e-10
    sr_star = sr_hat * math.sqrt((T - 1) / denom)

    emax = 0.0 if n_trials <= 1 else (1.0 - _EULER_GAMMA) * math.sqrt(2.0 * math.log(n_trials))

    return float(_norm.cdf(sr_star - emax))


_REGIME_TREND_WINDOW = 20
_REGIME_VOL_WINDOW = 20
_REGIME_LOOKBACK = 252
_REGIME_HIGH_VOL_PERCENTILE = 0.75
_REGIME_TRENDING_DRIFT_PERCENTILE = 0.5
_REGIME_MIN_BARS = 30


def regime_conditional_sharpe(returns: pd.Series, prices: pd.DataFrame) -> dict:
    """Classify each bar into a market regime and compute per-regime annualized Sharpe ratios.

    Uses only lagged data to avoid lookahead: regime label at bar t depends solely
    on prices through bar t-1, so the label is known before the return is captured.

    Regime classifier:
    - vol[t]: 20-bar rolling std of log-returns ending at bar t-1 (lagged one bar).
    - vol_pct[t]: percentile rank of vol[t] in the trailing 252-bar vol distribution.
    - drift_abs[t]: |close[t-1] - close[t-21]| / close[t-21] — 20-bar absolute price
      change, fully lagged.
    - drift_pct[t]: percentile rank of drift_abs[t] in the trailing 252-bar distribution.

    Labels (mutually exclusive, collectively exhaustive):
    - high_vol: vol_pct > 0.75 (top quartile of rolling volatility)
    - trending: drift_pct > 0.5 AND NOT high_vol (strong directional drift, low vol)
    - ranging: NOT high_vol AND NOT trending (weak drift, low vol)

    Regimes with fewer than _REGIME_MIN_BARS (30) valid return observations report NaN.

    Args:
        returns: Daily equity pct_change() aligned to prices.index.
        prices: OHLCV DataFrame with at least a 'close' column and the same
            DatetimeIndex as the equity series that produced returns.

    Returns:
        Dict with keys 'trending', 'ranging', 'high_vol' (float or NaN) and
        'regime_counts' (dict with integer counts that sum to len(prices)).
    """
    close = prices["close"]
    log_ret = np.log(close / close.shift(1))

    vol = log_ret.shift(1).rolling(_REGIME_VOL_WINDOW, min_periods=_REGIME_VOL_WINDOW).std()
    vol_pct = vol.rolling(_REGIME_LOOKBACK, min_periods=1).rank(pct=True)

    close_lag1 = close.shift(1)
    drift_abs = (close_lag1 - close_lag1.shift(_REGIME_TREND_WINDOW)).abs() / close_lag1.shift(_REGIME_TREND_WINDOW)
    drift_pct = drift_abs.rolling(_REGIME_LOOKBACK, min_periods=1).rank(pct=True)

    high_vol = vol_pct > _REGIME_HIGH_VOL_PERCENTILE
    trending = (drift_pct > _REGIME_TRENDING_DRIFT_PERCENTILE) & ~high_vol
    ranging = ~high_vol & ~trending

    result = {}
    for label, mask in [("high_vol", high_vol), ("trending", trending), ("ranging", ranging)]:
        r = returns[mask].dropna()
        if len(r) < _REGIME_MIN_BARS:
            result[label] = float("nan")
        else:
            std = r.std()
            result[label] = 0.0 if std == 0 else float(r.mean() / std * np.sqrt(ANNUALIZE))

    result["regime_counts"] = {
        "high_vol": int(high_vol.sum()),
        "trending": int(trending.sum()),
        "ranging": int(ranging.sum()),
    }
    return result


def cost_to_alpha_ratio(gross_equity: pd.Series, net_equity: pd.Series) -> float:
    """Fraction of gross edge consumed by transaction costs.

    Computes gross_alpha / (gross_alpha - net_alpha) where alpha is CAGR.
    Returns 1.0 when costs are zero (gross == net). Returns inf when net_alpha
    is non-positive — all edge is eaten by costs or the strategy loses money net.

    Blind spot: CAGR collapses path information. Two strategies with identical
    terminal CAGRs can have very different intraperiod cost profiles; this ratio
    only captures the cumulative endpoint difference.

    Args:
        gross_equity: Zero-cost equity curve (commission_bps=0, slippage_bps=0).
        net_equity: Post-cost equity curve (realistic frictions applied).

    Returns:
        float — 1.0 for zero-cost strategies, inf when net alpha is non-positive,
        and a value > 1 when there is positive net alpha with some friction.
    """
    gross_alpha = cagr(gross_equity)
    net_alpha = cagr(net_equity)
    if net_alpha <= 0.0:
        return float("inf")
    cost_absorbed = gross_alpha - net_alpha
    if cost_absorbed == 0.0:
        return 1.0
    return float(gross_alpha / cost_absorbed)


def sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple:
    """Bootstrap percentile confidence interval for the annualised Sharpe ratio.

    Blind spot covered: the standard error of a Sharpe estimate is large for
    finite samples (~1000 bars). A point-estimate Sharpe of 0.5 computed from
    1000 daily returns has a 95% CI of roughly [0.0, 1.0], making it
    statistically indistinguishable from zero. This function makes that
    uncertainty explicit so rankings can distinguish strategies with real edge
    (CI excludes zero) from those consistent with noise (CI straddles zero).

    Uses the percentile bootstrap (not studentized) — simpler and sufficient
    for ~1000-bar datasets, though it slightly undercovers for skewed
    distributions. Fixed seed ensures byte-for-byte reproducible CIs.

    Args:
        returns: Daily return series (e.g. equity.pct_change().dropna()).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Desired confidence level (e.g. 0.95 for 95% CI).
        seed: RNG seed for reproducibility. Uses numpy.random.default_rng
            (modern generator — do not substitute legacy np.random.seed).

    Returns:
        (lower, point, upper) — all annualised Sharpe values as floats.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    samples = rng.choice(returns.values, size=(n_bootstrap, n), replace=True)
    boot_sharpes = (samples.mean(axis=1) / samples.std(axis=1, ddof=1)) * np.sqrt(ANNUALIZE)
    lo = float(np.percentile(boot_sharpes, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(boot_sharpes, (1 + confidence) / 2 * 100))
    point = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(ANNUALIZE))
    return lo, point, hi


def compute_all(
    equity: pd.Series,
    positions: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute all metrics and return as a flat dict.

    Args:
        equity: Portfolio value series.
        positions: Position series aligned to equity index.
        risk_free_rate: Annualized risk-free rate as a decimal.

    Returns:
        Dict mapping metric name to scalar value.
    """
    return {
        "cagr": cagr(equity),
        "volatility": volatility(equity),
        "sharpe": sharpe(equity, risk_free_rate),
        "sortino": sortino(equity, risk_free_rate),
        "calmar": calmar(equity),
        "max_drawdown": max_drawdown(equity),
        "time_in_drawdown": time_in_drawdown(equity),
        "turnover": turnover(positions),
        "hit_rate": hit_rate(equity),
        "tail_ratio": tail_ratio(equity),
        "exposure": exposure(positions),
    }
