"""Metrics for evaluating backtest equity curves.

All functions operate on a pd.Series equity curve or auxiliary position series
and return a scalar. Keep all math here — backtest.py calls these; strategies never do.
"""

import numpy as np
import pandas as pd

ANNUALIZE = 252  # trading days per year


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
        Mean absolute daily position change.
    """
    return float(positions.diff().abs().dropna().mean())


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
