"""Generate synthetic OHLCV price CSVs for backtesting.

Run from the repo root:
    python data/generate_synthetic.py

Each series uses numpy.random.default_rng(seed) for full reproducibility.
Re-running this script with the same code regenerates identical CSVs.
Seeds and statistical properties are documented in data/README.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent
N_BARS = 1000
DT = 1 / 252  # one trading day in years
DATES = pd.bdate_range("2020-01-02", periods=N_BARS)


def _make_ohlcv(closes: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Construct a realistic OHLCV DataFrame from a close price series.

    Args:
        closes: Array of closing prices (length N_BARS, all positive).
        rng: Seeded random generator for intraday noise.

    Returns:
        DataFrame with columns [open, high, low, close, volume] and DATES index.
    """
    n = len(closes)
    opens = np.empty(n)
    opens[0] = closes[0]
    # Overnight gap: small uniform perturbation of previous close
    opens[1:] = closes[:-1] * (1.0 + rng.uniform(-0.003, 0.003, n - 1))

    intraday_range = np.abs(closes - opens) + closes * rng.uniform(0.002, 0.015, n)
    highs = np.maximum(opens, closes) + intraday_range * 0.5
    lows = np.minimum(opens, closes) - intraday_range * 0.5
    lows = np.maximum(lows, 0.01)  # prices must be positive

    volumes = rng.integers(200_000, 2_000_000, n)

    return pd.DataFrame(
        {
            "open": np.round(opens, 4),
            "high": np.round(highs, 4),
            "low": np.round(lows, 4),
            "close": np.round(closes, 4),
            "volume": volumes,
        },
        index=DATES,
    )


def generate_trend_gbm(seed: int = 42) -> pd.DataFrame:
    """GBM with positive drift: mu=0.15/yr, sigma=0.20/yr, S0=100.

    Args:
        seed: RNG seed for reproducibility.

    Returns:
        OHLCV DataFrame.
    """
    rng = np.random.default_rng(seed)
    mu, sigma, S0 = 0.15, 0.20, 100.0
    z = rng.standard_normal(N_BARS)
    log_rets = (mu - 0.5 * sigma**2) * DT + sigma * np.sqrt(DT) * z
    closes = S0 * np.exp(np.cumsum(log_rets))
    closes = np.insert(closes, 0, S0)[:-1]
    return _make_ohlcv(closes, rng)


def generate_mean_rev_ou(seed: int = 7) -> pd.DataFrame:
    """Ornstein-Uhlenbeck mean-reversion: theta=5/yr, mu=100, sigma=20/yr.

    Args:
        seed: RNG seed for reproducibility.

    Returns:
        OHLCV DataFrame.
    """
    rng = np.random.default_rng(seed)
    theta, mu_level, sigma, S0 = 5.0, 100.0, 20.0, 100.0
    sqrt_dt = np.sqrt(DT)
    closes = np.empty(N_BARS)
    closes[0] = S0
    for t in range(1, N_BARS):
        closes[t] = (
            closes[t - 1]
            + theta * (mu_level - closes[t - 1]) * DT
            + sigma * sqrt_dt * rng.standard_normal()
        )
        closes[t] = max(closes[t], 0.01)
    return _make_ohlcv(closes, rng)


def generate_regime_switch(seed: int = 123) -> pd.DataFrame:
    """Alternating trend (GBM mu=0.20/yr) and mean-reversion (OU) regimes.

    Regime boundaries every 200 bars; regimes alternate starting with trend.

    Args:
        seed: RNG seed for reproducibility.

    Returns:
        OHLCV DataFrame.
    """
    rng = np.random.default_rng(seed)
    REGIME_LEN = 200
    mu_gbm, sigma_gbm = 0.20, 0.20
    theta_ou, mu_ou, sigma_ou = 5.0, 0.0, 20.0  # OU drifts around 0 deviation
    sqrt_dt = np.sqrt(DT)
    S0 = 100.0

    closes = np.empty(N_BARS)
    closes[0] = S0
    for t in range(1, N_BARS):
        regime = (t // REGIME_LEN) % 2  # 0 = trend, 1 = mean-rev
        if regime == 0:
            z = rng.standard_normal()
            closes[t] = closes[t - 1] * np.exp(
                (mu_gbm - 0.5 * sigma_gbm**2) * DT + sigma_gbm * sqrt_dt * z
            )
        else:
            # OU around the price at the start of this regime window
            regime_start_idx = (t // REGIME_LEN) * REGIME_LEN
            anchor = closes[regime_start_idx]
            deviation = closes[t - 1] - anchor
            closes[t] = (
                closes[t - 1]
                + theta_ou * (mu_ou - deviation) * DT
                + sigma_ou * sqrt_dt * rng.standard_normal()
            )
        closes[t] = max(closes[t], 0.01)
    return _make_ohlcv(closes, rng)


def generate_fat_tail(seed: int = 999) -> pd.DataFrame:
    """Student-t(3) innovations, zero drift, sigma=0.20/yr.

    Fat tails make extreme daily moves far more common than GBM predicts.

    Args:
        seed: RNG seed for reproducibility.

    Returns:
        OHLCV DataFrame.
    """
    rng = np.random.default_rng(seed)
    sigma, S0, df = 0.20, 100.0, 3
    # Normalize t(3) to unit variance: var(t(df)) = df/(df-2), so divide by sqrt(df/(df-2))
    t_raw = rng.standard_t(df, size=N_BARS)
    t_normalized = t_raw / np.sqrt(df / (df - 2))
    log_rets = sigma * np.sqrt(DT) * t_normalized  # zero drift
    closes = S0 * np.exp(np.cumsum(log_rets))
    closes = np.insert(closes, 0, S0)[:-1]
    closes = np.maximum(closes, 0.01)
    return _make_ohlcv(closes, rng)


def main():
    """Generate all four synthetic CSVs and write to data/."""
    datasets = [
        ("trend_gbm.csv", generate_trend_gbm, 42),
        ("mean_rev_ou.csv", generate_mean_rev_ou, 7),
        ("regime_switch.csv", generate_regime_switch, 123),
        ("fat_tail.csv", generate_fat_tail, 999),
    ]
    for filename, generator, seed in datasets:
        df = generator(seed)
        path = OUT_DIR / filename
        df.index.name = "date"
        df.to_csv(path)
        print(f"Written {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
