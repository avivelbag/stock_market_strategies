"""Generate a synthetic co-integrated paired price dataset.

Two price series (A and B) share a common GBM log-price factor. Each series
has an additional idiosyncratic zero-mean OU noise component. The spread
log(A) - log(B) = eps_A - eps_B is a stationary OU process that mean-reverts
to zero — the theoretical foundation for pairs mean-reversion trading.

Generation parameters (seed=42):
    sigma_F  = 0.20/yr  — common-factor volatility (non-stationary component)
    sigma_eps = 0.10/yr — idiosyncratic OU volatility
    ou_theta  = 8.75/yr — mean-reversion speed; half-life ≈ 20 bars
    S0        = 100.0   — starting price for both series

Run from the repo root:
    python data/generate_paired.py

Produces data/paired_cointegrated.csv (1000 rows, seed=42).
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent
N_BARS = 1000
DT = 1 / 252
DATES = pd.bdate_range("2020-01-02", periods=N_BARS)

SEED = 42
S0 = 100.0
SIGMA_F = 0.20
SIGMA_EPS = 0.10
OU_THETA = 8.75


def generate_paired_cointegrated(seed: int = SEED) -> pd.DataFrame:
    """Generate two co-integrated price series sharing a common GBM log-price factor.

    Model:
        log F_t  = log F_{t-1} + sigma_F * sqrt(DT) * z_F        (common factor)
        eps_A_t  = (1 - theta*DT) * eps_A_{t-1} + sigma_eps * sqrt(DT) * z_A
        eps_B_t  = (1 - theta*DT) * eps_B_{t-1} + sigma_eps * sqrt(DT) * z_B
        log_A_t  = log(S0) + log_F_t + eps_A_t
        log_B_t  = log(S0) + log_F_t + eps_B_t
        spread_t = log_A_t - log_B_t = eps_A_t - eps_B_t          (stationary OU)

    The OHLCV ``close`` column equals close_A / close_B, which is always positive
    and whose logarithm equals the log-spread. The strategy's z-score is computed
    from log(close). The two individual series are preserved in ``close_a`` and
    ``close_b`` for documentation and test verification.

    Args:
        seed: RNG seed for full reproducibility.

    Returns:
        DataFrame with columns [open, high, low, close, volume, close_a, close_b]
        and a business-day DatetimeIndex. close = close_a / close_b.
    """
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(DT)

    log_F = np.zeros(N_BARS)
    for t in range(1, N_BARS):
        log_F[t] = log_F[t - 1] + SIGMA_F * sqrt_dt * rng.standard_normal()

    decay = 1.0 - OU_THETA * DT
    eps_A = np.zeros(N_BARS)
    eps_B = np.zeros(N_BARS)
    for t in range(1, N_BARS):
        eps_A[t] = decay * eps_A[t - 1] + SIGMA_EPS * sqrt_dt * rng.standard_normal()
        eps_B[t] = decay * eps_B[t - 1] + SIGMA_EPS * sqrt_dt * rng.standard_normal()

    log_close_A = np.log(S0) + log_F + eps_A
    log_close_B = np.log(S0) + log_F + eps_B
    close_a = np.exp(log_close_A)
    close_b = np.exp(log_close_B)

    spread_close = close_a / close_b

    n = N_BARS
    spread_open = np.empty(n)
    spread_open[0] = spread_close[0]
    spread_open[1:] = spread_close[:-1] * (1.0 + rng.uniform(-0.003, 0.003, n - 1))

    intraday_range = (
        np.abs(spread_close - spread_open) + spread_close * rng.uniform(0.001, 0.005, n)
    )
    spread_high = np.maximum(spread_open, spread_close) + intraday_range * 0.5
    spread_low = np.minimum(spread_open, spread_close) - intraday_range * 0.5
    spread_low = np.maximum(spread_low, 0.0001)

    volumes = rng.integers(200_000, 2_000_000, n)

    return pd.DataFrame(
        {
            "open": np.round(spread_open, 6),
            "high": np.round(spread_high, 6),
            "low": np.round(spread_low, 6),
            "close": np.round(spread_close, 6),
            "volume": volumes,
            "close_a": np.round(close_a, 4),
            "close_b": np.round(close_b, 4),
        },
        index=DATES,
    )


def main():
    """Generate paired_cointegrated.csv and write to data/."""
    df = generate_paired_cointegrated(SEED)
    path = OUT_DIR / "paired_cointegrated.csv"
    df.index.name = "date"
    df.to_csv(path)
    print(f"Written {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
