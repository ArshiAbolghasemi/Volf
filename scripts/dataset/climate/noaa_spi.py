from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

from src.util.path import DATA_DIR

# Scales converted to DAILY windows (not weekly)
SCALES = {"7d": 7, "1m": 30, "2m": 60, "3m": 90, "6m": 180, "12m": 365}

# Minimum sample size to fit distribution
MIN_SAMPLE_SIZE = 10


def _compute_spi(series: pd.Series) -> pd.Series:
    """Compute SPI for a precipitation series."""
    # Values used for fitting (non-zero only)
    non_zero = series[series > 0]

    # Avoid fitting on small samples
    if len(non_zero) < MIN_SAMPLE_SIZE:
        return pd.Series(np.nan, index=series.index)

    # Fit Gamma distribution (fix loc=0)
    shape, loc, scale = gamma.fit(non_zero, floc=0)

    # CDF (allow zeros)
    cdf = gamma.cdf(series, shape, loc=loc, scale=scale)

    # Bound CDF strictly to avoid inf in norm.ppf
    cdf = np.clip(cdf, 1e-8, 1 - 1e-8)

    # Convert to SPI
    spi = norm.ppf(cdf)

    return pd.Series(spi, index=series.index)


def main() -> None:
    df_daily = pd.read_csv(DATA_DIR / "climate" / "noaa_daily.csv")

    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.sort_values(["state", "date"])

    spi_states = []

    for state, df_state in df_daily.groupby("state"):
        # Avoid overwriting loop variable
        df_state_sorted = df_state.sort_values("date").copy()

        spi_df = df_state_sorted[["date"]].copy()
        spi_df["state"] = state

        for name, window in SCALES.items():
            roll = df_state_sorted["PRCP"].rolling(window=window, min_periods=window).sum()

            spi_df[f"SPI_{name}"] = _compute_spi(cast("pd.Series", roll))

        spi_df["week_date"] = spi_df["date"].dt.to_period("W").apply(lambda r: r.start_time)

        weekly_spi = spi_df.groupby(["state", "week_date"]).last().reset_index()

        spi_states.append(weekly_spi)

    df_spi = pd.concat(spi_states, ignore_index=True)
    df_spi.to_csv(DATA_DIR / "climate" / "spi_weekly_multiscale.csv", index=False)


if __name__ == "__main__":
    main()
