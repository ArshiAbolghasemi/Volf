import logging
import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.dataset.climate.crop_seasonal import crop_season_flag
from src.util.path import DATA_DIR

logger = logging.getLogger(__name__)

METRIC_COLUMNS = ("TMAX", "TMIN", "TAVG")
Q3_LOW = 0.75
Q3_HIGH = 0.90
LEFT_Q4_HIGH = 0.10
LEFT_Q3_HIGH = 0.25
CROPS = ("corn", "wheat", "soybean")
STATE_ABBREV_TO_NAME = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}


def _resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    msg = f"Input file not found: {path}"
    raise FileNotFoundError(msg)


def _resolve_production_dir() -> Path:
    path = DATA_DIR / "production_by_state"
    if path.exists():
        return path

    msg = f"Production directory not found: {path}"
    raise FileNotFoundError(msg)


def _resolve_spi_input() -> Path:
    path = DATA_DIR / "climate" / "spi_weekly_multiscale.csv"
    if path.exists():
        return path

    msg = f"SPI file not found: {path}"
    raise FileNotFoundError(msg)


def _merge_spi_1m(noaa_weekly: pd.DataFrame, spi_path: Path) -> pd.DataFrame:
    spi_df = pd.read_csv(spi_path)
    required_cols = {"state", "SPI_1m"}
    missing = sorted(required_cols - set(spi_df.columns))
    if missing:
        msg = f"Missing required SPI columns: {missing}"
        raise ValueError(msg)

    # NOAA weekly dates are week-start Mondays, so we prioritize week_date if present.
    if "week_date" in spi_df.columns:
        spi_df["merge_date"] = pd.to_datetime(spi_df["week_date"], errors="coerce")
    else:
        msg = "SPI input must have either 'week_date' or 'date' column."
        raise ValueError(msg)

    spi_weekly = spi_df[["state", "merge_date", "SPI_1m"]].copy()
    spi_weekly = spi_weekly.dropna(subset=["merge_date"]).drop_duplicates(
        subset=["state", "merge_date"], keep="last"
    )

    out = noaa_weekly.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if bool(out["date"].isna().any()):
        msg = "NOAA weekly column 'date' contains invalid values."
        raise ValueError(msg)

    out = out.merge(
        spi_weekly,
        left_on=["state", "date"],
        right_on=["state", "merge_date"],
        how="left",
    )
    return out.drop(columns=["merge_date"])


def construct_weekly_state_zscores(df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0915
    required = {"state", "date", *METRIC_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if bool(out["date"].isna().any()):
        msg = "Column 'date' contains invalid values."
        raise ValueError(msg)

    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)

    for metric in METRIC_COLUMNS:
        stats = cast(
            "pd.DataFrame",
            out.groupby(["state", "week_of_year"])[metric].agg(["mean", "std"]),
        )
        stats = stats.rename(
            columns={
                "mean": f"{metric}_week_mean",
                "std": f"{metric}_week_std",
            },
        )
        out = out.merge(
            stats.reset_index(),
            on=["state", "week_of_year"],
            how="left",
        )

        std_col = f"{metric}_week_std"
        mean_col = f"{metric}_week_mean"
        z_col = f"{metric}_zscore"
        out[z_col] = (out[metric] - out[mean_col]) / out[std_col].replace(0, np.nan)
        z_stats = cast(
            "pd.DataFrame",
            out.groupby(["state", "week_of_year"])[z_col].agg(["mean", "std"]),
        )
        z_stats = z_stats.rename(
            columns={
                "mean": f"{metric}_zscore_mean",
                "std": f"{metric}_zscore_std",
            }
        )
        out = out.merge(
            z_stats.reset_index(),
            on=["state", "week_of_year"],
            how="left",
        )

        z_mean_col = f"{metric}_zscore_mean"
        z_std_col = f"{metric}_zscore_std"
        q3_low_col = f"{metric}_q3_low"
        q3_high_col = f"{metric}_q3_high"
        z_values = out[z_col].to_numpy(dtype=float)
        z_mu = out[z_mean_col].to_numpy(dtype=float)
        z_sigma = out[z_std_col].to_numpy(dtype=float)
        valid_sigma = np.isfinite(z_sigma) & (z_sigma > 0.0)

        q3_low = np.full(len(out), np.nan, dtype=float)
        q3_high = np.full(len(out), np.nan, dtype=float)
        q3_low[valid_sigma] = norm.ppf(
            Q3_LOW,
            loc=z_mu[valid_sigma],
            scale=z_sigma[valid_sigma],
        )
        q3_high[valid_sigma] = norm.ppf(
            Q3_HIGH,
            loc=z_mu[valid_sigma],
            scale=z_sigma[valid_sigma],
        )
        out[q3_low_col] = q3_low
        out[q3_high_col] = q3_high

        in_q3 = (
            valid_sigma
            & np.isfinite(z_values)
            & (z_values >= q3_low)
            & (z_values <= q3_high)
        )
        in_q4 = valid_sigma & np.isfinite(z_values) & (z_values > q3_high)

        out[f"{metric}_q3_value"] = np.where(in_q3, z_values, 0.0)
        out[f"{metric}_q4_value"] = np.where(in_q4, z_values, 0.0)

        if metric == "TMIN":
            left_q4_high = np.full(len(out), np.nan, dtype=float)
            left_q3_high = np.full(len(out), np.nan, dtype=float)
            left_q4_high[valid_sigma] = norm.ppf(
                LEFT_Q4_HIGH,
                loc=z_mu[valid_sigma],
                scale=z_sigma[valid_sigma],
            )
            left_q3_high[valid_sigma] = norm.ppf(
                LEFT_Q3_HIGH,
                loc=z_mu[valid_sigma],
                scale=z_sigma[valid_sigma],
            )

            in_left_q3 = (
                valid_sigma
                & np.isfinite(z_values)
                & (z_values > left_q4_high)
                & (z_values <= left_q3_high)
            )
            in_left_q4 = valid_sigma & np.isfinite(z_values) & (z_values <= left_q4_high)

            out["TMIN_left_q3_value"] = np.where(in_left_q3, z_values, 0.0)
            out["TMIN_left_q4_value"] = np.where(in_left_q4, z_values, 0.0)

    return out


def _value_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.endswith("_value")]


def construct_crop_frameworks(zscore_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "week_of_year" not in zscore_df.columns:
        msg = "Column 'week_of_year' is required to build crop frameworks."
        raise ValueError(msg)

    value_cols = _value_feature_columns(zscore_df)
    if not value_cols:
        msg = "No '_value' columns found in zscore dataframe."
        raise ValueError(msg)

    frameworks: dict[str, pd.DataFrame] = {}
    for crop in CROPS:
        out = zscore_df.copy()
        seasonal_flags = out["week_of_year"].apply(
            lambda week, crop=crop: crop_season_flag(int(week), crop)
        )

        for col in value_cols:
            out[f"{col}_in_planting"] = out[col] * out["is_planting_week"]
            out[f"{col}_in_harvesting"] = out[col] * out["is_harvesting_week"]

        frameworks[crop] = out

    return frameworks


def _load_production_weights(crop: str, production_dir: Path) -> pd.DataFrame:
    path = production_dir / f"{crop}.csv"
    if not path.exists():
        msg = f"Production file not found for {crop}: {path}"
        raise FileNotFoundError(msg)

    production_df = pd.read_csv(path)
    if "state" not in production_df.columns:
        msg = f"Missing 'state' column in production file: {path}"
        raise ValueError(msg)

    pattern = re.compile(rf"^{crop.capitalize()}ProductionByState_(\d{{4}})$")
    year_cols = [col for col in production_df.columns if pattern.match(col)]
    if not year_cols:
        msg = f"No production-by-year columns found for {crop} in: {path}"
        raise ValueError(msg)

    long_df = production_df[["state", *year_cols]].melt(
        id_vars="state",
        value_vars=year_cols,
        var_name="year_col",
        value_name="production",
    )
    long_df["year"] = long_df["year_col"].str.extract(r"(\d{4})$").astype(float)
    long_df["year"] = long_df["year"].astype("Int64")
    long_df["production"] = pd.to_numeric(long_df["production"], errors="coerce")

    long_df["state"] = long_df["state"].astype(str).str.strip()
    long_df = long_df[long_df["state"] != "United States"].copy()
    long_df = long_df.dropna(subset=["year", "production"])
    long_df = long_df[long_df["production"] > 0].copy()
    long_df["total_production_year"] = long_df.groupby("year")["production"].transform(
        "sum"
    )
    long_df["state_weight"] = long_df["production"] / long_df["total_production_year"]

    return long_df[["state", "year", "state_weight"]].drop_duplicates()


def build_weighted_commodity_frame(
    crop_frame: pd.DataFrame,
    production_weights: pd.DataFrame,
) -> pd.DataFrame:
    required = {"state", "date"}
    missing = sorted(required - set(crop_frame.columns))
    if missing:
        msg = f"Missing required columns in crop frame: {missing}"
        raise ValueError(msg)

    out = crop_frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if bool(out["date"].isna().any()):
        msg = "Column 'date' contains invalid values."
        raise ValueError(msg)

    out["state_name"] = out["state"].map(STATE_ABBREV_TO_NAME)
    out["year"] = out["date"].dt.year.astype("Int64")
    merged = out.merge(
        production_weights,
        left_on=["state_name", "year"],
        right_on=["state", "year"],
        how="left",
    )
    merged["state_weight"] = merged["state_weight"].fillna(0.0)

    exclude_cols = {
        "week_of_year",
        "is_planting_week",
        "is_harvesting_week",
        "is_active_season",
        "year",
        "state_weight",
    }
    numeric_cols = [
        col
        for col in merged.columns
        if pd.api.types.is_numeric_dtype(merged[col]) and col not in exclude_cols
    ]
    if not numeric_cols:
        msg = "No numeric feature columns available for weighted aggregation."
        raise ValueError(msg)

    weighted_values = merged[numeric_cols].mul(merged["state_weight"], axis=0)
    weighted_totals = weighted_values.groupby(merged["date"], sort=True).sum()
    weighted_totals.index.name = "date"
    weighted_df = weighted_totals.reset_index()
    weight_sums = merged.groupby("date", sort=True)["state_weight"].sum()
    weighted_df["total_state_weight"] = weighted_df["date"].map(weight_sums)
    return weighted_df.sort_values("date").reset_index(drop=True)


def add_monthly_seasonal_features(
    weighted_df: pd.DataFrame,
    monthly_weeks: int = 4,
    seasonal_weeks: int = 13,
) -> pd.DataFrame:
    if "date" not in weighted_df.columns:
        msg = "Weighted dataframe must contain a 'date' column."
        raise ValueError(msg)

    out = weighted_df.sort_values("date").reset_index(drop=True).copy()
    numeric_cols = [
        col
        for col in out.columns
        if (
            pd.api.types.is_numeric_dtype(out[col])
            and col != "total_state_weight"
            and not col.endswith("_monthly")
            and not col.endswith("_seasonal")
        )
    ]

    aggregated: dict[str, pd.Series] = {}
    for col in numeric_cols:
        if col == "SPI_1m":
            continue
        # Use only past information: weekly t uses averages from previous weeks.
        shifted = out[col].shift(1)
        aggregated[f"{col}_monthly"] = shifted.rolling(
            window=monthly_weeks,
            min_periods=monthly_weeks,
        ).mean()
        aggregated[f"{col}_seasonal"] = shifted.rolling(
            window=seasonal_weeks,
            min_periods=seasonal_weeks,
        ).mean()

    return pd.concat([out, pd.DataFrame(aggregated, index=out.index)], axis=1)


def run_pipeline(
    input_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Path]:
    resolved_input_path = _resolve_input_path(str(input_path))
    resolved_production_dir = _resolve_production_dir()
    resolved_spi_input = _resolve_spi_input()
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading NOAA weekly data from %s", resolved_input_path)
    raw_df = pd.read_csv(resolved_input_path)
    raw_df = _merge_spi_1m(raw_df, resolved_spi_input)

    zscore_df = construct_weekly_state_zscores(raw_df)
    logger.info("Constructed dataframe with z-scores. rows=%d", len(zscore_df))
    logger.info("Columns added for metrics: %s", ", ".join(METRIC_COLUMNS))
    crop_frameworks = construct_crop_frameworks(zscore_df)

    outputs: dict[str, Path] = {}
    for crop, framework in crop_frameworks.items():
        production_weights = _load_production_weights(crop, resolved_production_dir)
        weighted_df = build_weighted_commodity_frame(framework, production_weights)
        weighted_df = add_monthly_seasonal_features(weighted_df)
        output_path = resolved_output_dir / f"cliamte_weekly_weighted_{crop}.csv"
        weighted_df.to_csv(output_path, index=False)
        logger.info(
            "Saved %s weighted dataset to %s (rows=%d, cols=%d)",
            crop,
            output_path,
            len(weighted_df),
            len(weighted_df.columns),
        )
        outputs[crop] = output_path

    return outputs
