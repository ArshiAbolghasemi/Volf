import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.dataset.climate.crop_seasonal import crop_season_flag
from src.dataset.util.path import require_path
from src.dataset.util.production_by_state import (
    STATE_ABBREV_TO_NAME,
    load_production_weights,
)
from src.util.path import DATA_DIR

logger = logging.getLogger(__name__)


CROPS = ("corn", "wheat", "soybean")
SPI_COLUMNS = ("SPI_1m", "SPI_3m", "SPI_7d")

# TMAX: right tail - heat stress
TMAX_Q3_LOW = 0.75  # 75th-90th pct -> moderate heat stress
TMAX_Q3_HIGH = 0.90  # above 90th pct -> severe heat stress

# TMIN: left tail - cold stress
TMIN_Q3_HIGH = 0.25  # 10th-25th pct -> moderate cold stress
TMIN_Q4_HIGH = 0.10  # below 10th pct -> severe cold stress

# AWND: right tail - wind stress
AWND_Q3_LOW = 0.80  # 80th-90th pct -> moderate wind stress
AWND_Q3_HIGH = 0.90  # above 90th pct -> severe wind stress


def _resolve_input_path(raw: str) -> Path:
    return require_path(Path(raw), "Input file")


def _resolve_production_dir() -> Path:
    return require_path(DATA_DIR / "production_by_state", "Production directory")


def _resolve_spi_input() -> Path:
    return require_path(DATA_DIR / "climate" / "spi_weekly_multiscale.csv", "SPI file")


def _merge_spi_features(noaa_weekly: pd.DataFrame, spi_path: Path) -> pd.DataFrame:
    spi_df = pd.read_csv(spi_path)

    missing = sorted({"state", *SPI_COLUMNS} - set(spi_df.columns))
    if missing:
        msg = f"Missing required SPI columns: {missing}"
        raise ValueError(msg)
    if "week_date" not in spi_df.columns:
        msg = "SPI input must have a 'week_date' column."
        raise ValueError(msg)

    spi_df["merge_date"] = pd.to_datetime(spi_df["week_date"], errors="coerce")
    spi_weekly = (
        cast("pd.DataFrame", spi_df[["state", "merge_date", *SPI_COLUMNS]])
        .dropna(subset=["merge_date"])
        .drop_duplicates(subset=["state", "merge_date"], keep="last")
    )

    out = noaa_weekly.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if bool(out["date"].isna().any()):
        msg = "NOAA weekly column 'date' contains invalid values."
        raise ValueError(msg)

    return out.merge(
        spi_weekly, left_on=["state", "date"], right_on=["state", "merge_date"], how="left"
    ).drop(columns=["merge_date"])


def aggregate_by_production_weights(
    state_df: pd.DataFrame,
    production_weights: pd.DataFrame,
    crop: str,
) -> pd.DataFrame:
    """Aggregate state-level raw metrics to national level using production weights.

    New approach: Aggregate FIRST, then compute z-scores and quantiles.
    This ensures q3 and q4 bands are mutually exclusive at national level.
    """
    merged = state_df.copy()
    merged["state_name"] = merged["state"].map(STATE_ABBREV_TO_NAME).fillna(merged["state"])
    merged = merged.merge(
        production_weights,
        left_on="state_name",
        right_on="state",
        how="left",
        suffixes=("", "_prod"),
    )
    merged = merged.drop(columns=["state_prod"], errors="ignore")
    merged["state_weight"] = pd.to_numeric(merged["state_weight"], errors="coerce").fillna(
        0.0
    )

    # Add temporal columns
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["week_of_year"] = merged["date"].dt.isocalendar().week.astype(int)
    merged["year"] = merged["date"].dt.year
    merged["month"] = merged["date"].dt.month

    # Apply seasonal flags
    flags = merged["week_of_year"].apply(lambda w: crop_season_flag(int(w), crop))
    merged["is_planting_week"] = flags.map(lambda f: f["is_planting_week"])
    merged["is_harvesting_week"] = flags.map(lambda f: f["is_harvesting_week"])

    # Metrics to aggregate
    base_metrics = ["TMAX", "TMIN", "AWND"]
    spi_metrics = [c for c in SPI_COLUMNS if c in merged.columns]
    all_metrics = base_metrics + spi_metrics

    # Weighted aggregation by date
    weighted = merged[
        [
            "date",
            "week_of_year",
            "year",
            "month",
            "is_planting_week",
            "is_harvesting_week",
            *all_metrics,
            "state_weight",
        ]
    ].copy()

    for metric in all_metrics:
        weighted[metric] = (
            pd.to_numeric(weighted[metric], errors="coerce").fillna(0.0)
            * weighted["state_weight"]
        )

    # Group by date and sum
    group_cols = [
        "date",
        "week_of_year",
        "year",
        "month",
        "is_planting_week",
        "is_harvesting_week",
    ]
    agg = weighted.groupby(group_cols, as_index=False).agg(
        {**dict.fromkeys(all_metrics, "sum"), "state_weight": "sum"}
    )

    # Normalize by total weight
    for metric in all_metrics:
        agg[metric] = np.where(
            agg["state_weight"] > 0, agg[metric] / agg["state_weight"], 0.0
        )

    return agg.drop(columns=["state_weight"])


def compute_national_zscores_and_extremes(national_df: pd.DataFrame) -> pd.DataFrame:  # noqa: PLR0912, C901
    """Compute z-scores and extreme values on national-level aggregated data.

    This ensures q3 and q4 bands are mutually exclusive (no overlaps).
    """
    out = national_df.copy()

    # Compute z-scores by week of year
    for metric in ["TMAX", "TMIN", "AWND"]:
        if metric not in out.columns:
            continue

        grp = out.groupby("week_of_year")[metric]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        out[f"{metric}_zscore"] = (out[metric] - mean) / std

    # Compute quantile thresholds by week of year
    for metric in ["TMAX", "TMIN", "AWND"]:
        zscore_col = f"{metric}_zscore"
        if zscore_col not in out.columns:
            continue

        grp = out.groupby("week_of_year")[zscore_col]

        if metric == "TMAX":
            out[f"{metric}_q3_thresh"] = grp.transform(lambda x: x.quantile(TMAX_Q3_LOW))
            out[f"{metric}_q4_thresh"] = grp.transform(lambda x: x.quantile(TMAX_Q3_HIGH))
        elif metric == "TMIN":
            out[f"{metric}_q4_thresh"] = grp.transform(lambda x: x.quantile(TMIN_Q4_HIGH))
            out[f"{metric}_q3_thresh"] = grp.transform(lambda x: x.quantile(TMIN_Q3_HIGH))
        elif metric == "AWND":
            out[f"{metric}_q3_thresh"] = grp.transform(lambda x: x.quantile(AWND_Q3_LOW))
            out[f"{metric}_q4_thresh"] = grp.transform(lambda x: x.quantile(AWND_Q3_HIGH))

    # Apply extreme value logic
    for metric in ["TMAX", "TMIN", "AWND"]:
        zscore_col = f"{metric}_zscore"
        if zscore_col not in out.columns:
            continue

        z = out[zscore_col].to_numpy(dtype=float)
        q3 = out[f"{metric}_q3_thresh"].to_numpy(dtype=float)
        q4 = out[f"{metric}_q4_thresh"].to_numpy(dtype=float)
        z_ok = np.isfinite(z)

        if metric in ["TMAX", "AWND"]:
            # Right tail
            out[f"{metric}_q3_value"] = np.where(z_ok & (z >= q3) & (z < q4), z, 0.0)
            out[f"{metric}_q4_value"] = np.where(z_ok & (z >= q4), z, 0.0)
        elif metric == "TMIN":
            # Left tail
            out[f"{metric}_q3_value"] = np.where(z_ok & (z > q4) & (z <= q3), z, 0.0)
            out[f"{metric}_q4_value"] = np.where(z_ok & (z <= q4), z, 0.0)

        # Apply seasonal masks
        out[f"{metric}_q3_value_in_planting"] = (
            out[f"{metric}_q3_value"] * out["is_planting_week"]
        )
        out[f"{metric}_q3_value_in_harvesting"] = (
            out[f"{metric}_q3_value"] * out["is_harvesting_week"]
        )
        out[f"{metric}_q4_value_in_planting"] = (
            out[f"{metric}_q4_value"] * out["is_planting_week"]
        )
        out[f"{metric}_q4_value_in_harvesting"] = (
            out[f"{metric}_q4_value"] * out["is_harvesting_week"]
        )

        # Drop intermediate columns
        out = out.drop(
            columns=[
                f"{metric}_q3_thresh",
                f"{metric}_q4_thresh",
                f"{metric}_q3_value",
                f"{metric}_q4_value",
            ],
            errors="ignore",
        )

    # Validate: q3 and q4 should be mutually exclusive
    for metric in ["TMAX", "TMIN", "AWND"]:
        for period in ["in_planting", "in_harvesting"]:
            q3_col = f"{metric}_q3_value_{period}"
            q4_col = f"{metric}_q4_value_{period}"
            if q3_col in out.columns and q4_col in out.columns:
                overlap = (out[q3_col] != 0) & (out[q4_col] != 0)
                if overlap.any():
                    msg = (
                        f"National-level overlap detected in {metric} {period}: "
                        f"{overlap.sum()} rows have both q3 and q4 non-zero."
                    )
                    raise ValueError(msg)

    logger.info(
        "✓ National-level extreme values validated: no overlaps between q3 and q4 bands"
    )

    return out


def add_rolling_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Add monthly (4-week) and seasonal (13-week) rolling averages."""
    out = df.copy()

    # Get all value columns
    value_cols = [
        col
        for col in out.columns
        if any(x in col for x in ["_q3_value_", "_q4_value_", "SPI_"])
    ]

    out = out.sort_values("date").reset_index(drop=True)

    for col in value_cols:
        out[f"monthly_{col}"] = out[col].rolling(window=4, min_periods=1).mean()
        out[f"seasonal_{col}"] = out[col].rolling(window=13, min_periods=1).mean()

    return out


def run_pipeline(input_path: Path | str, output_dir: Path | str) -> dict[str, Path]:
    resolved_input = _resolve_input_path(str(input_path))
    resolved_production = _resolve_production_dir()
    resolved_spi = _resolve_spi_input()
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    # Load state-level data
    noaa_weekly = pd.read_csv(resolved_input)
    noaa_with_spi = _merge_spi_features(noaa_weekly, resolved_spi)

    outputs: dict[str, Path] = {}

    for crop in CROPS:
        logger.info("Processing %s...", crop)

        # Load production weights
        production_weights = load_production_weights(crop, resolved_production)

        # Aggregate to national level
        national_df = aggregate_by_production_weights(
            noaa_with_spi, production_weights, crop
        )

        # Compute z-scores and extremes on national data
        extreme_df = compute_national_zscores_and_extremes(national_df)

        # Add rolling windows
        final_df = add_rolling_windows(extreme_df)

        # Save
        out_path = resolved_output / f"climate_weekly_weighted_{crop}.csv"
        final_df.to_csv(out_path, index=False)
        logger.info(
            "Saved %s → %s (rows=%d, cols=%d)",
            crop,
            out_path,
            len(final_df),
            len(final_df.columns),
        )
        outputs[crop] = out_path

    return outputs
