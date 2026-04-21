import logging
import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.dataset.climate.crop_seasonal import crop_season_flag
from src.util.path import DATA_DIR

logger = logging.getLogger(__name__)


CROPS = ("corn", "wheat", "soybean")
SPI_COLUMNS = ("SPI_1m", "SPI_3m")

# TMAX: right tail - heat stress
TMAX_Q3_LOW = 0.75  # 75th-90th pct -> moderate heat stress
TMAX_Q3_HIGH = 0.90  # above 90th pct -> severe heat stress

# TMIN: left tail - cold stress
TMIN_Q3_HIGH = 0.25  # 10th-25th pct -> moderate cold stress
TMIN_Q4_HIGH = 0.10  # below 10th pct -> severe cold stress

# AWND: right tail - wind stress
AWND_Q3_LOW = 0.80  # 80th-90th pct -> moderate wind stress
AWND_Q3_HIGH = 0.90  # above 90th pct -> severe wind stress

STATE_ABBREV_TO_NAME: dict[str, str] = {
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


def _require_path(path: Path, label: str) -> Path:
    if not path.exists():
        msg = f"{label} not found: {path}"
        raise FileNotFoundError(msg)
    return path


def _resolve_input_path(raw: str) -> Path:
    return _require_path(Path(raw), "Input file")


def _resolve_production_dir() -> Path:
    return _require_path(DATA_DIR / "production_by_state", "Production directory")


def _resolve_spi_input() -> Path:
    return _require_path(DATA_DIR / "climate" / "spi_weekly_multiscale.csv", "SPI file")


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


def _compute_zscores(df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted({"state", "date", "TMAX", "TMIN", "AWND"} - set(df.columns))
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month

    for metric in ("TMAX", "TMIN", "AWND"):
        grp = out.groupby(["state", "week_of_year"])[metric]
        mean = grp.transform("mean")
        std = grp.transform("std").replace(0, np.nan)
        out[f"{metric}_zscore"] = (out[metric] - mean) / std

    return out


def _compute_extreme_values(out: pd.DataFrame) -> pd.DataFrame:
    """Assign extreme z-score values from state/week quantile thresholds.

    For each (state, week_of_year), compute fixed quantile thresholds from the full
    historical z-score distribution, then assign band values:

      TMAX (right tail - heat stress):
        Q3 value : z-score if 75th <= z < 90th, else 0
        Q4 value : z-score if z >= 90th,         else 0

      TMIN (left tail - cold stress):
        Q3 value : z-score if 10th < z <= 25th,  else 0
        Q4 value : z-score if z <= 10th,          else 0

      AWND (right tail - wind stress):
        Q3 value : z-score if 80th <= z < 90th, else 0
        Q4 value : z-score if z >= 90th,        else 0
    """
    group_cols = ["state", "week_of_year"]
    tmax_group = out.groupby(group_cols)["TMAX_zscore"]
    tmin_group = out.groupby(group_cols)["TMIN_zscore"]
    awnd_group = out.groupby(group_cols)["AWND_zscore"]

    out["tmax_q3_thresh"] = tmax_group.transform(
        lambda values: values.quantile(TMAX_Q3_LOW)
    )
    out["tmax_q4_thresh"] = tmax_group.transform(
        lambda values: values.quantile(TMAX_Q3_HIGH)
    )
    out["tmin_q4_thresh"] = tmin_group.transform(
        lambda values: values.quantile(TMIN_Q4_HIGH)
    )
    out["tmin_q3_thresh"] = tmin_group.transform(
        lambda values: values.quantile(TMIN_Q3_HIGH)
    )
    out["awnd_q3_thresh"] = awnd_group.transform(
        lambda values: values.quantile(AWND_Q3_LOW)
    )
    out["awnd_q4_thresh"] = awnd_group.transform(
        lambda values: values.quantile(AWND_Q3_HIGH)
    )

    tmax_z = out["TMAX_zscore"].to_numpy(dtype=float)
    tmin_z = out["TMIN_zscore"].to_numpy(dtype=float)
    awnd_z = out["AWND_zscore"].to_numpy(dtype=float)
    tmax_ok = np.isfinite(tmax_z)
    tmin_ok = np.isfinite(tmin_z)
    awnd_ok = np.isfinite(awnd_z)

    tmax_q3 = out["tmax_q3_thresh"].to_numpy(dtype=float)
    tmax_q4 = out["tmax_q4_thresh"].to_numpy(dtype=float)
    tmin_q3 = out["tmin_q3_thresh"].to_numpy(dtype=float)
    tmin_q4 = out["tmin_q4_thresh"].to_numpy(dtype=float)
    awnd_q3 = out["awnd_q3_thresh"].to_numpy(dtype=float)
    awnd_q4 = out["awnd_q4_thresh"].to_numpy(dtype=float)

    # TMAX: right tail
    out["TMAX_q3_value"] = np.where(
        tmax_ok & (tmax_z >= tmax_q3) & (tmax_z < tmax_q4), tmax_z, 0.0
    )
    out["TMAX_q4_value"] = np.where(tmax_ok & (tmax_z >= tmax_q4), tmax_z, 0.0)

    # TMIN: left tail
    out["TMIN_q3_value"] = np.where(
        tmin_ok & (tmin_z > tmin_q4) & (tmin_z <= tmin_q3), tmin_z, 0.0
    )
    out["TMIN_q4_value"] = np.where(tmin_ok & (tmin_z <= tmin_q4), tmin_z, 0.0)

    # AWND: right tail
    out["AWND_q3_value"] = np.where(
        awnd_ok & (awnd_z >= awnd_q3) & (awnd_z < awnd_q4), awnd_z, 0.0
    )
    out["AWND_q4_value"] = np.where(awnd_ok & (awnd_z >= awnd_q4), awnd_z, 0.0)

    return out.drop(
        columns=[
            "tmax_q3_thresh",
            "tmax_q4_thresh",
            "tmin_q3_thresh",
            "tmin_q4_thresh",
            "awnd_q3_thresh",
            "awnd_q4_thresh",
        ]
    )


_EXTREME_VALUE_COLS = [
    "TMAX_q3_value",
    "TMAX_q4_value",
    "TMIN_q3_value",
    "TMIN_q4_value",
    "AWND_q3_value",
    "AWND_q4_value",
]


def _apply_seasonal_mask(df: pd.DataFrame, crop: str) -> pd.DataFrame:
    """Zero out extremes outside planting/harvesting windows.

    Adds _in_planting and _in_harvesting variants for each extreme value column.
    Only these masked columns are kept downstream - raw extremes are dropped.
    """
    out = df.copy()
    flags = out["week_of_year"].apply(lambda w: crop_season_flag(int(w), crop))

    out["is_planting_week"] = flags.map(lambda f: f["is_planting_week"])
    out["is_harvesting_week"] = flags.map(lambda f: f["is_harvesting_week"])

    for col in _EXTREME_VALUE_COLS:
        out[f"{col}_in_planting"] = out[col] * out["is_planting_week"]
        out[f"{col}_in_harvesting"] = out[col] * out["is_harvesting_week"]

    return out.drop(
        columns=["is_planting_week", "is_harvesting_week", *_EXTREME_VALUE_COLS]
    )


def _load_production_weights(crop: str, production_dir: Path) -> pd.DataFrame:
    path = _require_path(production_dir / f"{crop}.csv", f"{crop} production by state")

    production_df = pd.read_csv(path)
    if "state" not in production_df.columns:
        msg = f"Missing 'state' column in production file: {path}"
        raise ValueError(msg)

    pattern = re.compile(rf"^{crop.capitalize()}ProductionByState_(\d{{4}})$")
    year_cols = [col for col in production_df.columns if pattern.match(col)]
    if not year_cols:
        msg = f"No production-by-year columns found for {crop} in: {path}"
        raise ValueError(msg)

    long = production_df[["state", *year_cols]].melt(
        id_vars="state", value_vars=year_cols, var_name="year_col", value_name="production"
    )
    long["year"] = long["year_col"].str.extract(r"(\d{4})$").astype(float).astype("Int64")
    long["production"] = pd.to_numeric(long["production"], errors="coerce")
    long["state"] = long["state"].astype(str).str.strip()

    long = (
        long[long["state"] != "United States"]
        .dropna(subset=["year", "production"])
        .pipe(lambda d: d[d["production"] > 0])
        .copy()
    )
    # Use one stable state weight across all years:
    # 1) average each state's production over all available years
    # 2) normalize by sum of state averages
    state_avg = (
        long.groupby("state", as_index=False)["production"]
        .mean()
        .rename(columns={"production": "avg_production"})
    )
    total_avg = state_avg["avg_production"].sum()
    state_avg["state_weight"] = np.where(
        total_avg > 0,
        state_avg["avg_production"] / total_avg,
        0.0,
    )

    years = (
        cast("pd.Series", long["year"].dropna())
        .astype("Int64")
        .drop_duplicates()
        .sort_values()
    )
    years_df = pd.DataFrame({"year": years})
    state_avg["join_key"] = 1
    years_df["join_key"] = 1
    expanded = state_avg.merge(years_df, on="join_key", how="inner").drop(
        columns=["join_key", "avg_production"]
    )

    return cast("pd.DataFrame", expanded[["state", "year", "state_weight"]])


def _weighted_aggregate(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    weight_col: str,
) -> pd.DataFrame:
    """Compute a production-weighted mean: sum(value * weight) / sum(weight)."""
    working = df[group_cols + value_cols + [weight_col]].copy()
    working[weight_col] = working[weight_col].fillna(0.0)
    for col in value_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)

    weighted = working.copy()
    for col in value_cols:
        weighted[col] = weighted[col] * weighted[weight_col]

    agg = weighted.groupby(group_cols, as_index=False)[value_cols].sum()
    totals = (
        weighted.groupby(group_cols, as_index=False)[weight_col]
        .sum()
        .rename(columns={weight_col: "total_weight"})
    )
    agg = agg.merge(totals, on=group_cols, how="left")
    for col in value_cols:
        agg[col] = np.where(agg["total_weight"] > 0, agg[col] / agg["total_weight"], 0.0)

    return agg


def _seasonal_value_cols(df: pd.DataFrame) -> list[str]:
    """All _in_planting and _in_harvesting extreme value columns."""
    return [col for col in df.columns if col.endswith(("_in_planting", "_in_harvesting"))]


def build_weighted_commodity_frame(
    masked_frame: pd.DataFrame,
    production_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate state-level masked extremes into commodity-level signals.

    Weighted by each state's share of production.
    Then attach weekly and monthly re-aggregations, plus raw SPI columns.
    """
    out = masked_frame.copy()
    out["state_name"] = out["state"].map(STATE_ABBREV_TO_NAME).fillna(out["state"])

    out = out.merge(
        production_weights,
        left_on=["state_name", "year"],
        right_on=["state", "year"],
        how="left",
        suffixes=("", "_prod"),
    ).drop(columns=["state_prod"], errors="ignore")
    out["state_weight"] = pd.to_numeric(out["state_weight"], errors="coerce").fillna(0.0)

    value_cols = _seasonal_value_cols(out)
    base_group = ["date", "year", "month", "week_of_year"]
    spi_cols = [c for c in SPI_COLUMNS if c in out.columns]

    all_value_cols = value_cols + spi_cols

    aggregated = _weighted_aggregate(out, base_group, all_value_cols, "state_weight")

    # Trailing windows over the commodity-level weekly series:
    # weekly_*  -> last 4 weeks (including current week)
    # monthly_* -> last 13 weeks (including current week), approx 1 quarter
    aggregated = aggregated.sort_values("date").reset_index(drop=True)
    for col in value_cols:
        aggregated[f"weekly_{col}"] = (
            aggregated[col].rolling(window=4, min_periods=1).mean()
        )
        aggregated[f"monthly_{col}"] = (
            aggregated[col].rolling(window=13, min_periods=1).mean()
        )

    keep = (
        ["date"]
        + all_value_cols
        + [f"weekly_{c}" for c in value_cols]
        + [f"monthly_{c}" for c in value_cols]
    )
    return aggregated[keep].sort_values("date").reset_index(drop=True)


def run_pipeline(input_path: Path | str, output_dir: Path | str) -> dict[str, Path]:
    resolved_input = _resolve_input_path(str(input_path))
    resolved_production = _resolve_production_dir()
    resolved_spi = _resolve_spi_input()
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    noaa_weekly = pd.read_csv(resolved_input)
    noaa_with_spi = _merge_spi_features(noaa_weekly, resolved_spi)
    zscore_df = _compute_zscores(noaa_with_spi)
    extreme_df = _compute_extreme_values(zscore_df)

    outputs: dict[str, Path] = {}
    for crop in CROPS:
        masked_frame = _apply_seasonal_mask(extreme_df, crop)
        production_weights = _load_production_weights(crop, resolved_production)
        weighted_df = build_weighted_commodity_frame(masked_frame, production_weights)

        out_path = resolved_output / f"climate_weekly_weighted_{crop}.csv"
        weighted_df.to_csv(out_path, index=False)
        logger.info(
            "Saved %s → %s (rows=%d, cols=%d)",
            crop,
            out_path,
            len(weighted_df),
            len(weighted_df.columns),
        )
        outputs[crop] = out_path

    return outputs
