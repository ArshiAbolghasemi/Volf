import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from src.dataset.climate.crop_seasonal import crop_season_flag
from src.dataset.util.path import require_path
from src.dataset.util.production_by_state import load_production_weights

logger = logging.getLogger(__name__)

CROPS = ("corn", "wheat", "soybean")

MODERATE_WET_LOW = 0.80
WET_LOW = 0.90
DRY_HIGH = 0.10
MODERATE_DRY_HIGH = 0.20

FEATURE_COLUMNS = ("moderate_wet", "wet", "moderate_dry", "dry")
SEASON_PERIODS = ("in_planting", "in_harvesting")
SEASONAL_FEATURE_COLUMNS = tuple(
    [f"{col}_{period}" for col in FEATURE_COLUMNS for period in SEASON_PERIODS]
)
ROLLING_FEATURE_COLUMNS = tuple(
    [
        f"{prefix}_{col}"
        for col in SEASONAL_FEATURE_COLUMNS
        for prefix in ("monthly", "seasonal")
    ]
)
ALL_PDSI_COLUMNS = SEASONAL_FEATURE_COLUMNS + ROLLING_FEATURE_COLUMNS
LEGACY_ROLLING_FEATURE_COLUMNS = tuple(
    [f"{prefix}_{col}" for col in FEATURE_COLUMNS for prefix in ("monthly", "seasonal")]
)
LEGACY_ALL_PDSI_COLUMNS = FEATURE_COLUMNS + LEGACY_ROLLING_FEATURE_COLUMNS

STATE_DIR_PARTS = 2


def _load_station_series(path: Path, state: str) -> pd.DataFrame:
    station = pd.read_csv(path, skiprows=1, names=["date_raw", "pdsi"], header=None)
    station["date"] = pd.to_datetime(station["date_raw"], format="%Y%m%d", errors="coerce")
    station["pdsi"] = pd.to_numeric(station["pdsi"], errors="coerce")
    station["state"] = state
    # Convert to Monday of the week for consistent merging
    station_clean = station[["date", "state", "pdsi"]].dropna(subset=["date", "pdsi"])
    station_clean["date"] = station_clean["date"] - pd.to_timedelta(
        station_clean["date"].dt.dayofweek, unit="D"
    )
    return cast("pd.DataFrame", station_clean)


def build_state_pdsi_frame(palmer_dir: Path | str) -> pd.DataFrame:
    root = require_path(Path(palmer_dir), "Palmer directory")
    frames: list[pd.DataFrame] = []

    for state_dir in sorted(root.iterdir()):
        if not state_dir.is_dir():
            continue
        parts = state_dir.name.split("_", 1)
        if len(parts) != STATE_DIR_PARTS:
            continue
        state_name = parts[1].replace("_", " ")
        station_files = sorted(state_dir.glob("*.csv"))
        frames.extend(
            _load_station_series(station_file, state_name) for station_file in station_files
        )

    if not frames:
        msg = f"No station files found under {root}"
        raise ValueError(msg)

    stacked = pd.concat(frames, ignore_index=True)
    return cast(
        "pd.DataFrame",
        stacked.groupby(["date", "state"], as_index=False)["pdsi"]
        .mean()
        .sort_values(["date", "state"]),
    )


def aggregate_pdsi_by_production_weights(
    state_pdsi: pd.DataFrame,
    production_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate state-level PDSI to national level using production weights.

    New approach: Aggregate FIRST, then compute z-scores and quantiles.
    This ensures PDSI bands are mutually exclusive at national level.
    """
    merged = state_pdsi.merge(production_weights, on="state", how="left")
    merged["state_weight"] = pd.to_numeric(merged["state_weight"], errors="coerce").fillna(
        0.0
    )

    # Weighted aggregation by date
    weighted = merged[["date", "pdsi", "state_weight"]].copy()
    weighted["pdsi"] = (
        pd.to_numeric(weighted["pdsi"], errors="coerce").fillna(0.0)
        * weighted["state_weight"]
    )

    # Group by date and sum
    agg = weighted.groupby("date", as_index=False).agg(
        {"pdsi": "sum", "state_weight": "sum"}
    )

    # Normalize by total weight
    agg["pdsi"] = np.where(agg["state_weight"] > 0, agg["pdsi"] / agg["state_weight"], 0.0)

    return agg.drop(columns=["state_weight"]).sort_values("date").reset_index(drop=True)


def compute_national_pdsi_features(national_pdsi: pd.DataFrame) -> pd.DataFrame:
    """Compute z-scores and PDSI bands on national-level aggregated data.

    This ensures bands are mutually exclusive (no overlaps).
    """
    out = national_pdsi.copy()

    # Compute z-score
    mean = out["pdsi"].mean()
    std = out["pdsi"].std()
    out["zscore"] = (out["pdsi"] - mean) / std if std > 0 else 0.0

    # Compute quantile thresholds
    q80 = out["zscore"].quantile(MODERATE_WET_LOW)
    q90 = out["zscore"].quantile(WET_LOW)
    q10 = out["zscore"].quantile(DRY_HIGH)
    q20 = out["zscore"].quantile(MODERATE_DRY_HIGH)

    z = out["zscore"].to_numpy(dtype=float)

    # Apply quantile logic (mutually exclusive bands)
    out["moderate_wet"] = np.where((z >= q80) & (z < q90), z, 0.0)
    out["wet"] = np.where(z >= q90, z, 0.0)
    out["moderate_dry"] = np.where((z > q10) & (z <= q20), z, 0.0)
    out["dry"] = np.where(z <= q10, z, 0.0)

    # Validate: bands should be mutually exclusive
    wet_overlap = (out["moderate_wet"] != 0) & (out["wet"] != 0)
    dry_overlap = (out["moderate_dry"] != 0) & (out["dry"] != 0)

    if wet_overlap.any():
        msg = (
            f"National-level overlap detected in wet bands: "
            f"{wet_overlap.sum()} rows have both moderate_wet and wet non-zero."
        )
        raise ValueError(msg)

    if dry_overlap.any():
        msg = (
            f"National-level overlap detected in dry bands: "
            f"{dry_overlap.sum()} rows have both moderate_dry and dry non-zero."
        )
        raise ValueError(msg)

    logger.info(
        "✓ National-level PDSI bands validated: no overlaps within wet or dry categories"
    )

    return out.drop(columns=["zscore"])


def add_rolling_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Add monthly (4-week) and seasonal (13-week) rolling averages for PDSI features."""
    out = df.copy()
    for col in SEASONAL_FEATURE_COLUMNS:
        out[f"monthly_{col}"] = out[col].rolling(window=4, min_periods=1).mean()
        out[f"seasonal_{col}"] = out[col].rolling(window=13, min_periods=1).mean()
    return out


def apply_crop_season_masks(pdsi_features: pd.DataFrame, crop: str) -> pd.DataFrame:
    """Restrict PDSI features to planting/harvesting weeks for a given crop."""
    out = pdsi_features.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if bool(out["date"].isna().any()):
        msg = "PDSI feature frame has invalid values in 'date'."
        raise ValueError(msg)

    out["week_of_year"] = out["date"].dt.isocalendar().week.astype(int)
    flags = out["week_of_year"].apply(lambda w: crop_season_flag(int(w), crop))
    out["is_planting_week"] = flags.map(lambda f: f["is_planting_week"])
    out["is_harvesting_week"] = flags.map(lambda f: f["is_harvesting_week"])

    for feature in FEATURE_COLUMNS:
        out[f"{feature}_in_planting"] = out[feature] * out["is_planting_week"]
        out[f"{feature}_in_harvesting"] = out[feature] * out["is_harvesting_week"]

    return out.drop(
        columns=[
            *FEATURE_COLUMNS,
            "week_of_year",
            "is_planting_week",
            "is_harvesting_week",
        ],
        errors="ignore",
    )


def _align_pdsi_to_ag_rows(
    pdsi_features: pd.DataFrame, ag_df: pd.DataFrame, crop: str
) -> pd.DataFrame:
    if "date" in ag_df.columns:
        ag_dates = pd.to_datetime(ag_df["date"], errors="coerce")
        if bool(ag_dates.isna().any()):
            msg = f"Invalid values in data/ag/{crop}.csv 'date' column."
            raise ValueError(msg)
        keyed = pd.DataFrame({"date": ag_dates})
        merged = keyed.merge(
            pdsi_features[["date", *ALL_PDSI_COLUMNS]], on="date", how="left"
        )
        return merged[list(ALL_PDSI_COLUMNS)].fillna(0.0)

    if len(pdsi_features) < len(ag_df):
        msg = (
            f"PDSI rows ({len(pdsi_features)}) are fewer than "
            f"data/ag/{crop}.csv rows ({len(ag_df)}). "
            f"Cannot align by row order."
        )
        raise ValueError(msg)

    # data/ag files currently have no date column, so align from the end.
    return pdsi_features[list(ALL_PDSI_COLUMNS)].tail(len(ag_df)).reset_index(drop=True)


def _append_metrics_to_ag_file(
    crop: str, pdsi_features: pd.DataFrame, ag_dir: Path
) -> Path:
    ag_path = require_path(ag_dir / f"{crop}.csv", f"data/ag/{crop}.csv")
    ag_df = pd.read_csv(ag_path)
    aligned = _align_pdsi_to_ag_rows(pdsi_features, ag_df, crop)

    out = ag_df.drop(columns=list(LEGACY_ALL_PDSI_COLUMNS), errors="ignore").copy()
    for col in ALL_PDSI_COLUMNS:
        out[col] = aligned[col].to_numpy()

    out.to_csv(ag_path, index=False)
    return ag_path


def run_pipeline(
    palmer_dir: Path | str,
    production_dir: Path | str,
    ag_dir: Path | str,
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Run the PDSI pipeline with new approach:
    1. Build state-level PDSI frame
    2. Aggregate by production weights (national level)
    3. Compute z-scores and bands on aggregated data
    4. Apply crop-specific planting/harvesting masks
    5. Add rolling windows
    """
    state_pdsi = build_state_pdsi_frame(palmer_dir)

    resolved_production_dir = require_path(Path(production_dir), "Production directory")
    resolved_ag_dir = require_path(Path(ag_dir), "data/ag directory")
    resolved_output = Path(output_dir) if output_dir else None
    if resolved_output is not None:
        resolved_output.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    for crop in CROPS:
        logger.info(f"Processing {crop}...")

        # Load production weights
        weights = load_production_weights(crop, resolved_production_dir)

        # Aggregate to national level
        national_pdsi = aggregate_pdsi_by_production_weights(state_pdsi, weights)

        # Compute features on national data
        feature_df = compute_national_pdsi_features(national_pdsi)

        # Keep only planting/harvesting season signals for this crop
        masked_df = apply_crop_season_masks(feature_df, crop)

        # Add rolling windows
        crop_df = add_rolling_windows(masked_df)

        # Append to ag file
        ag_path = _append_metrics_to_ag_file(crop, crop_df, resolved_ag_dir)
        outputs[crop] = ag_path
        logger.info("Appended %s pdsi features to %s", crop, ag_path)

        # Save standalone file if output dir specified
        if resolved_output is not None:
            out_path = resolved_output / f"{crop}_pdsi.csv"
            crop_df.to_csv(out_path, index=False)
            logger.info("Saved %s pdsi features to %s", crop, out_path)

    return outputs
