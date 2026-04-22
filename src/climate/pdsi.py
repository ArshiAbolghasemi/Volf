import logging
import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CROPS = ("corn", "wheat", "soybean")

MODERATE_WET_LOW = 0.80
WET_LOW = 0.90
DRY_HIGH = 0.10
MODERATE_DRY_HIGH = 0.20

FEATURE_COLUMNS = ("moderate_wet", "wet", "moderate_dry", "dry")
STATE_DIR_PARTS = 2


def _require_path(path: Path, label: str) -> Path:
    if not path.exists():
        msg = f"{label} not found: {path}"
        raise FileNotFoundError(msg)
    return path


def _load_station_series(path: Path, state: str) -> pd.DataFrame:
    station = pd.read_csv(path, skiprows=1, names=["date_raw", "pdsi"], header=None)
    station["date"] = pd.to_datetime(station["date_raw"], format="%Y%m%d", errors="coerce")
    station["pdsi"] = pd.to_numeric(station["pdsi"], errors="coerce")
    station["state"] = state
    return cast(
        "pd.DataFrame", station[["date", "state", "pdsi"]].dropna(subset=["date", "pdsi"])
    )


def build_state_pdsi_frame(palmer_dir: Path | str) -> pd.DataFrame:
    root = _require_path(Path(palmer_dir), "Palmer directory")
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


def _add_zscore(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group = out.groupby("state")["pdsi"]
    mean = group.transform("mean")
    std = group.transform("std").replace(0, np.nan)
    out["zscore"] = ((out["pdsi"] - mean) / std).fillna(0.0)
    return out


def _add_quantile_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    state_group = out.groupby("state")["zscore"]

    out["q80"] = state_group.transform(lambda values: values.quantile(MODERATE_WET_LOW))
    out["q90"] = state_group.transform(lambda values: values.quantile(WET_LOW))
    out["q10"] = state_group.transform(lambda values: values.quantile(DRY_HIGH))
    out["q20"] = state_group.transform(lambda values: values.quantile(MODERATE_DRY_HIGH))

    z = out["zscore"].to_numpy(dtype=float)
    q80 = out["q80"].to_numpy(dtype=float)
    q90 = out["q90"].to_numpy(dtype=float)
    q10 = out["q10"].to_numpy(dtype=float)
    q20 = out["q20"].to_numpy(dtype=float)

    out["moderate_wet"] = np.where((z >= q80) & (z < q90), z, 0.0)
    out["wet"] = np.where(z >= q90, z, 0.0)
    out["moderate_dry"] = np.where((z >= q10) & (z < q20), z, 0.0)
    out["dry"] = np.where(z < q10, z, 0.0)

    return out.drop(columns=["q80", "q90", "q10", "q20"])


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
        id_vars="state",
        value_vars=year_cols,
        var_name="year_col",
        value_name="production",
    )
    long["production"] = pd.to_numeric(long["production"], errors="coerce")
    long["state"] = long["state"].astype(str).str.strip()
    long = (
        long[long["state"] != "United States"]
        .dropna(subset=["production"])
        .pipe(lambda d: d[d["production"] > 0])
    )

    state_avg = (
        long.groupby("state", as_index=False)["production"]
        .mean()
        .rename(columns={"production": "avg_production"})
    )
    total = state_avg["avg_production"].sum()
    state_avg["state_weight"] = np.where(
        total > 0, state_avg["avg_production"] / total, 0.0
    )

    return cast("pd.DataFrame", state_avg[["state", "state_weight"]])


def _weighted_daily_features(
    state_features: pd.DataFrame, weights: pd.DataFrame
) -> pd.DataFrame:
    merged = state_features.merge(weights, on="state", how="left")
    merged["state_weight"] = pd.to_numeric(merged["state_weight"], errors="coerce").fillna(
        0.0
    )

    weighted = merged[["date", *FEATURE_COLUMNS, "state_weight"]].copy()
    for col in FEATURE_COLUMNS:
        weighted[col] = (
            pd.to_numeric(weighted[col], errors="coerce").fillna(0.0)
            * weighted["state_weight"]
        )

    totals = weighted.groupby("date", as_index=False)["state_weight"].sum()
    agg = weighted.groupby("date", as_index=False)[list(FEATURE_COLUMNS)].sum()
    agg = agg.merge(totals, on="date", how="left")
    for col in FEATURE_COLUMNS:
        agg[col] = np.where(agg["state_weight"] > 0, agg[col] / agg["state_weight"], 0.0)

    return agg.drop(columns=["state_weight"]).sort_values("date").reset_index(drop=True)


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
            pdsi_features[["date", *FEATURE_COLUMNS]], on="date", how="left"
        )
        return merged[list(FEATURE_COLUMNS)].fillna(0.0)

    if len(pdsi_features) < len(ag_df):
        msg = (
            f"PDSI rows ({len(pdsi_features)}) are fewer than "
            f"data/ag/{crop}.csv rows ({len(ag_df)}). "
            f"Cannot align by row order."
        )
        raise ValueError(msg)

    # data/ag files currently have no date column, so align from the end.
    return pdsi_features[list(FEATURE_COLUMNS)].tail(len(ag_df)).reset_index(drop=True)


def _append_metrics_to_ag_file(
    crop: str, pdsi_features: pd.DataFrame, ag_dir: Path
) -> Path:
    ag_path = _require_path(ag_dir / f"{crop}.csv", f"data/ag/{crop}.csv")
    ag_df = pd.read_csv(ag_path)
    aligned = _align_pdsi_to_ag_rows(pdsi_features, ag_df, crop)

    out = ag_df.copy()
    for col in FEATURE_COLUMNS:
        out[col] = aligned[col].to_numpy()

    out.to_csv(ag_path, index=False)
    return ag_path


def run_pipeline(
    palmer_dir: Path | str,
    production_dir: Path | str,
    ag_dir: Path | str,
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    state_pdsi = build_state_pdsi_frame(palmer_dir)
    state_features = _add_quantile_features(_add_zscore(state_pdsi))

    resolved_production_dir = _require_path(Path(production_dir), "Production directory")
    resolved_ag_dir = _require_path(Path(ag_dir), "data/ag directory")
    resolved_output = Path(output_dir) if output_dir else None
    if resolved_output is not None:
        resolved_output.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    for crop in CROPS:
        weights = _load_production_weights(crop, resolved_production_dir)
        crop_df = _weighted_daily_features(state_features, weights)
        ag_path = _append_metrics_to_ag_file(crop, crop_df, resolved_ag_dir)
        outputs[crop] = ag_path
        logger.info("Appended %s pdsi features to %s", crop, ag_path)
        if resolved_output is not None:
            out_path = resolved_output / f"{crop}_pdsi.csv"
            crop_df.to_csv(out_path, index=False)
            logger.info("Saved %s pdsi features to %s", crop, out_path)

    return outputs
