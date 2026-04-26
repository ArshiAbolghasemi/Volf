import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.dataset.climate.crop_seasonal import crop_season_flag
from src.dataset.climate.pdsi import build_state_pdsi_frame
from src.dataset.util.production_by_state import (
    STATE_ABBREV_TO_NAME,
    load_production_weights,
)
from src.util.path import DATA_DIR

logger = logging.getLogger(__name__)

CROPS = ("corn", "soybean", "wheat")
NOAA_VARS = ("TMAX", "TMIN", "AWND")
SPI_VARS = ("SPI_7d", "SPI_1m", "SPI_3m")

NOAA_WEEKLY = DATA_DIR / "climate" / "noaa_weekly.csv"
SPI_WEEKLY = DATA_DIR / "climate" / "spi_weekly_multiscale.csv"
CO2_WEEKLY = DATA_DIR / "climate" / "co2_weekly_mlo.csv"
PALMER_DIR = DATA_DIR / "climate" / "palmer"
PRODUCTION_DIR = DATA_DIR / "production_by_state"
V6_FILE = DATA_DIR / "ag" / "v6.csv"

ALPHA = 0.05
MIN_ADF_OBS = 10
MIN_TREND_OBS = 2


def run_adf_test(series: pd.Series) -> tuple[bool, float]:
    clean_series = series.dropna()
    if len(clean_series) < MIN_ADF_OBS:
        return False, np.nan
    try:
        p_value = float(adfuller(clean_series, autolag="AIC")[1])
    except Exception:
        return False, np.nan
    return p_value >= ALPHA, p_value


def linear_detrend(series: pd.Series) -> tuple[pd.Series, np.ndarray | None]:
    clean_idx = series.dropna().index
    if len(clean_idx) < MIN_TREND_OBS:
        return series, None

    x = np.arange(len(clean_idx))
    y = series.loc[clean_idx].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    detrended = series.copy()
    detrended.loc[clean_idx] = y - (slope * x + intercept)

    full_x = np.arange(len(series))
    full_trend = slope * full_x + intercept
    return detrended, full_trend


def _standardize(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return pd.Series(np.nan, index=series.index)
    std = clean.std()
    if bool(pd.isna(std)) or std == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - clean.mean()) / std


def _build_quantile_masks(  # noqa: PLR0911
    zscore: pd.Series, variable_name: str
) -> dict[str, pd.Series]:
    clean = zscore.dropna()
    if clean.empty:
        return {}

    q10 = clean.quantile(0.10)
    q20 = clean.quantile(0.20)
    q25 = clean.quantile(0.25)
    q75 = clean.quantile(0.75)
    q80 = clean.quantile(0.80)
    q90 = clean.quantile(0.90)
    q95 = clean.quantile(0.95)

    if variable_name == "TMAX":
        return {
            "hot": (zscore >= q75) & (zscore <= q90),
            "very_hot": zscore > q90,
        }
    if variable_name == "TMIN":
        return {
            "cold": zscore < q10,
            "very_cold": (zscore >= q10) & (zscore <= q25),
        }
    if variable_name == "AWND":
        return {
            "moderate_high": (zscore >= q80) & (zscore <= q90),
            "extreme_high": zscore > q90,
        }
    if variable_name in SPI_VARS:
        return {
            "high": (zscore >= q80) & (zscore <= q90),
            "extreme": zscore > q90,
        }
    if variable_name == "pdsi":
        return {
            "extreme_dry": zscore < q10,
            "dry": (zscore >= q10) & (zscore <= q20),
            "wet": (zscore >= q80) & (zscore <= q90),
            "extreme_wet": zscore > q90,
        }
    if variable_name == "co2":
        return {"extreme": zscore > q95}
    return {}


def _state_season_flags(state_df: pd.DataFrame, crop: str) -> tuple[pd.Series, pd.Series]:
    flags = state_df["week_of_year"].apply(lambda week: crop_season_flag(int(week), crop))
    is_planting = flags.map(lambda item: item["is_planting_week"] == 1)
    is_harvesting = flags.map(lambda item: item["is_harvesting_week"] == 1)
    return cast("pd.Series", is_planting), cast("pd.Series", is_harvesting)


def _process_state_variable(
    state_df: pd.DataFrame,
    variable_name: str,
    crop: str,
) -> pd.DataFrame:
    series = pd.to_numeric(state_df[variable_name], errors="coerce")
    has_trend, p_value = run_adf_test(series)

    if has_trend:
        detrended, trend_values = linear_detrend(series)
        series_for_quantile = detrended
        logger.info(
            "Trend detected for %s (%s), p=%.5f",
            variable_name,
            state_df["state"].iloc[0],
            p_value,
        )
    else:
        trend_values = None
        series_for_quantile = series

    zscore = _standardize(series_for_quantile)
    quantile_masks = _build_quantile_masks(zscore, variable_name)
    planting_flag, harvesting_flag = _state_season_flags(state_df, crop)

    values_for_output = series_for_quantile.copy()
    if trend_values is not None:
        values_for_output = values_for_output + trend_values

    out = pd.DataFrame({"date": state_df["date"], "state": state_df["state"]})
    for label, mask in quantile_masks.items():
        selected = pd.Series(0.0, index=state_df.index, dtype=float)
        mask_filled = mask.fillna(value=False)
        selected[mask_filled] = values_for_output[mask_filled]

        planting_col = f"{label}_in_planting"
        harvesting_col = f"{label}_in_harvesting"

        out[planting_col] = np.where(planting_flag, selected, 0.0)
        out[harvesting_col] = np.where(harvesting_flag, selected, 0.0)

    return out


def _prepare_noaa_states(crop: str, noaa_path: Path) -> pd.DataFrame:
    noaa_df = pd.read_csv(noaa_path, parse_dates=["date"])
    noaa_df["week_of_year"] = noaa_df["date"].dt.isocalendar().week.astype(int)

    state_frames: list[pd.DataFrame] = []
    for state in sorted(noaa_df["state"].dropna().unique()):
        state_data = noaa_df[noaa_df["state"] == state].copy()
        if state_data.empty:
            continue

        merged_state = state_data[["date", "state"]].copy()
        for variable_name in NOAA_VARS:
            if variable_name not in state_data.columns:
                continue
            var_features = _process_state_variable(state_data, variable_name, crop)
            value_cols = [
                col for col in var_features.columns if col not in ("date", "state")
            ]
            merged_state = merged_state.merge(
                var_features[["date", "state", *value_cols]],
                on=["date", "state"],
                how="left",
            )
        state_frames.append(merged_state)

    if not state_frames:
        return pd.DataFrame(columns=["date", "state"])
    return pd.concat(state_frames, ignore_index=True)


def _prepare_spi_states(crop: str, spi_path: Path) -> pd.DataFrame:
    spi_df = pd.read_csv(spi_path, parse_dates=["week_date"])
    spi_df["date"] = spi_df["week_date"]
    spi_df["week_of_year"] = spi_df["date"].dt.isocalendar().week.astype(int)

    state_frames: list[pd.DataFrame] = []
    for state in sorted(spi_df["state"].dropna().unique()):
        state_data = spi_df[spi_df["state"] == state].copy()
        if state_data.empty:
            continue

        merged_state = state_data[["date", "state"]].copy()
        for variable_name in SPI_VARS:
            if variable_name not in state_data.columns:
                continue
            var_features = _process_state_variable(state_data, variable_name, crop)
            value_cols = [
                col for col in var_features.columns if col not in ("date", "state")
            ]
            merged_state = merged_state.merge(
                var_features[["date", "state", *value_cols]],
                on=["date", "state"],
                how="left",
            )
        state_frames.append(merged_state)

    if not state_frames:
        return pd.DataFrame(columns=["date", "state"])
    return pd.concat(state_frames, ignore_index=True)


def _prepare_pdsi_states(crop: str, palmer_dir: Path) -> pd.DataFrame:
    pdsi_df = build_state_pdsi_frame(palmer_dir)
    pdsi_df["week_of_year"] = pdsi_df["date"].dt.isocalendar().week.astype(int)

    state_frames: list[pd.DataFrame] = []
    for state in sorted(pdsi_df["state"].dropna().unique()):
        state_data = pdsi_df[pdsi_df["state"] == state].copy()
        if state_data.empty:
            continue
        state_frames.append(_process_state_variable(state_data, "pdsi", crop))

    if not state_frames:
        return pd.DataFrame(columns=["date", "state"])
    return pd.concat(state_frames, ignore_index=True)


def _prepare_co2_features(crop: str, co2_path: Path) -> pd.DataFrame:
    co2_df = pd.read_csv(co2_path, parse_dates=["date"])
    co2_value_col = next(
        column for column in co2_df.columns if column not in ("date", "", "Unnamed: 0")
    )
    co2_df["week_of_year"] = co2_df["date"].dt.isocalendar().week.astype(int)

    series = pd.to_numeric(co2_df[co2_value_col], errors="coerce")
    has_trend, _ = run_adf_test(series)
    if has_trend:
        detrended, trend_values = linear_detrend(series)
        series_for_quantile = detrended
    else:
        trend_values = None
        series_for_quantile = series

    zscore = _standardize(series_for_quantile)
    masks = _build_quantile_masks(zscore, "co2")
    selected = pd.Series(0.0, index=co2_df.index, dtype=float)
    if "extreme" in masks:
        extreme_mask = masks["extreme"].fillna(value=False)
        selected[extreme_mask] = series_for_quantile[extreme_mask]
    if trend_values is not None:
        non_zero = selected != 0
        selected[non_zero] = selected[non_zero] + trend_values[non_zero]

    flags = co2_df["week_of_year"].apply(lambda week: crop_season_flag(int(week), crop))
    is_planting = flags.map(lambda item: item["is_planting_week"] == 1)
    is_harvesting = flags.map(lambda item: item["is_harvesting_week"] == 1)

    out = co2_df[["date"]].copy()
    out["co2_extreme_in_planting"] = np.where(is_planting, selected, 0.0)
    out["co2_extreme_in_harvesting"] = np.where(is_harvesting, selected, 0.0)
    return out


def _state_key_from_abbrev_or_name(state_value: pd.Series) -> pd.Series:
    stripped = state_value.astype(str).str.strip()
    mapped = stripped.map(STATE_ABBREV_TO_NAME)
    return mapped.fillna(stripped)


def _weighted_aggregate_by_date(
    state_features: pd.DataFrame,
    crop: str,
    production_dir: Path,
) -> pd.DataFrame:
    if state_features.empty:
        return pd.DataFrame(columns=["date"])

    weights = load_production_weights(crop, production_dir).rename(
        columns={"state": "state_key"}
    )
    features = state_features.copy()
    features["state_key"] = _state_key_from_abbrev_or_name(features["state"])
    features = features.merge(weights, on="state_key", how="left")
    features["state_weight"] = pd.to_numeric(
        features["state_weight"], errors="coerce"
    ).fillna(0.0)

    value_cols = [
        column
        for column in features.columns
        if column not in ("date", "state", "state_key", "state_weight")
    ]
    features[value_cols] = (
        features[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )

    rows: list[dict[str, float | pd.Timestamp]] = []
    for date, group in features.groupby("date", sort=True):
        total_weight = group["state_weight"].sum()
        row: dict[str, float | pd.Timestamp] = {"date": date}
        for column in value_cols:
            if total_weight > 0:
                row[column] = float(
                    (group[column] * group["state_weight"]).sum() / total_weight
                )
            else:
                row[column] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _merge_state_feature_sources(
    crop: str, noaa_path: Path, spi_path: Path, palmer_dir: Path
) -> pd.DataFrame:
    noaa_state = _prepare_noaa_states(crop, noaa_path)
    spi_state = _prepare_spi_states(crop, spi_path)
    pdsi_state = _prepare_pdsi_states(crop, palmer_dir)

    merged = noaa_state.merge(spi_state, on=["date", "state"], how="outer")
    merged = merged.merge(pdsi_state, on=["date", "state"], how="outer")
    value_cols = [column for column in merged.columns if column not in ("date", "state")]
    merged[value_cols] = merged[value_cols].fillna(0.0)
    return merged


def process_crop(  # noqa: PLR0913
    crop: str,
    v6_df: pd.DataFrame,
    noaa_path: Path,
    spi_path: Path,
    co2_path: Path,
    palmer_dir: Path,
    production_dir: Path,
) -> pd.DataFrame:
    logger.info("Processing crop=%s", crop)

    state_features = _merge_state_feature_sources(crop, noaa_path, spi_path, palmer_dir)
    weighted = _weighted_aggregate_by_date(state_features, crop, production_dir)
    co2_features = _prepare_co2_features(crop, co2_path)

    climate = weighted.merge(co2_features, on="date", how="outer").sort_values("date")
    climate_cols = [column for column in climate.columns if column != "date"]
    climate = climate.rename(
        columns={column: f"{crop}_{column}" for column in climate_cols}
    )

    merged = v6_df.merge(climate, on="date", how="left")
    new_cols = [column for column in merged.columns if column.startswith(f"{crop}_")]
    merged[new_cols] = merged[new_cols].fillna(0.0)
    logger.info("Completed crop=%s rows=%d new_cols=%d", crop, len(merged), len(new_cols))
    return merged


def run_pipeline(  # noqa: PLR0913
    v6_path: Path | str = V6_FILE,
    output_dir: Path | str = DATA_DIR / "ag",
    noaa_path: Path | str = NOAA_WEEKLY,
    spi_path: Path | str = SPI_WEEKLY,
    co2_path: Path | str = CO2_WEEKLY,
    palmer_dir: Path | str = PALMER_DIR,
    production_dir: Path | str = PRODUCTION_DIR,
) -> dict[str, Path]:
    resolved_v6 = Path(v6_path)
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    v6_df = pd.read_csv(resolved_v6, parse_dates=["date"])
    outputs: dict[str, Path] = {}

    for crop in CROPS:
        crop_df = process_crop(
            crop=crop,
            v6_df=v6_df,
            noaa_path=Path(noaa_path),
            spi_path=Path(spi_path),
            co2_path=Path(co2_path),
            palmer_dir=Path(palmer_dir),
            production_dir=Path(production_dir),
        )
        out_path = resolved_output / f"{crop}.csv"
        crop_df.to_csv(out_path, index=False)
        outputs[crop] = out_path
        logger.info("Saved %s output: %s", crop, out_path)

    return outputs
