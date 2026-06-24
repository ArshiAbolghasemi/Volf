from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_TARGET = "wheat_weekly_rv"
DEFAULT_CORE_COLUMNS = ["wheat_weekly_rv", "wheat_monthly_rv", "wheat_seasonal_rv"]

CLIMATE_COLUMNS = [
    "ssta_elino",
    "ssta_lanina",
    "SOI_index",
    "NAO_index",
    "tmax_hot_in_planting",
    "tmax_hot_in_harvesting",
    "tmax_very_hot_in_planting",
    "tmax_very_hot_in_harvesting",
    "tmin_cold_in_planting",
    "tmin_cold_in_harvesting",
    "tmin_very_cold_in_planting",
    "tmin_very_cold_in_harvesting",
    "awnd_moderate_high_wind_in_planting",
    "awnd_moderate_high_wind_in_harvesting",
    "awnd_extreme_high_wind_in_planting",
    "awnd_extreme_high_wind_in_harvesting",
    "spi_7d_very_wet_in_planting",
    "spi_7d_very_wet_in_harvesting",
    "spi_7d_extreme_wet_in_planting",
    "spi_7d_extreme_wet_in_harvesting",
    "spi_7d_very_dry_in_planting",
    "spi_7d_very_dry_in_harvesting",
    "spi_7d_extreme_dry_in_planting",
    "spi_7d_extreme_dry_in_harvesting",
    "pdsi_very_wet_in_planting",
    "pdsi_very_wet_in_harvesting",
    "pdsi_extreme_wet_in_planting",
    "pdsi_extreme_wet_in_harvesting",
    "pdsi_extreme_drought_in_planting",
    "pdsi_extreme_drought_in_harvesting",
    "pdsi_severe_drought_in_planting",
    "pdsi_severe_drought_in_harvesting",
    "co2_extreme_in_planting",
    "co2_extreme_in_harvesting",
]

NEWS_BASE_COLUMNS = ["frbsf_sentiment", "Text_Climate_Anomaly", "epu_index"]
MACRO_COLUMNS = ["DJIA_Index", "WTI_Index", "Broad_Dollar_index", "Stock_Uncertainty"]

FEATURE_GROUP_COLUMNS = {
    "endo": [],
    "exo": [],
    "news": NEWS_BASE_COLUMNS,
    "macro": MACRO_COLUMNS,
    "climate": CLIMATE_COLUMNS,
}


def existing_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in data.columns]


def get_feature_group_columns(
    *,
    endo_columns: list[str] | None = None,
    exo_columns: list[str] | None = None,
    climate_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    return {
        "endo": list(endo_columns or []),
        "exo": list(exo_columns or []),
        "news": list(NEWS_BASE_COLUMNS),
        "macro": list(MACRO_COLUMNS),
        "climate": list(climate_columns or CLIMATE_COLUMNS),
    }


def classify_feature_group(
    feature_name: str,
    *,
    endo_columns: list[str] | None = None,
    exo_columns: list[str] | None = None,
    climate_columns: list[str] | None = None,
) -> str | None:
    for group_name, columns in get_feature_group_columns(
        endo_columns=endo_columns, exo_columns=exo_columns, climate_columns=climate_columns
    ).items():
        if feature_name in columns:
            return group_name
    return None


def normalize_target_mode(value: str) -> Literal["point", "mean"]:
    raw = value.strip().lower()
    if raw in {"point", "mean"}:
        return raw if raw == "point" else "mean"
    msg = f"Unknown target_mode '{value}'. expected one of: point, mean."
    raise ValueError(msg)


def infer_target_prefix(target_col: str) -> str:
    if target_col.endswith("_weekly_rv"):
        return target_col[: -len("_weekly_rv")]
    return target_col.split("_", 1)[0]


def benchmark_crop_dir_name(target_col: str) -> str:
    prefix = infer_target_prefix(target_col)
    if prefix == "soybeans":
        return "soybean"
    return prefix


def default_core_columns_for_target(target_col: str) -> list[str]:
    prefix = infer_target_prefix(target_col)
    return [f"{prefix}_weekly_rv", f"{prefix}_monthly_rv", f"{prefix}_seasonal_rv"]


def default_endo_columns_for_target(target_col: str) -> list[str]:
    prefix = infer_target_prefix(target_col)
    return [
        f"{prefix}_weekly_rvb",
        f"{prefix}_weekly_rvg",
        f"{prefix}_weekly_jumps",
    ]


def default_exo_columns_for_target(target_col: str) -> list[str]:
    target_prefix = infer_target_prefix(target_col)
    crop_prefixes = ("wheat", "corn", "soybeans")
    return [
        f"{crop_prefix}_weekly_rv"
        for crop_prefix in crop_prefixes
        if crop_prefix != target_prefix
    ]


def build_target_feature_sets(
    data: pd.DataFrame,
    *,
    target_col: str,
    core_columns: list[str] | None = None,
    climate_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    default_core = default_core_columns_for_target(target_col)
    _ = core_columns or existing_columns(data, default_core)
    endo = existing_columns(data, default_endo_columns_for_target(target_col))
    exo = existing_columns(data, default_exo_columns_for_target(target_col))
    climate = existing_columns(data, climate_columns or CLIMATE_COLUMNS)
    news = existing_columns(data, NEWS_BASE_COLUMNS)
    macro = existing_columns(data, MACRO_COLUMNS)

    feature_sets = {
        "har": [],
        "har_endo": endo,
        "har_endo_exo": endo + exo,
        "har_endo_exo_news": endo + exo + news,
        "har_endo_exo_macro": endo + exo + macro,
        "har_endo_exo_climate": endo + exo + climate,
        "har_endo_exo_climate_news": endo + exo + climate + news,
        "har_endo_exo_climate_macro": endo + exo + climate + macro,
        "har_endo_exo_news_macro": endo + exo + news + macro,
        "har_endo_exo_climate_news_macro": endo + exo + climate + news + macro,
    }

    cleaned: dict[str, list[str]] = {}
    for name, cols in feature_sets.items():
        seen: set[str] = set()
        unique_cols: list[str] = []
        for col in cols:
            if col in data.columns and col not in seen:
                seen.add(col)
                unique_cols.append(col)
        cleaned[name] = unique_cols
    return cleaned


def build_wheat_feature_sets(
    data: pd.DataFrame,
    *,
    core_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    return build_target_feature_sets(
        data,
        target_col=DEFAULT_TARGET,
        core_columns=core_columns,
        climate_columns=CLIMATE_COLUMNS,
    )
