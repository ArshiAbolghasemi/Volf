from __future__ import annotations

from typing import Literal

import pandas as pd  # noqa: TC002

DEFAULT_TARGET = "wheat_weekly_rv"
DEFAULT_CORE_COLUMNS = ["wheat_weekly_rv", "wheat_monthly_rv", "wheat_seasonal_rv"]

CLIMATE_COLUMNS = [
    "ssta_elino",
    "ssta_lanina",
    "dry",
    "wet",
    "SOI_index",
    "NAO_index",
    "Text_Climate_Anomaly",
]

NEWS_BASE_COLUMNS = ["frbsf_sentiment", "Text_Climate_Anomaly"]
MACRO_COLUMNS = ["DJIA_Index", "WTI_Index", "Broad_Dollar_index", "Stock_Uncertainty"]


def existing_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in data.columns]


def normalize_target_mode(value: str) -> Literal["point", "mean"]:
    raw = value.strip().lower()
    if raw in {"point", "mean"}:
        return raw if raw == "point" else "mean"
    msg = f"Unknown target_mode '{value}'. expected one of: point, mean."
    raise ValueError(msg)


def build_wheat_feature_sets(
    data: pd.DataFrame,
    *,
    core_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    core = core_columns or existing_columns(data, DEFAULT_CORE_COLUMNS)
    core_set = set(core)

    endo = sorted(
        [col for col in data.columns if col.startswith("wheat_") and col not in core_set]
    )
    exo = sorted([col for col in data.columns if col.startswith(("corn_", "soybeans_"))])
    climate = existing_columns(data, CLIMATE_COLUMNS)
    news = [*existing_columns(data, NEWS_BASE_COLUMNS), "epu_index"]
    macro = existing_columns(data, MACRO_COLUMNS)

    feature_sets = {
        "har": [],
        "har_endo": endo,
        "har_endo_exo": endo + exo,
        "har_endo_exogenous_climate": endo + exo + climate,
        "har_endo_exogenous_climate_news": endo + exo + climate + news,
        "har__all": endo + exo + climate + news + macro,
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
