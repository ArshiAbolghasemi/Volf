from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.model import (
    HARFeatureConfig,
    HARGridConfig,
    HARRunConfig,
    HARSelectionConfig,
    HARSplitConfig,
    run_har_feature_set_grid,
)
from src.util.path import DATA_DIR
from src.variable_selection import BSRSelectionConfig, LassoSelectionConfig

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = DATA_DIR / "ag" / "v3.csv"
DEFAULT_TARGET = "wheat_weekly_rv"
DEFAULT_CORE_COLUMNS = ["wheat_weekly_rv", "wheat_monthly_rv", "wheat_seasonal_rv"]

CLIMATE_COLUMNS = [
    "ssta_elino",
    "ssta_lanina",
    "dry",
    "normal",
    "wet",
    "SOI_index",
    "NAO_index",
]

NEWS_BASE_COLUMNS = ["frbsf_sentiment", "Text_Climate_Anomaly"]
MACRO_COLUMNS = ["DJIA_Index", "WTI_Index", "Broad_Dollar_index", "Stock_Uncertainty"]


@dataclass
class WheatHARBenchmarkConfig:
    csv_path: str = str(DEFAULT_DATA_PATH)
    target_col: str = DEFAULT_TARGET
    core_columns: list[str] | None = None
    target_horizon: int = 1
    run_configs: dict[str, HARRunConfig] | None = None


def _existing_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in data.columns]


def _build_wheat_endogenous_columns(
    data: pd.DataFrame, core_columns: list[str]
) -> list[str]:
    core_set = set(core_columns)
    return sorted(
        [col for col in data.columns if col.startswith("wheat_") and col not in core_set]
    )


def _build_exogenous_columns(data: pd.DataFrame) -> list[str]:
    return sorted([col for col in data.columns if col.startswith(("corn_", "soybeans_"))])


def _build_news_columns(data: pd.DataFrame) -> list[str]:
    epu_cols = sorted([col for col in data.columns if col.startswith("epu_")])
    base = _existing_columns(data, NEWS_BASE_COLUMNS)
    return base + epu_cols


def build_wheat_feature_sets(
    data: pd.DataFrame,
    *,
    core_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    logger.info("Building Wheat benchmark feature sets")
    core = core_columns or _existing_columns(data, DEFAULT_CORE_COLUMNS)

    endo = _build_wheat_endogenous_columns(data, core)
    exo = _build_exogenous_columns(data)
    climate = _existing_columns(data, CLIMATE_COLUMNS)
    news = _build_news_columns(data)
    macro = _existing_columns(data, MACRO_COLUMNS)

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
        logger.info("Feature set '%s' includes %d features", name, len(unique_cols))

    return cleaned


def default_run_configs() -> dict[str, HARRunConfig]:
    return {
        "ols": HARRunConfig(
            split=HARSplitConfig(val_size=0.2, test_size=0.2),
            selection=HARSelectionConfig(method="none"),
        ),
        "lasso": HARRunConfig(
            split=HARSplitConfig(val_size=0.2, test_size=0.2),
            selection=HARSelectionConfig(
                method="lasso",
                lasso=LassoSelectionConfig(n_splits=5),
            ),
        ),
        "bsr": HARRunConfig(
            split=HARSplitConfig(val_size=0.2, test_size=0.2),
            selection=HARSelectionConfig(
                method="bsr",
                bsr=BSRSelectionConfig(alpha=0.05),
            ),
        ),
    }


def run_wheat_har_benchmark(
    *,
    config: WheatHARBenchmarkConfig | None = None,
    data: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    cfg = config or WheatHARBenchmarkConfig()
    logger.info("Starting Wheat HAR benchmark")
    if data is None:
        logger.info("Loading benchmark data from %s", cfg.csv_path)
        data = pd.read_csv(cfg.csv_path)

    if "Date" in data.columns:
        data = data.sort_values("Date").reset_index(drop=True)
        logger.info("Data sorted by Date. rows=%d", len(data))

    core = cfg.core_columns or _existing_columns(data, DEFAULT_CORE_COLUMNS)
    if not core:
        msg = "No valid core columns found in data."
        raise ValueError(msg)
    logger.info("Using core columns: %s", core)

    feature_sets = build_wheat_feature_sets(data, core_columns=core)

    model_run_configs = cfg.run_configs or default_run_configs()
    all_results: dict[str, dict[str, Any]] = {}

    for model_name, run_cfg in model_run_configs.items():
        logger.info("Running benchmark model type: %s", model_name)
        grid_cfg = HARGridConfig(
            feature_sets=feature_sets,
            base_feature_config=HARFeatureConfig(
                target_col=cfg.target_col,
                core_columns=core,
                target_horizon=cfg.target_horizon,
            ),
        )
        all_results[model_name] = run_har_feature_set_grid(
            data,
            grid_config=grid_cfg,
            run_config=run_cfg,
        )
        logger.info("Completed benchmark model type: %s", model_name)

    logger.info("Wheat HAR benchmark finished. model_types=%d", len(all_results))
    return all_results


def benchmark_results_to_frame(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    logger.info("Converting benchmark results to summary dataframe")
    rows: list[dict[str, Any]] = []

    for model_name, model_results in results.items():
        for feature_set_name, result in model_results.items():
            row = {
                "model_type": model_name,
                "feature_set": feature_set_name,
                "n_selected": len(result.selected_features),
                "val_mse": result.metrics["val"]["mse"],
                "val_mae": result.metrics["val"]["mae"],
                "val_qlike": result.metrics["val"]["qlike"],
                "val_r2log": result.metrics["val"]["r2log"],
                "test_mse": result.metrics["test"]["mse"],
                "test_mae": result.metrics["test"]["mae"],
                "test_qlike": result.metrics["test"]["qlike"],
                "test_r2log": result.metrics["test"]["r2log"],
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        logger.warning("Benchmark result dataframe is empty")
        return out
    summary = out.sort_values(["model_type", "test_mse", "test_mae"]).reset_index(drop=True)
    logger.info("Benchmark summary rows=%d", len(summary))
    return summary
