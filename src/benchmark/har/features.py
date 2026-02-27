from __future__ import annotations

import logging

import pandas as pd  # noqa: TC002

from src.model import HARModelConfig, HARRunConfig, HARSelectionConfig, HARWalkForwardConfig
from src.variable_selection import BSRSelectionConfig, LassoSelectionConfig

from .types import (
    CLIMATE_COLUMNS,
    DEFAULT_CORE_COLUMNS,
    DEFAULT_INITIAL_TRAIN_SIZE,
    DEFAULT_ROLLING_WINDOW_SIZE,
    DEFAULT_STEP,
    DEFAULT_TEST_SIZE,
    MACRO_COLUMNS,
    NEWS_BASE_COLUMNS,
)

logger = logging.getLogger(__name__)


def existing_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
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
    epu_cols = ["epu_index"]
    base = existing_columns(data, NEWS_BASE_COLUMNS)
    return base + epu_cols


def build_wheat_feature_sets(
    data: pd.DataFrame,
    *,
    core_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    logger.info("Building Wheat benchmark feature sets")
    core = core_columns or existing_columns(data, DEFAULT_CORE_COLUMNS)

    endo = _build_wheat_endogenous_columns(data, core)
    exo = _build_exogenous_columns(data)
    climate = existing_columns(data, CLIMATE_COLUMNS)
    news = _build_news_columns(data)
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
        logger.info("Feature set '%s' includes %d features", name, len(unique_cols))

    return cleaned


def default_run_configs() -> dict[str, HARRunConfig]:
    return {
        "ols_expanding": HARRunConfig(
            walk_forward=HARWalkForwardConfig(
                window_type="expanding",
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
            ),
            selection=HARSelectionConfig(method="none"),
            model=HARModelConfig(standardize_features=False),
        ),
        "ols_rolling": HARRunConfig(
            walk_forward=HARWalkForwardConfig(
                window_type="rolling",
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
                rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE,
            ),
            selection=HARSelectionConfig(method="none"),
            model=HARModelConfig(standardize_features=False),
        ),
        "lasso_expanding": HARRunConfig(
            walk_forward=HARWalkForwardConfig(
                window_type="expanding",
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
            ),
            selection=HARSelectionConfig(
                method="lasso",
                lasso=LassoSelectionConfig(n_splits=5),
                refit_every_windows=4,
            ),
            model=HARModelConfig(standardize_features=True),
        ),
        "lasso_rolling": HARRunConfig(
            walk_forward=HARWalkForwardConfig(
                window_type="rolling",
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
                rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE,
            ),
            selection=HARSelectionConfig(
                method="lasso",
                lasso=LassoSelectionConfig(n_splits=5),
                refit_every_windows=4,
            ),
            model=HARModelConfig(standardize_features=True),
        ),
        "bsr_expanding": HARRunConfig(
            walk_forward=HARWalkForwardConfig(
                window_type="expanding",
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
            ),
            selection=HARSelectionConfig(
                method="bsr",
                bsr=BSRSelectionConfig(alpha=0.05),
                refit_every_windows=8,
            ),
            model=HARModelConfig(standardize_features=False),
        ),
        "bsr_rolling": HARRunConfig(
            walk_forward=HARWalkForwardConfig(
                window_type="rolling",
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
                rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE,
            ),
            selection=HARSelectionConfig(
                method="bsr",
                bsr=BSRSelectionConfig(alpha=0.05),
                refit_every_windows=8,
            ),
            model=HARModelConfig(standardize_features=False),
        ),
    }
