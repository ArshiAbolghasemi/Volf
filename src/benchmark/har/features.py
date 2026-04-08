from __future__ import annotations

import pandas as pd  # noqa: TC002

from src.benchmark.utils import (
    build_wheat_feature_sets as _build_wheat_feature_sets_common,
)
from src.benchmark.utils import existing_columns as _existing_columns_common
from src.model import HARModelConfig, HARRunConfig, HARSelectionConfig, HARWalkForwardConfig
from src.variable_selection import BSRSelectionConfig, LassoSelectionConfig

from .types import (
    DEFAULT_INITIAL_TRAIN_SIZE,
    DEFAULT_ROLLING_WINDOW_SIZE,
    DEFAULT_STEP,
    DEFAULT_TEST_SIZE,
)


def existing_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    return _existing_columns_common(data, columns)


def build_wheat_feature_sets(
    data: pd.DataFrame,
    *,
    core_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    return _build_wheat_feature_sets_common(data, core_columns=core_columns)


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
