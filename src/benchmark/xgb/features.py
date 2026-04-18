from __future__ import annotations

from src.benchmark.utils import build_wheat_feature_sets, existing_columns
from src.model import XGBModelConfig, XGBRunConfig, XGBWalkForwardConfig

from .types import (
    DEFAULT_INITIAL_TRAIN_SIZE,
    DEFAULT_ROLLING_WINDOW_SIZE,
    DEFAULT_STEP,
    DEFAULT_TEST_SIZE,
)


def default_run_configs() -> dict[str, XGBRunConfig]:
    return {
        "xgb_expanding": XGBRunConfig(
            walk_forward=XGBWalkForwardConfig(
                window_type="expanding",
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
            ),
            model=XGBModelConfig(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=1.0,
                random_state=42,
                n_jobs=-1,
                target_transform="log",
                log_transform_rv_features=True,
            ),
        ),
        "xgb_rolling": XGBRunConfig(
            walk_forward=XGBWalkForwardConfig(
                window_type="rolling",
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
                rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE,
            ),
            model=XGBModelConfig(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                min_child_weight=1.0,
                random_state=42,
                n_jobs=-1,
                target_transform="log",
                log_transform_rv_features=True,
            ),
        ),
    }


__all__ = ["build_wheat_feature_sets", "default_run_configs", "existing_columns"]
