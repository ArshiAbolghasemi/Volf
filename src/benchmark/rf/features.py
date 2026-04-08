from __future__ import annotations

from src.benchmark.utils import build_wheat_feature_sets, existing_columns
from src.model import RFModelConfig, RFRunConfig, RFWalkForwardConfig

from .types import (
    DEFAULT_INITIAL_TRAIN_SIZE,
    DEFAULT_ROLLING_WINDOW_SIZE,
    DEFAULT_STEP,
    DEFAULT_TEST_SIZE,
)


def default_run_configs() -> dict[str, RFRunConfig]:
    return {
        "rf_expanding": RFRunConfig(
            walk_forward=RFWalkForwardConfig(
                window_type="expanding",
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
            ),
            model=RFModelConfig(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
                target_transform="log",
                log_transform_rv_features=True,
            ),
        ),
        "rf_rolling": RFRunConfig(
            walk_forward=RFWalkForwardConfig(
                window_type="rolling",
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
                rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE,
            ),
            model=RFModelConfig(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
                target_transform="log",
                log_transform_rv_features=True,
            ),
        ),
    }


__all__ = ["build_wheat_feature_sets", "default_run_configs", "existing_columns"]
