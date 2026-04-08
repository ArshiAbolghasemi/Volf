from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd  # noqa: TC002


@dataclass
class RFFeatureConfig:
    target_col: str
    core_columns: list[str]
    target_horizon: int = 1
    extra_feature_cols: list[str] | None = None
    target_col_name: str = "RV_target"
    target_mode: Literal["point", "mean"] = "point"
    target_floor: float = 1e-10


@dataclass
class RFWalkForwardConfig:
    window_type: Literal["expanding", "rolling"] = "expanding"
    initial_train_size: int = 104
    test_size: int = 1
    step: int = 1
    rolling_window_size: int | None = None
    progress_bar: bool = True


@dataclass
class RFModelConfig:
    n_estimators: int = 400
    criterion: Literal[
        "squared_error",
        "absolute_error",
        "friedman_mse",
        "poisson",
    ] = "squared_error"
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Literal["sqrt", "log2"] | int | float | None = "sqrt"
    bootstrap: bool = True
    random_state: int | None = 42
    n_jobs: int | None = -1
    standardize_features: bool = False
    target_transform: Literal["none", "log"] = "log"
    prediction_floor: float = 1e-10
    log_transform_rv_features: bool = True
    feature_floor: float = 1e-10


@dataclass
class RFGridConfig:
    feature_sets: dict[str, list[str]]
    base_feature_config: RFFeatureConfig


@dataclass
class RFRunConfig:
    walk_forward: RFWalkForwardConfig | None = None
    model: RFModelConfig | None = None


@dataclass
class RFExperimentResult:
    selected_features: list[str]
    y_true_train: pd.Series
    y_pred_train: pd.Series
    y_true_test: pd.Series
    y_pred_test: pd.Series
    metrics: dict[str, dict[str, Any]]
    feature_importances: pd.Series
    model_info: dict[str, Any]
