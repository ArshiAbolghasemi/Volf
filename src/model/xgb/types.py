from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd  # noqa: TC002


@dataclass
class XGBFeatureConfig:
    target_col: str
    core_columns: list[str]
    target_horizon: int = 1
    extra_feature_cols: list[str] | None = None
    target_col_name: str = "RV_target"
    target_mode: Literal["point", "mean"] = "point"
    target_floor: float = 1e-10


@dataclass
class XGBWalkForwardConfig:
    window_type: Literal["expanding", "rolling"] = "expanding"
    initial_train_size: int = 104
    test_size: int = 1
    step: int = 1
    rolling_window_size: int | None = None
    progress_bar: bool = True


@dataclass
class XGBModelConfig:
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    objective: str = "reg:squarederror"
    random_state: int | None = 42
    n_jobs: int | None = -1
    standardize_features: bool = False
    target_transform: Literal["none", "log"] = "log"
    prediction_floor: float = 1e-10
    log_transform_rv_features: bool = True
    feature_floor: float = 1e-10


@dataclass
class XGBGridConfig:
    feature_sets: dict[str, list[str]]
    base_feature_config: XGBFeatureConfig


@dataclass
class XGBRunConfig:
    walk_forward: XGBWalkForwardConfig | None = None
    model: XGBModelConfig | None = None


@dataclass
class XGBExperimentResult:
    selected_features: list[str]
    y_true_train: pd.Series
    y_pred_train: pd.Series
    y_true_test: pd.Series
    y_pred_test: pd.Series
    metrics: dict[str, dict[str, Any]]
    feature_importances: pd.Series
    model_info: dict[str, Any]
