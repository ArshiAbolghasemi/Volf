from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd  # noqa: TC002

from src.variable_selection import (  # noqa: TC001
    BSRSelectionConfig,
    LassoSelectionConfig,
)


@dataclass
class HARFeatureConfig:
    target_col: str
    core_columns: list[str]
    target_horizon: int = 1
    extra_feature_cols: list[str] | None = None
    target_col_name: str = "RV_target"
    target_mode: Literal["point", "mean"] = "point"
    target_floor: float = 1e-10


@dataclass
class HARSelectionConfig:
    method: Literal["lasso", "bsr", "none"] = "lasso"
    lasso: LassoSelectionConfig | None = None
    bsr: BSRSelectionConfig | None = None
    refit_every_windows: int = 1


@dataclass
class HARWalkForwardConfig:
    window_type: Literal["expanding", "rolling"] = "expanding"
    initial_train_size: int = 104
    test_size: int = 1
    step: int = 1
    rolling_window_size: int | None = None
    progress_bar: bool = True


@dataclass
class HARModelConfig:
    add_constant: bool = True
    standardize_features: bool = False
    target_transform: Literal["none", "log"] = "log"
    prediction_floor: float = 1e-10
    log_transform_rv_features: bool = True
    feature_floor: float = 1e-10


@dataclass
class HARGridConfig:
    feature_sets: dict[str, list[str]]
    base_feature_config: HARFeatureConfig


@dataclass
class HARRunConfig:
    walk_forward: HARWalkForwardConfig | None = None
    selection: HARSelectionConfig | None = None
    model: HARModelConfig | None = None


@dataclass
class HARExperimentResult:
    selected_features: list[str]
    y_true_train: pd.Series
    y_pred_train: pd.Series
    y_true_test: pd.Series
    y_pred_test: pd.Series
    metrics: dict[str, dict[str, Any]]
    coefficients: pd.Series
    selection_info: dict[str, Any]
    model_info: dict[str, Any]
