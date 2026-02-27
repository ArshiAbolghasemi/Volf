from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from .types import (  # noqa: TC001
    HARFeatureConfig,
    HARModelConfig,
    HARWalkForwardConfig,
)

if TYPE_CHECKING:
    from statsmodels.regression.linear_model import RegressionResultsWrapper

MIN_OBS = 2


def _validate_feature_config(cfg: HARFeatureConfig, data: pd.DataFrame) -> None:
    if cfg.target_col not in data.columns:
        msg = f"target_col '{cfg.target_col}' is not in dataframe columns."
        raise ValueError(msg)

    if not cfg.core_columns:
        msg = "core_columns must not be empty."
        raise ValueError(msg)

    missing_core = [col for col in cfg.core_columns if col not in data.columns]
    if missing_core:
        msg = f"core_columns missing from dataframe: {missing_core}"
        raise ValueError(msg)

    if cfg.extra_feature_cols:
        missing_extra = [col for col in cfg.extra_feature_cols if col not in data.columns]
        if missing_extra:
            msg = f"extra_feature_cols missing from dataframe: {missing_extra}"
            raise ValueError(msg)

    if cfg.target_horizon < 0:
        msg = "target_horizon must be >= 0."
        raise ValueError(msg)


def build_har_design_matrix(
    data: pd.DataFrame,
    config: HARFeatureConfig,
) -> tuple[pd.DataFrame, list[str], str]:
    _validate_feature_config(config, data)

    feature_cols = config.core_columns.copy()
    if config.extra_feature_cols:
        feature_cols.extend(config.extra_feature_cols)

    design = cast("pd.DataFrame", data[feature_cols].copy())

    target_col = config.target_col_name
    if config.target_horizon == 0:
        design[target_col] = data[config.target_col]
    else:
        design[target_col] = data[config.target_col].shift(-config.target_horizon)

    return design, config.core_columns.copy(), target_col


def _is_rv_feature(col_name: str) -> bool:
    name = col_name.lower()
    return name == "rv" or name.endswith("_rv")


def log_transform_rv_features(
    x: pd.DataFrame,
    *,
    floor: float,
) -> tuple[pd.DataFrame, list[str]]:
    x_out = x.copy()
    transformed_cols: list[str] = []

    for col in x_out.columns:
        if _is_rv_feature(col):
            arr = np.clip(x_out[col].to_numpy(dtype=float), floor, None)
            x_out[col] = np.log(arr)
            transformed_cols.append(col)

    return x_out, transformed_cols


def get_xy_from_har_design(
    design: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in design.columns:
        msg = f"target_col '{target_col}' not found in design dataframe"
        raise ValueError(msg)

    x = design.drop(columns=[target_col])
    y = design[target_col]

    clean = pd.concat([x, y.to_frame(name=target_col)], axis=1).dropna()
    x_clean = clean.drop(columns=[target_col])
    y_clean = cast("pd.Series", clean[target_col])
    return x_clean, y_clean


def build_walk_forward_windows(
    n_obs: int,
    cfg: HARWalkForwardConfig,
) -> list[tuple[int, int, int, int]]:
    if cfg.initial_train_size < MIN_OBS:
        msg = f"initial_train_size must be >= {MIN_OBS}."
        raise ValueError(msg)
    if cfg.test_size < 1:
        msg = "test_size must be >= 1."
        raise ValueError(msg)
    if cfg.step < 1:
        msg = "step must be >= 1."
        raise ValueError(msg)
    if cfg.initial_train_size + cfg.test_size > n_obs:
        msg = (
            f"Not enough observations ({n_obs}) for "
            f"initial_train_size={cfg.initial_train_size} "
            f"and test_size={cfg.test_size}."
        )
        raise ValueError(msg)

    windows: list[tuple[int, int, int, int]] = []
    test_start = cfg.initial_train_size

    while test_start + cfg.test_size <= n_obs:
        if cfg.window_type == "expanding":
            train_start = 0
        else:
            rolling_size = cfg.rolling_window_size or cfg.initial_train_size
            if rolling_size < MIN_OBS:
                msg = f"rolling_window_size must be >= {MIN_OBS}."
                raise ValueError(msg)
            train_start = max(0, test_start - rolling_size)

        train_end = test_start
        test_end = test_start + cfg.test_size

        if train_end - train_start >= MIN_OBS:
            windows.append((train_start, train_end, test_start, test_end))

        test_start += cfg.step

    if not windows:
        msg = "No walk-forward windows were generated."
        raise ValueError(msg)

    return windows


def standardize_train_test(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(x_train)

    train_scaled = pd.DataFrame(
        scaler.transform(x_train),
        index=x_train.index,
        columns=x_train.columns,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        index=x_test.index,
        columns=x_test.columns,
    )
    return train_scaled, test_scaled, scaler


def fit_har_ols(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: list[str],
    *,
    add_constant: bool,
) -> RegressionResultsWrapper:
    x_fit = x_train[selected_features]
    if add_constant:
        x_fit = sm.add_constant(x_fit, has_constant="add")

    model = sm.OLS(y_train, x_fit)
    return model.fit()


def predict_har_ols(
    fitted_model: RegressionResultsWrapper,
    x: pd.DataFrame,
    selected_features: list[str],
    *,
    add_constant: bool,
) -> pd.Series:
    x_pred = x[selected_features]
    if add_constant:
        x_pred = sm.add_constant(x_pred, has_constant="add")

    pred = fitted_model.predict(x_pred)
    return pd.Series(pred, index=x.index, name="y_pred")


def transform_target(y: pd.Series, model_cfg: HARModelConfig) -> pd.Series:
    if model_cfg.target_transform == "none":
        return y
    clipped = np.clip(y.to_numpy(dtype=float), model_cfg.prediction_floor, None)
    return pd.Series(np.log(clipped), index=y.index, name=y.name)


def inverse_transform_prediction(
    y_pred: pd.Series,
    model_cfg: HARModelConfig,
) -> pd.Series:
    if model_cfg.target_transform == "none":
        pred_np = np.clip(y_pred.to_numpy(dtype=float), model_cfg.prediction_floor, None)
        return pd.Series(pred_np, index=y_pred.index, name=y_pred.name)

    pred_np = np.exp(y_pred.to_numpy(dtype=float))
    pred_np = np.clip(pred_np, model_cfg.prediction_floor, None)
    return pd.Series(pred_np, index=y_pred.index, name=y_pred.name)


def aggregate_predictions(
    y_true_parts: list[pd.Series],
    y_pred_parts: list[pd.Series],
) -> tuple[pd.Series, pd.Series]:
    y_true_all = cast(
        "pd.Series", pd.concat(y_true_parts).groupby(level=0).mean()
    ).sort_index()
    y_pred_all = cast(
        "pd.Series", pd.concat(y_pred_parts).groupby(level=0).mean()
    ).sort_index()

    aligned = (
        y_true_all.to_frame("y_true")
        .join(y_pred_all.to_frame("y_pred"), how="inner")
        .dropna()
    )

    return cast("pd.Series", aligned["y_true"]), cast("pd.Series", aligned["y_pred"])
