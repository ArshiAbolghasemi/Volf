from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from src.metrics import evaluate_statistical_metrics
from src.variable_selection import (
    BSRSelectionConfig,
    LassoSelectionConfig,
    backward_stepwise_feature_selection,
    lasso_time_series_feature_selection,
)

if TYPE_CHECKING:
    from statsmodels.regression.linear_model import RegressionResultsWrapper

logger = logging.getLogger(__name__)

MIN_OBS = 2


@dataclass
class HARFeatureConfig:
    target_col: str
    core_columns: list[str]
    target_horizon: int = 1
    extra_feature_cols: list[str] | None = None
    target_col_name: str = "RV_target"


@dataclass
class HARSelectionConfig:
    method: Literal["lasso", "bsr", "none"] = "lasso"
    lasso: LassoSelectionConfig | None = None
    bsr: BSRSelectionConfig | None = None


@dataclass
class HARWalkForwardConfig:
    window_type: Literal["expanding", "rolling"] = "expanding"
    initial_train_size: int = 104
    test_size: int = 1
    step: int = 1
    rolling_window_size: int | None = None


@dataclass
class HARModelConfig:
    add_constant: bool = True
    standardize_features: bool = False
    target_transform: Literal["none", "log"] = "log"
    prediction_floor: float = 1e-10
    log_transform_rv_features: bool = True
    feature_floor: float = 1e-10
    max_selected_features: int | None = 25
    min_train_feature_ratio: float = 5.0


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
    logger.info("Building HAR design matrix")
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

    logger.info(
        "HAR design matrix ready: rows=%d, n_features=%d, target=%s",
        len(design),
        len(design.columns) - 1,
        target_col,
    )
    return design, config.core_columns.copy(), target_col


def _is_rv_feature(col_name: str) -> bool:
    name = col_name.lower()
    return name == "rv" or name.endswith("_rv")


def _log_transform_rv_features(
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

    logger.info(
        "Prepared X/y: n_rows=%d, n_features=%d", len(x_clean), len(x_clean.columns)
    )
    return x_clean, y_clean


def _build_walk_forward_windows(
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


def _standardize_train_test(
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


def select_har_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    core_columns: list[str],
    config: HARSelectionConfig,
) -> tuple[list[str], dict[str, Any]]:
    missing_core = [col for col in core_columns if col not in x_train.columns]
    if missing_core:
        msg = f"core columns missing from training design matrix: {missing_core}"
        raise ValueError(msg)

    if config.method == "none":
        selected = x_train.columns.tolist()
        return selected, {"method": "none", "selected_features": selected}

    if config.method == "lasso":
        lasso_cfg = config.lasso or LassoSelectionConfig()
        lasso_cfg = replace(lasso_cfg, core_columns=core_columns)
        lasso_result = lasso_time_series_feature_selection(
            x_train,
            y_train,
            config=lasso_cfg,
        )
        info = {"method": "lasso", **lasso_result.info}
        return lasso_result.selected_features, info

    if config.method == "bsr":
        bsr_cfg = config.bsr or BSRSelectionConfig()
        bsr_cfg = replace(bsr_cfg, core_columns=tuple(core_columns))
        bsr_result = backward_stepwise_feature_selection(x_train, y_train, config=bsr_cfg)
        info = {"method": "bsr", **bsr_result.info}
        return bsr_result.selected_features, info

    msg = f"unsupported feature selection method: {config.method}"
    raise ValueError(msg)


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


def _enforce_feature_budget(
    selected_features: list[str],
    core_columns: list[str],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_cfg: HARModelConfig,
) -> list[str]:
    if not selected_features:
        return selected_features

    core_set = set(core_columns)
    core_kept = [feat for feat in selected_features if feat in core_set]
    non_core = [feat for feat in selected_features if feat not in core_set]

    max_selected = model_cfg.max_selected_features
    if max_selected is None:
        max_by_cap = len(selected_features)
    else:
        max_by_cap = max(max_selected, len(core_kept))

    max_by_ratio = max(
        len(core_kept),
        int(np.floor(len(y_train) / max(model_cfg.min_train_feature_ratio, 1.0))),
    )
    final_max = min(max_by_cap, max_by_ratio)
    final_max = max(final_max, len(core_kept))

    allowed_non_core = max(final_max - len(core_kept), 0)
    if len(non_core) <= allowed_non_core:
        return core_kept + non_core

    corr_scores: list[tuple[str, float]] = []
    y_np = y_train.to_numpy(dtype=float)
    for feat in non_core:
        x_np = x_train[feat].to_numpy(dtype=float)
        if np.std(x_np) == 0.0 or np.std(y_np) == 0.0:
            score = 0.0
        else:
            score = float(abs(np.corrcoef(x_np, y_np)[0, 1]))
            if not np.isfinite(score):
                score = 0.0
        corr_scores.append((feat, score))

    corr_scores.sort(key=lambda item: item[1], reverse=True)
    kept_non_core = [feat for feat, _ in corr_scores[:allowed_non_core]]
    return core_kept + kept_non_core


def _transform_target(y: pd.Series, model_cfg: HARModelConfig) -> pd.Series:
    if model_cfg.target_transform == "none":
        return y
    clipped = np.clip(y.to_numpy(dtype=float), model_cfg.prediction_floor, None)
    return pd.Series(np.log(clipped), index=y.index, name=y.name)


def _inverse_transform_prediction(
    y_pred: pd.Series,
    model_cfg: HARModelConfig,
) -> pd.Series:
    if model_cfg.target_transform == "none":
        pred_np = np.clip(y_pred.to_numpy(dtype=float), model_cfg.prediction_floor, None)
        return pd.Series(pred_np, index=y_pred.index, name=y_pred.name)

    pred_np = np.exp(y_pred.to_numpy(dtype=float))
    pred_np = np.clip(pred_np, model_cfg.prediction_floor, None)
    return pd.Series(pred_np, index=y_pred.index, name=y_pred.name)


def _aggregate_predictions(
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


def run_har_experiment_from_xy(  # noqa: PLR0915
    x: pd.DataFrame,
    y: pd.Series,
    core_columns: list[str],
    run_config: HARRunConfig | None = None,
) -> HARExperimentResult:
    logger.info("Starting HAR walk-forward experiment")
    cfg = run_config or HARRunConfig()
    wf_cfg = cfg.walk_forward or HARWalkForwardConfig()
    selection_cfg = cfg.selection or HARSelectionConfig()
    model_cfg = cfg.model or HARModelConfig()

    transformed_feature_columns: list[str] = []
    if model_cfg.log_transform_rv_features:
        x, transformed_feature_columns = _log_transform_rv_features(
            x,
            floor=model_cfg.feature_floor,
        )
        logger.info(
            "Log-transformed RV feature columns: n=%d",
            len(transformed_feature_columns),
        )

    windows = _build_walk_forward_windows(len(x), wf_cfg)
    logger.info("Generated %d windows (%s)", len(windows), wf_cfg.window_type)

    train_true_parts: list[pd.Series] = []
    train_pred_parts: list[pd.Series] = []
    test_true_parts: list[pd.Series] = []
    test_pred_parts: list[pd.Series] = []

    selected_union: list[str] = []
    selection_counts: dict[str, int] = {}
    last_coefficients = pd.Series(dtype=float, name="coefficient")
    last_selection_info: dict[str, Any] = {}

    window_rows: list[dict[str, Any]] = []

    for window_id, (train_start, train_end, test_start, test_end) in enumerate(windows):
        x_train = x.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        x_test = x.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        selected_features, selection_info = select_har_features(
            x_train=x_train,
            y_train=y_train,
            core_columns=core_columns,
            config=selection_cfg,
        )
        selected_features = _enforce_feature_budget(
            selected_features=selected_features,
            core_columns=core_columns,
            x_train=x_train,
            y_train=y_train,
            model_cfg=model_cfg,
        )

        for feat in selected_features:
            if feat not in selected_union:
                selected_union.append(feat)
            selection_counts[feat] = selection_counts.get(feat, 0) + 1

        x_train_sel = cast("pd.DataFrame", x_train[selected_features])
        x_test_sel = cast("pd.DataFrame", x_test[selected_features])

        if model_cfg.standardize_features:
            x_train_sel, x_test_sel, _ = _standardize_train_test(x_train_sel, x_test_sel)

        y_train_model = _transform_target(y_train, model_cfg)
        fitted = fit_har_ols(
            x_train=x_train_sel,
            y_train=y_train_model,
            selected_features=selected_features,
            add_constant=model_cfg.add_constant,
        )

        y_pred_train = predict_har_ols(
            fitted_model=fitted,
            x=x_train_sel,
            selected_features=selected_features,
            add_constant=model_cfg.add_constant,
        )
        y_pred_train = _inverse_transform_prediction(y_pred_train, model_cfg)
        y_pred_test = predict_har_ols(
            fitted_model=fitted,
            x=x_test_sel,
            selected_features=selected_features,
            add_constant=model_cfg.add_constant,
        )
        y_pred_test = _inverse_transform_prediction(y_pred_test, model_cfg)

        train_true_parts.append(y_train)
        train_pred_parts.append(y_pred_train)
        test_true_parts.append(y_test)
        test_pred_parts.append(y_pred_test)

        train_metrics_w = evaluate_statistical_metrics(y_train, y_pred_train)
        test_metrics_w = evaluate_statistical_metrics(y_test, y_pred_test)

        window_rows.append(
            {
                "window_id": window_id,
                "window_type": wf_cfg.window_type,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_selected": len(selected_features),
                "train_mse": train_metrics_w["mse"],
                "test_mse": test_metrics_w["mse"],
            }
        )

        last_coefficients = fitted.params.drop(labels=["const"], errors="ignore")
        last_coefficients.name = "coefficient"
        last_selection_info = selection_info

    y_true_train, y_pred_train = _aggregate_predictions(train_true_parts, train_pred_parts)
    y_true_test, y_pred_test = _aggregate_predictions(test_true_parts, test_pred_parts)

    metrics = {
        "train": evaluate_statistical_metrics(y_true_train, y_pred_train),
        "test": evaluate_statistical_metrics(y_true_test, y_pred_test),
    }

    model_info = {
        "window_type": wf_cfg.window_type,
        "initial_train_size": wf_cfg.initial_train_size,
        "test_size": wf_cfg.test_size,
        "step": wf_cfg.step,
        "rolling_window_size": wf_cfg.rolling_window_size,
        "n_windows": len(windows),
        "model_add_constant": model_cfg.add_constant,
        "model_standardize_features": model_cfg.standardize_features,
        "model_target_transform": model_cfg.target_transform,
        "model_prediction_floor": model_cfg.prediction_floor,
        "model_log_transform_rv_features": model_cfg.log_transform_rv_features,
        "model_feature_floor": model_cfg.feature_floor,
        "transformed_feature_columns": transformed_feature_columns,
        "model_max_selected_features": model_cfg.max_selected_features,
        "model_min_train_feature_ratio": model_cfg.min_train_feature_ratio,
        "window_report": pd.DataFrame(window_rows),
    }

    selection_frequency = {
        feat: count / len(windows) for feat, count in selection_counts.items()
    }
    selection_details = {
        **last_selection_info,
        "selection_frequency": selection_frequency,
        "final_selected_union": selected_union,
        "n_windows": len(windows),
    }

    logger.info("Train metrics: %s", metrics["train"])
    logger.info("Test metrics: %s", metrics["test"])

    return HARExperimentResult(
        selected_features=selected_union,
        y_true_train=y_true_train,
        y_pred_train=y_pred_train,
        y_true_test=y_true_test,
        y_pred_test=y_pred_test,
        metrics=metrics,
        coefficients=last_coefficients,
        selection_info=selection_details,
        model_info=model_info,
    )


def run_har_experiment_from_dataset(
    data: pd.DataFrame,
    *,
    feature_config: HARFeatureConfig,
    run_config: HARRunConfig | None = None,
) -> HARExperimentResult:
    cfg = run_config or HARRunConfig()
    wf_cfg = cfg.walk_forward or HARWalkForwardConfig()
    selection_cfg = cfg.selection or HARSelectionConfig()
    model_cfg = cfg.model or HARModelConfig()

    logger.info(
        "Starting HAR experiment from raw dataset with selection method %s",
        selection_cfg.method,
    )
    design, core_columns, target_col = build_har_design_matrix(data, feature_config)
    x, y = get_xy_from_har_design(design, target_col)

    result = run_har_experiment_from_xy(
        x=x,
        y=y,
        core_columns=core_columns,
        run_config=run_config,
    )

    result.model_info.update(
        {
            "target_col_raw": feature_config.target_col,
            "target_col_model": target_col,
            "target_horizon": feature_config.target_horizon,
            "core_columns": feature_config.core_columns,
            "extra_feature_cols": feature_config.extra_feature_cols or [],
            "selection_method": selection_cfg.method,
            "lasso_config": (
                vars(selection_cfg.lasso) if selection_cfg.lasso is not None else None
            ),
            "bsr_config": (
                vars(selection_cfg.bsr) if selection_cfg.bsr is not None else None
            ),
            "walk_forward_window_type": wf_cfg.window_type,
            "walk_forward_initial_train_size": wf_cfg.initial_train_size,
            "walk_forward_test_size": wf_cfg.test_size,
            "walk_forward_step": wf_cfg.step,
            "walk_forward_rolling_window_size": wf_cfg.rolling_window_size,
            "model_add_constant": model_cfg.add_constant,
            "model_standardize_features": model_cfg.standardize_features,
            "model_target_transform": model_cfg.target_transform,
            "model_prediction_floor": model_cfg.prediction_floor,
            "model_log_transform_rv_features": model_cfg.log_transform_rv_features,
            "model_feature_floor": model_cfg.feature_floor,
            "model_max_selected_features": model_cfg.max_selected_features,
            "model_min_train_feature_ratio": model_cfg.min_train_feature_ratio,
        }
    )
    return result


def run_har_feature_set_grid(
    data: pd.DataFrame,
    grid_config: HARGridConfig,
    *,
    run_config: HARRunConfig | None = None,
) -> dict[str, HARExperimentResult]:
    logger.info(
        "Running HAR feature-set grid: %d feature sets", len(grid_config.feature_sets)
    )
    base_cfg = grid_config.base_feature_config
    results: dict[str, HARExperimentResult] = {}

    for model_name, extra_cols in grid_config.feature_sets.items():
        logger.info(
            "Running feature set '%s' with %d additional features",
            model_name,
            len(extra_cols),
        )
        feature_cfg = replace(base_cfg, extra_feature_cols=extra_cols)
        results[model_name] = run_har_experiment_from_dataset(
            data,
            feature_config=feature_cfg,
            run_config=run_config,
        )

    return results
