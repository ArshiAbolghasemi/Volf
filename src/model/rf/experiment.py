from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, cast

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm

from src.metrics import evaluate_statistical_metrics
from src.model.har.utils import (
    aggregate_predictions,
    build_har_design_matrix,
    build_walk_forward_windows,
    get_xy_from_har_design,
    inverse_transform_prediction,
    log_transform_rv_features,
    standardize_train_test,
    transform_target,
)

from .types import (
    RFExperimentResult,
    RFFeatureConfig,
    RFGridConfig,
    RFModelConfig,
    RFRunConfig,
    RFWalkForwardConfig,
)

logger = logging.getLogger(__name__)


def _fit_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_cfg: RFModelConfig,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=model_cfg.n_estimators,
        criterion=model_cfg.criterion,
        max_depth=model_cfg.max_depth,
        min_samples_split=model_cfg.min_samples_split,
        min_samples_leaf=model_cfg.min_samples_leaf,
        max_features=model_cfg.max_features,  # type: ignore[arg-type]
        bootstrap=model_cfg.bootstrap,
        random_state=model_cfg.random_state,
        n_jobs=model_cfg.n_jobs,
    )
    model.fit(x_train, y_train)
    return model


def run_rf_experiment_from_xy(
    x: pd.DataFrame,
    y: pd.Series,
    run_config: RFRunConfig | None = None,
) -> RFExperimentResult:
    logger.info("Starting Random Forest walk-forward experiment")
    cfg = run_config or RFRunConfig()
    wf_cfg = cfg.walk_forward or RFWalkForwardConfig()
    model_cfg = cfg.model or RFModelConfig()

    transformed_feature_columns: list[str] = []
    if model_cfg.log_transform_rv_features:
        x, transformed_feature_columns = log_transform_rv_features(
            x,
            floor=model_cfg.feature_floor,
        )
        logger.info(
            "Log-transformed RV feature columns for RF: n=%d",
            len(transformed_feature_columns),
        )

    windows = build_walk_forward_windows(len(x), cfg=cast("Any", wf_cfg))
    logger.info("Generated %d RF windows (%s)", len(windows), wf_cfg.window_type)
    run_start = time.perf_counter()

    selected_features = x.columns.tolist()
    train_true_parts: list[pd.Series] = []
    train_pred_parts: list[pd.Series] = []
    test_true_parts: list[pd.Series] = []
    test_pred_parts: list[pd.Series] = []
    window_rows: list[dict[str, Any]] = []
    last_importances = pd.Series(dtype=float, name="importance")

    window_iterator = tqdm(
        enumerate(windows),
        total=len(windows),
        desc=f"RF walk-forward ({wf_cfg.window_type})",
        disable=not wf_cfg.progress_bar,
    )
    for window_id, (train_start, train_end, test_start, test_end) in window_iterator:
        if (not wf_cfg.progress_bar) and (
            window_id % 50 == 0 or window_id == len(windows) - 1
        ):
            elapsed = time.perf_counter() - run_start
            logger.info(
                "RF walk-forward progress: window %d/%d (%.1f%%), elapsed=%.1fs",
                window_id + 1,
                len(windows),
                100.0 * (window_id + 1) / len(windows),
                elapsed,
            )

        x_train = x.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        x_test = x.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        if model_cfg.standardize_features:
            x_train, x_test, _ = standardize_train_test(x_train, x_test)

        y_train_model = transform_target(y_train, cast("Any", model_cfg))
        fitted = _fit_random_forest(x_train, y_train_model, model_cfg)

        y_pred_train = pd.Series(
            fitted.predict(x_train), index=x_train.index, name="y_pred"
        )
        y_pred_train = inverse_transform_prediction(y_pred_train, cast("Any", model_cfg))
        y_pred_test = pd.Series(fitted.predict(x_test), index=x_test.index, name="y_pred")
        y_pred_test = inverse_transform_prediction(y_pred_test, cast("Any", model_cfg))

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

        last_importances = pd.Series(
            fitted.feature_importances_,
            index=selected_features,
            name="importance",
            dtype=float,
        ).sort_values(ascending=False)

    y_true_train, y_pred_train = aggregate_predictions(train_true_parts, train_pred_parts)
    y_true_test, y_pred_test = aggregate_predictions(test_true_parts, test_pred_parts)

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
        "rf_n_estimators": model_cfg.n_estimators,
        "rf_criterion": model_cfg.criterion,
        "rf_max_depth": model_cfg.max_depth,
        "rf_min_samples_split": model_cfg.min_samples_split,
        "rf_min_samples_leaf": model_cfg.min_samples_leaf,
        "rf_max_features": model_cfg.max_features,
        "rf_bootstrap": model_cfg.bootstrap,
        "rf_random_state": model_cfg.random_state,
        "rf_n_jobs": model_cfg.n_jobs,
        "model_standardize_features": model_cfg.standardize_features,
        "model_target_transform": model_cfg.target_transform,
        "model_prediction_floor": model_cfg.prediction_floor,
        "model_log_transform_rv_features": model_cfg.log_transform_rv_features,
        "model_feature_floor": model_cfg.feature_floor,
        "transformed_feature_columns": transformed_feature_columns,
        "window_report": pd.DataFrame(window_rows),
    }

    logger.info("RF train metrics: %s", metrics["train"])
    logger.info("RF test metrics: %s", metrics["test"])

    return RFExperimentResult(
        selected_features=selected_features,
        y_true_train=y_true_train,
        y_pred_train=y_pred_train,
        y_true_test=y_true_test,
        y_pred_test=y_pred_test,
        metrics=metrics,
        feature_importances=last_importances,
        model_info=model_info,
    )


def run_rf_experiment_from_dataset(
    data: pd.DataFrame,
    *,
    feature_config: RFFeatureConfig,
    run_config: RFRunConfig | None = None,
) -> RFExperimentResult:
    cfg = run_config or RFRunConfig()
    wf_cfg = cfg.walk_forward or RFWalkForwardConfig()
    model_cfg = cfg.model or RFModelConfig()

    logger.info(
        "Starting RF experiment from raw dataset: window_type=%s target_mode=%s horizon=%d",
        wf_cfg.window_type,
        feature_config.target_mode,
        feature_config.target_horizon,
    )
    design, _, target_col = build_har_design_matrix(
        data,
        cast("Any", feature_config),
        target_transform=model_cfg.target_transform,
    )
    x, y = get_xy_from_har_design(design, target_col)

    effective_run_config = run_config
    if feature_config.target_mode == "mean" and model_cfg.target_transform != "none":
        effective_model_cfg = replace(
            model_cfg,
            target_transform="none",
            prediction_floor=-1e12,
        )
        effective_run_config = replace(cfg, model=effective_model_cfg)
    else:
        effective_model_cfg = model_cfg

    result = run_rf_experiment_from_xy(
        x=x,
        y=y,
        run_config=effective_run_config,
    )

    mean_log_target = (
        feature_config.target_mode == "mean" and model_cfg.target_transform == "log"
    )
    if mean_log_target:
        result.y_true_train = inverse_transform_prediction(
            result.y_true_train.rename("y_true"),
            cast("Any", model_cfg),
        ).rename("y_true")
        result.y_pred_train = inverse_transform_prediction(
            result.y_pred_train.rename("y_pred"),
            cast("Any", model_cfg),
        ).rename("y_pred")
        result.y_true_test = inverse_transform_prediction(
            result.y_true_test.rename("y_true"),
            cast("Any", model_cfg),
        ).rename("y_true")
        result.y_pred_test = inverse_transform_prediction(
            result.y_pred_test.rename("y_pred"),
            cast("Any", model_cfg),
        ).rename("y_pred")
        result.metrics = {
            "train": evaluate_statistical_metrics(result.y_true_train, result.y_pred_train),
            "test": evaluate_statistical_metrics(result.y_true_test, result.y_pred_test),
        }

    result.model_info.update(
        {
            "target_col_raw": feature_config.target_col,
            "target_col_model": target_col,
            "target_horizon": feature_config.target_horizon,
            "target_mode": feature_config.target_mode,
            "target_floor": feature_config.target_floor,
            "core_columns": feature_config.core_columns,
            "extra_feature_cols": feature_config.extra_feature_cols or [],
            "walk_forward_window_type": wf_cfg.window_type,
            "walk_forward_initial_train_size": wf_cfg.initial_train_size,
            "walk_forward_test_size": wf_cfg.test_size,
            "walk_forward_step": wf_cfg.step,
            "walk_forward_rolling_window_size": wf_cfg.rolling_window_size,
            "model_target_transform_effective": effective_model_cfg.target_transform,
            "mean_log_target_inverse_applied": mean_log_target,
        }
    )
    return result


def run_rf_feature_set_grid(
    data: pd.DataFrame,
    grid_config: RFGridConfig,
    *,
    run_config: RFRunConfig | None = None,
) -> dict[str, RFExperimentResult]:
    logger.info(
        "Running RF feature-set grid: %d feature sets", len(grid_config.feature_sets)
    )
    base_cfg = grid_config.base_feature_config
    results: dict[str, RFExperimentResult] = {}
    cfg = run_config or RFRunConfig()
    wf_cfg = cfg.walk_forward or RFWalkForwardConfig()

    feature_iterator = tqdm(
        grid_config.feature_sets.items(),
        total=len(grid_config.feature_sets),
        desc="RF feature sets",
        disable=not wf_cfg.progress_bar,
    )
    for model_name, extra_cols in feature_iterator:
        logger.info(
            "Running RF feature set '%s' with %d additional features",
            model_name,
            len(extra_cols),
        )
        feature_cfg = replace(base_cfg, extra_feature_cols=extra_cols)
        results[model_name] = run_rf_experiment_from_dataset(
            data,
            feature_config=feature_cfg,
            run_config=run_config,
        )

    return results
