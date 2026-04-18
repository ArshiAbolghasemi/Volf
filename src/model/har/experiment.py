from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, cast

import pandas as pd
from tqdm.auto import tqdm

from src.metrics import evaluate_statistical_metrics
from src.model.common.preprocessing import (
    aggregate_predictions,
    build_forecasting_design_matrix,
    build_walk_forward_windows,
    inverse_transform_prediction,
    log_transform_rv_features,
    split_design_matrix_xy,
    standardize_train_test,
    transform_target,
)

from .selection import select_har_features
from .types import (
    HARExperimentResult,
    HARFeatureConfig,
    HARGridConfig,
    HARModelConfig,
    HARRunConfig,
    HARSelectionConfig,
    HARWalkForwardConfig,
)
from .utils import fit_har_ols, predict_har_ols

logger = logging.getLogger(__name__)


def run_har_experiment_from_xy(  # noqa: C901, PLR0912, PLR0915
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
        x, transformed_feature_columns = log_transform_rv_features(
            x,
            floor=model_cfg.feature_floor,
        )
        logger.info(
            "Log-transformed RV feature columns: n=%d",
            len(transformed_feature_columns),
        )

    windows = build_walk_forward_windows(len(x), wf_cfg)
    logger.info("Generated %d windows (%s)", len(windows), wf_cfg.window_type)
    run_start = time.perf_counter()

    train_true_parts: list[pd.Series] = []
    train_pred_parts: list[pd.Series] = []
    test_true_parts: list[pd.Series] = []
    test_pred_parts: list[pd.Series] = []

    selected_union: list[str] = []
    selection_counts: dict[str, int] = {}
    last_coefficients = pd.Series(dtype=float, name="coefficient")
    last_selection_info: dict[str, Any] = {}

    window_rows: list[dict[str, Any]] = []
    cached_selected_features: list[str] | None = None
    cached_selection_info: dict[str, Any] | None = None

    window_iterator = tqdm(
        enumerate(windows),
        total=len(windows),
        desc=f"HAR walk-forward ({wf_cfg.window_type})",
        disable=not wf_cfg.progress_bar,
    )
    for window_id, (train_start, train_end, test_start, test_end) in window_iterator:
        if (not wf_cfg.progress_bar) and (
            window_id % 50 == 0 or window_id == len(windows) - 1
        ):
            elapsed = time.perf_counter() - run_start
            logger.info(
                "Walk-forward progress: window %d/%d (%.1f%%), elapsed=%.1fs",
                window_id + 1,
                len(windows),
                100.0 * (window_id + 1) / len(windows),
                elapsed,
            )
        x_train = x.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        x_test = x.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        refit_every = max(selection_cfg.refit_every_windows, 1)
        should_refit = selection_cfg.method != "none" and (
            cached_selected_features is None or window_id % refit_every == 0
        )
        if should_refit:
            selected_features, selection_info = select_har_features(
                x_train=x_train,
                y_train=y_train,
                core_columns=core_columns,
                config=selection_cfg,
            )
            cached_selected_features = selected_features
            cached_selection_info = selection_info
        else:
            selected_features = cached_selected_features or x_train.columns.tolist()
            selection_info = cached_selection_info or {
                "method": selection_cfg.method,
                "reused_selection": True,
            }

        if wf_cfg.progress_bar:
            postfix: dict[str, Any] = {
                "method": selection_cfg.method,
                "refit": "Y" if should_refit else "N",
                "n_sel": len(selected_features),
            }
            if selection_cfg.method == "lasso":
                best_alpha = selection_info.get("best_alpha")
                if best_alpha is not None:
                    postfix["alpha"] = f"{float(best_alpha):.2e}"
            elif selection_cfg.method == "bsr":
                dropped = selection_info.get("total_dropped_features")
                if dropped is not None:
                    postfix["drop"] = int(dropped)
            window_iterator.set_postfix(postfix, refresh=False)

        for feat in selected_features:
            if feat not in selected_union:
                selected_union.append(feat)
            selection_counts[feat] = selection_counts.get(feat, 0) + 1

        x_train_sel = cast("pd.DataFrame", x_train[selected_features])
        x_test_sel = cast("pd.DataFrame", x_test[selected_features])

        if model_cfg.standardize_features:
            x_train_sel, x_test_sel, _ = standardize_train_test(x_train_sel, x_test_sel)

        y_train_model = transform_target(y_train, model_cfg)
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
        y_pred_train = inverse_transform_prediction(y_pred_train, model_cfg)
        y_pred_test = predict_har_ols(
            fitted_model=fitted,
            x=x_test_sel,
            selected_features=selected_features,
            add_constant=model_cfg.add_constant,
        )
        y_pred_test = inverse_transform_prediction(y_pred_test, model_cfg)

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
        "model_add_constant": model_cfg.add_constant,
        "model_standardize_features": model_cfg.standardize_features,
        "model_target_transform": model_cfg.target_transform,
        "model_prediction_floor": model_cfg.prediction_floor,
        "model_log_transform_rv_features": model_cfg.log_transform_rv_features,
        "model_feature_floor": model_cfg.feature_floor,
        "transformed_feature_columns": transformed_feature_columns,
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
    design, core_columns, target_col = build_forecasting_design_matrix(
        data,
        feature_config,
        target_transform=model_cfg.target_transform,
    )
    x, y = split_design_matrix_xy(design, target_col)

    effective_run_config = run_config
    if feature_config.target_mode == "mean" and model_cfg.target_transform != "none":
        effective_model_cfg = replace(
            model_cfg,
            target_transform="none",
            # Mean+log targets live on log scale; avoid clipping log predictions.
            prediction_floor=-1e12,
        )
        effective_run_config = replace(cfg, model=effective_model_cfg)
    else:
        effective_model_cfg = model_cfg

    result = run_har_experiment_from_xy(
        x=x,
        y=y,
        core_columns=core_columns,
        run_config=effective_run_config,
    )

    mean_log_target = (
        feature_config.target_mode == "mean" and model_cfg.target_transform == "log"
    )
    if mean_log_target:
        result.y_true_train = inverse_transform_prediction(
            result.y_true_train.rename("y_true"),
            model_cfg,
        ).rename("y_true")
        result.y_pred_train = inverse_transform_prediction(
            result.y_pred_train.rename("y_pred"),
            model_cfg,
        ).rename("y_pred")
        result.y_true_test = inverse_transform_prediction(
            result.y_true_test.rename("y_true"),
            model_cfg,
        ).rename("y_true")
        result.y_pred_test = inverse_transform_prediction(
            result.y_pred_test.rename("y_pred"),
            model_cfg,
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
            "selection_method": selection_cfg.method,
            "selection_refit_every_windows": selection_cfg.refit_every_windows,
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
            "model_target_transform_effective": effective_model_cfg.target_transform,
            "mean_log_target_inverse_applied": mean_log_target,
            "model_prediction_floor": model_cfg.prediction_floor,
            "model_log_transform_rv_features": model_cfg.log_transform_rv_features,
            "model_feature_floor": model_cfg.feature_floor,
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
    cfg = run_config or HARRunConfig()
    wf_cfg = cfg.walk_forward or HARWalkForwardConfig()

    feature_iterator = tqdm(
        grid_config.feature_sets.items(),
        total=len(grid_config.feature_sets),
        desc="HAR feature sets",
        disable=not wf_cfg.progress_bar,
    )
    for model_name, extra_cols in feature_iterator:
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
