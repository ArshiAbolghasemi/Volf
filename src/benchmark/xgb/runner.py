from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV

from src.model import (
    XGBExperimentResult,
    XGBFeatureConfig,
    XGBModelConfig,
    XGBRunConfig,
    XGBWalkForwardConfig,
    run_xgb_experiment_from_dataset,
)

from .cache import cache_key, dataset_signature, load_result_cache, save_result_cache
from .features import build_wheat_feature_sets, default_run_configs, existing_columns
from .types import (
    DEFAULT_CORE_COLUMNS,
    WheatXGBBenchmarkConfig,
    XGBGridSearchConfig,
    resolve_target_horizons,
)

logger = logging.getLogger(__name__)


def _grid_or_default(values: list[Any] | None, default: Any) -> list[Any]:
    if not values:
        return [default]
    return list(values)


def _build_param_candidates(
    base_run_cfg: XGBRunConfig,
    grid_cfg: XGBGridSearchConfig | None,
) -> list[dict[str, int | float]]:
    base_wf = base_run_cfg.walk_forward or XGBWalkForwardConfig()
    base_model = base_run_cfg.model or XGBModelConfig()
    if grid_cfg is None or not grid_cfg.enabled:
        return [
            {
                "initial_train_size": int(base_wf.initial_train_size),
                "test_size": int(base_wf.test_size),
                "step": int(base_wf.step),
                "n_estimators": int(base_model.n_estimators),
                "max_depth": int(base_model.max_depth),
                "learning_rate": float(base_model.learning_rate),
                "min_child_weight": float(base_model.min_child_weight),
            }
        ]

    initial_train_sizes = _grid_or_default(
        grid_cfg.initial_train_sizes,
        base_wf.initial_train_size,
    )
    test_sizes = _grid_or_default(grid_cfg.test_sizes, base_wf.test_size)
    steps = _grid_or_default(grid_cfg.steps, base_wf.step)
    n_estimators = _grid_or_default(grid_cfg.n_estimators, base_model.n_estimators)
    max_depths = _grid_or_default(grid_cfg.max_depths, base_model.max_depth)
    learning_rates = _grid_or_default(grid_cfg.learning_rates, base_model.learning_rate)
    min_child_weights = _grid_or_default(
        grid_cfg.min_child_weights,
        base_model.min_child_weight,
    )

    candidates: list[dict[str, int | float]] = []
    for initial_train_size in initial_train_sizes:
        for test_size in test_sizes:
            for step in steps:
                for n_estimator in n_estimators:
                    for max_depth in max_depths:
                        for learning_rate in learning_rates:
                            for min_child_weight in min_child_weights:
                                candidates.append(  # noqa: PERF401
                                    {
                                        "initial_train_size": int(initial_train_size),
                                        "test_size": int(test_size),
                                        "step": int(step),
                                        "n_estimators": int(n_estimator),
                                        "max_depth": int(max_depth),
                                        "learning_rate": float(learning_rate),
                                        "min_child_weight": float(min_child_weight),
                                    }
                                )
        if (
            grid_cfg.max_candidates is not None
            and len(candidates) >= grid_cfg.max_candidates
        ):
            break

    return candidates


def _run_cfg_from_candidate(
    base_cfg: XGBRunConfig,
    candidate: dict[str, int | float],
) -> XGBRunConfig:
    base_wf = base_cfg.walk_forward or XGBWalkForwardConfig()
    base_model = base_cfg.model or XGBModelConfig()
    initial_train_size = int(candidate["initial_train_size"])
    wf = replace(
        base_wf,
        initial_train_size=initial_train_size,
        test_size=int(candidate["test_size"]),
        step=int(candidate["step"]),
        rolling_window_size=(
            initial_train_size
            if base_wf.window_type == "rolling"
            else base_wf.rolling_window_size
        ),
    )
    model = replace(
        base_model,
        n_estimators=int(candidate["n_estimators"]),
        max_depth=int(candidate["max_depth"]),
        learning_rate=float(candidate["learning_rate"]),
        min_child_weight=float(candidate["min_child_weight"]),
    )
    return XGBRunConfig(walk_forward=wf, model=model)


def _resolve_metric_value(result: XGBExperimentResult, metric_name: str) -> float:
    if "_" not in metric_name:
        msg = f"grid metric must be prefixed with split, e.g. 'test_r2'. got={metric_name}"
        raise ValueError(msg)
    split, metric = metric_name.split("_", 1)
    split_metrics = result.metrics.get(split)
    if not isinstance(split_metrics, dict):
        msg = f"unknown split in metric '{metric_name}'"
        raise TypeError(msg)
    value = split_metrics.get(metric)
    if value is None:
        msg = f"metric '{metric_name}' not found in result metrics"
        raise ValueError(msg)
    return float(value)


def _run_single_with_cache(  # noqa: PLR0913
    *,
    data: pd.DataFrame,
    cache_dir: Path,
    cfg: WheatXGBBenchmarkConfig,
    model_name: str,
    feature_set_name: str,
    feature_cfg: XGBFeatureConfig,
    run_cfg: XGBRunConfig,
    data_signature_value: str,
    target_horizon: int,
) -> XGBExperimentResult:
    key = cache_key(
        model_name=model_name,
        feature_set_name=feature_set_name,
        feature_cfg=feature_cfg,
        run_cfg=run_cfg,
        data_signature_value=data_signature_value,
    )

    if cfg.use_cache and not cfg.cache_overwrite:
        cached = load_result_cache(cache_dir, key)
        if cached is not None:
            logger.info(
                "XGB cache hit for horizon=%d model=%s feature_set=%s at %s",
                target_horizon,
                model_name,
                feature_set_name,
                cache_dir / key,
            )
            return cached

    logger.info(
        "XGB cache miss for horizon=%d model=%s feature_set=%s; running training",
        target_horizon,
        model_name,
        feature_set_name,
    )
    result = run_xgb_experiment_from_dataset(
        data,
        feature_config=feature_cfg,
        run_config=run_cfg,
    )

    if cfg.use_cache:
        save_result_cache(
            cache_dir=cache_dir,
            key=key,
            result=result,
        )
        logger.info(
            "XGB cache saved for horizon=%d model=%s feature_set=%s at %s",
            target_horizon,
            model_name,
            feature_set_name,
            cache_dir / key,
        )
    return result


def _run_single_horizon(  # noqa: PLR0913
    *,
    data: pd.DataFrame,
    cfg: WheatXGBBenchmarkConfig,
    core: list[str],
    horizon: int,
    model_name: str,
    run_cfg: XGBRunConfig,
    feature_set_name: str,
    extra_cols: list[str],
    data_signature_value: str,
) -> XGBExperimentResult:
    cache_dir = Path(cfg.cache_dir) / f"target_horizon_{horizon}"
    feature_cfg = XGBFeatureConfig(
        target_col=cfg.target_col,
        core_columns=core,
        target_horizon=horizon,
        target_mode=cfg.target_mode,
        extra_feature_cols=extra_cols,
    )
    candidates = _build_param_candidates(run_cfg, cfg.grid_search)
    metric_name = cfg.grid_search.metric if cfg.grid_search else "test_r2"
    maximize = bool(cfg.grid_search.maximize_metric) if cfg.grid_search else True

    class _XGBEstimator(BaseEstimator):
        def __init__(  # noqa: PLR0913
            self,
            *,
            initial_train_size: int,
            test_size: int,
            step: int,
            n_estimators: int,
            max_depth: int,
            learning_rate: float,
            min_child_weight: float,
        ) -> None:
            self.initial_train_size = initial_train_size
            self.test_size = test_size
            self.step = step
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.min_child_weight = min_child_weight

        def fit(self, x_fit: pd.DataFrame, _y_fit: Any = None) -> _XGBEstimator:
            candidate_cfg = _run_cfg_from_candidate(
                run_cfg,
                {
                    "initial_train_size": int(self.initial_train_size),
                    "test_size": int(self.test_size),
                    "step": int(self.step),
                    "n_estimators": int(self.n_estimators),
                    "max_depth": int(self.max_depth),
                    "learning_rate": float(self.learning_rate),
                    "min_child_weight": float(self.min_child_weight),
                },
            )
            self.result_ = _run_single_with_cache(
                data=x_fit,
                cache_dir=cache_dir,
                cfg=cfg,
                model_name=model_name,
                feature_set_name=feature_set_name,
                feature_cfg=feature_cfg,
                run_cfg=candidate_cfg,
                data_signature_value=data_signature_value,
                target_horizon=horizon,
            )
            self.metric_value_ = _resolve_metric_value(self.result_, metric_name)
            return self

    def _scorer(estimator: _XGBEstimator, _x: pd.DataFrame, _y: Any) -> float:
        metric_value = float(estimator.metric_value_)
        return metric_value if maximize else -metric_value

    param_grid = [
        {
            "initial_train_size": [int(candidate["initial_train_size"])],
            "test_size": [int(candidate["test_size"])],
            "step": [int(candidate["step"])],
            "n_estimators": [int(candidate["n_estimators"])],
            "max_depth": [int(candidate["max_depth"])],
            "learning_rate": [float(candidate["learning_rate"])],
            "min_child_weight": [float(candidate["min_child_weight"])],
        }
        for candidate in candidates
    ]
    cv_split = [(np.arange(len(data), dtype=int), np.arange(len(data), dtype=int))]
    grid = GridSearchCV(
        estimator=_XGBEstimator(
            initial_train_size=int(candidates[0]["initial_train_size"]),
            test_size=int(candidates[0]["test_size"]),
            step=int(candidates[0]["step"]),
            n_estimators=int(candidates[0]["n_estimators"]),
            max_depth=int(candidates[0]["max_depth"]),
            learning_rate=float(candidates[0]["learning_rate"]),
            min_child_weight=float(candidates[0]["min_child_weight"]),
        ),
        param_grid=param_grid,
        scoring=_scorer,
        cv=cv_split,
        n_jobs=1,
        refit=True,
    )
    grid.fit(data, np.zeros(len(data), dtype=float))

    if not hasattr(grid.best_estimator_, "result_"):
        msg = "grid search failed to produce any candidate result."
        raise RuntimeError(msg)
    best_result = grid.best_estimator_.result_
    best_score = float(grid.best_estimator_.metric_value_)
    best_idx = int(grid.best_index_)

    best_result.model_info["grid_search_best_candidate_idx"] = best_idx
    best_result.model_info["grid_search_n_candidates"] = len(candidates)
    best_result.model_info["grid_search_metric"] = metric_name
    best_result.model_info["grid_search_metric_value"] = best_score

    logger.info(
        ("XGB selected horizon=%d model=%s feature_set=%s candidate=%d/%d %s=%.10f"),
        horizon,
        model_name,
        feature_set_name,
        best_idx + 1,
        len(candidates),
        metric_name,
        float(best_score),
    )
    return best_result


def run_wheat_xgb_benchmark_multi_horizon(
    *,
    config: WheatXGBBenchmarkConfig | None = None,
    data: pd.DataFrame | None = None,
) -> dict[int, dict[str, dict[str, XGBExperimentResult]]]:
    cfg = config or WheatXGBBenchmarkConfig()
    if data is None:
        data = pd.read_csv(cfg.csv_path)

    core = cfg.core_columns or existing_columns(data, DEFAULT_CORE_COLUMNS)
    feature_sets = build_wheat_feature_sets(data, core_columns=core)
    model_run_configs = cfg.run_configs or default_run_configs()
    target_horizons = resolve_target_horizons(cfg)
    data_signature_value = dataset_signature(data)

    results_by_horizon: dict[int, dict[str, dict[str, XGBExperimentResult]]] = {}
    for horizon in target_horizons:
        horizon_results: dict[str, dict[str, XGBExperimentResult]] = {}
        for model_name, run_cfg in model_run_configs.items():
            feature_results: dict[str, XGBExperimentResult] = {}
            for feature_set_name, extra_cols in feature_sets.items():
                feature_results[feature_set_name] = _run_single_horizon(
                    data=data,
                    cfg=cfg,
                    core=core,
                    horizon=horizon,
                    model_name=model_name,
                    run_cfg=run_cfg,
                    feature_set_name=feature_set_name,
                    extra_cols=extra_cols,
                    data_signature_value=data_signature_value,
                )
            horizon_results[model_name] = feature_results
        results_by_horizon[horizon] = horizon_results
    return results_by_horizon


def run_wheat_xgb_benchmark(
    *,
    config: WheatXGBBenchmarkConfig | None = None,
    data: pd.DataFrame | None = None,
) -> dict[str, dict[str, XGBExperimentResult]]:
    cfg = config or WheatXGBBenchmarkConfig()
    horizons = resolve_target_horizons(cfg)
    single_cfg = replace(cfg, target_horizons=[horizons[0]], target_horizon=horizons[0])
    results = run_wheat_xgb_benchmark_multi_horizon(config=single_cfg, data=data)
    return results[horizons[0]]


def benchmark_multi_horizon_results_to_frame(
    results_by_horizon: dict[int, dict[str, dict[str, XGBExperimentResult]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon in sorted(results_by_horizon):
        model_results = results_by_horizon[horizon]
        for model_name, feature_results in model_results.items():
            for feature_set_name, result in feature_results.items():
                model_info = result.model_info
                rows.append(
                    {
                        "target_horizon": horizon,
                        "model_type": model_name,
                        "feature_set": feature_set_name,
                        "window_type": model_info.get("window_type"),
                        "target_mode": model_info.get("target_mode"),
                        "target_col_raw": model_info.get("target_col_raw"),
                        "train_mse": result.metrics["train"]["mse"],
                        "train_mae": result.metrics["train"]["mae"],
                        "train_qlike": result.metrics["train"]["qlike"],
                        "train_r2": result.metrics["train"]["r2"],
                        "train_r2log": result.metrics["train"]["r2log"],
                        "test_mse": result.metrics["test"]["mse"],
                        "test_mae": result.metrics["test"]["mae"],
                        "test_qlike": result.metrics["test"]["qlike"],
                        "test_r2": result.metrics["test"]["r2"],
                        "test_r2log": result.metrics["test"]["r2log"],
                        "n_selected_features": len(result.selected_features),
                        "n_windows": model_info.get("n_windows"),
                        "initial_train_size": model_info.get("initial_train_size"),
                        "window_test_size": model_info.get("test_size"),
                        "window_step": model_info.get("step"),
                        "rolling_window_size": model_info.get("rolling_window_size"),
                        "xgb_n_estimators": model_info.get("xgb_n_estimators"),
                        "xgb_max_depth": model_info.get("xgb_max_depth"),
                        "xgb_learning_rate": model_info.get("xgb_learning_rate"),
                        "xgb_subsample": model_info.get("xgb_subsample"),
                        "xgb_colsample_bytree": model_info.get("xgb_colsample_bytree"),
                        "xgb_min_child_weight": model_info.get("xgb_min_child_weight"),
                        "xgb_reg_alpha": model_info.get("xgb_reg_alpha"),
                        "xgb_reg_lambda": model_info.get("xgb_reg_lambda"),
                        "xgb_objective": model_info.get("xgb_objective"),
                        "xgb_random_state": model_info.get("xgb_random_state"),
                        "xgb_n_jobs": model_info.get("xgb_n_jobs"),
                        "grid_search_metric": model_info.get("grid_search_metric"),
                        "grid_search_metric_value": model_info.get(
                            "grid_search_metric_value"
                        ),
                        "grid_search_n_candidates": model_info.get(
                            "grid_search_n_candidates"
                        ),
                        "grid_search_best_candidate_idx": model_info.get(
                            "grid_search_best_candidate_idx"
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["target_horizon", "test_r2"],
        ascending=[True, False],
    )


def benchmark_results_to_frame(
    results: dict[str, dict[str, XGBExperimentResult]],
) -> pd.DataFrame:
    return benchmark_multi_horizon_results_to_frame({0: results}).drop(
        columns=["target_horizon"]
    )
