from __future__ import annotations

import itertools
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.model import (
    HARExperimentResult,
    HARFeatureConfig,
    HARModelConfig,
    HARRunConfig,
    HARSelectionConfig,
    HARWalkForwardConfig,
    run_har_experiment_from_dataset,
)

from .cache import cache_key, dataset_signature, load_result_cache, save_result_cache
from .features import build_wheat_feature_sets, default_run_configs, existing_columns
from .types import (
    DEFAULT_CORE_COLUMNS,
    HARGridSearchConfig,
    WheatHARBenchmarkConfig,
    resolve_target_horizons,
)

logger = logging.getLogger(__name__)


def _grid_or_default(values: list[Any] | None, default: Any) -> list[Any]:
    if not values:
        return [default]
    return list(values)


def _build_run_config_candidates(
    base_run_cfg: HARRunConfig,
    grid_cfg: HARGridSearchConfig | None,
) -> list[HARRunConfig]:
    if grid_cfg is None or not grid_cfg.enabled:
        return [base_run_cfg]

    base_wf = base_run_cfg.walk_forward or HARWalkForwardConfig()
    base_sel = base_run_cfg.selection or HARSelectionConfig()
    base_model = base_run_cfg.model or HARModelConfig()

    initial_train_sizes = _grid_or_default(
        grid_cfg.initial_train_sizes,
        base_wf.initial_train_size,
    )
    test_sizes = _grid_or_default(grid_cfg.test_sizes, base_wf.test_size)
    steps = _grid_or_default(grid_cfg.steps, base_wf.step)

    combos = itertools.product(initial_train_sizes, test_sizes, steps)

    candidates: list[HARRunConfig] = []
    for initial_train_size, test_size, step in combos:
        wf = replace(
            base_wf,
            initial_train_size=int(initial_train_size),
            test_size=int(test_size),
            step=int(step),
            rolling_window_size=(
                int(initial_train_size)
                if base_wf.window_type == "rolling"
                else base_wf.rolling_window_size
            ),
        )
        candidates.append(
            HARRunConfig(walk_forward=wf, selection=base_sel, model=base_model)
        )
        if (
            grid_cfg.max_candidates is not None
            and len(candidates) >= grid_cfg.max_candidates
        ):
            break

    return candidates or [base_run_cfg]


def _resolve_metric_value(result: HARExperimentResult, metric_name: str) -> float:
    if "_" not in metric_name:
        msg = f"grid metric must be prefixed with split, e.g. 'test_mse'. got={metric_name}"
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
    cfg: WheatHARBenchmarkConfig,
    model_name: str,
    feature_set_name: str,
    feature_cfg: HARFeatureConfig,
    run_cfg: HARRunConfig,
    data_signature_value: str,
    target_horizon: int,
) -> HARExperimentResult:
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
                "Cache hit for horizon=%d model=%s feature_set=%s at %s",
                target_horizon,
                model_name,
                feature_set_name,
                cache_dir / key,
            )
            return cached

    logger.info(
        "Cache miss for horizon=%d model=%s feature_set=%s; running training",
        target_horizon,
        model_name,
        feature_set_name,
    )
    result = run_har_experiment_from_dataset(
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
            "Saved cache for horizon=%d model=%s feature_set=%s at %s",
            target_horizon,
            model_name,
            feature_set_name,
            cache_dir / key,
        )
    return result


def _run_benchmark_task(  # noqa: PLR0913
    *,
    data: pd.DataFrame,
    cfg: WheatHARBenchmarkConfig,
    core: list[str],
    data_signature_value: str,
    horizon: int,
    model_name: str,
    run_cfg: HARRunConfig,
    feature_set_name: str,
    extra_cols: list[str],
) -> tuple[int, str, str, HARExperimentResult]:
    cache_dir = Path(cfg.cache_dir) / f"target_horizon_{horizon}"
    feature_cfg = HARFeatureConfig(
        target_col=cfg.target_col,
        core_columns=core,
        target_horizon=horizon,
        extra_feature_cols=extra_cols,
    )
    candidates = _build_run_config_candidates(run_cfg, cfg.grid_search)
    best_result: HARExperimentResult | None = None
    best_score: float | None = None
    best_idx = -1
    metric_name = cfg.grid_search.metric if cfg.grid_search else "test_mse"
    maximize = bool(cfg.grid_search.maximize_metric) if cfg.grid_search else False

    if len(candidates) > 1:
        logger.info(
            (
                "Grid search active for horizon=%d model=%s "
                "feature_set=%s candidates=%d metric=%s"
            ),
            horizon,
            model_name,
            feature_set_name,
            len(candidates),
            metric_name,
        )

    for idx, candidate_cfg in enumerate(candidates):
        candidate_wf = candidate_cfg.walk_forward or HARWalkForwardConfig()
        candidate_model = candidate_cfg.model or HARModelConfig()
        logger.info(
            (
                "Grid candidate %d/%d for horizon=%d model=%s feature_set=%s: "
                "window_type=%s initial_train_size=%d test_size=%d step=%d "
                "rolling_window_size=%s std=%s target_transform=%s"
            ),
            idx + 1,
            len(candidates),
            horizon,
            model_name,
            feature_set_name,
            candidate_wf.window_type,
            candidate_wf.initial_train_size,
            candidate_wf.test_size,
            candidate_wf.step,
            str(candidate_wf.rolling_window_size),
            candidate_model.standardize_features,
            candidate_model.target_transform,
        )
        result = _run_single_with_cache(
            data=data,
            cache_dir=cache_dir,
            cfg=cfg,
            model_name=model_name,
            feature_set_name=feature_set_name,
            feature_cfg=feature_cfg,
            run_cfg=candidate_cfg,
            data_signature_value=data_signature_value,
            target_horizon=horizon,
        )
        score = _resolve_metric_value(result, metric_name)
        logger.info(
            ("Grid candidate %d/%d score for horizon=%d model=%s feature_set=%s: %s=%.10f"),
            idx + 1,
            len(candidates),
            horizon,
            model_name,
            feature_set_name,
            metric_name,
            score,
        )
        is_better = best_score is None or (
            score > best_score if maximize else score < best_score
        )
        if is_better:
            best_score = score
            best_result = result
            best_idx = idx
            logger.info(
                (
                    "Grid best updated for horizon=%d model=%s feature_set=%s: "
                    "candidate=%d %s=%.10f"
                ),
                horizon,
                model_name,
                feature_set_name,
                best_idx + 1,
                metric_name,
                best_score,
            )

    if best_result is None:
        msg = "grid search failed to produce any candidate result."
        raise RuntimeError(msg)

    best_result.model_info["grid_search_best_candidate_idx"] = best_idx
    best_result.model_info["grid_search_n_candidates"] = len(candidates)
    best_result.model_info["grid_search_metric"] = metric_name
    best_result.model_info["grid_search_metric_value"] = best_score
    logger.info(
        ("Grid selected for horizon=%d model=%s feature_set=%s: candidate=%d/%d %s=%.10f"),
        horizon,
        model_name,
        feature_set_name,
        best_idx + 1,
        len(candidates),
        metric_name,
        float(best_score or 0.0),
    )
    return horizon, model_name, feature_set_name, best_result


def run_wheat_har_benchmark_multi_horizon(
    *,
    config: WheatHARBenchmarkConfig | None = None,
    data: pd.DataFrame | None = None,
) -> dict[int, dict[str, dict[str, Any]]]:
    cfg = config or WheatHARBenchmarkConfig()
    logger.info("Starting Wheat HAR benchmark")
    if data is None:
        logger.info("Loading benchmark data from %s", cfg.csv_path)
        data = pd.read_csv(cfg.csv_path)

    if "Date" in data.columns:
        data = data.sort_values("Date").reset_index(drop=True)
        logger.info("Data sorted by Date. rows=%d", len(data))

    core = cfg.core_columns or existing_columns(data, DEFAULT_CORE_COLUMNS)
    if not core:
        msg = "No valid core columns found in data."
        raise ValueError(msg)
    logger.info("Using core columns: %s", core)

    feature_sets = build_wheat_feature_sets(data, core_columns=core)
    data_signature_value = dataset_signature(data)
    model_run_configs = cfg.run_configs or default_run_configs()
    horizons = resolve_target_horizons(cfg)
    results_by_horizon: dict[int, dict[str, dict[str, Any]]] = {
        horizon: {model_name: {} for model_name in model_run_configs}
        for horizon in horizons
    }

    tasks: list[tuple[int, str, HARRunConfig, str, list[str]]] = []
    for horizon in horizons:
        for model_name, run_cfg in model_run_configs.items():
            for feature_set_name, extra_cols in feature_sets.items():
                tasks.append((horizon, model_name, run_cfg, feature_set_name, extra_cols))

    logger.info(
        "Dispatching %d benchmark tasks with parallel_jobs=%d",
        len(tasks),
        cfg.parallel_jobs,
    )

    if cfg.parallel_jobs <= 1:
        for horizon, model_name, run_cfg, feature_set_name, extra_cols in tasks:
            h, m, f, result = _run_benchmark_task(
                data=data,
                cfg=cfg,
                core=core,
                data_signature_value=data_signature_value,
                horizon=horizon,
                model_name=model_name,
                run_cfg=run_cfg,
                feature_set_name=feature_set_name,
                extra_cols=extra_cols,
            )
            results_by_horizon[h][m][f] = result
    else:
        with ThreadPoolExecutor(max_workers=cfg.parallel_jobs) as executor:
            futures = [
                executor.submit(
                    _run_benchmark_task,
                    data=data,
                    cfg=cfg,
                    core=core,
                    data_signature_value=data_signature_value,
                    horizon=horizon,
                    model_name=model_name,
                    run_cfg=run_cfg,
                    feature_set_name=feature_set_name,
                    extra_cols=extra_cols,
                )
                for horizon, model_name, run_cfg, feature_set_name, extra_cols in tasks
            ]
            for future in as_completed(futures):
                h, m, f, result = future.result()
                results_by_horizon[h][m][f] = result

    logger.info("Wheat HAR benchmark finished. horizons=%d", len(results_by_horizon))
    return results_by_horizon


def run_wheat_har_benchmark(
    *,
    config: WheatHARBenchmarkConfig | None = None,
    data: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    cfg = config or WheatHARBenchmarkConfig()
    horizons = resolve_target_horizons(cfg)
    if len(horizons) > 1:
        logger.warning(
            "Multiple target horizons %s provided; run_wheat_har_benchmark uses "
            "the first (%d). Use run_wheat_har_benchmark_multi_horizon for all.",
            horizons,
            horizons[0],
        )

    single_cfg = replace(
        cfg,
        target_horizon=horizons[0],
        target_horizons=[horizons[0]],
    )
    results = run_wheat_har_benchmark_multi_horizon(config=single_cfg, data=data)
    return results[horizons[0]]


def benchmark_results_to_frame(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    logger.info("Converting benchmark results to summary dataframe")
    rows: list[dict[str, Any]] = []

    for model_name, model_results in results.items():
        for feature_set_name, result in model_results.items():
            model_info = result.model_info
            selection_info = result.selection_info
            row = {
                "model_type": model_name,
                "feature_set": feature_set_name,
                "n_selected": len(result.selected_features),
                "selected_features": ",".join(result.selected_features),
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
                "target_col_raw": model_info.get("target_col_raw"),
                "target_col_model": model_info.get("target_col_model"),
                "target_horizon": model_info.get("target_horizon"),
                "core_columns": ",".join(model_info.get("core_columns", [])),
                "extra_feature_cols": ",".join(model_info.get("extra_feature_cols", [])),
                "window_type": model_info.get("walk_forward_window_type"),
                "initial_train_size": model_info.get("walk_forward_initial_train_size"),
                "window_test_size": model_info.get("walk_forward_test_size"),
                "window_step": model_info.get("walk_forward_step"),
                "rolling_window_size": model_info.get("walk_forward_rolling_window_size"),
                "n_windows": model_info.get("n_windows"),
                "selection_method": model_info.get("selection_method"),
                "model_add_constant": model_info.get("model_add_constant"),
                "model_standardize_features": model_info.get("model_standardize_features"),
                "model_target_transform": model_info.get("model_target_transform"),
                "model_prediction_floor": model_info.get("model_prediction_floor"),
                "model_log_transform_rv_features": model_info.get(
                    "model_log_transform_rv_features"
                ),
                "model_feature_floor": model_info.get("model_feature_floor"),
                "lasso_best_alpha": selection_info.get("best_alpha"),
                "bsr_alpha": selection_info.get("alpha"),
                "bsr_window_type": selection_info.get("window_type"),
                "bsr_window_size": selection_info.get("window_size"),
                "bsr_step": selection_info.get("step"),
                "grid_search_best_candidate_idx": model_info.get(
                    "grid_search_best_candidate_idx"
                ),
                "grid_search_n_candidates": model_info.get("grid_search_n_candidates"),
                "grid_search_metric": model_info.get("grid_search_metric"),
                "grid_search_metric_value": model_info.get("grid_search_metric_value"),
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        logger.warning("Benchmark result dataframe is empty")
        return out
    summary = out.sort_values(["model_type", "test_mse", "test_mae"]).reset_index(drop=True)
    logger.info("Benchmark summary rows=%d", len(summary))
    return summary


def benchmark_multi_horizon_results_to_frame(
    results_by_horizon: dict[int, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for horizon, results in sorted(results_by_horizon.items()):
        frame = benchmark_results_to_frame(results)
        if frame.empty:
            continue
        frame = frame.copy()
        if "target_horizon" not in frame.columns:
            frame["target_horizon"] = horizon
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=0, ignore_index=True)
    return out.sort_values(
        ["target_horizon", "model_type", "test_mse", "test_mae"],
    ).reset_index(drop=True)
