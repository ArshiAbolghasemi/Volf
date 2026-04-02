from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.benchmark import (
    HARGridSearchConfig,
    ShapConfig,
    ShapJobConfig,
    WheatHARBenchmarkConfig,
    build_wheat_feature_sets,
    default_run_configs,
    resolve_run_config_for_shap_job,
    run_linear_shap_for_job,
    run_wheat_har_benchmark_multi_horizon,
    save_shap_job_outputs,
)
from src.benchmark.har.features import existing_columns
from src.benchmark.har.types import DEFAULT_CORE_COLUMNS
from src.model import (
    HARFeatureConfig,
    HARModelConfig,
    HARRunConfig,
    HARSelectionConfig,
    HARWalkForwardConfig,
)
from src.util.path import DATA_DIR
from src.variable_selection import BSRSelectionConfig, LassoSelectionConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SHAP values for selected benchmark models/horizons",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to SHAP JSON config",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def _build_run_config_from_dict(cfg: dict[str, Any]) -> HARRunConfig:
    walk_forward_cfg = None
    if isinstance(cfg.get("walk_forward"), dict):
        walk_forward_cfg = HARWalkForwardConfig(**cfg["walk_forward"])

    selection_cfg = None
    if isinstance(cfg.get("selection"), dict):
        selection_raw = cfg["selection"]
        lasso_cfg = None
        bsr_cfg = None
        if isinstance(selection_raw.get("lasso"), dict):
            lasso_cfg = LassoSelectionConfig(**selection_raw["lasso"])
        if isinstance(selection_raw.get("bsr"), dict):
            bsr_cfg = BSRSelectionConfig(**selection_raw["bsr"])

        selection_cfg = HARSelectionConfig(
            method=selection_raw.get("method", "none"),
            lasso=lasso_cfg,
            bsr=bsr_cfg,
            refit_every_windows=int(selection_raw.get("refit_every_windows", 1)),
        )

    model_cfg = None
    if isinstance(cfg.get("model"), dict):
        model_cfg = HARModelConfig(**cfg["model"])

    return HARRunConfig(
        walk_forward=walk_forward_cfg,
        selection=selection_cfg,
        model=model_cfg,
    )


def _load_benchmark_config_from_json(path: str) -> WheatHARBenchmarkConfig:
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)

    run_configs = None
    if isinstance(raw.get("run_configs"), dict):
        run_configs = {
            name: _build_run_config_from_dict(cfg)
            for name, cfg in raw["run_configs"].items()
            if isinstance(cfg, dict)
        }

    return WheatHARBenchmarkConfig(
        csv_path=raw.get("csv_path", str(DATA_DIR / "ag" / "v4.csv")),
        target_col=raw.get("target_col", "wheat_weekly_rv"),
        core_columns=raw.get("core_columns"),
        target_horizon=int(raw.get("target_horizon", 1)),
        target_horizons=(
            [int(v) for v in raw["target_horizons"]]
            if isinstance(raw.get("target_horizons"), list)
            else None
        ),
        run_configs=cast("dict[str, HARRunConfig] | None", run_configs),
        grid_search=(
            HARGridSearchConfig(**raw["grid_search"])
            if isinstance(raw.get("grid_search"), dict)
            else None
        ),
        parallel_jobs=int(raw.get("parallel_jobs", 1)),
        use_cache=bool(raw.get("use_cache", True)),
        cache_dir=str(raw.get("cache_dir", ".cache/benchmark")),
        cache_overwrite=bool(raw.get("cache_overwrite", False)),
    )


def _load_shap_config(path: str) -> tuple[WheatHARBenchmarkConfig, ShapConfig]:
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)

    benchmark_config_path = raw.get("benchmark_config")
    if not benchmark_config_path:
        msg = "SHAP config requires 'benchmark_config' path (e.g., config/har.json)."
        raise ValueError(msg)

    benchmark_cfg = _load_benchmark_config_from_json(str(benchmark_config_path))

    jobs_raw = raw.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        msg = "SHAP config must include a non-empty 'jobs' list."
        raise ValueError(msg)

    shap_cfg = ShapConfig(
        jobs=[ShapJobConfig(**job) for job in jobs_raw],
        output_subdir=str(raw.get("output_subdir", "shap")),
    )
    return benchmark_cfg, shap_cfg


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    benchmark_cfg, shap_cfg = _load_shap_config(args.config)

    data = pd.read_csv(benchmark_cfg.csv_path)
    if "Date" in data.columns:
        data = data.sort_values("Date").reset_index(drop=True)

    model_run_configs = benchmark_cfg.run_configs or default_run_configs()
    core_columns = benchmark_cfg.core_columns or existing_columns(
        data,
        DEFAULT_CORE_COLUMNS,
    )
    if not core_columns:
        msg = "No valid core columns found in data for SHAP run."
        raise ValueError(msg)

    feature_sets = build_wheat_feature_sets(data, core_columns=core_columns)

    horizons = sorted({int(job.target_horizon) for job in shap_cfg.jobs})
    benchmark_cfg.target_horizons = horizons
    benchmark_cfg.target_horizon = horizons[0]

    logger.info("Running benchmark once to resolve best configs and predictions")
    results_by_horizon = run_wheat_har_benchmark_multi_horizon(
        config=benchmark_cfg,
        data=data,
    )

    output_root = DATA_DIR / "benchmark"
    for job in shap_cfg.jobs:
        if job.model_type not in model_run_configs:
            msg = f"model_type '{job.model_type}' not found in benchmark run configs."
            raise ValueError(msg)
        if job.feature_set not in feature_sets:
            msg = f"feature_set '{job.feature_set}' not found in available feature sets."
            raise ValueError(msg)

        horizon = int(job.target_horizon)
        horizon_results = results_by_horizon.get(horizon)
        if horizon_results is None:
            msg = f"No benchmark results available for horizon={horizon}."
            raise ValueError(msg)

        model_results = horizon_results.get(job.model_type)
        if model_results is None or job.feature_set not in model_results:
            msg = (
                f"Missing benchmark result for horizon={horizon}, "
                f"model={job.model_type}, feature_set={job.feature_set}."
            )
            raise ValueError(msg)

        result = model_results[job.feature_set]
        run_cfg = model_run_configs[job.model_type]

        resolved_run_cfg = resolve_run_config_for_shap_job(
            base_run_cfg=run_cfg,
            model_info=result.model_info,
        )

        feature_cfg = HARFeatureConfig(
            target_col=benchmark_cfg.target_col,
            core_columns=core_columns,
            target_horizon=horizon,
            extra_feature_cols=feature_sets[job.feature_set],
        )

        logger.info(
            "Computing SHAP for horizon=%d model=%s feature_set=%s split=%s",
            horizon,
            job.model_type,
            job.feature_set,
            job.split,
        )
        shap_result = run_linear_shap_for_job(
            data=data,
            feature_cfg=feature_cfg,
            core_columns=core_columns,
            run_cfg=resolved_run_cfg,
            job=job,
        )

        job_root = output_root / f"target_horizon_{horizon}" / shap_cfg.output_subdir
        saved = save_shap_job_outputs(
            result=shap_result,
            job=job,
            output_root=job_root,
        )
        logger.info("Saved SHAP outputs to %s", saved["dir"])


if __name__ == "__main__":
    main()
