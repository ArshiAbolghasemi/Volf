from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.benchmark.checkpoints import load_checkpoint_model_info
from src.benchmark.rf import (
    RFGridSearchConfig,
    WheatRFBenchmarkConfig,
)
from src.benchmark.rf import (
    build_wheat_feature_sets as build_rf_feature_sets,
)
from src.benchmark.rf import (
    default_run_configs as default_rf_run_configs,
)
from src.benchmark.tree_shap import (
    TreeShapConfig,
    TreeShapJobConfig,
    resolve_rf_run_config_for_shap_job,
    resolve_xgb_run_config_for_shap_job,
    run_rf_shap_for_job,
    run_xgb_shap_for_job,
    save_tree_shap_job_outputs,
)
from src.benchmark.utils import (
    DEFAULT_CORE_COLUMNS,
    existing_columns,
    normalize_target_mode,
)
from src.benchmark.xgb import (
    WheatXGBBenchmarkConfig,
    XGBGridSearchConfig,
)
from src.benchmark.xgb import (
    build_wheat_feature_sets as build_xgb_feature_sets,
)
from src.benchmark.xgb import (
    default_run_configs as default_xgb_run_configs,
)
from src.model import (
    RFFeatureConfig,
    RFModelConfig,
    RFRunConfig,
    RFWalkForwardConfig,
    XGBFeatureConfig,
    XGBModelConfig,
    XGBRunConfig,
    XGBWalkForwardConfig,
)
from src.util.path import DATA_DIR

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SHAP values for Random Forest and XGBoost benchmarks",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to SHAP JSON config"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def _build_rf_run_config_from_dict(cfg: dict[str, Any]) -> RFRunConfig:
    walk_forward_cfg = None
    if isinstance(cfg.get("walk_forward"), dict):
        walk_forward_cfg = RFWalkForwardConfig(**cfg["walk_forward"])
    model_cfg = None
    if isinstance(cfg.get("model"), dict):
        model_cfg = RFModelConfig(**cfg["model"])
    return RFRunConfig(walk_forward=walk_forward_cfg, model=model_cfg)


def _build_xgb_run_config_from_dict(cfg: dict[str, Any]) -> XGBRunConfig:
    walk_forward_cfg = None
    if isinstance(cfg.get("walk_forward"), dict):
        walk_forward_cfg = XGBWalkForwardConfig(**cfg["walk_forward"])
    model_cfg = None
    if isinstance(cfg.get("model"), dict):
        model_cfg = XGBModelConfig(**cfg["model"])
    return XGBRunConfig(walk_forward=walk_forward_cfg, model=model_cfg)


def _load_rf_benchmark_config_from_json(path: str) -> WheatRFBenchmarkConfig:
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)
    run_configs = None
    if isinstance(raw.get("run_configs"), dict):
        run_configs = {
            name: _build_rf_run_config_from_dict(cfg)
            for name, cfg in raw["run_configs"].items()
            if isinstance(cfg, dict)
        }
    return WheatRFBenchmarkConfig(
        csv_path=raw.get("csv_path", str(DATA_DIR / "ag" / "v4.csv")),
        target_col=raw.get("target_col", "wheat_weekly_rv"),
        core_columns=raw.get("core_columns"),
        core_columns_by_target=raw.get("core_columns_by_target"),
        climate_columns=raw.get("climate_columns"),
        target_horizon=int(raw.get("target_horizon", 1)),
        target_horizons=(
            [int(v) for v in raw["target_horizons"]]
            if isinstance(raw.get("target_horizons"), list)
            else None
        ),
        target_mode=normalize_target_mode(str(raw.get("target_mode", "point"))),
        run_configs=cast("dict[str, RFRunConfig] | None", run_configs),
        grid_search=(
            RFGridSearchConfig(**raw["grid_search"])
            if isinstance(raw.get("grid_search"), dict)
            else None
        ),
        use_cache=bool(raw.get("use_cache", True)),
        cache_dir=str(raw.get("cache_dir", ".cache/benchmark")),
        cache_overwrite=bool(raw.get("cache_overwrite", False)),
    )


def _load_xgb_benchmark_config_from_json(path: str) -> WheatXGBBenchmarkConfig:
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)
    run_configs = None
    if isinstance(raw.get("run_configs"), dict):
        run_configs = {
            name: _build_xgb_run_config_from_dict(cfg)
            for name, cfg in raw["run_configs"].items()
            if isinstance(cfg, dict)
        }
    return WheatXGBBenchmarkConfig(
        csv_path=raw.get("csv_path", str(DATA_DIR / "ag" / "v4.csv")),
        target_col=raw.get("target_col", "wheat_weekly_rv"),
        core_columns=raw.get("core_columns"),
        core_columns_by_target=raw.get("core_columns_by_target"),
        climate_columns=raw.get("climate_columns"),
        target_horizon=int(raw.get("target_horizon", 1)),
        target_horizons=(
            [int(v) for v in raw["target_horizons"]]
            if isinstance(raw.get("target_horizons"), list)
            else None
        ),
        target_mode=normalize_target_mode(str(raw.get("target_mode", "point"))),
        run_configs=cast("dict[str, XGBRunConfig] | None", run_configs),
        grid_search=(
            XGBGridSearchConfig(**raw["grid_search"])
            if isinstance(raw.get("grid_search"), dict)
            else None
        ),
        use_cache=bool(raw.get("use_cache", True)),
        cache_dir=str(raw.get("cache_dir", ".cache/benchmark")),
        cache_overwrite=bool(raw.get("cache_overwrite", False)),
    )


def _resolve_output_root(raw_value: str | None, *, default_subpath: str) -> Path:
    if raw_value is None or not str(raw_value).strip():
        return DATA_DIR / "benchmark" / default_subpath
    raw_path = Path(str(raw_value))
    return raw_path if raw_path.is_absolute() else (DATA_DIR / "benchmark" / raw_path)


def _load_tree_shap_config(
    path: str,
) -> tuple[
    str, WheatRFBenchmarkConfig | WheatXGBBenchmarkConfig, TreeShapConfig, Path, Path
]:
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)

    benchmark_type = str(raw.get("benchmark_type", "")).strip().lower()
    benchmark_config_path = raw.get("benchmark_config")
    if benchmark_type not in {"rf", "xgb"}:
        msg = "SHAP config requires benchmark_type='rf' or 'xgb'."
        raise ValueError(msg)
    if not benchmark_config_path:
        msg = "SHAP config requires 'benchmark_config' path."
        raise ValueError(msg)

    benchmark_cfg = (
        _load_rf_benchmark_config_from_json(str(benchmark_config_path))
        if benchmark_type == "rf"
        else _load_xgb_benchmark_config_from_json(str(benchmark_config_path))
    )

    jobs_raw = raw.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        msg = "SHAP config must include a non-empty 'jobs' list."
        raise ValueError(msg)

    shap_cfg = TreeShapConfig(
        jobs=[TreeShapJobConfig(**job) for job in jobs_raw],
        output_subdir=str(raw.get("output_subdir", "shap")),
    )
    output_root = _resolve_output_root(
        cast("str | None", raw.get("output_root")),
        default_subpath=f"{benchmark_type}/{benchmark_cfg.target_mode}",
    )
    checkpoint_root = _resolve_output_root(
        cast("str | None", raw.get("checkpoint_root")),
        default_subpath=f"{benchmark_type}/{benchmark_cfg.target_mode}",
    )
    return benchmark_type, benchmark_cfg, shap_cfg, output_root, checkpoint_root


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    benchmark_type, benchmark_cfg, shap_cfg, output_root, checkpoint_root = (
        _load_tree_shap_config(args.config)
    )

    data = pd.read_csv(benchmark_cfg.csv_path)
    if "Date" in data.columns:
        data = data.sort_values("Date").reset_index(drop=True)

    core_columns = benchmark_cfg.core_columns or existing_columns(
        data, DEFAULT_CORE_COLUMNS
    )
    if not core_columns:
        msg = "No valid core columns found in data for SHAP run."
        raise ValueError(msg)

    if benchmark_type == "rf":
        model_run_configs = benchmark_cfg.run_configs or default_rf_run_configs()
        feature_sets = build_rf_feature_sets(
            data,
            core_columns=core_columns,
            core_columns_by_target=benchmark_cfg.core_columns_by_target,
            target_col=benchmark_cfg.target_col,
            climate_columns=benchmark_cfg.climate_columns,
        )
    else:
        model_run_configs = benchmark_cfg.run_configs or default_xgb_run_configs()
        feature_sets = build_xgb_feature_sets(
            data,
            core_columns=core_columns,
            core_columns_by_target=benchmark_cfg.core_columns_by_target,
            target_col=benchmark_cfg.target_col,
            climate_columns=benchmark_cfg.climate_columns,
        )

    required_jobs = {
        (int(job.target_horizon), job.model_type, job.feature_set) for job in shap_cfg.jobs
    }
    resolved_model_info: dict[tuple[int, str, str], dict[str, Any]] = {}
    for horizon, model_type, feature_set in sorted(required_jobs):
        resolved_model_info[(horizon, model_type, feature_set)] = (
            load_checkpoint_model_info(
                checkpoint_root=checkpoint_root,
                target_horizon=horizon,
                target_mode=benchmark_cfg.target_mode,
                model_type=model_type,
                feature_set=feature_set,
            )
        )

    for job in shap_cfg.jobs:
        if job.model_type not in model_run_configs:
            msg = f"model_type '{job.model_type}' not found in benchmark run configs."
            raise ValueError(msg)
        if job.feature_set not in feature_sets:
            msg = f"feature_set '{job.feature_set}' not found in available feature sets."
            raise ValueError(msg)

        horizon = int(job.target_horizon)
        model_info = resolved_model_info[(horizon, job.model_type, job.feature_set)]

        logger.info(
            "Computing tree SHAP for type=%s horizon=%d model=%s feature_set=%s split=%s",
            benchmark_type,
            horizon,
            job.model_type,
            job.feature_set,
            job.split,
        )

        if benchmark_type == "rf":
            resolved_run_cfg = resolve_rf_run_config_for_shap_job(
                base_run_cfg=cast("RFRunConfig", model_run_configs[job.model_type]),
                model_info=model_info,
            )
            feature_cfg = RFFeatureConfig(
                target_col=benchmark_cfg.target_col,
                core_columns=core_columns,
                target_horizon=horizon,
                target_mode=benchmark_cfg.target_mode,
                extra_feature_cols=feature_sets[job.feature_set],
            )
            shap_result = run_rf_shap_for_job(
                data=data,
                feature_cfg=feature_cfg,
                run_cfg=resolved_run_cfg,
                job=job,
            )
        else:
            resolved_run_cfg = resolve_xgb_run_config_for_shap_job(
                base_run_cfg=cast("XGBRunConfig", model_run_configs[job.model_type]),
                model_info=model_info,
            )
            feature_cfg = XGBFeatureConfig(
                target_col=benchmark_cfg.target_col,
                core_columns=core_columns,
                target_horizon=horizon,
                target_mode=benchmark_cfg.target_mode,
                extra_feature_cols=feature_sets[job.feature_set],
            )
            shap_result = run_xgb_shap_for_job(
                data=data,
                feature_cfg=feature_cfg,
                run_cfg=resolved_run_cfg,
                job=job,
            )

        job_root = output_root / f"target_horizon_{horizon}" / shap_cfg.output_subdir
        saved = save_tree_shap_job_outputs(
            result=shap_result,
            job=job,
            output_root=job_root,
        )
        logger.info("Saved SHAP outputs to %s", saved["dir"])


if __name__ == "__main__":
    main()
