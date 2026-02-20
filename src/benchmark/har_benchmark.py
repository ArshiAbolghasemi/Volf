from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

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
from src.util.path import DATA_DIR
from src.variable_selection import BSRSelectionConfig, LassoSelectionConfig

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = DATA_DIR / "ag" / "v4.csv"
DEFAULT_TARGET = "wheat_weekly_rv"
DEFAULT_CORE_COLUMNS = ["wheat_weekly_rv", "wheat_monthly_rv", "wheat_seasonal_rv"]

CLIMATE_COLUMNS = [
    "ssta_elino",
    "ssta_lanina",
    "dry",
    "wet",
    "SOI_index",
    "NAO_index",
    "Text_Climate_Anomaly",
]

NEWS_BASE_COLUMNS = ["frbsf_sentiment", "Text_Climate_Anomaly"]
MACRO_COLUMNS = ["DJIA_Index", "WTI_Index", "Broad_Dollar_index", "Stock_Uncertainty"]


@dataclass
class WheatHARBenchmarkConfig:
    csv_path: str = str(DEFAULT_DATA_PATH)
    target_col: str = DEFAULT_TARGET
    core_columns: list[str] | None = None
    target_horizon: int = 1
    run_configs: dict[str, HARRunConfig] | None = None
    use_cache: bool = True
    cache_dir: str = ".cache/benchmark"
    cache_overwrite: bool = False


def _existing_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in data.columns]


def _build_wheat_endogenous_columns(
    data: pd.DataFrame, core_columns: list[str]
) -> list[str]:
    core_set = set(core_columns)
    return sorted(
        [col for col in data.columns if col.startswith("wheat_") and col not in core_set]
    )


def _build_exogenous_columns(data: pd.DataFrame) -> list[str]:
    return sorted([col for col in data.columns if col.startswith(("corn_", "soybeans_"))])


def _build_news_columns(data: pd.DataFrame) -> list[str]:
    epu_cols = sorted([col for col in data.columns if col.startswith("epu_")])
    base = _existing_columns(data, NEWS_BASE_COLUMNS)
    return base + epu_cols


def build_wheat_feature_sets(
    data: pd.DataFrame,
    *,
    core_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    logger.info("Building Wheat benchmark feature sets")
    core = core_columns or _existing_columns(data, DEFAULT_CORE_COLUMNS)

    endo = _build_wheat_endogenous_columns(data, core)
    exo = _build_exogenous_columns(data)
    climate = _existing_columns(data, CLIMATE_COLUMNS)
    news = _build_news_columns(data)
    macro = _existing_columns(data, MACRO_COLUMNS)

    feature_sets = {
        "har": [],
        "har_endo": endo,
        "har_endo_exo": endo + exo,
        "har_endo_exogenous_climate": endo + exo + climate,
        "har_endo_exogenous_climate_news": endo + exo + climate + news,
        "har__all": endo + exo + climate + news + macro,
    }

    cleaned: dict[str, list[str]] = {}
    for name, cols in feature_sets.items():
        seen: set[str] = set()
        unique_cols: list[str] = []
        for col in cols:
            if col in data.columns and col not in seen:
                seen.add(col)
                unique_cols.append(col)
        cleaned[name] = unique_cols
        logger.info("Feature set '%s' includes %d features", name, len(unique_cols))

    return cleaned


def default_run_configs() -> dict[str, HARRunConfig]:
    return {
        "ols_expanding": HARRunConfig(
            walk_forward=HARWalkForwardConfig(window_type="expanding"),
            selection=HARSelectionConfig(method="none"),
            model=HARModelConfig(standardize_features=False),
        ),
        "ols_rolling": HARRunConfig(
            walk_forward=HARWalkForwardConfig(
                window_type="rolling",
                rolling_window_size=104,
            ),
            selection=HARSelectionConfig(method="none"),
            model=HARModelConfig(standardize_features=False),
        ),
        "lasso_expanding": HARRunConfig(
            walk_forward=HARWalkForwardConfig(window_type="expanding"),
            selection=HARSelectionConfig(
                method="lasso",
                lasso=LassoSelectionConfig(n_splits=5),
                refit_every_windows=4,
            ),
            model=HARModelConfig(standardize_features=True),
        ),
        "lasso_rolling": HARRunConfig(
            walk_forward=HARWalkForwardConfig(
                window_type="rolling",
                rolling_window_size=104,
            ),
            selection=HARSelectionConfig(
                method="lasso",
                lasso=LassoSelectionConfig(n_splits=5),
                refit_every_windows=4,
            ),
            model=HARModelConfig(standardize_features=True),
        ),
        "bsr_expanding": HARRunConfig(
            walk_forward=HARWalkForwardConfig(window_type="expanding"),
            selection=HARSelectionConfig(
                method="bsr",
                bsr=BSRSelectionConfig(alpha=0.05),
                refit_every_windows=8,
            ),
            model=HARModelConfig(standardize_features=False),
        ),
        "bsr_rolling": HARRunConfig(
            walk_forward=HARWalkForwardConfig(
                window_type="rolling",
                rolling_window_size=104,
            ),
            selection=HARSelectionConfig(
                method="bsr",
                bsr=BSRSelectionConfig(alpha=0.05),
                refit_every_windows=8,
            ),
            model=HARModelConfig(standardize_features=False),
        ),
    }


def _dataset_signature(data: pd.DataFrame) -> str:
    payload = data.to_csv(index=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, default=str)


def _cache_key(
    *,
    model_name: str,
    feature_set_name: str,
    feature_cfg: HARFeatureConfig,
    run_cfg: HARRunConfig,
    data_signature: str,
) -> str:
    payload = {
        "cache_version": "v1",
        "model_name": model_name,
        "feature_set_name": feature_set_name,
        "feature_config": asdict(feature_cfg),
        "run_config": asdict(run_cfg),
        "data_signature": data_signature,
    }
    digest = hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()
    return f"{model_name}__{feature_set_name}__{digest[:16]}"


def _cache_paths(cache_dir: Path, cache_key: str) -> dict[str, Path]:
    base = cache_dir / cache_key
    return {
        "base": base,
        "meta": base / "meta.parquet",
        "train_pred": base / "train_pred.parquet",
        "test_pred": base / "test_pred.parquet",
        "coefficients": base / "coefficients.parquet",
        "window_report": base / "window_report.parquet",
    }


def _serialize_model_info(model_info: dict[str, Any]) -> tuple[str, bool]:
    info_copy = dict(model_info)
    has_window_report = "window_report" in info_copy
    info_copy.pop("window_report", None)
    return _json_dumps(info_copy), has_window_report


def _save_result_cache(
    *,
    cache_dir: Path,
    cache_key: str,
    result: HARExperimentResult,
) -> None:
    paths = _cache_paths(cache_dir, cache_key)
    paths["base"].mkdir(parents=True, exist_ok=True)

    model_info_json, has_window_report = _serialize_model_info(result.model_info)
    meta = pd.DataFrame(
        [
            {
                "selected_features_json": _json_dumps(result.selected_features),
                "metrics_json": _json_dumps(result.metrics),
                "selection_info_json": _json_dumps(result.selection_info),
                "model_info_json": model_info_json,
                "has_window_report": has_window_report,
            }
        ]
    )
    meta.to_parquet(paths["meta"], index=False)

    pd.DataFrame(
        {"y_true": result.y_true_train, "y_pred": result.y_pred_train},
        index=result.y_true_train.index,
    ).to_parquet(paths["train_pred"], index=True)
    pd.DataFrame(
        {"y_true": result.y_true_test, "y_pred": result.y_pred_test},
        index=result.y_true_test.index,
    ).to_parquet(paths["test_pred"], index=True)
    result.coefficients.to_frame(name="coefficient").to_parquet(
        paths["coefficients"],
        index=True,
    )

    window_report = result.model_info.get("window_report")
    if isinstance(window_report, pd.DataFrame):
        window_report.to_parquet(paths["window_report"], index=False)


def _load_result_cache(cache_dir: Path, cache_key: str) -> HARExperimentResult | None:
    paths = _cache_paths(cache_dir, cache_key)
    if not paths["meta"].exists():
        return None
    required = ["train_pred", "test_pred", "coefficients"]
    if any(not paths[name].exists() for name in required):
        return None

    meta = pd.read_parquet(paths["meta"])
    if meta.empty:
        return None
    meta_row = meta.iloc[0]

    train_pred = pd.read_parquet(paths["train_pred"])
    test_pred = pd.read_parquet(paths["test_pred"])
    coeff = pd.read_parquet(paths["coefficients"])

    y_true_train = train_pred["y_true"].copy()
    y_true_train.name = "y_true"
    y_pred_train = train_pred["y_pred"].copy()
    y_pred_train.name = "y_pred"
    y_true_test = test_pred["y_true"].copy()
    y_true_test.name = "y_true"
    y_pred_test = test_pred["y_pred"].copy()
    y_pred_test.name = "y_pred"

    coefficients = coeff["coefficient"].copy()
    coefficients.name = "coefficient"

    model_info = json.loads(str(meta_row["model_info_json"]))
    if bool(meta_row.get("has_window_report", False)) and paths["window_report"].exists():
        model_info["window_report"] = pd.read_parquet(paths["window_report"])

    return HARExperimentResult(
        selected_features=list(json.loads(str(meta_row["selected_features_json"]))),
        y_true_train=cast("pd.Series", y_true_train),
        y_pred_train=cast("pd.Series", y_pred_train),
        y_true_test=cast("pd.Series", y_true_test),
        y_pred_test=cast("pd.Series", y_pred_test),
        metrics=dict(json.loads(str(meta_row["metrics_json"]))),
        coefficients=cast("pd.Series", coefficients),
        selection_info=dict(json.loads(str(meta_row["selection_info_json"]))),
        model_info=model_info,
    )


def run_wheat_har_benchmark(
    *,
    config: WheatHARBenchmarkConfig | None = None,
    data: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    cfg = config or WheatHARBenchmarkConfig()
    logger.info("Starting Wheat HAR benchmark")
    if data is None:
        logger.info("Loading benchmark data from %s", cfg.csv_path)
        data = pd.read_csv(cfg.csv_path)

    if "Date" in data.columns:
        data = data.sort_values("Date").reset_index(drop=True)
        logger.info("Data sorted by Date. rows=%d", len(data))

    core = cfg.core_columns or _existing_columns(data, DEFAULT_CORE_COLUMNS)
    if not core:
        msg = "No valid core columns found in data."
        raise ValueError(msg)
    logger.info("Using core columns: %s", core)

    feature_sets = build_wheat_feature_sets(data, core_columns=core)
    data_signature = _dataset_signature(data)
    cache_dir = Path(cfg.cache_dir)

    model_run_configs = cfg.run_configs or default_run_configs()
    all_results: dict[str, dict[str, Any]] = {}

    for model_name, run_cfg in model_run_configs.items():
        logger.info("Running benchmark model type: %s", model_name)
        model_results: dict[str, HARExperimentResult] = {}
        base_feature_cfg = HARFeatureConfig(
            target_col=cfg.target_col,
            core_columns=core,
            target_horizon=cfg.target_horizon,
        )
        for feature_set_name, extra_cols in feature_sets.items():
            feature_cfg = replace(base_feature_cfg, extra_feature_cols=extra_cols)
            cache_key = _cache_key(
                model_name=model_name,
                feature_set_name=feature_set_name,
                feature_cfg=feature_cfg,
                run_cfg=run_cfg,
                data_signature=data_signature,
            )

            if cfg.use_cache and not cfg.cache_overwrite:
                cached = _load_result_cache(cache_dir, cache_key)
                if cached is not None:
                    logger.info(
                        "Cache hit for model=%s feature_set=%s at %s",
                        model_name,
                        feature_set_name,
                        cache_dir / cache_key,
                    )
                    model_results[feature_set_name] = cached
                    continue

            logger.info(
                "Cache miss for model=%s feature_set=%s; running training",
                model_name,
                feature_set_name,
            )
            result = run_har_experiment_from_dataset(
                data,
                feature_config=feature_cfg,
                run_config=run_cfg,
            )
            model_results[feature_set_name] = result

            if cfg.use_cache:
                _save_result_cache(
                    cache_dir=cache_dir,
                    cache_key=cache_key,
                    result=result,
                )
                logger.info(
                    "Saved cache for model=%s feature_set=%s at %s",
                    model_name,
                    feature_set_name,
                    cache_dir / cache_key,
                )

        all_results[model_name] = model_results
        logger.info("Completed benchmark model type: %s", model_name)

    logger.info("Wheat HAR benchmark finished. model_types=%d", len(all_results))
    return all_results


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
                "model_max_selected_features": model_info.get(
                    "model_max_selected_features"
                ),
                "model_min_train_feature_ratio": model_info.get(
                    "model_min_train_feature_ratio"
                ),
                "lasso_best_alpha": selection_info.get("best_alpha"),
                "bsr_alpha": selection_info.get("alpha"),
                "bsr_window_type": selection_info.get("window_type"),
                "bsr_window_size": selection_info.get("window_size"),
                "bsr_step": selection_info.get("step"),
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        logger.warning("Benchmark result dataframe is empty")
        return out
    summary = out.sort_values(["model_type", "test_mse", "test_mae"]).reset_index(drop=True)
    logger.info("Benchmark summary rows=%d", len(summary))
    return summary
