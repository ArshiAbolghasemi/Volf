from __future__ import annotations

import hashlib
import itertools
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
DEFAULT_INITIAL_TRAIN_SIZE = 260
DEFAULT_TEST_SIZE = 4
DEFAULT_STEP = 4
DEFAULT_ROLLING_WINDOW_SIZE = 260


@dataclass
class HARGridSearchConfig:
    enabled: bool = False
    metric: str = "test_mse"
    maximize_metric: bool = False
    initial_train_sizes: list[int] | None = None
    test_sizes: list[int] | None = None
    steps: list[int] | None = None
    max_candidates: int | None = 50


@dataclass
class WheatHARBenchmarkConfig:
    csv_path: str = str(DEFAULT_DATA_PATH)
    target_col: str = DEFAULT_TARGET
    core_columns: list[str] | None = None
    target_horizon: int = 1
    target_horizons: list[int] | None = None
    run_configs: dict[str, HARRunConfig] | None = None
    grid_search: HARGridSearchConfig | None = None
    use_cache: bool = True
    cache_dir: str = ".cache/benchmark"
    cache_overwrite: bool = False


def _existing_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in data.columns]


def _resolve_target_horizons(cfg: WheatHARBenchmarkConfig) -> list[int]:
    horizons = cfg.target_horizons or [cfg.target_horizon]
    unique_horizons = sorted({int(h) for h in horizons})
    if not unique_horizons:
        msg = "target_horizons cannot be empty."
        raise ValueError(msg)
    if any(h < 0 for h in unique_horizons):
        msg = f"target_horizons must be >= 0. got={unique_horizons}"
        raise ValueError(msg)
    return unique_horizons


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
    epu_cols = ["epu_index"]
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
            walk_forward=HARWalkForwardConfig(
                window_type="expanding",
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
            ),
            selection=HARSelectionConfig(method="none"),
            model=HARModelConfig(standardize_features=False),
        ),
        "ols_rolling": HARRunConfig(
            walk_forward=HARWalkForwardConfig(
                window_type="rolling",
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
                rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE,
            ),
            selection=HARSelectionConfig(method="none"),
            model=HARModelConfig(standardize_features=False),
        ),
        "lasso_expanding": HARRunConfig(
            walk_forward=HARWalkForwardConfig(
                window_type="expanding",
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
            ),
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
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
                rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE,
            ),
            selection=HARSelectionConfig(
                method="lasso",
                lasso=LassoSelectionConfig(n_splits=5),
                refit_every_windows=4,
            ),
            model=HARModelConfig(standardize_features=True),
        ),
        "bsr_expanding": HARRunConfig(
            walk_forward=HARWalkForwardConfig(
                window_type="expanding",
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
            ),
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
                initial_train_size=DEFAULT_INITIAL_TRAIN_SIZE,
                test_size=DEFAULT_TEST_SIZE,
                step=DEFAULT_STEP,
                rolling_window_size=DEFAULT_ROLLING_WINDOW_SIZE,
            ),
            selection=HARSelectionConfig(
                method="bsr",
                bsr=BSRSelectionConfig(alpha=0.05),
                refit_every_windows=8,
            ),
            model=HARModelConfig(standardize_features=False),
        ),
    }


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
    data_signature: str,
    target_horizon: int,
) -> HARExperimentResult:
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
                "Cache hit for horizon=%d model=%s feature_set=%s at %s",
                target_horizon,
                model_name,
                feature_set_name,
                cache_dir / cache_key,
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
        _save_result_cache(
            cache_dir=cache_dir,
            cache_key=cache_key,
            result=result,
        )
        logger.info(
            "Saved cache for horizon=%d model=%s feature_set=%s at %s",
            target_horizon,
            model_name,
            feature_set_name,
            cache_dir / cache_key,
        )
    return result


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


def _run_wheat_har_benchmark_for_horizon(  # noqa: PLR0913
    *,
    data: pd.DataFrame,
    cfg: WheatHARBenchmarkConfig,
    core: list[str],
    feature_sets: dict[str, list[str]],
    model_run_configs: dict[str, HARRunConfig],
    data_signature: str,
    target_horizon: int,
) -> dict[str, dict[str, Any]]:
    cache_dir = Path(cfg.cache_dir) / f"target_horizon_{target_horizon}"
    all_results: dict[str, dict[str, Any]] = {}

    logger.info("Starting horizon=%d benchmark", target_horizon)
    for model_name, run_cfg in model_run_configs.items():
        logger.info(
            "Running benchmark model type: %s (target_horizon=%d)",
            model_name,
            target_horizon,
        )
        model_results: dict[str, HARExperimentResult] = {}
        base_feature_cfg = HARFeatureConfig(
            target_col=cfg.target_col,
            core_columns=core,
            target_horizon=target_horizon,
        )
        for feature_set_name, extra_cols in feature_sets.items():
            feature_cfg = replace(base_feature_cfg, extra_feature_cols=extra_cols)
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
                    target_horizon,
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
                    target_horizon,
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
                    data_signature=data_signature,
                    target_horizon=target_horizon,
                )
                score = _resolve_metric_value(result, metric_name)
                logger.info(
                    (
                        "Grid candidate %d/%d score for horizon=%d model=%s "
                        "feature_set=%s: %s=%.10f"
                    ),
                    idx + 1,
                    len(candidates),
                    target_horizon,
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
                        target_horizon,
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
                (
                    "Grid selected for horizon=%d model=%s feature_set=%s: "
                    "candidate=%d/%d %s=%.10f"
                ),
                target_horizon,
                model_name,
                feature_set_name,
                best_idx + 1,
                len(candidates),
                metric_name,
                float(best_score),
            )
            model_results[feature_set_name] = best_result

        all_results[model_name] = model_results
        logger.info(
            "Completed benchmark model type: %s (target_horizon=%d)",
            model_name,
            target_horizon,
        )
    return all_results


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

    core = cfg.core_columns or _existing_columns(data, DEFAULT_CORE_COLUMNS)
    if not core:
        msg = "No valid core columns found in data."
        raise ValueError(msg)
    logger.info("Using core columns: %s", core)

    feature_sets = build_wheat_feature_sets(data, core_columns=core)
    data_signature = _dataset_signature(data)
    model_run_configs = cfg.run_configs or default_run_configs()
    results_by_horizon: dict[int, dict[str, dict[str, Any]]] = {}
    for horizon in _resolve_target_horizons(cfg):
        results_by_horizon[horizon] = _run_wheat_har_benchmark_for_horizon(
            data=data,
            cfg=cfg,
            core=core,
            feature_sets=feature_sets,
            model_run_configs=model_run_configs,
            data_signature=data_signature,
            target_horizon=horizon,
        )

    logger.info(
        "Wheat HAR benchmark finished. horizons=%d",
        len(results_by_horizon),
    )
    return results_by_horizon


def run_wheat_har_benchmark(
    *,
    config: WheatHARBenchmarkConfig | None = None,
    data: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    cfg = config or WheatHARBenchmarkConfig()
    horizons = _resolve_target_horizons(cfg)
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
