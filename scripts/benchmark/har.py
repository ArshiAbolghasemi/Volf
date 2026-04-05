import argparse
import errno
import json
import logging
from pathlib import Path
from typing import Any, cast

from src.benchmark import (
    HARGridSearchConfig,
    WheatHARBenchmarkConfig,
    benchmark_multi_horizon_results_to_frame,
    benchmark_results_to_frame,
    run_wheat_har_benchmark,
    run_wheat_har_benchmark_multi_horizon,
)
from src.benchmark.har.types import normalize_target_mode
from src.model import (
    HARModelConfig,
    HARRunConfig,
    HARSelectionConfig,
    HARWalkForwardConfig,
)
from src.util.path import DATA_DIR
from src.variable_selection import BSRSelectionConfig, LassoSelectionConfig

logger = logging.getLogger(__name__)


def parse_args(
    *,
    default_config: str | None = None,
    default_output: str = str(DATA_DIR / "benchmark" / "har.csv"),
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wheat HAR benchmark on ag/v4.csv")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to JSON config file for benchmark",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_DIR / "ag" / "v4.csv"),
        help="Path to benchmark input CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help="Path to save benchmark summary CSV",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="wheat_weekly_rv",
        help="Target column to forecast",
    )
    parser.add_argument(
        "--target_horizon",
        type=int,
        default=1,
        help="Forecast horizon (weeks)",
    )
    parser.add_argument(
        "--target_horizons",
        type=str,
        default=None,
        help="Comma-separated horizons (e.g. '1,4,8'). Overrides --target_horizon.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--print_hyperparams",
        action="store_true",
        help="Print hyperparameter table in logs",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached benchmark outputs when available",
    )
    cache_group.add_argument(
        "--no_cache",
        dest="use_cache",
        action="store_false",
        help="Disable cache and retrain all runs",
    )
    parser.set_defaults(use_cache=True)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".cache/benchmark",
        help="Cache directory for benchmark artifacts",
    )
    parser.add_argument(
        "--cache_overwrite",
        action="store_true",
        help="Overwrite existing cache entries and retrain",
    )
    parser.add_argument(
        "--parallel_jobs",
        type=int,
        default=None,
        help="Number of concurrent benchmark tasks",
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


def _load_config_from_json(path: str) -> WheatHARBenchmarkConfig:
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
        target_mode=normalize_target_mode(str(raw.get("target_mode", "point"))),
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


def _parse_target_horizons_arg(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return sorted({int(token.strip()) for token in value.split(",") if token.strip()})


def main(  # noqa: C901, PLR0912, PLR0915
    *,
    default_config: str | None = None,
    default_output: str = str(DATA_DIR / "benchmark" / "har.csv"),
) -> None:
    args = parse_args(
        default_config=default_config,
        default_output=default_output,
    )

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.config:
        cfg = _load_config_from_json(args.config)
        logger.info("Loaded benchmark config from %s", args.config)
    else:
        cfg = WheatHARBenchmarkConfig(
            csv_path=args.input,
            target_col=args.target_col,
            target_horizon=args.target_horizon,
            parallel_jobs=(
                int(args.parallel_jobs) if args.parallel_jobs is not None else 1
            ),
            use_cache=bool(args.use_cache),
            cache_dir=args.cache_dir,
            cache_overwrite=bool(args.cache_overwrite),
        )
        logger.info("Using default/CLI benchmark config")
    if args.config and args.parallel_jobs is not None:
        cfg.parallel_jobs = int(args.parallel_jobs)

    cli_horizons = _parse_target_horizons_arg(args.target_horizons)
    if cli_horizons is not None:
        cfg.target_horizons = cli_horizons
        cfg.target_horizon = cli_horizons[0]
    elif cfg.target_horizons is None:
        cfg.target_horizons = [cfg.target_horizon]

    logger.info("Running benchmark with input=%s", cfg.csv_path)
    if cfg.target_horizons and len(cfg.target_horizons) > 1:
        results_by_horizon = run_wheat_har_benchmark_multi_horizon(config=cfg)
        summary = benchmark_multi_horizon_results_to_frame(results_by_horizon)
    else:
        results = run_wheat_har_benchmark(config=cfg)
        summary = benchmark_results_to_frame(results)

    output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        fallback_path = DATA_DIR / "benchmark" / output_path.name
        if exc.errno in {errno.EROFS, errno.EACCES, errno.ENOENT}:
            logger.warning(
                "Output path %s is not writable/available (%s). Falling back to %s",
                output_path,
                exc,
                fallback_path,
            )
            output_path = fallback_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            raise
    logger.info("Benchmark rows=%d", len(summary))

    output_name = output_path.name
    horizons = cfg.target_horizons or [cfg.target_horizon]
    for horizon in horizons:
        horizon_df = summary[summary["target_horizon"] == horizon].copy()
        if horizon_df.empty:
            continue
        horizon_dir = output_path.parent / f"target_horizon_{horizon}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        horizon_out = horizon_dir / output_name
        horizon_df.to_csv(horizon_out, index=False)
        logger.info("Saved horizon=%d summary to %s", horizon, horizon_out)

    metric_cols = [
        "model_type",
        "feature_set",
        "window_type",
        "train_mse",
        "train_mae",
        "train_qlike",
        "train_r2",
        "train_r2log",
        "test_mse",
        "test_mae",
        "test_qlike",
        "test_r2",
        "test_r2log",
    ]
    logger.info(
        "Top results by test_mse:\\n%s",
        summary[metric_cols].head(15).to_string(index=False),
    )

    if args.print_hyperparams:
        hp_cols = [
            "model_type",
            "feature_set",
            "selection_method",
            "window_type",
            "initial_train_size",
            "window_test_size",
            "window_step",
            "rolling_window_size",
            "n_windows",
            "target_col_raw",
            "target_horizon",
            "core_columns",
            "extra_feature_cols",
            "model_add_constant",
            "model_standardize_features",
            "model_target_transform",
            "model_prediction_floor",
            "model_log_transform_rv_features",
            "model_feature_floor",
            "lasso_best_alpha",
            "bsr_alpha",
            "bsr_window_type",
            "bsr_window_size",
            "bsr_step",
            "grid_search_best_candidate_idx",
            "grid_search_n_candidates",
            "grid_search_metric",
            "grid_search_metric_value",
        ]
        available_hp_cols = [col for col in hp_cols if col in summary.columns]
        logger.info(
            "Hyperparameter table:\\n%s",
            summary[available_hp_cols].to_string(index=False),
        )


if __name__ == "__main__":
    main()
