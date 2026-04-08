from __future__ import annotations

import argparse
import errno
import json
import logging
from pathlib import Path
from typing import Any, cast

from src.benchmark.rf import (
    RFGridSearchConfig,
    WheatRFBenchmarkConfig,
    benchmark_multi_horizon_results_to_frame,
    benchmark_results_to_frame,
    run_wheat_rf_benchmark,
    run_wheat_rf_benchmark_multi_horizon,
)
from src.benchmark.utils import normalize_target_mode
from src.model import RFModelConfig, RFRunConfig, RFWalkForwardConfig
from src.util.path import DATA_DIR

logger = logging.getLogger(__name__)


def parse_args(
    *,
    default_config: str | None = None,
    default_output: str = str(DATA_DIR / "benchmark" / "random_forest.csv"),
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wheat RF benchmark on ag/v4.csv")
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
        help="Cache directory for RF benchmark artifacts",
    )
    parser.add_argument(
        "--cache_overwrite",
        action="store_true",
        help="Overwrite existing cache entries and retrain",
    )
    return parser.parse_args()


def _parse_target_horizons_arg(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return sorted({int(token.strip()) for token in value.split(",") if token.strip()})


def _target_mode_output_dir(mode: str) -> Path:
    return DATA_DIR / "benchmark" / "rf" / mode


def _build_run_config_from_dict(cfg: dict[str, Any]) -> RFRunConfig:
    walk_forward_cfg = None
    if isinstance(cfg.get("walk_forward"), dict):
        walk_forward_cfg = RFWalkForwardConfig(**cfg["walk_forward"])

    model_cfg = None
    if isinstance(cfg.get("model"), dict):
        model_cfg = RFModelConfig(**cfg["model"])

    return RFRunConfig(walk_forward=walk_forward_cfg, model=model_cfg)


def _load_config_from_json(path: str) -> WheatRFBenchmarkConfig:
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)

    run_configs = None
    if isinstance(raw.get("run_configs"), dict):
        run_configs = {
            name: _build_run_config_from_dict(cfg)
            for name, cfg in raw["run_configs"].items()
            if isinstance(cfg, dict)
        }

    return WheatRFBenchmarkConfig(
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


def main(  # noqa: C901, PLR0912
    *,
    default_config: str | None = None,
    default_output: str = str(DATA_DIR / "benchmark" / "random_forest.csv"),
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
        logger.info("Loaded RF benchmark config from %s", args.config)
    else:
        cfg = WheatRFBenchmarkConfig(
            csv_path=args.input,
            target_col=args.target_col,
            target_horizon=args.target_horizon,
            use_cache=bool(args.use_cache),
            cache_dir=args.cache_dir,
            cache_overwrite=bool(args.cache_overwrite),
        )
        logger.info("Using default/CLI RF benchmark config")
    if args.config:
        cfg.use_cache = bool(args.use_cache)
        cfg.cache_dir = args.cache_dir
        cfg.cache_overwrite = bool(args.cache_overwrite)

    cli_horizons = _parse_target_horizons_arg(args.target_horizons)
    if cli_horizons is not None:
        cfg.target_horizons = cli_horizons
        cfg.target_horizon = cli_horizons[0]
    elif cfg.target_horizons is None:
        cfg.target_horizons = [cfg.target_horizon]

    if cfg.target_horizons and len(cfg.target_horizons) > 1:
        results_by_horizon = run_wheat_rf_benchmark_multi_horizon(config=cfg)
        summary = benchmark_multi_horizon_results_to_frame(results_by_horizon)
    else:
        results = run_wheat_rf_benchmark(config=cfg)
        summary = benchmark_results_to_frame(results)

    output_path = Path(args.output)
    mode_output_root = _target_mode_output_dir(cfg.target_mode)
    if output_path.parent == (DATA_DIR / "benchmark"):
        output_path = mode_output_root / output_path.name

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        fallback_path = mode_output_root / output_path.name
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
        logger.info("Saved horizon=%d RF summary to %s", horizon, horizon_out)

        logger.info(
            "Top RF results by test_r2:\n%s",
            summary[
                [
                    "target_horizon",
                    "model_type",
                    "feature_set",
                    "window_type",
                    "test_r2",
                    "test_mse",
                    "grid_search_metric_value",
                ]
            ].to_string(index=False),
        )


if __name__ == "__main__":
    main()
