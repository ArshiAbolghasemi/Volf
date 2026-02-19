import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.benchmark import (
    WheatHARBenchmarkConfig,
    benchmark_results_to_frame,
    run_wheat_har_benchmark,
)
from src.model import HARModelConfig, HARRunConfig, HARSelectionConfig, HARSplitConfig
from src.util.path import DATA_DIR
from src.variable_selection import BSRSelectionConfig, LassoSelectionConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wheat HAR benchmark on ag/v4.csv")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
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
        default=str(DATA_DIR / "ag" / "har_benchmark_summary.csv"),
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
    return parser.parse_args()


def _build_run_config_from_dict(cfg: dict[str, Any]) -> HARRunConfig:
    split_cfg = None
    if isinstance(cfg.get("split"), dict):
        split_cfg = HARSplitConfig(**cfg["split"])

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
        )

    model_cfg = None
    if isinstance(cfg.get("model"), dict):
        model_cfg = HARModelConfig(**cfg["model"])

    return HARRunConfig(split=split_cfg, selection=selection_cfg, model=model_cfg)


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
        run_configs=run_configs,
    )


def main() -> None:
    args = parse_args()

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
        )
        logger.info("Using default/CLI benchmark config")

    logger.info("Running benchmark with input=%s", cfg.csv_path)
    results = run_wheat_har_benchmark(config=cfg)

    summary = benchmark_results_to_frame(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    logger.info("Benchmark summary saved to %s", output_path)
    logger.info("Benchmark rows=%d", len(summary))

    metric_cols = [
        "model_type",
        "feature_set",
        "test_mse",
        "test_mae",
        "test_qlike",
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
            "target_col_raw",
            "target_horizon",
            "core_columns",
            "extra_feature_cols",
            "split_val_size",
            "split_test_size",
            "model_add_constant",
            "model_standardize_features",
            "model_refit_on_train_val",
            "lasso_best_alpha",
            "bsr_alpha",
            "bsr_window_type",
            "bsr_window_size",
            "bsr_step",
        ]
        available_hp_cols = [col for col in hp_cols if col in summary.columns]
        logger.info(
            "Hyperparameter table:\\n%s",
            summary[available_hp_cols].to_string(index=False),
        )


if __name__ == "__main__":
    main()
