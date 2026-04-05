from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast

from src.benchmark import (
    ClarkWestConfig,
    ClarkWestPairConfig,
    HARGridSearchConfig,
    WheatHARBenchmarkConfig,
    run_clark_west_by_pairs,
    run_wheat_har_benchmark_multi_horizon,
)
from src.benchmark.har.types import normalize_target_mode
from src.model import HARModelConfig, HARRunConfig, HARSelectionConfig, HARWalkForwardConfig
from src.util.path import DATA_DIR
from src.variable_selection import BSRSelectionConfig, LassoSelectionConfig

logger = logging.getLogger(__name__)
OUTPUT_FILENAME = "clark_west.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run Clark-West tests for selected benchmark model/feature-set pairs")
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to Clark-West JSON config",
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


def _load_clark_west_config(path: str) -> tuple[WheatHARBenchmarkConfig, ClarkWestConfig]:
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)

    benchmark_config_path = raw.get("benchmark_config")
    if benchmark_config_path:
        benchmark_cfg = _load_benchmark_config_from_json(str(benchmark_config_path))
    else:
        benchmark_cfg = WheatHARBenchmarkConfig(
            csv_path=str(raw.get("csv_path", DATA_DIR / "ag" / "v4.csv")),
            target_col=str(raw.get("target_col", "wheat_weekly_rv")),
            target_horizon=int(raw.get("target_horizon", 1)),
            target_horizons=(
                [int(v) for v in raw["target_horizons"]]
                if isinstance(raw.get("target_horizons"), list)
                else None
            ),
            target_mode=normalize_target_mode(str(raw.get("target_mode", "point"))),
            use_cache=bool(raw.get("use_cache", True)),
            cache_dir=str(raw.get("cache_dir", ".cache/benchmark")),
            cache_overwrite=bool(raw.get("cache_overwrite", False)),
            parallel_jobs=int(raw.get("parallel_jobs", 1)),
        )

    if isinstance(raw.get("benchmark_overrides"), dict):
        overrides = raw["benchmark_overrides"]
        for key, value in overrides.items():
            if hasattr(benchmark_cfg, key):
                setattr(benchmark_cfg, key, value)

    pairs_raw = raw.get("pairs")
    if not isinstance(pairs_raw, list) or not pairs_raw:
        msg = "Clark-West config must include a non-empty 'pairs' list."
        raise ValueError(msg)

    pairs = [ClarkWestPairConfig(**pair) for pair in pairs_raw]

    cw_cfg = ClarkWestConfig(
        pairs=pairs,
        hac_maxlags=raw.get("hac_maxlags"),
    )

    return benchmark_cfg, cw_cfg


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    benchmark_cfg, cw_cfg = _load_clark_west_config(args.config)

    logger.info("Running base benchmark to collect forecast series for CW tests")
    results_by_horizon = run_wheat_har_benchmark_multi_horizon(config=benchmark_cfg)

    logger.info(
        "Running Clark-West tests for %d pair(s)",
        len(cw_cfg.pairs),
    )
    cw_frame = run_clark_west_by_pairs(results_by_horizon, cw_cfg)

    if cw_frame.empty:
        logger.warning("Clark-West output is empty")
        return

    root_parent = DATA_DIR / "benchmark" / benchmark_cfg.target_mode

    horizons = sorted({int(h) for h in cw_frame["target_horizon"].tolist()})
    for horizon in horizons:
        horizon_df = cw_frame[cw_frame["target_horizon"] == horizon].copy()
        if horizon_df.empty:
            continue
        horizon_dir = root_parent / f"target_horizon_{horizon}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        horizon_out = horizon_dir / OUTPUT_FILENAME
        horizon_df.to_csv(horizon_out, index=False)
        logger.info(
            "Saved Clark-West CSV report for horizon=%d to %s",
            horizon,
            horizon_out,
        )

    display_cols = [
        "pair_name",
        "target_horizon",
        "base_test_r2",
        "augmented_test_r2",
        "delta_test_r2",
        "test_cw_stat",
        "test_p_value_one_sided",
        "test_augmented_better_at_5pct",
    ]
    logger.info("Top Clark-West rows:\n%s", cw_frame[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
