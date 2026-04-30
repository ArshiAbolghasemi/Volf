from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast

from src.benchmark.checkpoints import load_har_checkpoint_result
from src.benchmark.har import (
    ClarkWestConfig,
    ClarkWestPairConfig,
    HARGridSearchConfig,
    WheatHARBenchmarkConfig,
    default_run_configs,
    run_clark_west_by_pairs,
)
from src.benchmark.utils import normalize_target_mode
from src.model import (
    HARModelConfig,
    HARRunConfig,
    HARSelectionConfig,
    HARWalkForwardConfig,
)
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


def _resolve_output_root(raw_value: str | None, *, default_subpath: str) -> Path:
    if raw_value is None or not str(raw_value).strip():
        return DATA_DIR / "benchmark" / default_subpath
    raw_path = Path(str(raw_value))
    return raw_path if raw_path.is_absolute() else (DATA_DIR / "benchmark" / raw_path)


def _load_clark_west_config(
    path: str,
) -> tuple[WheatHARBenchmarkConfig, ClarkWestConfig, Path, Path]:
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
    pairs: list[ClarkWestPairConfig] = []
    if isinstance(pairs_raw, list) and pairs_raw:
        pairs = [ClarkWestPairConfig(**pair) for pair in pairs_raw]
    elif isinstance(raw.get("pair_template"), dict):
        template = cast("dict[str, Any]", raw["pair_template"])
        horizons = template.get("horizons") or benchmark_cfg.target_horizons
        model_types = template.get("model_types")
        feature_pairs = template.get("feature_pairs")
        if not isinstance(horizons, list) or not horizons:
            msg = "pair_template.horizons must be a non-empty list."
            raise ValueError(msg)
        if not isinstance(model_types, list) or not model_types:
            msg = "pair_template.model_types must be a non-empty list."
            raise ValueError(msg)
        if not isinstance(feature_pairs, list) or not feature_pairs:
            msg = "pair_template.feature_pairs must be a non-empty list."
            raise ValueError(msg)
        for horizon in horizons:
            for model_type in model_types:
                for feature_pair in feature_pairs:
                    if not isinstance(feature_pair, dict):
                        continue
                    base_fs = feature_pair.get("base_feature_set")
                    aug_fs = feature_pair.get("augmented_feature_set")
                    if not isinstance(base_fs, str) or not isinstance(aug_fs, str):
                        continue
                    pair_name = feature_pair.get("name")
                    if not isinstance(pair_name, str) or not pair_name.strip():
                        pair_name = f"{base_fs}->{aug_fs}"
                    pair_hac = feature_pair.get("hac_maxlags")
                    pairs.append(
                        ClarkWestPairConfig(
                            model_type=str(model_type),
                            base_feature_set=base_fs,
                            augmented_feature_set=aug_fs,
                            target_horizon=int(horizon),
                            name=f"{model_type}:{pair_name}:h{int(horizon)}",
                            hac_maxlags=(
                                None if pair_hac is None else int(cast("int", pair_hac))
                            ),
                        )
                    )
    else:
        msg = "Clark-West config must include non-empty 'pairs' or 'pair_template'."
        raise ValueError(msg)

    if not pairs:
        msg = "Clark-West config produced an empty pair list."
        raise ValueError(msg)

    cw_cfg = ClarkWestConfig(
        pairs=pairs,
        hac_maxlags=raw.get("hac_maxlags"),
    )

    output_root = _resolve_output_root(
        cast("str | None", raw.get("output_root")),
        default_subpath=f"har/{benchmark_cfg.target_mode}",
    )
    checkpoint_root = _resolve_output_root(
        cast("str | None", raw.get("checkpoint_root")),
        default_subpath=f"har/{benchmark_cfg.target_mode}",
    )
    return benchmark_cfg, cw_cfg, output_root, checkpoint_root


def _collect_pair_results(
    *,
    benchmark_cfg: WheatHARBenchmarkConfig,
    cw_cfg: ClarkWestConfig,
    checkpoint_root: Path,
) -> dict[int, dict[str, dict[str, Any]]]:
    run_configs = benchmark_cfg.run_configs or default_run_configs()

    required_jobs: set[tuple[int, str, str]] = set()
    for pair in cw_cfg.pairs:
        required_jobs.add(
            (int(pair.target_horizon), pair.model_type, pair.base_feature_set)
        )
        required_jobs.add(
            (int(pair.target_horizon), pair.model_type, pair.augmented_feature_set)
        )

    results_by_horizon: dict[int, dict[str, dict[str, Any]]] = {}
    logger.info("Loading %d HAR checkpoint bundles for Clark-West", len(required_jobs))

    for horizon, model_type, feature_set in sorted(required_jobs):
        if model_type not in run_configs:
            msg = (
                f"model_type='{model_type}' not found in benchmark run configs. "
                f"available={sorted(run_configs)}"
            )
            raise ValueError(msg)
        logger.info(
            "Loading CW input from checkpoint: horizon=%d model=%s feature_set=%s",
            horizon,
            model_type,
            feature_set,
        )
        result = load_har_checkpoint_result(
            checkpoint_root=checkpoint_root,
            target_horizon=horizon,
            target_mode=benchmark_cfg.target_mode,
            model_type=model_type,
            feature_set=feature_set,
        )
        results_by_horizon.setdefault(horizon, {}).setdefault(model_type, {})[
            feature_set
        ] = result

    return results_by_horizon


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    benchmark_cfg, cw_cfg, output_root, checkpoint_root = _load_clark_west_config(
        args.config
    )

    logger.info("Loading requested Clark-West inputs from saved checkpoints")
    results_by_horizon = _collect_pair_results(
        benchmark_cfg=benchmark_cfg,
        cw_cfg=cw_cfg,
        checkpoint_root=checkpoint_root,
    )

    logger.info(
        "Running Clark-West tests for %d pair(s)",
        len(cw_cfg.pairs),
    )
    cw_frame = run_clark_west_by_pairs(results_by_horizon, cw_cfg)

    if cw_frame.empty:
        logger.warning("Clark-West output is empty")
        return

    horizons = sorted({int(h) for h in cw_frame["target_horizon"].tolist()})
    for horizon in horizons:
        horizon_df = cw_frame[cw_frame["target_horizon"] == horizon].copy()
        if horizon_df.empty:
            continue
        horizon_dir = output_root / f"target_horizon_{horizon}"
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
