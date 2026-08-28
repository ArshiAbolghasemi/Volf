from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from src.benchmark.utils import (
    classify_feature_group,
    default_core_columns_for_target,
    default_endo_columns_for_target,
    default_exo_columns_for_target,
)
from src.util.path import DATA_DIR

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)
FAMILY_CHOICES = ("har", "rf", "xgb")
CROP_TARGETS = {
    "wheat": "wheat_weekly_rv",
    "corn": "corn_weekly_rv",
    "soybean": "soybeans_weekly_rv",
}
GROUPS = ("core", "endo", "exo", "climate", "news", "macro")
JOB_MARKERS = {"har": "group_lasso_expanding", "rf": "rf_expanding"}
META_COLUMNS = {"Date", "base_value", "prediction_transformed", "prediction"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot exact feature-group SHAP values across forecast horizons"
    )
    parser.add_argument(
        "--crops", nargs="+", choices=CROP_TARGETS, default=list(CROP_TARGETS)
    )
    parser.add_argument(
        "--include_families",
        nargs="+",
        choices=FAMILY_CHOICES,
        default=["har", "rf"],
    )
    parser.add_argument(
        "--exclude_families", nargs="+", choices=FAMILY_CHOICES, default=[]
    )
    parser.add_argument(
        "--output_dir",
        default=str(DATA_DIR / "benchmark" / "plots" / "shap_horizons"),
    )
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def _selected_families(
    included: Iterable[str], excluded: Iterable[str]
) -> list[str]:
    blocked = set(excluded)
    selected = [family for family in included if family not in blocked]
    if not selected:
        raise ValueError("No model families selected.")
    return selected


def _group_for_feature(feature: str, target: str) -> str | None:
    if feature in default_core_columns_for_target(target):
        return "core"
    return classify_feature_group(
        feature,
        endo_columns=default_endo_columns_for_target(target),
        exo_columns=default_exo_columns_for_target(target),
    )


def _group_shap_summary(path: Path, target: str) -> pd.DataFrame:
    values = pd.read_csv(path, index_col=0)
    feature_columns = [column for column in values.columns if column not in META_COLUMNS]
    summary_path = path.with_name("summary.csv")
    if summary_path.exists() and len(feature_columns) < len(pd.read_csv(summary_path)):
        raise ValueError(f"{path} contains only reported top features; regenerate SHAP")
    grouped = pd.DataFrame(0.0, index=values.index, columns=GROUPS)
    for feature in feature_columns:
        group = _group_for_feature(feature, target)
        if group is not None:
            grouped[group] += pd.to_numeric(values[feature], errors="coerce").fillna(0.0)
    return pd.DataFrame(
        {
            "group": GROUPS,
            "mean_abs_shap": [grouped[group].abs().mean() for group in GROUPS],
            "mean_shap": [grouped[group].mean() for group in GROUPS],
        }
    )


def _discover_horizon_values(crop: str, family: str) -> list[tuple[int, str, Path]]:
    root = DATA_DIR / "benchmark" / crop / family / "mean"
    marker = JOB_MARKERS.get(family)
    found: list[tuple[int, str, Path]] = []
    for horizon_dir in sorted(root.glob("target_horizon_*")):
        try:
            horizon = int(horizon_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        jobs = sorted((horizon_dir / "shap").glob("*/shap_values.csv"))
        jobs = [path for path in jobs if marker is None or marker in path.parent.name]
        if jobs:
            found.append((horizon, jobs[0].parent.name, jobs[0]))
    return sorted(found)


def _build_trends(crop: str, family: str) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[pd.DataFrame] = []
    jobs: list[dict] = []
    for horizon, job_name, path in _discover_horizon_values(crop, family):
        try:
            frame = _group_shap_summary(path, CROP_TARGETS[crop])
        except ValueError as exc:
            logger.warning("%s", exc)
            continue
        frame["horizon"] = horizon
        rows.append(frame)
        jobs.append(
            {"horizon": horizon, "job_name": job_name, "shap_values_path": str(path)}
        )
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), jobs


def _plot_trends(frame: pd.DataFrame, crop: str, family: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    for group in GROUPS:
        group_frame = frame[frame["group"].eq(group)].sort_values("horizon")
        axis.plot(
            group_frame["horizon"],
            group_frame["mean_abs_shap"],
            marker="o",
            linewidth=2,
            label=group,
        )
    axis.set(
        title=f"{crop.title()} {family.upper()} group SHAP trends",
        xlabel="Forecast horizon (weeks)",
        ylabel="mean |sum of group SHAP values|",
    )
    axis.set_xticks(sorted(frame["horizon"].unique()))
    axis.grid(visible=True, linestyle="--", alpha=0.3)
    axis.legend(title="Feature group")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    output_root = Path(args.output_dir)
    for crop in args.crops:
        for family in _selected_families(
            args.include_families, args.exclude_families
        ):
            trends, jobs = _build_trends(crop, family)
            if trends.empty:
                logger.info("No matching SHAP values for crop=%s family=%s", crop, family)
                continue
            output_dir = output_root / crop / family
            plot_path = output_dir / "group_shap_trends.png"
            _plot_trends(trends, crop, family, plot_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            trends.to_csv(output_dir / "group_shap_trends.csv", index=False)
            (output_dir / "plot_metadata.json").write_text(
                json.dumps(
                    {
                        "crop": crop,
                        "family": family,
                        "aggregation": "mean(abs(sum(signed feature SHAP within group)))",
                        "groups": list(GROUPS),
                        "jobs": jobs,
                        "plot_path": str(plot_path),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("Saved group SHAP trends to %s", plot_path)


if __name__ == "__main__":
    main()
