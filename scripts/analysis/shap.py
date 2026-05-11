from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from src.benchmark.utils import get_feature_group_columns
from src.util.path import DATA_DIR

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

FAMILY_CHOICES = ("har", "rf", "xgb")
CROP_CHOICES = ("wheat", "corn", "soybean")
MODE_CHOICES = ("mean",)
TRACKED_GROUPS = ("news", "macro", "climate")
CLIMATE_REFERENCE_HORIZON = 16
CLIMATE_TOP_N_FEATURES = 10


@dataclass(frozen=True)
class ShapSummaryRecord:
    crop: str
    family: str
    target_mode: str
    target_horizon: int
    job_name: str
    summary_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot SHAP summary values across horizons for selected crops/model families"
        )
    )
    parser.add_argument(
        "--crops",
        nargs="+",
        choices=CROP_CHOICES,
        default=list(CROP_CHOICES),
        help="Crops to include",
    )
    parser.add_argument(
        "--include_families",
        nargs="+",
        choices=FAMILY_CHOICES,
        default=list(FAMILY_CHOICES),
        help="Model families to include",
    )
    parser.add_argument(
        "--exclude_families",
        nargs="+",
        choices=FAMILY_CHOICES,
        default=[],
        help="Model families to exclude",
    )
    parser.add_argument(
        "--top_n_features",
        type=int,
        default=20,
        help="Number of top features to show in each plot",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Optional explicit feature list to plot",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DATA_DIR / "benchmark" / "plots" / "shap_horizons"),
        help="Directory to save horizon plots",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def _normalize_family_selection(
    include_families: Iterable[str],
    exclude_families: Iterable[str],
) -> list[str]:
    excluded = {value.strip().lower() for value in exclude_families}
    selected = [value for value in include_families if value not in excluded]
    if not selected:
        msg = "No model families selected after applying exclusions."
        raise ValueError(msg)
    return selected


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def _family_mode_roots(crop: str, family: str, target_mode: str) -> list[Path]:
    roots = [DATA_DIR / "benchmark" / crop / family / target_mode]
    legacy_root = DATA_DIR / "benchmark" / crop / target_mode
    if legacy_root not in roots:
        roots.append(legacy_root)
    return roots


def _discover_summary_records(
    *,
    crop: str,
    family: str,
) -> list[ShapSummaryRecord]:
    records: list[ShapSummaryRecord] = []
    for target_mode in MODE_CHOICES:
        for root in _family_mode_roots(crop, family, target_mode):
            if not root.exists():
                continue
            for horizon_dir in sorted(root.glob("target_horizon_*")):
                if not horizon_dir.is_dir():
                    continue
                try:
                    horizon = int(horizon_dir.name.split("_")[-1])
                except ValueError:
                    continue
                shap_root = horizon_dir / "shap"
                if not shap_root.exists():
                    continue
                job_dirs = sorted(path for path in shap_root.iterdir() if path.is_dir())
                for job_dir in job_dirs:
                    summary_path = job_dir / "summary.csv"
                    if not summary_path.exists():
                        continue
                    records.append(
                        ShapSummaryRecord(
                            crop=crop,
                            family=family,
                            target_mode=target_mode,
                            target_horizon=horizon,
                            job_name=job_dir.name,
                            summary_path=summary_path,
                        )
                    )
            break
    return records


def _load_best_summary_per_horizon(
    records: list[ShapSummaryRecord],
) -> list[tuple[ShapSummaryRecord, pd.DataFrame]]:
    grouped: dict[tuple[str, int], list[ShapSummaryRecord]] = {}
    for record in records:
        grouped.setdefault((record.target_mode, record.target_horizon), []).append(record)

    selected: list[tuple[ShapSummaryRecord, pd.DataFrame]] = []
    for _, options in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        best_record: ShapSummaryRecord | None = None
        best_frame: pd.DataFrame | None = None
        best_score = float("-inf")
        for record in options:
            frame = pd.read_csv(record.summary_path)
            if frame.empty or "mean_abs_shap" not in frame.columns:
                continue
            score = float(frame["mean_abs_shap"].sum())
            if score > best_score:
                best_score = score
                best_record = record
                best_frame = frame
        if best_record is not None and best_frame is not None:
            selected.append((best_record, best_frame))
    return selected


def _resolve_feature_order(
    frames: list[pd.DataFrame],
    *,
    explicit_features: list[str] | None,
    top_n_features: int,
    allowed_features: set[str] | None = None,
) -> list[str]:
    if explicit_features:
        if allowed_features is None:
            return explicit_features
        return [feature for feature in explicit_features if feature in allowed_features]

    aggregate: dict[str, float] = {}
    for frame in frames:
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            feature_name = str(row["feature"])
            if allowed_features is not None and feature_name not in allowed_features:
                continue
            aggregate[feature_name] = aggregate.get(feature_name, 0.0) + float(
                row["mean_abs_shap"]
            )

    ordered = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ordered[: max(top_n_features, 1)]]


def _resolve_climate_feature_order_from_reference_horizon(
    records_and_frames: list[tuple[ShapSummaryRecord, pd.DataFrame]],
    *,
    allowed_features: set[str],
) -> list[str]:
    reference_frames = [
        frame
        for record, frame in records_and_frames
        if record.target_horizon == CLIMATE_REFERENCE_HORIZON
    ]
    if reference_frames:
        return _resolve_feature_order(
            reference_frames,
            explicit_features=None,
            top_n_features=CLIMATE_TOP_N_FEATURES,
            allowed_features=allowed_features,
        )

    logger.warning(
        "No SHAP summary found for climate reference horizon h%d; "
        "falling back to all horizons",
        CLIMATE_REFERENCE_HORIZON,
    )
    return _resolve_feature_order(
        [frame for _, frame in records_and_frames],
        explicit_features=None,
        top_n_features=CLIMATE_TOP_N_FEATURES,
        allowed_features=allowed_features,
    )


def _build_mode_horizon_frame(
    records_and_frames: list[tuple[ShapSummaryRecord, pd.DataFrame]],
    *,
    feature_order: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    horizon_counts: dict[int, int] = {}
    for record, _ in records_and_frames:
        horizon_counts[record.target_horizon] = (
            horizon_counts.get(record.target_horizon, 0) + 1
        )

    ordered_records = sorted(
        records_and_frames,
        key=lambda item: (
            item[0].target_horizon,
            item[0].job_name,
        ),
    )
    for plot_order, (record, frame) in enumerate(ordered_records):
        ordered_frame = frame.set_index("feature").reindex(feature_order).fillna(0.0)
        horizon_label = f"h{record.target_horizon}"
        for feature_name, row in ordered_frame.iterrows():
            rows.append(
                {
                    "feature": feature_name,
                    "target_horizon": record.target_horizon,
                    "target_mode": record.target_mode,
                    "horizon_label": horizon_label,
                    "plot_order": plot_order,
                    "mean_abs_shap": float(row["mean_abs_shap"]),
                    "mean_shap": float(row["mean_shap"]) if "mean_shap" in row else 0.0,
                    "job_name": record.job_name,
                }
            )
    return pd.DataFrame(rows)


def _plot_group_horizon_lines(  # noqa: PLR0913
    *,
    crop: str,
    family: str,
    records_and_frames: list[tuple[ShapSummaryRecord, pd.DataFrame]],
    output_dir: Path,
    top_n_features: int,
    explicit_features: list[str] | None,
) -> tuple[list[Path], list[dict[str, object]]]:
    saved_paths: list[Path] = []
    trend_rows: list[dict[str, object]] = []
    if not records_and_frames:
        return saved_paths, trend_rows

    feature_groups = get_feature_group_columns()
    for group_name in TRACKED_GROUPS:
        allowed_features = set(feature_groups[group_name])
        if group_name == "climate" and explicit_features is None:
            feature_order = _resolve_climate_feature_order_from_reference_horizon(
                records_and_frames,
                allowed_features=allowed_features,
            )
        else:
            feature_order = _resolve_feature_order(
                [frame for _, frame in records_and_frames],
                explicit_features=explicit_features,
                top_n_features=top_n_features,
                allowed_features=allowed_features,
            )
        if not feature_order:
            logger.info(
                "No %s features found for crop=%s family=%s",
                group_name,
                crop,
                family,
            )
            continue

        plot_frame = _build_mode_horizon_frame(
            records_and_frames,
            feature_order=feature_order,
        )
        if plot_frame.empty:
            continue

        fig, axis = plt.subplots(figsize=(10, max(5.5, 1.0 + 0.55 * len(feature_order))))
        tick_positions = (
            plot_frame[["plot_order", "horizon_label"]]
            .drop_duplicates()
            .sort_values("plot_order")
        )
        for feature_name in feature_order:
            feature_frame = plot_frame[plot_frame["feature"] == feature_name].sort_values(
                "plot_order"
            )
            if feature_frame.empty:
                continue

            plot_positions = feature_frame["plot_order"].astype(int).to_numpy()
            shap_values = feature_frame["mean_abs_shap"].astype(float).to_numpy()
            start_row = feature_frame.iloc[0]
            end_row = feature_frame.iloc[-1]
            delta = float(shap_values[-1] - shap_values[0]) if len(shap_values) > 1 else 0.0
            trend = "increased" if delta > 0 else "decreased" if delta < 0 else "flat"
            trend_rows.append(
                {
                    "crop": crop,
                    "family": family,
                    "feature_group": group_name,
                    "feature": feature_name,
                    "start_horizon": int(start_row["target_horizon"]),
                    "start_label": str(start_row["horizon_label"]),
                    "end_horizon": int(end_row["target_horizon"]),
                    "end_label": str(end_row["horizon_label"]),
                    "start_mean_abs_shap": float(shap_values[0]),
                    "end_mean_abs_shap": float(shap_values[-1]),
                    "delta_mean_abs_shap": delta,
                    "trend": trend,
                }
            )
            axis.plot(
                plot_positions,
                shap_values,
                marker="o",
                linewidth=2,
                label=f"{feature_name} ({trend})",
            )

        axis.set_title(
            f"{crop.title()} {family.upper()} {group_name.title()} SHAP trend",
            fontsize=13,
        )
        axis.set_xlabel("horizon")
        axis.set_ylabel("mean |SHAP|")
        axis.grid(visible=True, linestyle="--", alpha=0.3)
        axis.set_xticks(tick_positions["plot_order"].astype(int).to_list())
        axis.set_xticklabels(
            tick_positions["horizon_label"].to_list(), rotation=30, ha="right"
        )
        axis.legend(loc="best", fontsize=8)
        fig.tight_layout()

        mode_output_dir = output_dir / crop / family
        mode_output_dir.mkdir(parents=True, exist_ok=True)
        out_path = mode_output_dir / f"shap_trend_{group_name}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)
    return saved_paths, trend_rows


def _write_metadata(  # noqa: PLR0913
    *,
    crop: str,
    family: str,
    records_and_frames: list[tuple[ShapSummaryRecord, pd.DataFrame]],
    saved_paths: list[Path],
    trend_rows: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    metadata = {
        "crop": crop,
        "family": family,
        "selected_jobs": [
            {
                "target_mode": record.target_mode,
                "target_horizon": record.target_horizon,
                "job_name": record.job_name,
                "summary_path": str(record.summary_path),
            }
            for record, _ in records_and_frames
        ],
        "plot_paths": [str(path) for path in saved_paths],
        "tracked_groups": list(TRACKED_GROUPS),
        "trends": trend_rows,
    }
    meta_dir = output_dir / crop / family
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / "plot_metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_path


def _write_trend_summary(
    *,
    crop: str,
    family: str,
    trend_rows: list[dict[str, object]],
    output_dir: Path,
) -> Path:
    out_dir = output_dir / crop / family
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shap_trend_summary.csv"
    pd.DataFrame(trend_rows).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    families = _normalize_family_selection(
        include_families=args.include_families,
        exclude_families=args.exclude_families,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for crop in args.crops:
        for family in families:
            records = _discover_summary_records(crop=crop, family=family)
            if not records:
                logger.info("No SHAP summaries found for crop=%s family=%s", crop, family)
                continue

            records_and_frames = _load_best_summary_per_horizon(records)
            if not records_and_frames:
                logger.info(
                    "No usable SHAP summary frames found for crop=%s family=%s",
                    crop,
                    family,
                )
                continue

            saved_paths, trend_rows = _plot_group_horizon_lines(
                crop=crop,
                family=family,
                records_and_frames=records_and_frames,
                output_dir=output_dir,
                top_n_features=args.top_n_features,
                explicit_features=args.features,
            )
            trend_summary_path = _write_trend_summary(
                crop=crop,
                family=family,
                trend_rows=trend_rows,
                output_dir=output_dir,
            )
            metadata_path = _write_metadata(
                crop=crop,
                family=family,
                records_and_frames=records_and_frames,
                saved_paths=saved_paths,
                trend_rows=trend_rows,
                output_dir=output_dir,
            )
            logger.info(
                "Saved %d SHAP trend plots for crop=%s family=%s; trends=%s metadata=%s",
                len(saved_paths),
                crop,
                family,
                trend_summary_path,
                metadata_path,
            )


if __name__ == "__main__":
    main()
