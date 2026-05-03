from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add adjusted R^2 columns to every HAR benchmark summary CSV."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/benchmark"),
        help="Root directory to search for har.csv files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be updated without writing changes.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def adjusted_r2(r2: float, n_obs: int, n_predictors: int) -> float:
    if math.isnan(r2):
        return math.nan
    if n_obs <= n_predictors + 1:
        return math.nan
    factor = (n_obs - 1) / (n_obs - n_predictors - 1)
    return 1.0 - (1.0 - float(r2)) * factor


def _sample_size_from_metrics(checkpoint_dir: Path, split: str) -> int:
    metrics_path = checkpoint_dir / "metrics.json"
    if not metrics_path.exists():
        msg = f"Missing metrics file: {metrics_path}"
        raise FileNotFoundError(msg)
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    n_obs = int(metrics[split]["n_obs"])
    logger.debug(
        "Loaded %s sample size from %s: n_obs=%d",
        split,
        metrics_path,
        n_obs,
    )
    return n_obs


def _checkpoint_candidates(csv_path: Path, row: dict[str, str]) -> list[Path]:
    model_type = row["model_type"]
    feature_set = row["feature_set"]
    run_dir_name = f"{model_type}__{feature_set}"
    return [
        csv_path.parent / "checkpoints" / run_dir_name,
        csv_path.parent / "checkpoints" / run_dir_name,
    ]


def _resolve_checkpoint_dir(csv_path: Path, row: dict[str, str]) -> Path:
    for candidate in _checkpoint_candidates(csv_path, row):
        if candidate.exists():
            logger.debug(
                "Resolved checkpoint dir for %s/%s via canonical candidate: %s",
                row["model_type"],
                row["feature_set"],
                candidate,
            )
            return candidate

    run_dir_name = f"{row['model_type']}__{row['feature_set']}"
    matches = sorted(
        candidate
        for candidate in (csv_path.parent / "checkpoints").rglob(run_dir_name)
        if candidate.is_dir()
    )
    if len(matches) == 1:
        logger.debug(
            "Resolved checkpoint dir for %s/%s via recursive search: %s",
            row["model_type"],
            row["feature_set"],
            matches[0],
        )
        return matches[0]
    if not matches:
        msg = f"Could not find checkpoint dir for row '{run_dir_name}' in {csv_path}"
        raise FileNotFoundError(msg)
    msg = (
        f"Multiple checkpoint dirs found for row '{run_dir_name}' in {csv_path}: {matches}"
    )
    raise RuntimeError(msg)


def _parse_int(value: str) -> int:
    return int(float(value.strip()))


def _parse_float(value: str) -> float:
    text = value.strip()
    if not text:
        return math.nan
    return float(text)


def _format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    return repr(value)


def _fallback_train_sample_size(row: dict[str, str]) -> int:
    n_windows = _parse_int(row["n_windows"])
    train_size = _parse_int(row["initial_train_size"])
    step = _parse_int(row["window_step"])
    window_type = row["window_type"].strip()
    rolling_size_raw = row.get("rolling_window_size", "").strip()

    if window_type == "rolling":
        rolling_size = _parse_int(rolling_size_raw) if rolling_size_raw else train_size
        train_n = n_windows * rolling_size
    else:
        train_n = n_windows * train_size + step * n_windows * (n_windows - 1) // 2

    logger.debug(
        "Derived fallback train sample size for %s/%s: train_n=%d n_windows=%d",
        row["model_type"],
        row["feature_set"],
        train_n,
        n_windows,
    )
    return train_n


def update_har_csv(csv_path: Path) -> None:
    logger.info("Processing %s", csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    logger.info("Loaded %d rows from %s", len(rows), csv_path)

    if not rows:
        output_fieldnames = fieldnames.copy()
        if "train_adjusted_r2" not in output_fieldnames:
            output_fieldnames.append("train_adjusted_r2")
        if "test_adjusted_r2" in output_fieldnames:
            output_fieldnames.remove("test_adjusted_r2")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
            writer.writeheader()
        logger.info("Wrote empty HAR summary with adjusted R^2 columns to %s", csv_path)
        return

    fallback_count = 0
    for row_idx, row in enumerate(rows, start=1):
        n_predictors = int(row["n_selected"])
        try:
            checkpoint_dir = _resolve_checkpoint_dir(csv_path, row)
            train_n = _sample_size_from_metrics(checkpoint_dir, "train")
        except FileNotFoundError:
            train_n = _fallback_train_sample_size(row)
            fallback_count += 1
            logger.warning(
                (
                    "Falling back to derived train sample size for %s "
                    "row=%d model=%s feature_set=%s"
                ),
                csv_path,
                row_idx,
                row["model_type"],
                row["feature_set"],
            )
        train_adjusted_r2 = adjusted_r2(
            _parse_float(row["train_r2"]),
            train_n,
            n_predictors,
        )
        row["train_adjusted_r2"] = _format_float(train_adjusted_r2)
        row.pop("test_adjusted_r2", None)
        logger.debug(
            ("Updated row=%d model=%s feature_set=%s p=%d train_n=%d train_adj_r2=%s"),
            row_idx,
            row["model_type"],
            row["feature_set"],
            n_predictors,
            train_n,
            row["train_adjusted_r2"] or "nan",
        )

    output_fieldnames = fieldnames.copy()
    if "train_adjusted_r2" not in output_fieldnames:
        output_fieldnames.append("train_adjusted_r2")
    if "test_adjusted_r2" in output_fieldnames:
        output_fieldnames.remove("test_adjusted_r2")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(
        "Updated %s with training adjusted R^2 for %d rows (%d fallback rows)",
        csv_path,
        len(rows),
        fallback_count,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    csv_paths = sorted(args.root.rglob("har.csv"))
    logger.info("Discovered %d HAR summary files under %s", len(csv_paths), args.root)

    if args.dry_run:
        for csv_path in csv_paths:
            logger.info("Would update %s", csv_path)
        return

    for csv_path in csv_paths:
        update_har_csv(csv_path)
    logger.info("Finished updating %d HAR summary files", len(csv_paths))


if __name__ == "__main__":
    main()
