import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset.climate.crop_seasonal import crop_season_flag
from src.dataset.climate.noaa_preprocessing import (
    STATE_ABBREV_TO_NAME,
    _apply_seasonal_mask,
    _compute_extreme_values,
    _compute_zscores,
    _load_production_weights,
    _merge_spi_features,
    build_weighted_commodity_frame,
)
from src.util.path import DATA_DIR

MAX_DISPLAY_MISMATCHES = 30
NUMERIC_TYPES = (int, float, np.number)


def _write(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def _resolve_result_csv(path: Path | None, crop: str) -> Path:
    if path is not None:
        if not path.exists():
            msg = f"Result CSV not found: {path}"
            raise FileNotFoundError(msg)
        return path

    candidates = [
        DATA_DIR / "climate" / f"noaa_weekly_weighted_{crop}.csv",
        DATA_DIR / "climate" / f"cliamte_weekly_weighted_{crop}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    msg = "Could not find result CSV automatically. Tried: " + ", ".join(
        str(candidate) for candidate in candidates
    )
    raise FileNotFoundError(msg)


def _compute_expected_frame(crop: str, noaa_path: Path, spi_path: Path) -> pd.DataFrame:
    noaa_df = pd.read_csv(noaa_path)
    merged = _merge_spi_features(noaa_df, spi_path)
    zscores = _compute_zscores(merged)
    extremes = _compute_extreme_values(zscores)
    masked = _apply_seasonal_mask(extremes, crop)
    production = _load_production_weights(crop, DATA_DIR / "production_by_state")
    return build_weighted_commodity_frame(masked, production)


def _build_state_diagnostics(
    crop: str,
    noaa_path: Path,
    spi_path: Path,
    sample_date: pd.Timestamp,
) -> pd.DataFrame:
    noaa_df = pd.read_csv(noaa_path)
    merged = _merge_spi_features(noaa_df, spi_path)
    zscores = _compute_zscores(merged)
    extremes = _compute_extreme_values(zscores)
    masked = _apply_seasonal_mask(extremes, crop)
    production = _load_production_weights(crop, DATA_DIR / "production_by_state")

    group_cols = ["state", "week_of_year"]
    tmax_group = zscores.groupby(group_cols)["TMAX_zscore"]
    tmin_group = zscores.groupby(group_cols)["TMIN_zscore"]
    thresholds = zscores[group_cols].drop_duplicates().copy()
    thresholds["TMAX_q3_thresh"] = tmax_group.transform(lambda s: s.quantile(0.75))
    thresholds["TMAX_q4_thresh"] = tmax_group.transform(lambda s: s.quantile(0.90))
    thresholds["TMIN_q4_thresh"] = tmin_group.transform(lambda s: s.quantile(0.10))
    thresholds["TMIN_q3_thresh"] = tmin_group.transform(lambda s: s.quantile(0.25))

    detailed = masked.merge(
        thresholds.drop_duplicates(subset=group_cols),
        on=group_cols,
        how="left",
    )
    detailed["state_name"] = (
        detailed["state"].map(STATE_ABBREV_TO_NAME).fillna(detailed["state"])
    )
    detailed = detailed.merge(
        production,
        left_on=["state_name", "year"],
        right_on=["state", "year"],
        how="left",
        suffixes=("", "_prod"),
    )
    detailed["state_weight"] = pd.to_numeric(
        detailed["state_weight"], errors="coerce"
    ).fillna(0.0)

    week_flags = detailed["week_of_year"].apply(lambda w: crop_season_flag(int(w), crop))
    detailed["is_planting_week"] = week_flags.map(lambda f: f["is_planting_week"])
    detailed["is_harvesting_week"] = week_flags.map(lambda f: f["is_harvesting_week"])

    weighted_base_cols = [
        c for c in detailed.columns if c.endswith(("_in_planting", "_in_harvesting"))
    ]
    for col in weighted_base_cols:
        detailed[f"{col}_weighted"] = detailed[col] * detailed["state_weight"]

    return detailed.loc[detailed["date"] == sample_date].copy()


def _pick_date(
    expected: pd.DataFrame,
    result: pd.DataFrame,
    raw_date: str | None,
) -> pd.Timestamp:
    if raw_date is not None:
        selected = pd.Timestamp(raw_date)
        if expected["date"].eq(selected).any() and result["date"].eq(selected).any():
            return selected
        msg = f"Date {selected.date()} not found in both expected and result datasets."
        raise ValueError(msg)

    overlap = sorted(set(expected["date"]).intersection(set(result["date"])))
    if not overlap:
        msg = "No overlapping dates found between expected and result datasets."
        raise ValueError(msg)
    return overlap[0]


def _is_numeric(value: object) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, NUMERIC_TYPES)


def _compare_rows(
    expected_row: pd.Series,
    result_row: pd.Series,
    shared_cols: list[str],
    atol: float,
) -> list[str]:
    mismatches: list[str] = []
    for col in shared_cols:
        left = expected_row[col]
        right = result_row[col]
        if pd.isna(left) and pd.isna(right):
            continue
        if _is_numeric(left) or _is_numeric(right):
            if not np.isclose(left, right, equal_nan=True, atol=atol, rtol=0.0):
                mismatches.append(f"{col}: expected={left}, result={right}")
            continue
        if left != right:
            mismatches.append(f"{col}: expected={left}, result={right}")
    return mismatches


def _log_row_snapshot(expected_row: pd.Series, result_row: pd.Series) -> None:
    focus_cols = [
        "TMAX_q3_value_in_planting",
        "TMAX_q4_value_in_planting",
        "TMAX_q3_value_in_harvesting",
        "TMAX_q4_value_in_harvesting",
        "TMIN_q3_value_in_planting",
        "TMIN_q4_value_in_planting",
        "TMIN_q3_value_in_harvesting",
        "TMIN_q4_value_in_harvesting",
        "SPI_1m",
        "SPI_3m",
    ]
    _write("")
    _write("Expected vs result sample row (key fields):")
    for col in focus_cols:
        if col in expected_row.index and col in result_row.index:
            _write(f"- {col}: expected={expected_row[col]}, result={result_row[col]}")


def _log_state_diagnostics(state_details: pd.DataFrame) -> None:
    if state_details.empty:
        _write("")
        _write("No state-level diagnostics available for sample date.")
        return

    display_cols = [
        "state",
        "state_name",
        "week_of_year",
        "TMAX_zscore",
        "TMAX_q3_thresh",
        "TMAX_q4_thresh",
        "TMAX_q3_value",
        "TMAX_q4_value",
        "TMIN_zscore",
        "TMIN_q4_thresh",
        "TMIN_q3_thresh",
        "TMIN_q3_value",
        "TMIN_q4_value",
        "is_planting_week",
        "is_harvesting_week",
        "state_weight",
        "TMAX_q3_value_in_planting_weighted",
        "TMAX_q4_value_in_planting_weighted",
        "TMAX_q3_value_in_harvesting_weighted",
        "TMAX_q4_value_in_harvesting_weighted",
        "TMIN_q3_value_in_planting_weighted",
        "TMIN_q4_value_in_planting_weighted",
        "TMIN_q3_value_in_harvesting_weighted",
        "TMIN_q4_value_in_harvesting_weighted",
        "SPI_1m",
        "SPI_1m_weighted",
        "SPI_3m",
        "SPI_3m_weighted",
    ]
    keep = [col for col in display_cols if col in state_details.columns]
    printable = state_details[keep].sort_values("state").reset_index(drop=True)

    _write("")
    _write("State-level diagnostics for sample date:")
    _write(printable.to_string(index=False, max_colwidth=30))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one sample date in an existing weighted NOAA result CSV by "
            "recomputing values from source inputs."
        )
    )
    parser.add_argument(
        "--crop",
        default="corn",
        choices=["corn", "wheat", "soybean"],
        help="Commodity to validate.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date to validate in YYYY-MM-DD. If omitted, first common date is used.",
    )
    parser.add_argument(
        "--noaa_input",
        default=str(DATA_DIR / "climate" / "noaa_weekly.csv"),
        help="Path to NOAA weekly input CSV.",
    )
    parser.add_argument(
        "--spi_input",
        default=str(DATA_DIR / "climate" / "spi_weekly_multiscale.csv"),
        help="Path to SPI CSV.",
    )
    parser.add_argument(
        "--result_csv",
        default=None,
        help="Path to result CSV. If omitted, auto-detects under data/climate.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for numeric comparisons.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result_path = _resolve_result_csv(
        Path(args.result_csv) if args.result_csv is not None else None,
        args.crop,
    )
    expected = _compute_expected_frame(
        args.crop,
        Path(args.noaa_input),
        Path(args.spi_input),
    )
    result = pd.read_csv(result_path)

    expected["date"] = pd.to_datetime(expected["date"], errors="coerce")
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    sample_date = _pick_date(expected, result, args.date)

    expected_row = expected.loc[expected["date"] == sample_date].iloc[0]
    result_row = result.loc[result["date"] == sample_date].iloc[0]
    state_details = _build_state_diagnostics(
        args.crop,
        Path(args.noaa_input),
        Path(args.spi_input),
        sample_date,
    )

    shared_cols = sorted(set(expected.columns).intersection(set(result.columns)) - {"date"})
    extra_expected = sorted(set(expected.columns) - set(result.columns))
    extra_result = sorted(set(result.columns) - set(expected.columns))
    mismatches = _compare_rows(expected_row, result_row, shared_cols, args.atol)

    _write(f"Crop: {args.crop}")
    _write(f"Result CSV: {result_path}")
    _write(f"Sample date checked: {sample_date.date()}")
    _write(f"Shared columns compared: {len(shared_cols)}")
    _write(f"Columns only in expected: {len(extra_expected)}")
    _write(f"Columns only in result: {len(extra_result)}")
    if extra_expected:
        _write(f"Expected-only columns: {', '.join(extra_expected)}")
    if extra_result:
        _write(f"Result-only columns: {', '.join(extra_result)}")
    _log_row_snapshot(expected_row, result_row)
    _log_state_diagnostics(state_details)

    if mismatches:
        _write("")
        _write("Status: FAILED")
        _write(f"Mismatched columns: {len(mismatches)}")
        for item in mismatches[:MAX_DISPLAY_MISMATCHES]:
            _write(f"- {item}")
        if len(mismatches) > MAX_DISPLAY_MISMATCHES:
            remaining = len(mismatches) - MAX_DISPLAY_MISMATCHES
            _write(f"... and {remaining} more")
        raise SystemExit(1)

    _write("")
    _write("Status: OK - all shared columns match for this sample date.")


if __name__ == "__main__":
    main()
