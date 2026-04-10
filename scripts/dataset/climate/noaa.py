import argparse
import logging
from datetime import date

import pandas as pd

from src.dataset.climate.noaa import (
    aggregate_weekly,
    clean_and_aggregate_daily_by_states,
    fetch_all_data,
    metrics,
)
from src.util.path import DATA_DIR

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def save_result(daily_df: pd.DataFrame, weekly_df: pd.DataFrame) -> None:
    output_path = DATA_DIR / "climate"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(output_path / "noaa_daily.csv", index=False)
    weekly_df.to_csv(output_path / "noaa_weekly.csv", index=False)
    logger.info("Saved result at %s", output_path)


def parse_args() -> tuple[int, int, int]:
    parser = argparse.ArgumentParser(description="Fetch NOAA GHCND climate data")
    parser.add_argument("--startyear", required=True, type=int)
    parser.add_argument("--endyear", required=True, type=int)
    parser.add_argument("--workers", type=int, default=2)

    args = parser.parse_args()
    if args.startyear > args.endyear:
        msg = f"startyear must be <= endyear. got {args.startyear} > {args.endyear}"
        raise ValueError(msg)
    return args.startyear, args.endyear, args.workers


def build_year_intervals(startyear: int, endyear: int) -> list[tuple[str, str]]:
    intervals: list[tuple[str, str]] = []
    for year in range(startyear, endyear + 1):
        start = date(year, 1, 1).strftime("%Y-%m-%d")
        end = date(year, 12, 30).strftime("%Y-%m-%d")
        intervals.append((start, end))
    return intervals


def main() -> None:
    startyear, endyear, workers = parse_args()
    intervals = build_year_intervals(startyear, endyear)

    logger.info(
        "Fetching NOAA data by yearly intervals from %d to %d (%d intervals)",
        startyear,
        endyear,
        len(intervals),
    )
    raw_chunks: list[pd.DataFrame] = []
    for startdate, enddate in intervals:
        logger.info("Fetching interval %s -> %s", startdate, enddate)
        interval_df = fetch_all_data(startdate, enddate, workers)
        logger.info("Interval %s -> %s rows=%d", startdate, enddate, len(interval_df))
        raw_chunks.append(interval_df)

    raw_df = pd.concat(raw_chunks, ignore_index=True) if raw_chunks else pd.DataFrame()
    logger.info("Total fetched rows across all intervals: %d", len(raw_df))
    daily_df = clean_and_aggregate_daily_by_states(raw_df)
    weekly_df = aggregate_weekly(daily_df)

    save_result(daily_df, weekly_df)

    logger.info("==== RUN SUMMARY ====")
    logger.info("API requests: %d", metrics["api_requests"])
    logger.info("Success: %d", metrics["success_api_requests"])
    logger.info("Cache hits: %d", metrics["cache_hits"])
    logger.info("Failures: %d", metrics["failures"])
    logger.info("Token switches: %d", metrics.get("token_switches", 0))


if __name__ == "__main__":
    main()
