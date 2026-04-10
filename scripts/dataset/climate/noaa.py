import argparse
import logging

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


def parse_args() -> tuple[str, str, int]:
    parser = argparse.ArgumentParser(description="Fetch NOAA GHCND climate data")
    parser.add_argument("--startdate", required=True)
    parser.add_argument("--enddate", required=True)
    parser.add_argument("--workers", type=int, default=2)

    args = parser.parse_args()
    return args.startdate, args.enddate, args.workers


def main() -> None:
    startdate, enddate, workers = parse_args()

    raw_df = fetch_all_data(startdate, enddate, workers)
    daily_df = clean_and_aggregate_daily_by_states(raw_df)
    weekly_df = aggregate_weekly(daily_df)

    save_result(daily_df, weekly_df)

    logger.info("==== RUN SUMMARY ====")
    logger.info("API requests: %d", metrics["api_requests"])
    logger.info("Success: %d", metrics["success_api_requests"])
    logger.info("Cache hits: %d", metrics["cache_hits"])
    logger.info("Failures: %d", metrics["failures"])


if __name__ == "__main__":
    main()
