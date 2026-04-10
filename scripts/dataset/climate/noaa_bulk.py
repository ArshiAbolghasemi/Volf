import argparse
import logging

import pandas as pd

from src.dataset.climate.noaa import aggregate_weekly, clean_and_aggregate_daily_by_states
from src.dataset.climate.noaa_bulk import fetch_bulk_data
from src.util.path import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def save_result(daily_df: pd.DataFrame, weekly_df: pd.DataFrame) -> None:
    output_path = DATA_DIR / "climate"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(output_path / "noaa_bulk_daily.csv", index=False)
    weekly_df.to_csv(output_path / "noaa_bulk_weekly.csv", index=False)
    logger.info("Saved bulk NOAA results at %s", output_path)


def parse_args() -> tuple[int, int, int]:
    parser = argparse.ArgumentParser(description="Fetch NOAA bulk GHCND climate data")
    parser.add_argument("--startyear", required=True, type=int)
    parser.add_argument("--endyear", required=True, type=int)
    parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()
    if args.startyear > args.endyear:
        msg = f"startyear must be <= endyear. got {args.startyear} > {args.endyear}"
        raise ValueError(msg)
    return args.startyear, args.endyear, args.workers


def main() -> None:
    startyear, endyear, workers = parse_args()
    logger.info(
        "Running NOAA bulk pipeline | startyear=%d endyear=%d workers=%d",
        startyear,
        endyear,
        workers,
    )

    result_df = fetch_bulk_data(startyear, endyear, workers=workers)
    logger.info("Raw bulk result rows=%d", len(result_df))

    daily_df = clean_and_aggregate_daily_by_states(result_df)
    weekly_df = aggregate_weekly(daily_df)

    save_result(daily_df, weekly_df)

    logger.info("==== BULK RUN SUMMARY ====")
    logger.info("Raw rows: %d", len(result_df))
    logger.info("Daily rows: %d", len(daily_df))
    logger.info("Weekly rows: %d", len(weekly_df))


if __name__ == "__main__":
    main()
