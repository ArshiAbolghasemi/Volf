import argparse
import logging
import os
from pathlib import Path

from src.dataset.news.dataset import DATA_DIR, build_weekly_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="argument parser for news features retriever"
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default="2009-04-13",
        help="start date for retreiving new features",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2025-03-17",
        help="end date for retreiving new features",
    )
    parser.add_argument(
        "--include_gdelt",
        type=bool,
        default=False,
        help="For including GDELT news features",
    )

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    include_gdelt = args.include_gdelt

    df = build_weekly_dataset(
        start_date=start_date, end_date=end_date, include_gdelt=include_gdelt
    )
    Path.mkdir(DATA_DIR / "features.csv", exist_ok=True)
    df.to_csv(DATA_DIR / "news.csv", index=False)

    logger.info("Wrote %d rows -> %s", len(df), "../data/features.csv")
    logger.info("Columns: %s", list(df.columns))


if __name__ == "__main__":
    main()
