import argparse
import logging
import os

from src.dataset.news.dataset import build_dataset
from src.util.path import DATA_DIR

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

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date

    df = build_dataset(start_date=start_date, end_date=end_date)
    output_path = DATA_DIR / "news" / "news.csv"
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("Wrote %d rows -> %s", len(df), "../data/news/news.csv")
    logger.info("Columns: %s", list(df.columns))


if __name__ == "__main__":
    main()
