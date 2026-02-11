import logging
import time
from pathlib import Path

from pandas.core.indexing import sys

from src.dataset.climate.config import config
from src.dataset.climate.prism import (
    construct_download_url,
    download_file,
    generate_date_range,
    get_filename,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "climate"


def main() -> int:  # noqa: PLR0915
    logger.info("=" * 80)
    logger.info("Climate Data Downloader Started")
    logger.info("=" * 80)
    logger.info("Date range: %s to %s", config.start_date, config.end_date)
    logger.info("Region: %s", config.region)
    logger.info("Resolution: %s", config.resolution)
    logger.info("Format: %s", config.format)
    logger.info("Elements: %s", config.elements)
    logger.info("=" * 80)

    dates = generate_date_range(config.start_date, config.end_date)
    logger.info("Total days to process: %s", len(dates))

    total_files = len(dates) * len(config.elements)
    successful_downloads = 0
    failed_downloads = 0
    skipped_files = 0

    logger.info("Total files to download: %s", total_files)
    logger.info("-" * 80)

    start_time = time.time()

    for idx, date in enumerate(dates, 1):
        logger.info("Processing date %d/%d: %s", idx, len(dates), date.strftime("%Y-%m-%d"))

        for element in config.elements:
            url = construct_download_url(date, element)

            filename = get_filename(date)
            output_path = DATA_DIR / element / filename

            if output_path.exists():
                logger.info("File already exists, skipping: %s", filename)
                skipped_files += 1
                continue

            try:
                download_file(url, output_path)
            except Exception:
                logger.exception("failed to download %s", url)
                failed_downloads += 1
                continue
            else:
                successful_downloads += 1

            time.sleep(config.delay_between_downloads)

        if idx % config.progress_interval == 0:
            elapsed = time.time() - start_time
            avg_time_per_day = elapsed / idx
            remaining_days = len(dates) - idx
            estimated_remaining = avg_time_per_day * remaining_days

            logger.info(
                "Progress: %d/%d days (%.1f%%)", idx, len(dates), idx / len(dates) * 100
            )
            logger.info(
                "Successful: %d, Failed: %d, Skipped: %d",
                successful_downloads,
                failed_downloads,
                skipped_files,
            )
            logger.info(
                "Estimated time remaining: %.1f hours",
                estimated_remaining / 3600,
            )
            logger.info("-" * 80)

    elapsed_time = time.time() - start_time

    logger.info("=" * 80)
    logger.info("Download Complete!")
    logger.info("=" * 80)
    logger.info("Total files attempted: %d", total_files)
    logger.info("Successfully downloaded: %d", successful_downloads)
    logger.info("Failed downloads: %d", failed_downloads)
    logger.info("Skipped (already exist): %d", skipped_files)
    logger.info("Total time elapsed: %.2f hours", elapsed_time / 3600)
    logger.info("Average time per file: %.2f  seconds", elapsed_time / total_files)
    logger.info("=" * 80)

    if failed_downloads > 0:
        logger.warning("Process completed with %d failures", failed_downloads)
        return 1
    logger.info("All downloads completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
