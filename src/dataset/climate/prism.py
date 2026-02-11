import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from src.dataset.climate.config import config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def generate_date_range(start_date: str, end_date: str) -> list[datetime]:
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)

    return dates


def get_filename(date: datetime) -> str:
    return config.filename_template.format(
        region=config.region,
        resolution=config.resolution,
        time_period=date.strftime("%Y%m%d"),
    )


def construct_download_url(date: datetime, element: str) -> str:
    return config.url_template.format(
        base_url=config.base_url,
        region=config.region,
        resolution=config.resolution,
        element=element,
        date=date.strftime("%Y%m%d"),
        format=config.format,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(config.max_retries),
    wait=wait_exponential(min=1, max=10),
)
def download_file(url: str, output_path: Path) -> None:
    logger.info("download %s", url)

    response = requests.get(url, timeout=config.timeout, stream=True)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Path.open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=config.chunk_size):
            if chunk:
                f.write(chunk)

    file_size = output_path.stat().st_size
    logger.info("Successfully downloaded: %s (%d bytes)", url, file_size)
