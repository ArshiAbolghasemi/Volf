import logging
import time
from pathlib import Path

import pandas as pd
from pytrends import dailydata
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

_MONTH_COUNT = 12


def _month_iter(
    start_year: int, start_mon: int, end_year: int, end_mon: int
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    y, m = start_year, start_mon
    while (y < end_year) or (y == end_year and m <= end_mon):
        out.append((y, m))
        m += 1
        if m == _MONTH_COUNT + 1:
            m = 1
            y += 1
    return out


def _chunk_months(
    months: list[tuple[int, int]], chunk_months: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    chunks: list[tuple[tuple[int, int], tuple[int, int]]] = []
    i = 0
    while i < len(months):
        y1, m1 = months[i]
        y2, m2 = months[min(i + chunk_months - 1, len(months) - 1)]
        chunks.append(((y1, m1), (y2, m2)))
        i += chunk_months
    return chunks


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential(min=1, max=10),
)
def _fetch_data(kw: str, geo: str, y1: int, m1: int, y2: int, m2: int) -> pd.DataFrame:  # noqa: PLR0913
    logger.info(
        "Fetching Google Trends daily data: kw=%s geo=%s %04d-%02d .. %04d-%02d",
        kw,
        geo,
        y1,
        m1,
        y2,
        m2,
    )
    df = dailydata.get_daily_data(
        kw,
        start_year=y1,
        start_mon=m1,
        stop_year=y2,
        stop_mon=m2,
        geo=geo,
    )

    logger.info(
        "Fetched rows=%d cols=%d for %04d-%02d .. %04d-%02d",
        len(df),
        df.shape[1],
        y1,
        m1,
        y2,
        m2,
    )
    return df


def _read_cache(path: Path) -> pd.DataFrame:
    logger.info("Reading cache (parquet): %s", path)
    return pd.read_parquet(path, engine="pyarrow")


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    logger.info("Writing cache (parquet): %s", path)
    df.to_parquet(path, engine="pyarrow")


def get_text_climate_anomaly_w_mon(
    *,
    start_year: int,
    start_mon: int,
    stop_year: int,
    stop_mon: int,
    chunk_months: int = 6,
) -> pd.DataFrame:
    kw = "climate change"
    geo = "US"

    cache_dir = Path(".cache/climate_anomaly_trends")
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Cache directory: %s", cache_dir)

    months = _month_iter(start_year, start_mon, stop_year, stop_mon)
    chunks = _chunk_months(months, chunk_months)

    logger.info(
        "Preparing climate anomaly series: %04d-%02d .. %04d-%02d, chunk_months=%d, chunks=%d",  # noqa: E501
        start_year,
        start_mon,
        stop_year,
        stop_mon,
        chunk_months,
        len(chunks),
    )

    parts: list[pd.DataFrame] = []

    for (y1, m1), (y2, m2) in chunks:
        stem = f"climate_change_US_{y1:04d}{m1:02d}_{y2:04d}{m2:02d}"
        cache_file = cache_dir / (stem + ".parquet")

        if cache_file.exists():
            logger.info("Cache hit: %s", cache_file.name)
            part = _read_cache(cache_file)
        else:
            logger.info("Cache miss: %s", cache_file.name)
            part = _fetch_data(kw, geo, y1, m1, y2, m2)

            if part.index.name != "date":
                part.index.name = "date"

            _write_cache(part, cache_file)

        parts.append(part)
        logger.info("Sleeping 1s to be polite with upstream.")
        time.sleep(1.0)

    logger.info("Concatenating %d parts.", len(parts))
    df = pd.concat(parts).sort_index()

    mean = float(df[kw].mean())
    std = float(df[kw].std())
    df["Text_Climate_Anomaly"] = (df[kw] - mean) / std

    logger.info("Resampling to weekly (W-MON) mean.")
    weekly = (
        df["Text_Climate_Anomaly"]
        .resample("W-MON")
        .mean()
        .rename("Text_Climate_Anomaly")
        .to_frame()
        .reset_index()
    )

    out = weekly[["date", "Text_Climate_Anomaly"]]
    logger.info("Done. Output rows=%d cols=%d", len(out), out.shape[1])
    return out
