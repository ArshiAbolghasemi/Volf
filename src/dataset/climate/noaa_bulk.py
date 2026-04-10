import logging
import math
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from src.dataset.climate.noaa import STATE_FIPS

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GHCN_BULK_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd_all.tar.gz"
STATION_META_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
MISSING_DLY_VALUE = -9999
DEFAULT_DOWNLOAD_WORKERS = 8
RANGE_PART_SIZE_BYTES = 64 * 1024 * 1024

RAW_DIR = Path("data/noaa_raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BulkParseStats:
    total_station_files: int = 0
    eligible_station_files: int = 0
    parsed_station_files: int = 0
    skipped_station_files: int = 0
    failed_station_files: int = 0
    parsed_rows: int = 0


def download_bulk_dataset_stream() -> Path:
    """Stream download the full GHCND archive (~3-4GB).

    Uses streaming so RAM usage stays small.
    """
    path = RAW_DIR / "ghcnd_all.tar.gz"

    logger.info("Downloading GHCND bulk dataset to %s", path)
    start_ts = time.perf_counter()
    supports_range, size_bytes = _probe_range_support(GHCN_BULK_URL)

    if supports_range and size_bytes is not None:
        logger.info(
            ("Using concurrent ranged download | size=%.2f GB workers=%d part_size=%d MB"),
            size_bytes / (1024**3),
            DEFAULT_DOWNLOAD_WORKERS,
            RANGE_PART_SIZE_BYTES // (1024 * 1024),
        )
        _download_concurrent_ranges(
            url=GHCN_BULK_URL,
            output_path=path,
            total_size=size_bytes,
            workers=DEFAULT_DOWNLOAD_WORKERS,
            part_size=RANGE_PART_SIZE_BYTES,
        )
    else:
        logger.info("Server does not support range requests; falling back to streaming")
        _download_stream_fallback(GHCN_BULK_URL, path)

    elapsed = max(time.perf_counter() - start_ts, 1e-9)
    size_on_disk = path.stat().st_size if path.exists() else 0
    logger.info(
        "Download finished: %s (%.2f GB in %.1fs, %.2f MB/s)",
        path,
        size_on_disk / (1024**3),
        elapsed,
        (size_on_disk / (1024**2)) / elapsed,
    )

    return path


def _probe_range_support(url: str) -> tuple[bool, int | None]:
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("HEAD probe failed, using fallback stream download: %s", exc)
        return False, None

    accept_ranges = response.headers.get("Accept-Ranges", "").lower()
    content_length = response.headers.get("Content-Length")
    total_size = (
        int(content_length) if content_length and content_length.isdigit() else None
    )
    return (
        accept_ranges == "bytes" and total_size is not None and total_size > 0,
        total_size,
    )


def _download_stream_fallback(url: str, output_path: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with output_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)


def _download_range_part(
    *,
    url: str,
    part_path: Path,
    start_byte: int,
    end_byte: int,
) -> int:
    headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        if response.status_code not in {200, 206}:
            msg = f"Unexpected status code for ranged download: {response.status_code}"
            raise RuntimeError(msg)

        bytes_written = 0
        with part_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)
                    bytes_written += len(chunk)
    return bytes_written


def _download_concurrent_ranges(
    *,
    url: str,
    output_path: Path,
    total_size: int,
    workers: int,
    part_size: int,
) -> None:
    n_parts = max(1, math.ceil(total_size / part_size))
    parts_dir = output_path.parent / f"{output_path.name}.parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    ranges: list[tuple[int, int, int]] = []
    for part_idx in range(n_parts):
        start_byte = part_idx * part_size
        end_byte = min(total_size - 1, (part_idx + 1) * part_size - 1)
        ranges.append((part_idx, start_byte, end_byte))

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _download_range_part,
                    url=url,
                    part_path=parts_dir / f"part_{part_idx:05d}.bin",
                    start_byte=start_byte,
                    end_byte=end_byte,
                ): part_idx
                for part_idx, start_byte, end_byte in ranges
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="NOAA bulk parts",
            ):
                future.result()

        with output_path.open("wb") as output_file:
            for part_idx in range(n_parts):
                part_path = parts_dir / f"part_{part_idx:05d}.bin"
                with part_path.open("rb") as part_file:
                    shutil.copyfileobj(part_file, output_file)
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)


def extract_bulk_dataset(tar_path: Path) -> Path:
    """Extract archive then delete the tar.gz file."""
    out_dir = RAW_DIR / "ghcnd_all"

    logger.info("Extracting GHCND archive %s into %s", tar_path, out_dir)

    with tarfile.open(tar_path, "r:gz") as tar:
        # Archive source is NOAA official dataset endpoint.
        tar.extractall(out_dir)  # noqa: S202

    logger.info("Extraction finished: %s", out_dir)

    try:
        tar_path.unlink()
        logger.info("Deleted bulk archive to save disk space")
    except OSError as exc:
        logger.warning("Could not delete archive: %s", exc)

    return out_dir


def load_station_states() -> tuple[dict[str, str], Path]:
    """Load station -> state mapping."""
    path = RAW_DIR / "ghcnd-stations.txt"

    if not path.exists():
        logger.info("Downloading station metadata")

        r = requests.get(STATION_META_URL, timeout=60)
        r.raise_for_status()

        path.write_text(r.text)

    station_state: dict[str, str] = {}

    with path.open() as f:
        for line in f:
            station = line[0:11].strip()
            state = line[38:40].strip()

            if state:
                station_state[station] = state

    logger.info("Loaded station metadata: %d station->state mappings", len(station_state))
    return station_state, path


def parse_dly_file(
    filepath: Path,
    station: str,
    state: str,
    start_year: int,
    end_year: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    elements = {"TMIN", "TMAX", "PRCP"}

    with filepath.open() as f:
        for line in f:
            year = int(line[11:15])
            month = int(line[15:17])
            element = line[17:21].strip()

            if element not in elements:
                continue

            if year < start_year or year > end_year:
                continue

            for day in range(31):
                value = int(line[21 + day * 8 : 26 + day * 8])

                if value == MISSING_DLY_VALUE:
                    continue

                try:
                    date = f"{year}-{month:02d}-{day + 1:02d}"
                    pd.Timestamp(date)
                except ValueError:
                    continue

                rows.append(
                    {
                        "station": station,
                        "state": state,
                        "date": date,
                        "datatype": element,
                        "value": value,
                    }
                )

    return rows


def _cleanup_intermediate_files(
    *,
    extracted_dir: Path | None,
    station_meta_path: Path | None,
) -> None:
    if extracted_dir is not None and extracted_dir.exists():
        try:
            shutil.rmtree(extracted_dir)
            logger.info("Deleted extracted NOAA directory: %s", extracted_dir)
        except OSError as exc:
            logger.warning(
                "Could not delete extracted NOAA directory %s: %s", extracted_dir, exc
            )

    if station_meta_path is not None and station_meta_path.exists():
        try:
            station_meta_path.unlink()
            logger.info("Deleted station metadata file: %s", station_meta_path)
        except OSError as exc:
            logger.warning(
                "Could not delete station metadata file %s: %s", station_meta_path, exc
            )


def fetch_bulk_data(
    start_year: int,
    end_year: int,
    workers: int = 8,
) -> pd.DataFrame:
    logger.info(
        "Starting bulk NOAA GHCND pipeline | start_year=%d end_year=%d workers=%d",
        start_year,
        end_year,
        workers,
    )
    data_dir: Path | None = None
    station_meta_path: Path | None = None
    stats = BulkParseStats()
    try:
        tar_path = download_bulk_dataset_stream()
        data_dir = extract_bulk_dataset(tar_path)
        station_states, station_meta_path = load_station_states()

        station_files = list(data_dir.glob("*.dly"))
        stats.total_station_files = len(station_files)
        stats.eligible_station_files = sum(
            1 for path in station_files if station_states.get(path.stem) in STATE_FIPS
        )
        logger.info("Discovered %d .dly station files", stats.total_station_files)
        logger.info(
            "Eligible station files for supported states: %d",
            stats.eligible_station_files,
        )
        rows: list[dict[str, Any]] = []

        def process(path: Path) -> list[dict[str, Any]]:
            station = path.stem
            state = station_states.get(station)
            if state not in STATE_FIPS:
                return []

            return parse_dly_file(
                path,
                station,
                state,
                start_year,
                end_year,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process, p): p for p in station_files}

            for future in tqdm(as_completed(futures), total=len(futures)):
                source_path = futures[future]
                try:
                    parsed_rows = future.result()
                except Exception:
                    stats.failed_station_files += 1
                    logger.exception("Failed parsing %s", source_path.name)
                    continue

                if not parsed_rows:
                    stats.skipped_station_files += 1
                    continue

                stats.parsed_station_files += 1
                stats.parsed_rows += len(parsed_rows)
                rows.extend(parsed_rows)

        result_df = pd.DataFrame(rows)
        logger.info(
            (
                "Bulk parse summary | files_total=%d parsed=%d skipped=%d "
                "failed=%d rows=%d result_rows=%d"
            ),
            stats.total_station_files,
            stats.parsed_station_files,
            stats.skipped_station_files,
            stats.failed_station_files,
            stats.parsed_rows,
            len(result_df),
        )
        return result_df
    finally:
        _cleanup_intermediate_files(
            extracted_dir=data_dir,
            station_meta_path=station_meta_path,
        )
