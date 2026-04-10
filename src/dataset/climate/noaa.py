import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

TOKEN: str | None = os.getenv("NOAA_TOKEN")
if not TOKEN:
    msg = "NOAA_TOKEN is missing"
    raise ValueError(msg)

BASE_URL: str | None = os.getenv("NOAA_BASE_URL")
if not BASE_URL:
    msg = "NOAA_BASE_URL is missing"
    raise ValueError(msg)

PROXY: str | None = os.getenv("PROXY")
PROXIES: dict[str, str] | None = {"http": PROXY, "https": PROXY} if PROXY else None

HEADERS: dict[str, str] = {"token": TOKEN}

DATATYPES: list[str] = ["PRCP", "TMAX", "TMIN", "TAVG"]

STATE_FIPS: dict[str, str] = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}

CACHE_DIR = Path(".cache/noaa")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

session = requests.Session()
session.headers.update(HEADERS)

metrics: dict[str, int] = {
    "api_requests": 0,
    "success_api_requests": 0,
    "cache_hits": 0,
    "failures": 0,
}

metrics_lock = threading.Lock()


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _request(url: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        metrics["api_requests"] += 1
        r = session.get(url, params=params, proxies=PROXIES, timeout=30)
        r.raise_for_status()

        with metrics_lock:
            metrics["success_api_requests"] += 1

        return r.json()

    except requests.exceptions.RequestException:
        with metrics_lock:
            metrics["failures"] += 1
        raise


def cache_key(url: str, params: dict[str, Any]) -> str:
    key = url + json.dumps(params, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()


def cached_request(url: str, params: dict[str, Any]) -> pd.DataFrame:
    key = cache_key(url, params)
    path = CACHE_DIR / f"{key}.parquet"

    if path.exists():
        with metrics_lock:
            metrics["cache_hits"] += 1

        return pd.read_parquet(path)

    data = _request(url, params)

    results = data.get("results", [])
    result_df = pd.DataFrame(results)

    result_df.to_parquet(path, index=False)

    return result_df


def fetch_state_datatype(
    state: str,
    fips: str,
    datatype: str,
    startdate: str,
    enddate: str,
) -> list[dict[str, Any]]:
    logger.info("Fetching %s for %s", datatype, state)

    params = {
        "datasetid": "GHCND",
        "locationid": f"FIPS:{fips}",
        "datatypeid": datatype,
        "startdate": startdate,
        "enddate": enddate,
        "limit": 1000,
    }

    rows: list[dict[str, Any]] = []
    offset = 1

    while True:
        params["offset"] = offset

        result_df = cached_request(f"{BASE_URL}/data", params)

        if result_df.empty:
            break

        result_df["state"] = state

        rows.extend(result_df.to_dict("records"))

        offset += len(result_df)

    return rows


def fetch_all_data(startdate: str, enddate: str, workers: int) -> pd.DataFrame:
    tasks = [
        (state, fips, datatype)
        for state, fips in STATE_FIPS.items()
        for datatype in DATATYPES
    ]

    rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_state_datatype,
                state,
                fips,
                datatype,
                startdate,
                enddate,
            ): (state, datatype)
            for state, fips, datatype in tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            rows.extend(future.result())

    return pd.DataFrame(rows)


def clean_and_aggregate_daily_by_states(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    working_df = df.copy()

    working_df["date"] = pd.to_datetime(working_df["date"])
    working_df["value"] = working_df["value"] / 10

    return working_df.pivot_table(
        index=["date", "state"],
        columns="datatype",
        values="value",
        aggfunc="mean",
    ).reset_index()


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    working_df = df.copy()
    working_df["date"] = pd.to_datetime(working_df["date"])

    agg_map = {
        c: v
        for c, v in {
            "PRCP": "sum",
            "TMAX": "mean",
            "TMIN": "mean",
            "TAVG": "mean",
        }.items()
        if c in working_df.columns
    }

    result = (
        working_df.set_index("date")
        .groupby("state")
        .resample("W-MON", label="left", closed="left")
        .agg(agg_map)
        .reset_index()
    )

    result["date"] = result["date"].dt.strftime("%Y-%m-%d")

    return result
