import logging
from typing import cast

import pandas as pd
from pandas.core.dtypes.dtypes import NaTType

from src.dataset.news.epu import (
    calculate_categorical_epu_features,
    calculate_epu_index_feature,
    load_categorical_epu,
    load_epu_daily,
)
from src.dataset.news.frbsf import calculate_frbsf_sentiment_feature, load_frbsf_sentiment
from src.dataset.news.gdelt import (
    calculate_gdelt_news_features,
    fetch_gdelt_agriculture_news,
    fetch_gdelt_commodity_news,
    fetch_gdelt_total_news,
)
from src.util.path import DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def generate_week_starts(start_date: str, end_date: str) -> list[pd.Timestamp | NaTType]:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    start_monday = start - pd.Timedelta(days=start.weekday())
    end_monday = end - pd.Timedelta(days=end.weekday())

    weeks = pd.date_range(start=start_monday, end=end_monday, freq="W-MON")
    return [pd.Timestamp(w) for w in weeks]


def build_weekly_dataset(
    *, start_date: str, end_date: str, include_gdelt: bool
) -> pd.DataFrame:
    logger.info("Loading local sources...")
    frbsf_df = load_frbsf_sentiment(DATA_DIR / "frbsf.csv")
    epu_daily_df = load_epu_daily(DATA_DIR / "epu_daily.csv")
    epu_cat_df = load_categorical_epu(DATA_DIR / "categorical_epu_indices.csv")

    wheat = corn = soy = ag = total = None
    if include_gdelt:
        logger.info("Querying GDELT from BigQuery (with caching)...")
        wheat = fetch_gdelt_commodity_news(
            start_date, end_date, "wheat", cache_dir=DATA_DIR
        )
        corn = fetch_gdelt_commodity_news(start_date, end_date, "corn", cache_dir=DATA_DIR)
        soy = fetch_gdelt_commodity_news(
            start_date, end_date, "soybeans", cache_dir=DATA_DIR
        )
        ag = fetch_gdelt_agriculture_news(start_date, end_date, cache_dir=DATA_DIR)
        total = fetch_gdelt_total_news(start_date, end_date, cache_dir=DATA_DIR)

    week_starts = generate_week_starts(start_date, end_date)
    logger.info("Building weekly rows: %d weeks", len(week_starts))

    rows: list[dict] = []
    for ws in week_starts:
        if isinstance(ws, NaTType):
            continue

        row: dict = {"Date": ws}

        row["frbsf_sentiment"] = calculate_frbsf_sentiment_feature(ws, frbsf_df)

        row["epu_index"] = calculate_epu_index_feature(ws, epu_daily_df)
        row.update(calculate_categorical_epu_features(ws, epu_cat_df))

        if include_gdelt:
            datasets = {
                "wheat_news_data": wheat,
                "corn_news_data": corn,
                "soybeans_news_data": soy,
                "ag_news_data": ag,
                "total_news_data": total,
            }
            row.update(calculate_gdelt_news_features(week_start=ws, datasets=datasets))

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

    df = df[
        (df["Date"] >= pd.to_datetime(start_date))
        & (df["Date"] <= pd.to_datetime(end_date))
    ].copy()

    return cast("pd.DataFrame", df)
