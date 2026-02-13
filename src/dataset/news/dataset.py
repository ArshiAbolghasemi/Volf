import logging
from typing import cast

import pandas as pd

from src.dataset.news.epu import (
    calculate_weekly_categorical_epu,
    calculate_weekly_epu_index,
    load_categorical_epu,
    load_epu_daily,
)
from src.dataset.news.frbsf import (
    calculate_weekly_frbsf_sentiment,
    load_frbsf_sentiment,
)
from src.util.path import DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def build_dataset(*, start_date: str, end_date: str) -> pd.DataFrame:
    logger.info("Loading local sources...")
    frbsf_daily = load_frbsf_sentiment(DATA_DIR / "news" / "frbsf.csv")
    epu_daily = load_epu_daily(DATA_DIR / "news" / "epu_daily.csv")
    epu_monthly = load_categorical_epu(DATA_DIR / "news" / "categorical_epu_indices.csv")

    logger.info("Resampling to weekly (W-MON)...")
    frbsf_weekly = calculate_weekly_frbsf_sentiment(frbsf_daily)
    epu_weekly = calculate_weekly_epu_index(epu_daily)
    epu_cat_weekly = calculate_weekly_categorical_epu(epu_monthly)

    if frbsf_weekly is None:
        msg = "frbsf_weekly dataset is empty"
        raise ValueError(msg)
    df_frbsf = frbsf_weekly.reset_index().rename(columns={"sentiment": "frbsf_sentiment"})

    if epu_weekly is None:
        msg = "epu_weekly dataset is empty"
        raise ValueError(msg)
    df_epu = epu_weekly.reset_index().rename(columns={"daily_policy_index": "epu_index"})

    if epu_cat_weekly is None:
        msg = "epu_cat_weekly dataset is empty"
        raise ValueError(msg)
    df_epu_cat = epu_cat_weekly.reset_index()

    df = df_frbsf.merge(df_epu, on="Date", how="outer").merge(
        df_epu_cat, on="Date", how="outer"
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()

    logger.info("Built dataset: %d rows, %d columns", len(df), df.shape[1])
    return cast("pd.DataFrame", df)
