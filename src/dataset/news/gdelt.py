import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery

from src.dataset.news.bq_query import agriculture_query, commodity_query, total_news_query

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _get_cache_path(
    data_dir: Path,
    data_type: str,
    start_date: str,
    end_date: str,
) -> Path:
    cache_dir = data_dir / "gdelt_cache"
    cache_dir.mkdir(exist_ok=True)

    filename = f"gdelt_{data_type}_{start_date}_{end_date}.parquet"
    return cache_dir / filename


def _load_from_cache(cache_path: Path) -> pd.DataFrame | None:
    if cache_path.exists():
        return None
    try:
        logger.info("Loading from cache: %s", cache_path.name)
        gdelt_df = pd.read_parquet(cache_path)
        logger.info("✓ Loaded %s rows from cache", len(gdelt_df))
    except Exception:
        logger.exception("Error loading cache from %s", cache_path.name)
        return None
    else:
        return gdelt_df


def _save_to_cache(gdelt_df: pd.DataFrame, cache_path: Path) -> None:
    try:
        gdelt_df.to_parquet(cache_path, index=False)
        logger.info("✓ Saved to cache: %s", cache_path.name)
    except Exception:
        logger.exception("Error saving cache to %s", cache_path.name)


def fetch_gdelt_commodity_news(
    start_date: str, end_date: str, commodity: str, cache_dir: Path | None = None
) -> pd.DataFrame | None:
    cache_path = None
    if cache_dir is not None:
        cache_path = _get_cache_path(cache_dir, commodity, start_date, end_date)
        cached_df = _load_from_cache(cache_path)
        if cached_df is not None:
            return cached_df

    logger.info(
        "Fetching %s news from GDELT (BigQuery): %s to %s", commodity, start_date, end_date
    )
    logger.info("  This may take a few minutes...")

    try:
        client = bigquery.Client()

        query = commodity_query(
            start_date=start_date, end_date=end_date, commodity=commodity
        )

        gdelt_df = client.query(query).to_dataframe()
        gdelt_df["date"] = pd.to_datetime(gdelt_df["date"])
        gdelt_df = gdelt_df.set_index("date")

        logger.info("✓ Fetched %s news: %d days with data", commodity, len(gdelt_df))

        if cache_path is not None:
            _save_to_cache(gdelt_df, cache_path)
    except Exception:
        logger.exception("✗ Error fetching %d news from GDELT", commodity)
        return None
    else:
        return gdelt_df


def fetch_gdelt_agriculture_news(
    start_date: str, end_date: str, cache_dir: Path | None = None
) -> pd.DataFrame | None:
    cache_path = None
    if cache_dir is not None:
        cache_path = _get_cache_path(cache_dir, "agriculture", start_date, end_date)
        cached_df = _load_from_cache(cache_path)
        if cached_df is not None:
            return cached_df

    logger.info("Fetching agriculture news from GDELT...")

    try:
        client = bigquery.Client()

        query = agriculture_query(start_date=start_date, end_date=end_date)

        gdelt_df = client.query(query).to_dataframe()
        gdelt_df["date"] = pd.to_datetime(gdelt_df["date"])
        gdelt_df = gdelt_df.set_index("date")

        logger.info("✓ Fetched agriculture news: %d days with data", len(gdelt_df))

        if cache_path is not None:
            _save_to_cache(gdelt_df, cache_path)
    except Exception:
        logger.exception("✗ Error fetching agriculture news from GDELT")
        return None
    else:
        return gdelt_df


def fetch_gdelt_total_news(
    start_date: str, end_date: str, cache_dir: Path | None = None
) -> pd.DataFrame | None:
    cache_path = None
    if cache_dir is not None:
        cache_path = _get_cache_path(cache_dir, "total", start_date, end_date)
        cached_df = _load_from_cache(cache_path)
        if cached_df is not None:
            return cached_df

    logger.info("Fetching total news volume from GDELT...")

    try:
        client = bigquery.Client()

        query = total_news_query(start_date=start_date, end_date=end_date)

        gdelt_df = client.query(query).to_dataframe()
        gdelt_df["date"] = pd.to_datetime(gdelt_df["date"])
        gdelt_df = gdelt_df.set_index("date")

        logger.info("✓ Fetched total news volume: %d days with data", len(gdelt_df))

        if cache_path is not None:
            _save_to_cache(gdelt_df, cache_path)

    except Exception:
        logger.exception("✗ Error fetching total news volume from GDELT")
        return None
    else:
        return gdelt_df


def fetch_all_gdelt_data(
    start_date: str, end_date: str, cache_dir: Path | None = None
) -> dict[str, pd.DataFrame | None]:
    logger.info("=" * 70)
    logger.info("Fetching all GDELT data...")
    logger.info("=" * 70)

    wheat_data = fetch_gdelt_commodity_news(start_date, end_date, "wheat", cache_dir)
    corn_data = fetch_gdelt_commodity_news(start_date, end_date, "corn", cache_dir)
    soybeans_data = fetch_gdelt_commodity_news(start_date, end_date, "soybeans", cache_dir)

    ag_data = fetch_gdelt_agriculture_news(start_date, end_date, cache_dir)
    total_data = fetch_gdelt_total_news(start_date, end_date, cache_dir)

    logger.info("=" * 70)
    logger.info("✓ GDELT data fetch complete")
    logger.info("=" * 70)

    return {
        "wheat": wheat_data,
        "corn": corn_data,
        "soybeans": soybeans_data,
        "agriculture": ag_data,
        "total": total_data,
    }


def calculate_gdelt_news_features(
    *,
    week_start: pd.Timestamp,
    datasets: dict[str, pd.DataFrame | None],
) -> dict[str, float]:
    """Calculate GDELT news features for the week for wheat, corn, and soybeans.

    Args:
        week_start: Monday of the week
        wheat_news_data: DataFrame with daily wheat news data
        corn_news_data: DataFrame with daily corn news data
        soybeans_news_data: DataFrame with daily soybeans news data
        ag_news_data: DataFrame with daily agriculture news volume
        total_news_data: DataFrame with daily total news volume

    Returns:
        Dictionary of GDELT news features for all three commodities

    """
    week_end = week_start + timedelta(days=6)
    features = {}

    wheat_news_data = datasets.get("wheat_news_data")
    corn_news_data = datasets.get("corn_news_data")
    soybeans_news_data = datasets.get("soybeans_news_data")
    ag_news_data = datasets.get("ag_news_data")
    total_news_data = datasets.get("total_news_data")

    if ag_news_data is None or total_news_data is None:
        features["news_volume_agriculture"] = np.nan
        features["agricultural_topic_share"] = np.nan
    else:
        ag_mask = (ag_news_data.index >= week_start) & (ag_news_data.index <= week_end)
        total_mask = (total_news_data.index >= week_start) & (
            total_news_data.index <= week_end
        )

        week_ag = ag_news_data.loc[ag_mask]
        week_total = total_news_data.loc[total_mask]

        if len(week_ag) > 0 and not week_ag["ag_volume"].isna().all():
            features["news_volume_agriculture"] = int(week_ag["ag_volume"].sum(skipna=True))
        else:
            features["news_volume_agriculture"] = 0

        if len(week_total) > 0 and len(week_ag) > 0:
            total_vol = week_total["total_volume"].sum(skipna=True)
            ag_vol = week_ag["ag_volume"].sum(skipna=True)
            features["agricultural_topic_share"] = float(
                ag_vol / total_vol if total_vol > 0 else 0.0
            )
        else:
            features["agricultural_topic_share"] = 0.0

    features.update(
        _calculate_commodity_features(
            commodity="wheat",
            week_start=week_start,
            week_end=week_end,
            news_data=wheat_news_data,
        )
    )

    features.update(
        _calculate_commodity_features(
            commodity="corn",
            week_start=week_start,
            week_end=week_end,
            news_data=corn_news_data,
        )
    )

    features.update(
        _calculate_commodity_features(
            commodity="soybeans",
            week_start=week_start,
            week_end=week_end,
            news_data=soybeans_news_data,
        )
    )

    return features


def _calculate_commodity_features(
    *,
    commodity: str,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    news_data: pd.DataFrame | None,
) -> dict[str, float]:
    """Calculate news features for a specific commodity.

    Args:
        commodity: Name of commodity (wheat, corn, soybeans)
        week_start: Monday of the week
        week_end: Sunday of the week
        news_data: DataFrame with daily commodity news data

    Returns:
        Dictionary of news features for the commodity

    """
    features = {}
    prefix = commodity.lower()

    if news_data is None:
        features[f"{prefix}_news_volume"] = np.nan
        features[f"{prefix}_news_sentiment"] = np.nan
        features[f"{prefix}_news_sentiment_volatility"] = np.nan
        features[f"{prefix}_negative_news_count"] = np.nan
        features[f"{prefix}_positive_news_count"] = np.nan
        return features

    mask = (news_data.index >= week_start) & (news_data.index <= week_end)
    week_data = news_data.loc[mask]

    if len(week_data) > 0 and not week_data["volume"].isna().all():
        features[f"{prefix}_news_volume"] = int(week_data["volume"].sum(skipna=True))

        if week_data["volume"].sum(skipna=True) > 0:
            features[f"{prefix}_news_sentiment"] = float(
                week_data["tone"].mean(skipna=True) / 100
            )
        else:
            features[f"{prefix}_news_sentiment"] = 0.0

        if "tone_std" in week_data.columns:
            features[f"{prefix}_news_sentiment_volatility"] = float(
                week_data["tone_std"].mean(skipna=True) / 100
            )
        else:
            features[f"{prefix}_news_sentiment_volatility"] = 0.0

        features[f"{prefix}_negative_news_count"] = int(
            week_data["negative_count"].sum(skipna=True)
        )
        features[f"{prefix}_positive_news_count"] = int(
            week_data["positive_count"].sum(skipna=True)
        )
    else:
        features[f"{prefix}_news_volume"] = 0
        features[f"{prefix}_news_sentiment"] = 0.0
        features[f"{prefix}_news_sentiment_volatility"] = 0.0
        features[f"{prefix}_negative_news_count"] = 0
        features[f"{prefix}_positive_news_count"] = 0

    return features
