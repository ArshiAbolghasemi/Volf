import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_frbsf_sentiment(filepath: Path) -> pd.DataFrame | None:
    """Load FRBSF Daily News Sentiment Index from local CSV file.

    Args:
        filepath: Path to news_sentiment.csv file

    Returns:
        DataFrame with date index and sentiment values, or None if loading fails

    """
    logger.info("Loading FRBSF sentiment from %s...", filepath)

    if not Path.exists(filepath):
        logger.error("File not found: %s", filepath)
        return None

    try:
        df = pd.read_csv(filepath)
        df["date"] = (
            df["date"]
            .astype(str)
            .str.strip()
            .pipe(pd.to_datetime, format="%m/%d/%Y", errors="coerce")
        )
        df = df.set_index("date")
        df = df.rename(columns={"News Sentiment": "sentiment"})
        logger.info(
            "Loaded FRBSF data: %s days from %s to %s",
            len(df),
            df.index.min(),
            df.index.max(),
        )
    except Exception:
        logger.exception("Error loading FRBSF data")
        return None
    else:
        return df


def calculate_frbsf_sentiment_feature(
    week_start: pd.Timestamp, frbsf_data: pd.DataFrame | None
) -> float:
    """Calculate weekly FRBSF sentiment feature.

    Args:
        week_start: Monday of the week
        frbsf_data: DataFrame with daily FRBSF sentiment

    Returns:
        Weekly average sentiment value, or NaN if data unavailable

    """
    if frbsf_data is None:
        return np.nan

    week_end = week_start + timedelta(days=6)

    try:
        mask = (frbsf_data.index >= week_start) & (frbsf_data.index <= week_end)
        week_data = frbsf_data.loc[mask]

        if len(week_data) > 0 and not week_data["sentiment"].isna().all():
            return float(week_data["sentiment"].mean(skipna=True))
    except Exception:
        logger.exception("Error calculating FRBSF sentiment for week %s", week_start)
        return np.nan
    else:
        return np.nan
