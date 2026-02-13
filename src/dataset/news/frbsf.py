import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_frbsf_sentiment(filepath: Path) -> pd.DataFrame | None:
    logger.info("Loading FRBSF sentiment from %s...", filepath)

    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        return None

    try:
        df = pd.read_csv(filepath)

        df["date"] = pd.to_datetime(
            df["date"].astype(str).str.strip(),
            dayfirst=True,
            errors="coerce",
        )

        df = (
            df.dropna(subset=["date"])
            .rename(columns={"News Sentiment": "sentiment"})
            .set_index("date")
            .sort_index()
        )

        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")

        logger.info(
            "Loaded FRBSF data: %s rows from %s to %s",
            len(df),
            df.index.min(),
            df.index.max(),
        )

    except Exception:
        logger.exception("Error loading FRBSF data")
        return None
    else:
        return df


def calculate_weekly_frbsf_sentiment(
    frbsf_daily: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if frbsf_daily is None or frbsf_daily.empty:
        logger.warning("FRBSF daily data is empty or None")
        return None

    try:
        s = frbsf_daily[["sentiment"]].sort_index()

        full_daily = s.resample("D").asfreq()

        full_daily["sentiment"] = full_daily["sentiment"].interpolate(method="time")

        weekly = full_daily.resample("W-MON").mean().rename_axis("Date")

        logger.info("Computed weekly FRBSF sentiment: %s weeks", len(weekly))

    except Exception:
        logger.exception("Error computing weekly FRBSF sentiment")
        return None
    else:
        return weekly
