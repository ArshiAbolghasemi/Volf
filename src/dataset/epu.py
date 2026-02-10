import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_epu_daily(filepath: Path) -> pd.DataFrame | None:
    """Load Daily Economic Policy Uncertainty Index from local CSV file.

    Args:
        filepath: Path to epu_daily.csv file

    Returns:
        DataFrame with date index and EPU values, or None if loading fails

    """
    logger.info("Loading daily EPU from %s...", filepath)

    if not Path.exists(filepath):
        logger.error("File not found: %s", filepath)
        return None

    try:
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        df = df.set_index("date")

        logger.info(
            "Loaded EPU daily data: %s days from %s to %s",
            len(df),
            df.index.min(),
            df.index.max(),
        )
    except Exception:
        logger.exception("Error loading EPU data")
        return None
    else:
        return df


def load_categorical_epu(filepath: Path) -> pd.DataFrame | None:
    """Load categorical EPU indices from local CSV file.

    Args:
        filepath: Path to categorical_epu_indices.csv file

    Returns:
        DataFrame with monthly categorical EPU indices, or None if loading fails

    """
    logger.info("Loading categorical EPU from %s...", filepath)

    if not Path.exists(filepath):
        logger.error("File not found: %s", filepath)
        return None

    try:
        df = pd.read_csv(filepath)
        if df.iloc[-1].isna().all() or "Source:" in str(df.iloc[-1, 0]):
            df = df.iloc[:-1]
        df["date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
        df = df.set_index("date")
        df = df.drop(["Year", "Month"], axis=1, errors="ignore")
        df.columns = df.columns.str.strip()

        column_mapping = {
            "1. Economic Policy Uncertainty": "epu_overall",
            "2. Monetary policy": "epu_monetary",
            "Fiscal Policy (Taxes OR Spending)": "epu_fiscal",
            "3. Taxes": "epu_taxes",
            "4. Government spending": "epu_govt_spending",
            "5. Health care": "epu_healthcare",
            "6. National security": "epu_national_security",
            "7. Entitlement programs": "epu_entitlement",
            "8. Regulation": "epu_regulation",
            "Financial Regulation": "epu_financial_reg",
            "9. Trade policy": "epu_trade",
            "10. Sovereign debt, currency crises": "epu_sovereign_debt",
        }

        df = df.rename(columns=column_mapping)

        logger.info(
            "Loaded categorical EPU: %s months from %s to %s",
            len(df),
            df.index.min(),
            df.index.max(),
        )
        logger.info("  Categories: %s", list(df.columns))
    except Exception:
        logger.exception("Error loading categorical EPU data")
        return None
    else:
        return df


def calculate_epu_index_feature(
    week_start: pd.Timestamp, epu_daily_data: pd.DataFrame | None
) -> float:
    """Calculate weekly EPU index feature.

    Args:
        week_start: Monday of the week
        epu_daily_data: DataFrame with daily EPU index

    Returns:
        Weekly average EPU index, or NaN if data unavailable

    """
    if epu_daily_data is None:
        return np.nan

    week_end = week_start + timedelta(days=6)

    try:
        mask = (epu_daily_data.index >= week_start) & (epu_daily_data.index <= week_end)
        week_data = epu_daily_data.loc[mask]

        if len(week_data) > 0 and not week_data["daily_policy_index"].isna().all():
            return float(week_data["daily_policy_index"].mean(skipna=True))
    except Exception:
        logger.warning("Error calculating EPU index for week %s", week_start)
        return np.nan
    else:
        return np.nan


def calculate_categorical_epu_features(
    week_start: pd.Timestamp, categorical_epu_data: pd.DataFrame | None
) -> dict[str, float]:
    """Calculate categorical EPU features for the week.

    Args:
        week_start: Monday of the week
        categorical_epu_data: DataFrame with monthly categorical EPU indices

    Returns:
        Dictionary of categorical EPU features

    """
    features = {}

    if categorical_epu_data is None:
        epu_cols = [
            "epu_overall",
            "epu_monetary",
            "epu_fiscal",
            "epu_taxes",
            "epu_govt_spending",
            "epu_healthcare",
            "epu_national_security",
            "epu_entitlement",
            "epu_regulation",
            "epu_financial_reg",
            "epu_trade",
            "epu_sovereign_debt",
        ]
        for col in epu_cols:
            features[col] = np.nan
        return features

    month_start = pd.Timestamp(week_start.year, week_start.month, 1)

    if month_start in categorical_epu_data.index:
        month_epu = categorical_epu_data.loc[month_start]
        for col in categorical_epu_data.columns:
            features[col] = float(month_epu[col])
    else:
        prev_months = categorical_epu_data.index[categorical_epu_data.index <= month_start]
        if len(prev_months) > 0:
            closest_month = prev_months[-1]
            month_epu = categorical_epu_data.loc[closest_month]
            for col in categorical_epu_data.columns:
                features[col] = float(month_epu[col])
        else:
            for col in categorical_epu_data.columns:
                features[col] = np.nan

    return features
