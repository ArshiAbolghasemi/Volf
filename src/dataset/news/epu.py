import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_epu_daily(filepath: Path) -> pd.DataFrame | None:
    logger.info("Loading daily EPU from %s...", filepath)

    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        return None

    try:
        epu_df = pd.read_csv(filepath)
        epu_df["date"] = pd.to_datetime(epu_df[["year", "month", "day"]], errors="coerce")
        epu_df = epu_df.dropna(subset=["date"]).set_index("date").sort_index()

        logger.info(
            "Loaded EPU daily data: %s days from %s to %s",
            len(epu_df),
            epu_df.index.min(),
            epu_df.index.max(),
        )
    except Exception:
        logger.exception("Error loading EPU data")
        return None
    else:
        return epu_df


def load_categorical_epu(filepath: Path) -> pd.DataFrame | None:
    """Load categorical EPU indices from local CSV file (monthly)."""
    logger.info("Loading categorical EPU from %s...", filepath)

    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        return None

    try:
        epu_df = pd.read_csv(filepath)

        if len(epu_df) > 0 and (
            epu_df.iloc[-1].isna().all() or "Source:" in str(epu_df.iloc[-1, 0])
        ):
            epu_df = epu_df.iloc[:-1]

        epu_df["date"] = pd.to_datetime(
            epu_df[["Year", "Month"]].assign(Day=1), errors="coerce"
        )
        epu_df = (
            epu_df.dropna(subset=["date"])
            .set_index("date")
            .sort_index()
            .drop(["Year", "Month"], axis=1, errors="ignore")
        )
        epu_df.columns = epu_df.columns.str.strip()

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
        epu_df = epu_df.rename(columns=column_mapping)

        logger.info(
            "Loaded categorical EPU: %s months from %s to %s",
            len(epu_df),
            epu_df.index.min(),
            epu_df.index.max(),
        )
        logger.info("  Categories: %s", list(epu_df.columns))
    except Exception:
        logger.exception("Error loading categorical EPU data")
        return None
    else:
        return epu_df


def calculate_weekly_epu_index(epu_daily_data: pd.DataFrame | None) -> pd.DataFrame | None:
    """Weekly mean of the daily EPU index using resample (W-MON)."""
    if epu_daily_data is None or epu_daily_data.empty:
        logger.warning("EPU daily data is empty or None")
        return None

    if "daily_policy_index" not in epu_daily_data.columns:
        logger.error("Missing required column: daily_policy_index")
        return None

    try:
        weekly = (
            epu_daily_data[["daily_policy_index"]]
            .resample("W-MON")
            .mean()
            .rename_axis("Date")
        )
        logger.info("Computed weekly EPU index: %s weeks", len(weekly))
    except Exception:
        logger.exception("Error resampling EPU daily index")
        return None
    else:
        return weekly


def calculate_weekly_categorical_epu(
    categorical_epu_data: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if categorical_epu_data is None or categorical_epu_data.empty:
        logger.warning("Categorical EPU data is empty or None")
        return None

    try:
        monthly = categorical_epu_data.sort_index()

        daily = monthly.resample("D").asfreq()

        daily = daily.interpolate(method="linear")

        weekly = daily.resample("W-MON").mean().rename_axis("Date")

        logger.info(
            "Computed weekly categorical EPU (linear interpolation): %s weeks", len(weekly)
        )

    except Exception:
        logger.exception("Error resampling categorical EPU")
        return None
    else:
        return weekly
