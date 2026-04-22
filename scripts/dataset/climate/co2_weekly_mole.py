import argparse
import logging
from pathlib import Path
from typing import cast

import pandas as pd

logger = logging.getLogger(__name__)


def _add_co2_features(co2: pd.DataFrame) -> pd.DataFrame:
    co2 = co2.copy()
    values = co2["co2_molfrac_ppm"]
    co2["zscore_co2"] = (values - values.mean()) / values.std()
    co2["co2_weekly_mean"] = co2["zscore_co2"].shift(1).rolling(4).mean()
    co2["co2_monthly_mean"] = co2["zscore_co2"].shift(1).rolling(13).mean()
    return cast(
        "pd.DataFrame", co2[["date", "zscore_co2", "co2_weekly_mean", "co2_monthly_mean"]]
    )


def _merge(dataset: pd.DataFrame, co2: pd.DataFrame) -> pd.DataFrame:
    features = _add_co2_features(co2)
    merged = dataset.merge(features, on="date", how="left")
    missing = (
        merged[["zscore_co2", "co2_weekly_mean", "co2_monthly_mean"]]
        .isna()
        .any(axis=1)
        .sum()
    )
    if missing:
        logger.warning("%d row(s) in dataset had no matching CO2 date.", missing)
    return merged


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Append CO2 features (zscore_co2, co2_weekly_mean, co2_monthly_mean)"
    )
    parser.add_argument(
        "--dataset", default="data/ag/v5.csv", help="Path to v6.csv dataset."
    )
    parser.add_argument(
        "--co2",
        default="data/climate/co2_weekly_mlo.csv",
        help="Path to weekly CO2 source CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/ag/v6.csv",
        help="Output path for merged dataset.",
    )
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset, parse_dates=["date"])
    co2 = pd.read_csv(args.co2, parse_dates=["date"])

    merged = _merge(dataset, co2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    logger.info("Saved: %s", output_path)
    logger.info("Rows: %d", len(merged))
    logger.info("Columns: %d", len(merged.columns))


if __name__ == "__main__":
    main()
