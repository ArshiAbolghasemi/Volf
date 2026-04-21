import argparse
import sys
from pathlib import Path
from typing import cast

import pandas as pd


def _write(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def _add_co2_features(co2: pd.DataFrame) -> pd.DataFrame:
    co2 = co2.copy()
    values = co2["co2_molfrac_ppm"]
    co2["zscore_co2"] = (values - values.mean()) / values.std()
    co2["co2_weekly_mean"] = values.shift(1).rolling(4).mean()
    co2["co2_monthly_mean"] = values.shift(1).rolling(13).mean()
    return cast(
        "pd.DataFrame", co2[["date", "zscore_co2", "co2_weekly_mean", "co2_monthly_mean"]]
    )


def _merge(dataset: pd.DataFrame, co2: pd.DataFrame) -> pd.DataFrame:
    if len(co2) < len(dataset):
        msg = f"CO2 rows ({len(co2)}) are fewer than v6 rows ({len(dataset)})."
        raise ValueError(msg)

    features = _add_co2_features(co2).tail(len(dataset)).reset_index(drop=True)
    feature_cols = ["zscore_co2", "co2_weekly_mean", "co2_monthly_mean"]
    return pd.concat([dataset.reset_index(drop=True), features[feature_cols]], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append CO2 features (zscore_co2, co2_weekly_mean, co2_monthly_mean)"
    )
    parser.add_argument(
        "--dataset", default="data/ag/v6.csv", help="Path to v6.csv dataset."
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

    dataset = pd.read_csv(args.dataset)
    co2 = pd.read_csv(args.co2)
    merged = _merge(dataset, co2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    _write(f"Saved: {output_path}")
    _write(f"Rows: {len(merged)}")
    _write(f"Columns: {len(merged.columns)}")


if __name__ == "__main__":
    main()
