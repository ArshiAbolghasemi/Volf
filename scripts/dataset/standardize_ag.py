import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.util.path import DATA_DIR

logger = logging.getLogger(__name__)

CROPS = ("corn", "wheat", "soybean")

# Features that are already standardized (z-scores) - DO NOT standardize again
ALREADY_STANDARDIZED = [
    "zscore_co2",
    "co2_weekly_mean",
    "co2_monthly_mean",
    # Climate extreme values (NOAA preprocessing outputs z-scores)
    "TMAX_q3_value_in_planting",
    "TMAX_q3_value_in_harvesting",
    "TMAX_q4_value_in_planting",
    "TMAX_q4_value_in_harvesting",
    "TMIN_q3_value_in_planting",
    "TMIN_q3_value_in_harvesting",
    "TMIN_q4_value_in_planting",
    "TMIN_q4_value_in_harvesting",
    "AWND_q3_value_in_planting",
    "AWND_q3_value_in_harvesting",
    "AWND_q4_value_in_planting",
    "AWND_q4_value_in_harvesting",
    "weekly_TMAX_q3_value_in_planting",
    "weekly_TMAX_q3_value_in_harvesting",
    "weekly_TMAX_q4_value_in_planting",
    "weekly_TMAX_q4_value_in_harvesting",
    "weekly_TMIN_q3_value_in_planting",
    "weekly_TMIN_q3_value_in_harvesting",
    "weekly_TMIN_q4_value_in_planting",
    "weekly_TMIN_q4_value_in_harvesting",
    "weekly_AWND_q3_value_in_planting",
    "weekly_AWND_q3_value_in_harvesting",
    "weekly_AWND_q4_value_in_planting",
    "weekly_AWND_q4_value_in_harvesting",
    "monthly_TMAX_q3_value_in_planting",
    "monthly_TMAX_q3_value_in_harvesting",
    "monthly_TMAX_q4_value_in_planting",
    "monthly_TMAX_q4_value_in_harvesting",
    "monthly_TMIN_q3_value_in_planting",
    "monthly_TMIN_q3_value_in_harvesting",
    "monthly_TMIN_q4_value_in_planting",
    "monthly_TMIN_q4_value_in_harvesting",
    "monthly_AWND_q3_value_in_planting",
    "monthly_AWND_q3_value_in_harvesting",
    "monthly_AWND_q4_value_in_planting",
    "monthly_AWND_q4_value_in_harvesting",
    # PDSI features (already z-score based)
    "moderate_wet",
    "wet",
    "moderate_dry",
    "dry",
    "weekly_moderate_wet",
    "monthly_moderate_wet",
    "weekly_wet",
    "monthly_wet",
    "weekly_moderate_dry",
    "monthly_moderate_dry",
    "weekly_dry",
    "monthly_dry",
]


def standardize_crop_features(ag_dir: Path, output_dir: Path) -> dict[str, Path]:
    ag_dir = Path(ag_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for crop in CROPS:
        input_path = ag_dir / f"{crop}.csv"
        if not input_path.exists():
            logger.warning("Skipping %s: file not found at %s", crop, input_path)
            continue

        crop_df = pd.read_csv(input_path)
        numeric_cols = crop_df.select_dtypes(include=["number"]).columns.tolist()

        # Identify columns to standardize
        cols_to_standardize = [
            col for col in numeric_cols if col not in ALREADY_STANDARDIZED
        ]

        logger.info(
            "Processing %s: %d total columns, %d to standardize, %d already standardized",
            crop,
            len(crop_df.columns),
            len(cols_to_standardize),
            len([c for c in numeric_cols if c in ALREADY_STANDARDIZED]),
        )

        # Standardize
        scaler = StandardScaler()
        crop_df[cols_to_standardize] = scaler.fit_transform(crop_df[cols_to_standardize])

        output_path = output_dir / f"{crop}_standardized.csv"
        crop_df.to_csv(output_path, index=False)
        outputs[crop] = output_path
        logger.info("Saved standardized %s to %s", crop, output_path)

    return outputs


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    standardize_crop_features(ag_dir=DATA_DIR / "ag", output_dir=DATA_DIR / "ag")


if __name__ == "__main__":
    main()
