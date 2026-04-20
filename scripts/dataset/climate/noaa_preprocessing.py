import argparse
import logging

from src.dataset.climate.noaa_preprocessing import run_pipeline
from src.util.path import DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct weekly NOAA z-scores per state for TMAX, TMIN, and TAVG. "
            "Create crop-specific seasonal features and production-weighted outputs."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_DIR / "climate" / "noaa_weekly.csv"),
        help="Path to input NOAA weekly CSV.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DATA_DIR / "climate"),
        help="Directory where weighted commodity CSVs are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
