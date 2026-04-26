import argparse
import logging

from src.dataset.climate.preprocessing import run_pipeline
from src.util.path import DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified climate preprocessing for NOAA/SPI/PDSI/CO2. "
            "Outputs crop-specific files merged with v6 features."
        )
    )
    parser.add_argument(
        "--v6_path",
        type=str,
        default=str(DATA_DIR / "ag" / "v6.csv"),
        help="Path to input v6.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DATA_DIR / "ag"),
        help="Output directory for corn.csv, soybean.csv, wheat.csv.",
    )
    parser.add_argument(
        "--noaa_path",
        type=str,
        default=str(DATA_DIR / "climate" / "noaa_weekly.csv"),
        help="Path to NOAA weekly file.",
    )
    parser.add_argument(
        "--spi_path",
        type=str,
        default=str(DATA_DIR / "climate" / "spi_weekly_multiscale.csv"),
        help="Path to SPI weekly file.",
    )
    parser.add_argument(
        "--co2_path",
        type=str,
        default=str(DATA_DIR / "climate" / "co2_weekly_mlo.csv"),
        help="Path to weekly CO2 file.",
    )
    parser.add_argument(
        "--palmer_dir",
        type=str,
        default=str(DATA_DIR / "climate" / "palmer"),
        help="Path to Palmer directory.",
    )
    parser.add_argument(
        "--production_dir",
        type=str,
        default=str(DATA_DIR / "production_by_state"),
        help="Path to production-by-state directory.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_pipeline(
        v6_path=args.v6_path,
        output_dir=args.output_dir,
        noaa_path=args.noaa_path,
        spi_path=args.spi_path,
        co2_path=args.co2_path,
        palmer_dir=args.palmer_dir,
        production_dir=args.production_dir,
    )


if __name__ == "__main__":
    main()
