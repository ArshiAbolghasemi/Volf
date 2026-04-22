import argparse
import logging

from src.dataset.climate.pdsi import run_pipeline
from src.util.path import DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build crop-level PDSI features from state station files and append "
            "them to data/ag crop datasets."
        )
    )
    parser.add_argument(
        "--palmer_dir",
        type=str,
        default=str(DATA_DIR / "climate" / "palmer"),
        help="Directory containing per-state Palmer station CSV files.",
    )
    parser.add_argument(
        "--production_dir",
        type=str,
        default=str(DATA_DIR / "production_by_state"),
        help="Directory containing crop production-by-state CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional directory where standalone crop PDSI CSVs are written.",
    )
    parser.add_argument(
        "--ag_dir",
        type=str,
        default=str(DATA_DIR / "ag"),
        help=("Directory containing data/ag/{corn,wheat,soybean}.csv to append metrics."),
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
        palmer_dir=args.palmer_dir,
        production_dir=args.production_dir,
        ag_dir=args.ag_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
