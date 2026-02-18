import argparse
import logging
from pathlib import Path

from src.benchmark import (
    WheatHARBenchmarkConfig,
    benchmark_results_to_frame,
    run_wheat_har_benchmark,
)
from src.util.path import DATA_DIR

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wheat HAR benchmark on ag/v3.csv")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_DIR / "ag" / "v3.csv"),
        help="Path to benchmark input CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "ag" / "har_benchmark_summary.csv"),
        help="Path to save benchmark summary CSV",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="wheat_weekly_rv",
        help="Target column to forecast",
    )
    parser.add_argument(
        "--target_horizon",
        type=int,
        default=1,
        help="Forecast horizon (weeks)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = WheatHARBenchmarkConfig(
        csv_path=args.input,
        target_col=args.target_col,
        target_horizon=args.target_horizon,
    )

    logger.info("Running benchmark with input=%s", cfg.csv_path)
    results = run_wheat_har_benchmark(config=cfg)

    summary = benchmark_results_to_frame(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    logger.info("Benchmark summary saved to %s", output_path)


if __name__ == "__main__":
    main()
