import argparse
import logging
from dataclasses import dataclass

from google.cloud import bigquery

from dataset.bq_query import agriculture_query, commodity_query, total_news_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


PRICE_PER_TB_USD = 5.0
BYTES_PER_GB = 1_000_000_000
BYTES_PER_TB = 1_000_000_000_000


@dataclass(frozen=True)
class QueryCost:
    bytes_processed: int
    gb_processed: float
    tb_processed: float
    estimated_cost_usd: float


def estimate_query_cost(
    client: bigquery.Client,
    query: str,
    label: str,
) -> QueryCost:
    job_config = bigquery.QueryJobConfig(
        dry_run=True,
        use_query_cache=False,
    )
    job = client.query(query, job_config=job_config)

    bytes_processed = int(job.total_bytes_processed or 0)
    gb = bytes_processed / BYTES_PER_GB
    tb = bytes_processed / BYTES_PER_TB
    cost = tb * PRICE_PER_TB_USD

    logger.info(
        "[DRY RUN] %-15s → %8.2f GB  (~$%5.2f)",
        label,
        gb,
        cost,
    )

    return QueryCost(
        bytes_processed=bytes_processed,
        gb_processed=gb,
        tb_processed=tb,
        estimated_cost_usd=cost,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="argument parser for GDELT features cost estimation"
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default="2009-04-13",
        help="start date for retreiving new features",
    )
    parser.add_argument(
        "--end_date",
        action="store_true",
        type=str,
        default="2025-03-17",
        help="start date for retreiving new features",
    )

    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date

    client = bigquery.Client()

    queries: dict[str, str] = {
        "wheat": commodity_query(start_date, end_date, "wheat"),
        "corn": commodity_query(start_date, end_date, "corn"),
        "soybeans": commodity_query(start_date, end_date, "soybeans"),
        "agriculture": agriculture_query(start_date, end_date),
        "total": total_news_query(start_date, end_date),
    }

    logger.info("=" * 72)
    logger.info("BigQuery GDELT COST ESTIMATION")
    logger.info("=" * 72)

    total_cost = 0.0
    total_gb = 0.0

    for name, q in queries.items():
        qc = estimate_query_cost(client, q, name)
        total_cost += qc.estimated_cost_usd
        total_gb += qc.gb_processed

    logger.info("-" * 72)
    logger.info(
        "TOTAL ESTIMATE → %8.2f GB  (~$%5.2f)",
        total_gb,
        total_cost,
    )
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
