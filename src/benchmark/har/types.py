from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.util.path import DATA_DIR

if TYPE_CHECKING:
    from src.model import HARRunConfig

DEFAULT_DATA_PATH = DATA_DIR / "ag" / "v4.csv"
DEFAULT_TARGET = "wheat_weekly_rv"
DEFAULT_CORE_COLUMNS = ["wheat_weekly_rv", "wheat_monthly_rv", "wheat_seasonal_rv"]

CLIMATE_COLUMNS = [
    "ssta_elino",
    "ssta_lanina",
    "dry",
    "wet",
    "SOI_index",
    "NAO_index",
    "Text_Climate_Anomaly",
]

NEWS_BASE_COLUMNS = ["frbsf_sentiment", "Text_Climate_Anomaly"]
MACRO_COLUMNS = ["DJIA_Index", "WTI_Index", "Broad_Dollar_index", "Stock_Uncertainty"]
DEFAULT_INITIAL_TRAIN_SIZE = 260
DEFAULT_TEST_SIZE = 4
DEFAULT_STEP = 4
DEFAULT_ROLLING_WINDOW_SIZE = 260


@dataclass
class HARGridSearchConfig:
    enabled: bool = False
    metric: str = "test_mse"
    maximize_metric: bool = False
    initial_train_sizes: list[int] | None = None
    test_sizes: list[int] | None = None
    steps: list[int] | None = None
    max_candidates: int | None = 50


@dataclass
class WheatHARBenchmarkConfig:
    csv_path: str = str(DEFAULT_DATA_PATH)
    target_col: str = DEFAULT_TARGET
    core_columns: list[str] | None = None
    target_horizon: int = 1
    target_horizons: list[int] | None = None
    run_configs: dict[str, HARRunConfig] | None = None
    grid_search: HARGridSearchConfig | None = None
    parallel_jobs: int = 1
    use_cache: bool = True
    cache_dir: str = ".cache/benchmark"
    cache_overwrite: bool = False


def resolve_target_horizons(cfg: WheatHARBenchmarkConfig) -> list[int]:
    horizons = cfg.target_horizons or [cfg.target_horizon]
    unique_horizons = sorted({int(h) for h in horizons})
    if not unique_horizons:
        msg = "target_horizons cannot be empty."
        raise ValueError(msg)
    if any(h < 0 for h in unique_horizons):
        msg = f"target_horizons must be >= 0. got={unique_horizons}"
        raise ValueError(msg)
    return unique_horizons
