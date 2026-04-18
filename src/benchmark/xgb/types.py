from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.benchmark.utils import DEFAULT_CORE_COLUMNS, DEFAULT_TARGET
from src.util.path import DATA_DIR

if TYPE_CHECKING:
    from src.model import XGBRunConfig

DEFAULT_DATA_PATH = DATA_DIR / "ag" / "v4.csv"
DEFAULT_INITIAL_TRAIN_SIZE = 260
DEFAULT_TEST_SIZE = 4
DEFAULT_STEP = 4
DEFAULT_ROLLING_WINDOW_SIZE = 260


@dataclass
class XGBGridSearchConfig:
    enabled: bool = True
    metric: str = "test_r2"
    maximize_metric: bool = True
    initial_train_sizes: list[int] | None = None
    test_sizes: list[int] | None = None
    steps: list[int] | None = None
    n_estimators: list[int] | None = None
    max_depths: list[int] | None = None
    learning_rates: list[float] | None = None
    min_child_weights: list[float] | None = None
    max_candidates: int | None = 60


@dataclass
class WheatXGBBenchmarkConfig:
    csv_path: str = str(DEFAULT_DATA_PATH)
    target_col: str = DEFAULT_TARGET
    core_columns: list[str] | None = None
    target_horizon: int = 1
    target_horizons: list[int] | None = None
    target_mode: Literal["point", "mean"] = "point"
    run_configs: dict[str, XGBRunConfig] | None = None
    grid_search: XGBGridSearchConfig | None = None
    use_cache: bool = True
    cache_dir: str = ".cache/benchmark"
    cache_overwrite: bool = False


def resolve_target_horizons(cfg: WheatXGBBenchmarkConfig) -> list[int]:
    horizons = cfg.target_horizons or [cfg.target_horizon]
    unique_horizons = sorted({int(h) for h in horizons})
    if not unique_horizons:
        msg = "target_horizons cannot be empty."
        raise ValueError(msg)
    if any(h < 0 for h in unique_horizons):
        msg = f"target_horizons must be >= 0. got={unique_horizons}"
        raise ValueError(msg)
    return unique_horizons


__all__ = [
    "DEFAULT_CORE_COLUMNS",
    "DEFAULT_DATA_PATH",
    "DEFAULT_INITIAL_TRAIN_SIZE",
    "DEFAULT_ROLLING_WINDOW_SIZE",
    "DEFAULT_STEP",
    "DEFAULT_TARGET",
    "DEFAULT_TEST_SIZE",
    "WheatXGBBenchmarkConfig",
    "XGBGridSearchConfig",
    "resolve_target_horizons",
]
