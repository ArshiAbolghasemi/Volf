from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.metrics import clark_west_test

if TYPE_CHECKING:
    from src.model import HARExperimentResult

logger = logging.getLogger(__name__)


@dataclass
class ClarkWestPairConfig:
    model_type: str
    base_feature_set: str
    augmented_feature_set: str
    target_horizon: int
    name: str | None = None
    hac_maxlags: int | None = None


@dataclass
class ClarkWestConfig:
    pairs: list[ClarkWestPairConfig]
    hac_maxlags: int | None = None


def _pair_name(pair: ClarkWestPairConfig) -> str:
    if pair.name:
        return pair.name
    return f"{pair.model_type}:{pair.base_feature_set}->{pair.augmented_feature_set}"


def _is_feature_nested(
    base_result: HARExperimentResult,
    augmented_result: HARExperimentResult,
) -> bool:
    base_cols = set(base_result.model_info.get("extra_feature_cols", []))
    augmented_cols = set(augmented_result.model_info.get("extra_feature_cols", []))
    return base_cols.issubset(augmented_cols)


def _compute_cw_for_split(
    split_name: str,
    y_true: pd.Series,
    y_pred_base: pd.Series,
    y_pred_augmented: pd.Series,
    *,
    hac_maxlags: int | None,
) -> dict[str, Any]:
    cw = clark_west_test(
        y_true=y_true,
        y_pred_base=y_pred_base,
        y_pred_augmented=y_pred_augmented,
        hac_maxlags=hac_maxlags,
    )
    return {f"{split_name}_{k}": v for k, v in cw.items()}


def run_clark_west_by_pairs(
    results_by_horizon: dict[int, dict[str, dict[str, HARExperimentResult]]],
    config: ClarkWestConfig,
) -> pd.DataFrame:
    if not config.pairs:
        msg = "At least one Clark-West pair config is required."
        raise ValueError(msg)

    rows: list[dict[str, Any]] = []

    for pair in config.pairs:
        horizon = int(pair.target_horizon)
        if horizon not in results_by_horizon:
            msg = (
                f"target_horizon={horizon} not found in benchmark results. "
                f"available={sorted(results_by_horizon)}"
            )
            raise ValueError(msg)
        horizon_results = results_by_horizon[horizon]
        if pair.model_type not in horizon_results:
            msg = (
                f"model_type='{pair.model_type}' not found for horizon={horizon}. "
                f"available={sorted(horizon_results)}"
            )
            raise ValueError(msg)

        model_results = horizon_results[pair.model_type]
        if pair.base_feature_set not in model_results:
            msg = (
                f"base_feature_set='{pair.base_feature_set}' missing "
                f"for model={pair.model_type}, horizon={horizon}."
            )
            raise ValueError(msg)
        if pair.augmented_feature_set not in model_results:
            msg = (
                f"augmented_feature_set='{pair.augmented_feature_set}' missing "
                f"for model={pair.model_type}, horizon={horizon}."
            )
            raise ValueError(msg)

        base_result = model_results[pair.base_feature_set]
        augmented_result = model_results[pair.augmented_feature_set]

        pair_hac_maxlags = (
            pair.hac_maxlags if pair.hac_maxlags is not None else config.hac_maxlags
        )

        test_cw = _compute_cw_for_split(
            split_name="test",
            y_true=base_result.y_true_test,
            y_pred_base=base_result.y_pred_test,
            y_pred_augmented=augmented_result.y_pred_test,
            hac_maxlags=pair_hac_maxlags,
        )
        train_cw = _compute_cw_for_split(
            split_name="train",
            y_true=base_result.y_true_train,
            y_pred_base=base_result.y_pred_train,
            y_pred_augmented=augmented_result.y_pred_train,
            hac_maxlags=pair_hac_maxlags,
        )

        row: dict[str, Any] = {
            "pair_name": _pair_name(pair),
            "target_horizon": horizon,
            "model_type": pair.model_type,
            "base_feature_set": pair.base_feature_set,
            "augmented_feature_set": pair.augmented_feature_set,
            "base_train_r2": float(base_result.metrics["train"]["r2"]),
            "augmented_train_r2": float(augmented_result.metrics["train"]["r2"]),
            "delta_train_r2": float(
                augmented_result.metrics["train"]["r2"] - base_result.metrics["train"]["r2"]
            ),
            "base_test_r2": float(base_result.metrics["test"]["r2"]),
            "augmented_test_r2": float(augmented_result.metrics["test"]["r2"]),
            "delta_test_r2": float(
                augmented_result.metrics["test"]["r2"] - base_result.metrics["test"]["r2"]
            ),
            "base_n_selected": len(base_result.selected_features),
            "augmented_n_selected": len(augmented_result.selected_features),
            "is_feature_nested": _is_feature_nested(base_result, augmented_result),
            "hac_maxlags": pair_hac_maxlags,
        }
        row.update(test_cw)
        row.update(train_cw)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    return frame.sort_values(["target_horizon", "pair_name"]).reset_index(drop=True)
