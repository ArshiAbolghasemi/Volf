from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from src.variable_selection import (
    BSRSelectionConfig,
    GroupLassoSelectionConfig,
    LassoSelectionConfig,
    backward_stepwise_feature_selection,
    group_lasso_time_series_feature_selection,
    lasso_time_series_feature_selection,
)

if TYPE_CHECKING:
    import pandas as pd

    from .types import HARSelectionConfig, HARWalkForwardConfig


def _resolve_lasso_max_train_size(
    lasso_cfg: LassoSelectionConfig,
    walk_forward_config: HARWalkForwardConfig | None,
) -> int | None:
    if lasso_cfg.max_train_size is not None or walk_forward_config is None:
        return lasso_cfg.max_train_size
    if walk_forward_config.rolling_window_size is not None:
        return walk_forward_config.rolling_window_size
    return walk_forward_config.initial_train_size


def _resolve_group_lasso_max_train_size(
    group_lasso_cfg: GroupLassoSelectionConfig,
    walk_forward_config: HARWalkForwardConfig | None,
) -> int | None:
    if group_lasso_cfg.max_train_size is not None or walk_forward_config is None:
        return group_lasso_cfg.max_train_size
    if walk_forward_config.rolling_window_size is not None:
        return walk_forward_config.rolling_window_size
    return walk_forward_config.initial_train_size


def select_har_features(  # noqa: PLR0913
    x_train: pd.DataFrame,
    y_train: pd.Series,
    core_columns: list[str],
    feature_groups: dict[str, list[str]] | None,
    config: HARSelectionConfig,
    walk_forward_config: HARWalkForwardConfig | None = None,
) -> tuple[list[str], dict[str, Any]]:
    missing_core = [col for col in core_columns if col not in x_train.columns]
    if missing_core:
        msg = f"core columns missing from training design matrix: {missing_core}"
        raise ValueError(msg)

    if config.method == "none":
        selected = x_train.columns.tolist()
        return selected, {"method": "none", "selected_features": selected}

    if config.method == "lasso":
        lasso_cfg = config.lasso or LassoSelectionConfig()
        lasso_cfg = replace(
            lasso_cfg,
            core_columns=core_columns,
            max_train_size=_resolve_lasso_max_train_size(
                lasso_cfg,
                walk_forward_config,
            ),
            progress_bar=False,
        )
        lasso_result = lasso_time_series_feature_selection(
            x_train,
            y_train,
            config=lasso_cfg,
        )
        info = {"method": "lasso", **lasso_result.info}
        return lasso_result.selected_features, info

    if config.method == "group_lasso":
        group_lasso_cfg = config.group_lasso or GroupLassoSelectionConfig()
        group_lasso_cfg = replace(
            group_lasso_cfg,
            core_columns=core_columns,
            feature_groups=feature_groups,
            max_train_size=_resolve_group_lasso_max_train_size(
                group_lasso_cfg,
                walk_forward_config,
            ),
            progress_bar=False,
        )
        group_lasso_result = group_lasso_time_series_feature_selection(
            x_train,
            y_train,
            config=group_lasso_cfg,
        )
        info = {"method": "group_lasso", **group_lasso_result.info}
        return group_lasso_result.selected_features, info

    if config.method == "bsr":
        bsr_cfg = config.bsr or BSRSelectionConfig()
        # In HAR walk-forward, run BSR on current train window only.
        bsr_cfg = replace(
            bsr_cfg,
            core_columns=tuple(core_columns),
            window_type="full",
            progress_bar=False,
        )
        bsr_result = backward_stepwise_feature_selection(x_train, y_train, config=bsr_cfg)
        info = {"method": "bsr", **bsr_result.info}
        return bsr_result.selected_features, info

    msg = f"unsupported feature selection method: {config.method}"
    raise ValueError(msg)
