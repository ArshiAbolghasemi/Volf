from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd  # noqa: TC002

from src.variable_selection import (
    BSRSelectionConfig,
    LassoSelectionConfig,
    backward_stepwise_feature_selection,
    lasso_time_series_feature_selection,
)

from .types import HARSelectionConfig  # noqa: TC001


def select_har_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    core_columns: list[str],
    config: HARSelectionConfig,
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
            progress_bar=False,
        )
        lasso_result = lasso_time_series_feature_selection(
            x_train,
            y_train,
            config=lasso_cfg,
        )
        info = {"method": "lasso", **lasso_result.info}
        return lasso_result.selected_features, info

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
