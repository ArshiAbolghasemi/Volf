from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import statsmodels.api as sm

from src.model.common.preprocessing import (
    build_forecasting_design_matrix,
    split_design_matrix_xy,
)

if TYPE_CHECKING:
    from statsmodels.regression.linear_model import RegressionResultsWrapper

    from .types import HARFeatureConfig


# Backward-compatible aliases while callers migrate to shared naming.
def build_har_design_matrix(
    data: pd.DataFrame,
    config: HARFeatureConfig,
    *,
    target_transform: str = "none",
) -> tuple[pd.DataFrame, list[str], str]:
    return build_forecasting_design_matrix(
        data,
        config,
        target_transform=target_transform,
    )


# Backward-compatible alias while callers migrate to shared naming.
def get_xy_from_har_design(
    design: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    return split_design_matrix_xy(design, target_col)


def fit_har_ols(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: list[str],
    *,
    add_constant: bool,
) -> RegressionResultsWrapper:
    x_fit = x_train[selected_features]
    if add_constant:
        x_fit = sm.add_constant(x_fit, has_constant="add")

    model = sm.OLS(y_train, x_fit)
    return model.fit()


def predict_har_ols(
    fitted_model: RegressionResultsWrapper,
    x: pd.DataFrame,
    selected_features: list[str],
    *,
    add_constant: bool,
) -> pd.Series:
    x_pred = x[selected_features]
    if add_constant:
        x_pred = sm.add_constant(x_pred, has_constant="add")

    pred = fitted_model.predict(x_pred)
    return pd.Series(pred, index=x.index, name="y_pred")
