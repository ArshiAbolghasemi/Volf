from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

EPS = 1e-12


def _to_series(
    values: pd.Series | pd.DataFrame | np.ndarray | list[float], name: str
) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float)

    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            msg = f"{name} DataFrame must have exactly one column."
            raise ValueError(msg)
        return values.iloc[:, 0].astype(float)

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        msg = f"{name} must be 1-dimensional."
        raise ValueError(msg)
    return pd.Series(arr, name=name, dtype=float)


def _align_and_clean(
    y_true: pd.Series | pd.DataFrame | np.ndarray | list[float],
    y_pred: pd.Series | pd.DataFrame | np.ndarray | list[float],
) -> tuple[pd.Series, pd.Series]:
    y_true_s = _to_series(y_true, "y_true")
    y_pred_s = _to_series(y_pred, "y_pred")

    merged = pd.concat([y_true_s.rename("y_true"), y_pred_s.rename("y_pred")], axis=1)
    merged = merged.dropna()

    if merged.empty:
        msg = "No valid observations after alignment and NaN removal."
        raise ValueError(msg)

    y_pred_aligned = cast("pd.Series", merged["y_true"])
    y_true_aligned = cast("pd.Series", merged["y_pred"])

    return y_true_aligned, y_pred_aligned


def mse(
    y_true: pd.Series | pd.DataFrame | np.ndarray | list[float],
    y_pred: pd.Series | pd.DataFrame | np.ndarray | list[float],
) -> float:
    actual, forecast = _align_and_clean(y_true, y_pred)
    return float(np.mean((actual - forecast) ** 2))


def mae(
    y_true: pd.Series | pd.DataFrame | np.ndarray | list[float],
    y_pred: pd.Series | pd.DataFrame | np.ndarray | list[float],
) -> float:
    actual, forecast = _align_and_clean(y_true, y_pred)
    return float(np.mean(np.abs(actual - forecast)))


def qlike(
    y_true: pd.Series | pd.DataFrame | np.ndarray | list[float],
    y_pred: pd.Series | pd.DataFrame | np.ndarray | list[float],
    *,
    eps: float = EPS,
) -> float:
    """QLIKE loss for volatility forecasts: mean(log(h) + rv / h)."""
    actual, forecast = _align_and_clean(y_true, y_pred)

    actual_clipped = np.clip(actual.to_numpy(dtype=float), eps, None)
    forecast_clipped = np.clip(forecast.to_numpy(dtype=float), eps, None)

    loss = np.log(forecast_clipped) + (actual_clipped / forecast_clipped)
    return float(np.mean(loss))


def r2log(
    y_true: pd.Series | pd.DataFrame | np.ndarray | list[float],
    y_pred: pd.Series | pd.DataFrame | np.ndarray | list[float],
    *,
    eps: float = EPS,
) -> float:
    """R-squared on log scale, commonly used in RV forecasting."""
    actual, forecast = _align_and_clean(y_true, y_pred)

    log_actual = np.log(np.clip(actual.to_numpy(dtype=float), eps, None))
    log_forecast = np.log(np.clip(forecast.to_numpy(dtype=float), eps, None))

    ss_res = float(np.sum((log_actual - log_forecast) ** 2))
    ss_tot = float(np.sum((log_actual - np.mean(log_actual)) ** 2))

    if ss_tot <= 0.0:
        return float("nan")

    return float(1.0 - (ss_res / ss_tot))


def evaluate_statistical_metrics(
    y_true: pd.Series | pd.DataFrame | np.ndarray | list[float],
    y_pred: pd.Series | pd.DataFrame | np.ndarray | list[float],
    *,
    eps: float = EPS,
) -> dict[str, Any]:
    actual, forecast = _align_and_clean(y_true, y_pred)

    return {
        "mse": mse(actual, forecast),
        "mae": mae(actual, forecast),
        "qlike": qlike(actual, forecast, eps=eps),
        "r2log": r2log(actual, forecast, eps=eps),
        "n_obs": len(actual),
    }
