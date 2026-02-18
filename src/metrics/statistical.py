from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm

EPS = 1e-12
MIN_CW_OBS = 2
SIGNIFICANCE_5PCT = 0.05


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

    y_true_aligned = cast("pd.Series", merged["y_true"])
    y_pred_aligned = cast("pd.Series", merged["y_pred"])

    return y_true_aligned, y_pred_aligned


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def clark_west_test(
    y_true: pd.Series | pd.DataFrame | np.ndarray | list[float],
    y_pred_base: pd.Series | pd.DataFrame | np.ndarray | list[float],
    y_pred_augmented: pd.Series | pd.DataFrame | np.ndarray | list[float],
    *,
    hac_maxlags: int | None = None,
) -> dict[str, float | int | bool]:
    """Clark-West test for nested forecast models.

    Null hypothesis: augmented model is not better than base model.
    Alternative (one-sided): augmented model has lower expected MSPE.
    """
    y_true_s = _to_series(y_true, "y_true").rename("y_true")
    y_base_s = _to_series(y_pred_base, "y_pred_base").rename("y_pred_base")
    y_aug_s = _to_series(y_pred_augmented, "y_pred_augmented").rename("y_pred_augmented")

    merged = pd.concat([y_true_s, y_base_s, y_aug_s], axis=1).dropna()
    if merged.empty:
        msg = "No valid observations after alignment and NaN removal."
        raise ValueError(msg)

    y = cast("pd.Series", merged["y_true"])
    f_base = cast("pd.Series", merged["y_pred_base"])
    f_aug = cast("pd.Series", merged["y_pred_augmented"])

    e_base = y - f_base
    e_aug = y - f_aug

    adjusted_loss_diff = (e_base**2) - ((e_aug**2) - ((f_base - f_aug) ** 2))
    mean_adjusted_diff = float(adjusted_loss_diff.mean())
    n_obs = len(adjusted_loss_diff)

    if n_obs < MIN_CW_OBS:
        return {
            "cw_stat": float("nan"),
            "p_value_one_sided": float("nan"),
            "p_value_two_sided": float("nan"),
            "mean_adjusted_loss_diff": mean_adjusted_diff,
            "n_obs": n_obs,
            "augmented_better_at_5pct": False,
        }

    if hac_maxlags is None:
        se_mean = float(adjusted_loss_diff.std(ddof=1) / np.sqrt(n_obs))
        if not np.isfinite(se_mean) or se_mean <= 0.0:
            cw_stat = float("nan")
        else:
            cw_stat = mean_adjusted_diff / se_mean
    else:
        x_const = np.ones((n_obs, 1), dtype=float)
        fit = sm.OLS(adjusted_loss_diff.to_numpy(dtype=float), x_const).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": int(hac_maxlags)},
        )
        cw_stat = float(fit.tvalues[0])

    if np.isnan(cw_stat):
        p_one_sided = float("nan")
        p_two_sided = float("nan")
        reject = False
    else:
        p_one_sided = float(1.0 - _normal_cdf(cw_stat))
        p_two_sided = float(2.0 * (1.0 - _normal_cdf(abs(cw_stat))))
        reject = bool(p_one_sided < SIGNIFICANCE_5PCT)

    return {
        "cw_stat": float(cw_stat),
        "p_value_one_sided": p_one_sided,
        "p_value_two_sided": p_two_sided,
        "mean_adjusted_loss_diff": mean_adjusted_diff,
        "n_obs": n_obs,
        "augmented_better_at_5pct": reject,
    }


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
