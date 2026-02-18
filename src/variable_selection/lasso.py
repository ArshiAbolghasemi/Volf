from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MIN_SPLITS = 2


@dataclass
class LassoSelectionResult:
    selected_features: list[str]
    coefficients: pd.Series
    info: dict[str, Any]


@dataclass
class LassoSelectionConfig:
    core_columns: list[str] | None = None
    n_splits: int = 5
    alphas: np.ndarray | int = 100
    max_iter: int = 50_000
    coef_threshold: float = 1e-10
    random_state: int | None = 42
    n_jobs: int | None = -1


def _build_lasso_pipeline(
    *,
    n_splits: int,
    alphas: np.ndarray | int,
    max_iter: int,
    random_state: int | None,
    n_jobs: int | None,
) -> Pipeline:
    cv = TimeSeriesSplit(n_splits=n_splits)
    lasso = LassoCV(
        alphas=alphas,  # type: ignore[arg-type]
        cv=cv,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", lasso),
        ]
    )


def _validate_forced_features(
    feature_cols: list[str],
    core_columns: list[str],
) -> None:
    missing_forced = [f for f in core_columns if f not in feature_cols]
    if missing_forced:
        msg = f"forced features {missing_forced} is missed"
        raise ValueError(msg)


def _to_target_series(y: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    if y.shape[1] != 1:
        msg = "y DataFrame must have exactly one column."
        raise ValueError(msg)
    return y.iloc[:, 0]


def lasso_time_series_feature_selection(
    x: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    config: LassoSelectionConfig | None = None,
) -> LassoSelectionResult:
    """Run LASSO feature selection with time-series cross-validation.

    Parameters
    ----------
    x:
        DataFrame of candidate features.
    y:
        Target as a pandas Series or single-column DataFrame.
    config:
        Optional LassoSelectionConfig for CV and regularization settings.

    Returns
    -------
    LassoSelectionResult
        selected_features:
            Final features after union of LASSO-selected and forced features.
        coefficients:
            Coefficients indexed by feature name.
        info:
            Diagnostics about CV path, alpha, selected sets, and dropped features.

    """
    cfg = config or LassoSelectionConfig()
    y_series = _to_target_series(y)
    feature_cols = x.columns.tolist()
    if not feature_cols:
        msg = "x has no feature columns."
        raise ValueError(msg)

    if cfg.n_splits < MIN_SPLITS:
        msg = f"n_splits must be >= {MIN_SPLITS}."
        raise ValueError(msg)

    if cfg.core_columns:
        _validate_forced_features(feature_cols, cfg.core_columns)

    aligned = x.copy()
    aligned["__target__"] = y_series
    clean_df = aligned.dropna()
    if clean_df.empty:
        msg = "No rows left after dropping NaNs."
        raise ValueError(msg)

    if len(clean_df) <= cfg.n_splits:
        msg = (
            f"Not enough observations ({len(clean_df)}) for TimeSeriesSplit "
            f"with n_splits={cfg.n_splits}."
        )
        raise ValueError(msg)

    x_clean = clean_df[feature_cols]
    y_clean = clean_df["__target__"]

    model = _build_lasso_pipeline(
        n_splits=cfg.n_splits,
        alphas=cfg.alphas,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    model.fit(x_clean, y_clean)

    lasso: LassoCV = model.named_steps["lasso"]
    coefs = pd.Series(lasso.coef_, index=feature_cols, name="coefficient")

    lasso_selected: list[str] = [
        feature_cols[i]
        for i, coef in enumerate(coefs)
        if abs(float(coef)) > cfg.coef_threshold
    ]

    selected_features: list[str] = sorted(set(lasso_selected).union(cfg.core_columns or []))
    dropped_features = sorted(set(feature_cols) - set(selected_features))

    mse_path = lasso.mse_path_
    mse_mean = mse_path.mean(axis=1)
    mse_std = mse_path.std(axis=1)

    cv_path = pd.DataFrame(
        {
            "alpha": lasso.alphas_,
            "cv_mse_mean": mse_mean,
            "cv_mse_std": mse_std,
        }
    ).sort_values("alpha", ascending=False)

    info: dict[str, Any] = {
        "target_col": y_series.name if y_series.name is not None else "__target__",
        "n_samples_used": len(clean_df),
        "n_candidate_features": len(feature_cols),
        "n_selected_by_lasso": len(lasso_selected),
        "n_selected_total": len(selected_features),
        "best_alpha": float(lasso.alpha_),
        "lasso_selected_features": lasso_selected,
        "dropped_features": dropped_features,
        "cv_path": cv_path,
    }

    return LassoSelectionResult(
        selected_features=selected_features,
        coefficients=coefs.sort_values(key=np.abs, ascending=False),
        info=info,
    )


def summarize_lasso_selection(result: LassoSelectionResult) -> pd.DataFrame:
    """Return a compact summary table for selected features."""
    selected_set = set(result.selected_features)
    summary = result.coefficients.to_frame()
    summary["selected"] = summary.index.map(lambda col: col in selected_set)
    return summary
