from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

MIN_SPLITS = 2
DEFAULT_START_TRAIN_SIZE = 52
DEFAULT_CORE_COLUMNS = ("RV_weekly", "RV_monthly", "RV_seasonal")


@dataclass
class BSRSelectionResult:
    selected_features: list[str]
    coefficients: pd.Series
    info: dict[str, Any]


@dataclass
class BSRSelectionConfig:
    core_columns: tuple[str, ...] | None = None
    alpha: float = 0.05
    add_constant: bool = True
    hac_maxlags: int | None = None
    window_type: str = "expanding"
    window_size: int | None = None
    step: int = 1
    start_train_size: int | None = None
    min_features: int = 0
    selection_threshold: float = 0.5


def _to_target_series(y: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    if y.shape[1] != 1:
        msg = "y DataFrame must have exactly one column."
        raise ValueError(msg)
    return y.iloc[:, 0]


def _validate_required_columns(
    x: pd.DataFrame,
    required_cols: list[str],
) -> None:
    missing = [col for col in required_cols if col not in x.columns]
    if missing:
        msg = f"missing required columns in x: {missing}"
        raise ValueError(msg)


def _build_windows(
    n_obs: int,
    *,
    window_type: str,
    start_train_size: int,
    step: int,
    window_size: int | None,
) -> list[tuple[int, int]]:
    if step < 1:
        msg = "step must be >= 1."
        raise ValueError(msg)

    if start_train_size >= n_obs:
        msg = (
            f"start_train_size ({start_train_size}) must be smaller than number "
            f"of observations ({n_obs})."
        )
        raise ValueError(msg)

    windows: list[tuple[int, int]] = []
    for train_end in range(start_train_size, n_obs + 1, step):
        if window_type == "expanding":
            train_start = 0
        elif window_type == "rolling":
            if window_size is None or window_size < MIN_SPLITS:
                msg = f"window_size must be >= {MIN_SPLITS} for rolling windows."
                raise ValueError(msg)
            train_start = max(0, train_end - window_size)
        else:
            msg = "window_type must be either 'expanding' or 'rolling'."
            raise ValueError(msg)

        if train_end - train_start >= MIN_SPLITS:
            windows.append((train_start, train_end))

    return windows


def _fit_ols_with_optional_hac(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    add_constant: bool,
    hac_maxlags: int | None,
) -> Any:
    design_matrix = x_train
    if add_constant:
        design_matrix = sm.add_constant(design_matrix, has_constant="add")

    model = sm.OLS(y_train, design_matrix)
    if hac_maxlags is None:
        return model.fit()
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})


def _run_window_backward_elimination(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    forced_columns: list[str],
    candidate_columns: list[str],
    cfg: BSRSelectionConfig,
) -> tuple[list[str], list[str], str | None]:
    remaining_candidates = candidate_columns.copy()
    dropped_features: list[str] = []

    while True:
        model_columns = forced_columns + remaining_candidates
        used_columns = model_columns.copy()

        window_df = x_train[used_columns].copy()
        window_df["__target__"] = y_train
        window_df = window_df.dropna()

        if window_df.empty:
            return [], dropped_features, "window has no usable rows after dropna"

        x_window = cast("pd.DataFrame", window_df[used_columns])
        y_window = cast("pd.Series", window_df["__target__"])

        try:
            fit_result = _fit_ols_with_optional_hac(
                x_train=x_window,
                y_train=y_window,
                add_constant=cfg.add_constant,
                hac_maxlags=cfg.hac_maxlags,
            )
        except Exception as exc:
            return [], dropped_features, f"regression failed: {exc}"

        candidate_pvalues: dict[str, float] = {}
        for feature in remaining_candidates:
            p_value = fit_result.pvalues.get(feature, np.nan)
            if not np.isnan(p_value):
                candidate_pvalues[feature] = float(p_value)

        if not candidate_pvalues:
            return remaining_candidates, dropped_features, None

        worst_feature = max(
            candidate_pvalues, key=lambda feature: candidate_pvalues[feature]
        )
        worst_p_value = candidate_pvalues[worst_feature]

        if worst_p_value <= cfg.alpha:
            return remaining_candidates, dropped_features, None

        if len(remaining_candidates) - 1 < cfg.min_features:
            return remaining_candidates, dropped_features, None

        remaining_candidates.remove(worst_feature)
        dropped_features.append(worst_feature)


def _fit_final_coefficients(
    x: pd.DataFrame,
    y: pd.Series,
    selected_features: list[str],
    *,
    add_constant: bool,
    hac_maxlags: int | None,
) -> pd.Series:
    if not selected_features:
        return pd.Series(dtype=float, name="coefficient")

    fit_df = x[selected_features].copy()
    fit_df["__target__"] = y
    fit_df = fit_df.dropna()
    if fit_df.empty:
        return pd.Series(dtype=float, name="coefficient")

    x_fit = cast("pd.DataFrame", fit_df[selected_features])
    y_fit = cast("pd.Series", fit_df["__target__"])

    fit_result = _fit_ols_with_optional_hac(
        x_train=x_fit,
        y_train=y_fit,
        add_constant=add_constant,
        hac_maxlags=hac_maxlags,
    )

    coefficients = fit_result.params.drop(labels=["const"], errors="ignore")
    coefficients.name = "coefficient"
    return coefficients.sort_values(key=np.abs, ascending=False)


def backward_stepwise_feature_selection(
    x: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    config: BSRSelectionConfig | None = None,
) -> BSRSelectionResult:
    """Time-series-safe p-value backward elimination aggregated across windows."""
    cfg = config or BSRSelectionConfig()
    y_series = _to_target_series(y)

    common_index = x.index.intersection(y_series.index)
    if common_index.empty:
        msg = "x and y have no overlapping index."
        raise ValueError(msg)

    x_aligned = x.loc[common_index].sort_index()
    y_aligned = y_series.loc[common_index].sort_index()

    forced_columns = list(cfg.core_columns or DEFAULT_CORE_COLUMNS)
    _validate_required_columns(x_aligned, forced_columns)

    candidate_columns = [c for c in x_aligned.columns if c not in forced_columns]

    start_train_size = cfg.start_train_size or DEFAULT_START_TRAIN_SIZE
    windows = _build_windows(
        n_obs=len(x_aligned),
        window_type=cfg.window_type,
        start_train_size=start_train_size,
        step=cfg.step,
        window_size=cfg.window_size,
    )

    selection_counter: Counter[str] = Counter()
    total_dropped = 0
    failures_count = 0
    warnings: list[str] = []
    per_window_selected: list[list[str]] = []

    for window_idx, (train_start, train_end) in enumerate(windows):
        x_train = x_aligned.iloc[train_start:train_end]
        y_train = y_aligned.iloc[train_start:train_end]

        kept_candidates, dropped_in_window, failure_message = (
            _run_window_backward_elimination(
                x_train=x_train,
                y_train=y_train,
                forced_columns=forced_columns,
                candidate_columns=candidate_columns,
                cfg=cfg,
            )
        )

        total_dropped += len(dropped_in_window)

        if failure_message is not None:
            failures_count += 1
            warning_msg = (
                f"window {window_idx} [{train_start}:{train_end}] {failure_message}"
            )
            warnings.append(warning_msg)
            logger.warning(warning_msg)
            continue

        for feature in kept_candidates:
            selection_counter[feature] += 1
        per_window_selected.append(forced_columns + kept_candidates)

    successful_windows = len(per_window_selected)

    selection_frequency: dict[str, float] = {}
    for feature in forced_columns:
        selection_frequency[feature] = 1.0 if successful_windows > 0 else 0.0
    for feature in candidate_columns:
        if successful_windows == 0:
            selection_frequency[feature] = 0.0
        else:
            selection_frequency[feature] = selection_counter[feature] / successful_windows

    final_candidates = [
        feature
        for feature in candidate_columns
        if selection_frequency.get(feature, 0.0) >= cfg.selection_threshold
    ]
    final_selected_features = forced_columns + final_candidates

    final_coefficients = _fit_final_coefficients(
        x=x_aligned,
        y=y_aligned,
        selected_features=final_selected_features,
        add_constant=cfg.add_constant,
        hac_maxlags=cfg.hac_maxlags,
    )

    info: dict[str, Any] = {
        "mode": "backward_elimination_pvalue_ts",
        "alpha": cfg.alpha,
        "add_constant": cfg.add_constant,
        "hac_maxlags": cfg.hac_maxlags,
        "window_type": cfg.window_type,
        "window_size": cfg.window_size,
        "step": cfg.step,
        "start_train_size": start_train_size,
        "n_windows_run": len(windows),
        "n_windows_successful": successful_windows,
        "total_dropped_features": total_dropped,
        "final_n_selected": len(final_selected_features),
        "selection_frequency": selection_frequency,
        "final_selected_features": final_selected_features,
        "warnings": warnings,
        "failures_count": failures_count,
        "selection_threshold": cfg.selection_threshold,
        "per_window_selected": per_window_selected,
    }

    return BSRSelectionResult(
        selected_features=final_selected_features,
        coefficients=final_coefficients,
        info=info,
    )


def summarize_bsr_selection(result: BSRSelectionResult) -> pd.DataFrame:
    """Return a compact summary table for selected features."""
    selected_set = set(result.selected_features)
    summary = result.coefficients.to_frame()
    summary["selected"] = summary.index.map(lambda col: col in selected_set)
    return summary
