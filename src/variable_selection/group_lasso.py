from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from tqdm.auto import tqdm

MIN_SPLITS = 2


@dataclass
class GroupLassoSelectionResult:
    selected_features: list[str]
    coefficients: pd.Series
    info: dict[str, Any]


@dataclass
class GroupLassoSelectionConfig:
    core_columns: list[str] | None = None
    feature_groups: dict[str, list[str]] | None = None
    n_splits: int = 5
    max_train_size: int | None = None
    alphas: np.ndarray | list[float] | int = 30
    max_iter: int = 5_000
    tol: float = 1e-6
    eps: float = 1e-3
    group_weight_power: float = 0.5
    coef_threshold: float = 1e-8
    step_size: float | None = None
    progress_bar: bool = False


@dataclass(frozen=True)
class _GroupPenalty:
    indices: np.ndarray
    weight: float


@dataclass(frozen=True)
class _PreparedGroupDesign:
    feature_cols: list[str]
    core_features: list[str]
    penalized_features: list[str]
    ordered_groups: list[str]
    feature_to_group: dict[str, str]
    group_features: dict[str, list[str]]
    penalties: list[_GroupPenalty]
    x_scaled: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_centered: np.ndarray
    y_mean: float


@dataclass(frozen=True)
class _PredictionTask:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_eval: pd.DataFrame
    alpha: float
    feature_groups: dict[str, list[str]] | None


@dataclass(frozen=True)
class _GroupLassoResultContext:
    y_series: pd.Series
    feature_cols: list[str]
    clean_df: pd.DataFrame
    prepared_full: _PreparedGroupDesign
    cfg: GroupLassoSelectionConfig
    coefficients: pd.Series
    best_alpha: float
    best_mse: float
    best_iteration_count: int
    best_converged: bool
    final_iterations: int
    final_converged: bool
    cv_path: pd.DataFrame


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


def _normalize_feature_groups(
    *,
    penalized_features: list[str],
    feature_groups: dict[str, list[str]] | None,
) -> tuple[list[str], dict[str, str]]:
    normalized_groups: list[str] = []
    feature_to_group: dict[str, str] = {}

    if feature_groups is not None:
        seen_features: set[str] = set()
        for group_name, columns in feature_groups.items():
            available = [col for col in columns if col in penalized_features]
            if not available:
                continue
            duplicate = [col for col in available if col in seen_features]
            if duplicate:
                msg = f"features assigned to multiple groups: {duplicate}"
                raise ValueError(msg)
            normalized_groups.append(group_name)
            for col in available:
                seen_features.add(col)
                feature_to_group[col] = group_name

    for feature in penalized_features:
        if feature in feature_to_group:
            continue
        group_name = f"__singleton__:{feature}"
        normalized_groups.append(group_name)
        feature_to_group[feature] = group_name

    ordered_groups = [
        group_name
        for group_name in normalized_groups
        if any(feature_to_group[col] == group_name for col in penalized_features)
    ]
    return ordered_groups, feature_to_group


def _build_group_index(
    *,
    penalized_features: list[str],
    ordered_groups: list[str],
    feature_to_group: dict[str, str],
    feature_offset: int,
) -> tuple[list[_GroupPenalty], dict[str, list[str]]]:
    group_features: dict[str, list[str]] = {group_name: [] for group_name in ordered_groups}
    for feature in penalized_features:
        group_features[feature_to_group[feature]].append(feature)

    penalties: list[_GroupPenalty] = []
    for group_name in ordered_groups:
        features = group_features[group_name]
        indices = np.array(
            [feature_offset + penalized_features.index(feature) for feature in features],
            dtype=int,
        )
        # Store the raw group cardinality; the configurable exponent
        # (group_weight_power, default 0.5 -> sqrt(size)) is applied in
        # _prepare_design. Pre-raising to **0.5 here double-applied the power.
        penalties.append(_GroupPenalty(indices=indices, weight=float(len(features))))

    return penalties, group_features


def _prepare_design(
    x: pd.DataFrame,
    y: pd.Series,
    core_columns: list[str],
    feature_groups: dict[str, list[str]] | None,
    group_weight_power: float,
) -> _PreparedGroupDesign:
    core_features = [col for col in x.columns if col in core_columns]
    penalized_features = [col for col in x.columns if col not in core_columns]
    ordered_groups, feature_to_group = _normalize_feature_groups(
        penalized_features=penalized_features,
        feature_groups=feature_groups,
    )
    penalties, group_features = _build_group_index(
        penalized_features=penalized_features,
        ordered_groups=ordered_groups,
        feature_to_group=feature_to_group,
        feature_offset=len(core_features),
    )

    feature_cols = core_features + penalized_features
    x_ordered = x[feature_cols].astype(float)
    x_values = x_ordered.to_numpy(dtype=float)
    x_mean = x_values.mean(axis=0)
    x_scale = x_values.std(axis=0, ddof=0)
    x_scale = np.where(x_scale <= 0.0, 1.0, x_scale)
    x_scaled = (x_values - x_mean) / x_scale

    y_values = y.to_numpy(dtype=float)
    y_mean = float(y_values.mean())
    y_centered = y_values - y_mean

    scaled_penalties = [
        _GroupPenalty(
            indices=penalty.indices,
            weight=float(penalty.weight**group_weight_power),
        )
        for penalty in penalties
    ]

    return _PreparedGroupDesign(
        feature_cols=feature_cols,
        core_features=core_features,
        penalized_features=penalized_features,
        ordered_groups=ordered_groups,
        feature_to_group=feature_to_group,
        group_features=group_features,
        penalties=scaled_penalties,
        x_scaled=x_scaled,
        x_mean=x_mean,
        x_scale=x_scale,
        y_centered=y_centered,
        y_mean=y_mean,
    )


def _fit_core_only(
    x_core: np.ndarray,
    y_centered: np.ndarray,
) -> np.ndarray:
    if x_core.size == 0:
        return np.zeros(0, dtype=float)
    coef, *_ = np.linalg.lstsq(x_core, y_centered, rcond=None)
    return coef.astype(float, copy=False)


def _build_alpha_grid(
    prepared: _PreparedGroupDesign,
    cfg: GroupLassoSelectionConfig,
) -> np.ndarray:
    if not isinstance(cfg.alphas, int):
        alpha_values = np.asarray(cfg.alphas, dtype=float)
        if alpha_values.ndim != 1 or alpha_values.size == 0:
            msg = "alphas must be a non-empty 1D array-like."
            raise ValueError(msg)
        return np.sort(alpha_values)[::-1]

    n_alphas = int(cfg.alphas)
    if n_alphas < 1:
        msg = "alphas must be >= 1 when passed as an integer."
        raise ValueError(msg)

    core_dim = len(prepared.core_features)
    x_scaled = prepared.x_scaled
    y_centered = prepared.y_centered

    if not prepared.penalties:
        return np.array([0.0], dtype=float)

    core_coef = _fit_core_only(x_scaled[:, :core_dim], y_centered)
    residual = y_centered.copy()
    if core_dim > 0:
        residual = residual - x_scaled[:, :core_dim] @ core_coef

    n_obs = x_scaled.shape[0]
    alpha_max = 0.0
    for penalty in prepared.penalties:
        group_grad = x_scaled[:, penalty.indices].T @ residual / n_obs
        candidate = float(np.linalg.norm(group_grad, ord=2) / max(penalty.weight, 1e-12))
        alpha_max = max(alpha_max, candidate)

    alpha_max = max(alpha_max, 1e-6)
    alpha_min = max(alpha_max * cfg.eps, 1e-8)
    return np.geomspace(alpha_max, alpha_min, num=n_alphas)


def _lipschitz_constant(x: np.ndarray) -> float:
    spectral_norm = float(np.linalg.norm(x, ord=2))
    n_obs = max(int(x.shape[0]), 1)
    lipschitz = (spectral_norm * spectral_norm) / n_obs
    return max(lipschitz, 1e-8)


def _prox_group_lasso(
    values: np.ndarray,
    *,
    alpha: float,
    step_size: float,
    penalties: list[_GroupPenalty],
    core_dim: int,
) -> np.ndarray:
    updated = values.copy()
    if core_dim > 0:
        updated[:core_dim] = values[:core_dim]

    for penalty in penalties:
        block = values[penalty.indices]
        block_norm = float(np.linalg.norm(block, ord=2))
        threshold = alpha * step_size * penalty.weight
        if block_norm <= threshold:
            updated[penalty.indices] = 0.0
            continue
        updated[penalty.indices] = (1.0 - threshold / block_norm) * block

    return updated


def _fit_group_lasso_coefficients(
    prepared: _PreparedGroupDesign,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
    step_size: float | None,
) -> tuple[np.ndarray, int, bool]:
    x_scaled = prepared.x_scaled
    y_centered = prepared.y_centered
    coef = np.zeros(x_scaled.shape[1], dtype=float)
    aux = coef.copy()
    momentum = 1.0
    effective_step = (
        float(step_size) if step_size is not None else 1.0 / _lipschitz_constant(x_scaled)
    )
    core_dim = len(prepared.core_features)

    converged = False
    iteration = 0
    for _iteration in range(1, max_iter + 1):
        iteration = _iteration
        gradient = x_scaled.T @ (x_scaled @ aux - y_centered) / x_scaled.shape[0]
        prox_input = aux - effective_step * gradient
        updated = _prox_group_lasso(
            prox_input,
            alpha=alpha,
            step_size=effective_step,
            penalties=prepared.penalties,
            core_dim=core_dim,
        )

        max_delta = float(np.max(np.abs(updated - coef))) if updated.size else 0.0
        scale = max(1.0, float(np.max(np.abs(coef))) if coef.size else 0.0)
        if max_delta <= tol * scale:
            coef = updated
            converged = True
            break

        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
        aux = updated + ((momentum - 1.0) / next_momentum) * (updated - coef)
        coef = updated
        momentum = next_momentum

    return coef, iteration, converged


def _predict_from_prepared(
    task: _PredictionTask,
    cfg: GroupLassoSelectionConfig,
) -> tuple[np.ndarray, int, bool]:
    prepared = _prepare_design(
        task.x_train,
        task.y_train,
        cfg.core_columns or [],
        task.feature_groups,
        cfg.group_weight_power,
    )
    coef_scaled, iterations, converged = _fit_group_lasso_coefficients(
        prepared,
        alpha=task.alpha,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        step_size=cfg.step_size,
    )

    x_eval_values = task.x_eval[prepared.feature_cols].astype(float).to_numpy(dtype=float)
    x_eval_scaled = (x_eval_values - prepared.x_mean) / prepared.x_scale
    y_pred = x_eval_scaled @ coef_scaled + prepared.y_mean
    return y_pred, iterations, converged


def _validate_inputs(
    x: pd.DataFrame,
    y: pd.Series,
    cfg: GroupLassoSelectionConfig,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_cols = x.columns.tolist()
    if not feature_cols:
        msg = "x has no feature columns."
        raise ValueError(msg)

    if cfg.n_splits < MIN_SPLITS:
        msg = f"n_splits must be >= {MIN_SPLITS}."
        raise ValueError(msg)
    if cfg.max_train_size is not None and cfg.max_train_size <= 0:
        msg = "max_train_size must be > 0 when provided."
        raise ValueError(msg)

    if cfg.core_columns:
        _validate_forced_features(feature_cols, cfg.core_columns)

    aligned = x.copy()
    aligned["__target__"] = y
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

    return clean_df[feature_cols], clean_df["__target__"], feature_cols


def _run_group_lasso_cv(
    x_clean: pd.DataFrame,
    y_clean: pd.Series,
    prepared_full: _PreparedGroupDesign,
    cfg: GroupLassoSelectionConfig,
) -> tuple[pd.DataFrame, float, float, int, bool]:
    alpha_grid = _build_alpha_grid(prepared_full, cfg)
    cv = TimeSeriesSplit(
        n_splits=cfg.n_splits,
        max_train_size=cfg.max_train_size,
    )

    cv_rows: list[dict[str, float]] = []
    best_alpha = float(alpha_grid[0])
    best_mse = float("inf")
    best_iteration_count = 0
    best_converged = True

    for alpha in tqdm(
        alpha_grid,
        total=len(alpha_grid),
        desc="Group LASSO CV",
        disable=not cfg.progress_bar,
    ):
        fold_mse: list[float] = []
        fold_iterations: list[int] = []
        fold_converged: list[bool] = []
        for train_idx, test_idx in cv.split(x_clean):
            x_train = x_clean.iloc[train_idx]
            y_train = y_clean.iloc[train_idx]
            x_test = x_clean.iloc[test_idx]
            y_test = y_clean.iloc[test_idx]
            y_pred, iterations, converged = _predict_from_prepared(
                _PredictionTask(
                    x_train=x_train,
                    y_train=y_train,
                    x_eval=x_test,
                    alpha=float(alpha),
                    feature_groups=cfg.feature_groups,
                ),
                cfg,
            )
            fold_mse.append(float(np.mean((y_test.to_numpy(dtype=float) - y_pred) ** 2)))
            fold_iterations.append(iterations)
            fold_converged.append(converged)

        mse_mean = float(np.mean(fold_mse))
        mse_std = float(np.std(fold_mse))
        cv_rows.append(
            {
                "alpha": float(alpha),
                "cv_mse_mean": mse_mean,
                "cv_mse_std": mse_std,
            }
        )
        if mse_mean < best_mse:
            best_mse = mse_mean
            best_alpha = float(alpha)
            best_iteration_count = int(np.max(fold_iterations)) if fold_iterations else 0
            best_converged = all(fold_converged) if fold_converged else True

    cv_path = pd.DataFrame(cv_rows).sort_values("alpha", ascending=False)
    return cv_path, best_alpha, best_mse, best_iteration_count, best_converged


def _build_result_info(ctx: _GroupLassoResultContext) -> GroupLassoSelectionResult:
    group_norms: dict[str, float] = {}
    selected_groups: list[str] = []
    for group_name, penalty in zip(
        ctx.prepared_full.ordered_groups,
        ctx.prepared_full.penalties,
        strict=True,
    ):
        norm_value = float(
            np.linalg.norm(
                ctx.coefficients.loc[ctx.prepared_full.feature_cols].to_numpy()[
                    penalty.indices
                ]
                * ctx.prepared_full.x_scale[penalty.indices],
                ord=2,
            )
        )
        group_norms[group_name] = norm_value
        if norm_value > ctx.cfg.coef_threshold:
            selected_groups.append(group_name)

    group_selected_features: list[str] = []
    for group_name in selected_groups:
        group_selected_features.extend(ctx.prepared_full.group_features[group_name])

    selected_features = sorted(
        set(group_selected_features).union(ctx.cfg.core_columns or [])
    )
    dropped_features = sorted(set(ctx.feature_cols) - set(selected_features))

    info: dict[str, Any] = {
        "target_col": (
            ctx.y_series.name if ctx.y_series.name is not None else "__target__"
        ),
        "n_samples_used": len(ctx.clean_df),
        "n_candidate_features": len(ctx.feature_cols),
        "n_penalized_features": len(ctx.prepared_full.penalized_features),
        "n_selected_groups": len(selected_groups),
        "n_selected_by_group_lasso": len(group_selected_features),
        "n_selected_total": len(selected_features),
        "best_alpha": ctx.best_alpha,
        "cv_max_train_size": ctx.cfg.max_train_size,
        "selected_groups": selected_groups,
        "feature_groups": ctx.prepared_full.group_features,
        "feature_to_group": ctx.prepared_full.feature_to_group,
        "group_norms": group_norms,
        "group_lasso_selected_features": sorted(group_selected_features),
        "dropped_features": dropped_features,
        "cv_path": ctx.cv_path,
        "cv_best_mse": ctx.best_mse,
        "cv_best_fit_max_iter": ctx.best_iteration_count,
        "cv_best_fit_converged": ctx.best_converged,
        "final_fit_iterations": ctx.final_iterations,
        "final_fit_converged": ctx.final_converged,
    }

    return GroupLassoSelectionResult(
        selected_features=selected_features,
        coefficients=ctx.coefficients.sort_values(key=np.abs, ascending=False),
        info=info,
    )


def group_lasso_time_series_feature_selection(
    x: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    config: GroupLassoSelectionConfig | None = None,
) -> GroupLassoSelectionResult:
    cfg = config or GroupLassoSelectionConfig()
    y_series = _to_target_series(y)
    x_clean, y_clean, feature_cols = _validate_inputs(x, y_series, cfg)
    clean_df = pd.concat([x_clean, y_clean.rename("__target__")], axis=1)
    prepared_full = _prepare_design(
        x_clean,
        y_clean,
        cfg.core_columns or [],
        cfg.feature_groups,
        cfg.group_weight_power,
    )
    cv_path, best_alpha, best_mse, best_iteration_count, best_converged = (
        _run_group_lasso_cv(
            x_clean,
            y_clean,
            prepared_full,
            cfg,
        )
    )

    coef_scaled, final_iterations, final_converged = _fit_group_lasso_coefficients(
        prepared_full,
        alpha=best_alpha,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
        step_size=cfg.step_size,
    )
    coef_original = coef_scaled / prepared_full.x_scale
    coefficients = pd.Series(
        coef_original,
        index=prepared_full.feature_cols,
        name="coefficient",
    )
    return _build_result_info(
        _GroupLassoResultContext(
            y_series=y_series,
            feature_cols=feature_cols,
            clean_df=clean_df,
            prepared_full=prepared_full,
            cfg=cfg,
            coefficients=coefficients,
            best_alpha=best_alpha,
            best_mse=best_mse,
            best_iteration_count=best_iteration_count,
            best_converged=best_converged,
            final_iterations=final_iterations,
            final_converged=final_converged,
            cv_path=cv_path,
        )
    )


def summarize_group_lasso_selection(
    result: GroupLassoSelectionResult,
) -> pd.DataFrame:
    selected_set = set(result.selected_features)
    feature_to_group = result.info.get("feature_to_group", {})
    summary = result.coefficients.to_frame()
    summary["selected"] = summary.index.map(lambda col: col in selected_set)
    summary["group"] = summary.index.map(feature_to_group.get)
    return summary
