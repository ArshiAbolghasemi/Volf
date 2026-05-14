from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBRFRegressor

from src.model import (
    RFFeatureConfig,
    RFModelConfig,
    RFRunConfig,
    RFWalkForwardConfig,
    XGBFeatureConfig,
    XGBModelConfig,
    XGBRunConfig,
    XGBWalkForwardConfig,
)
from src.model.common.preprocessing import (
    build_forecasting_design_matrix,
    build_walk_forward_windows,
    inverse_transform_prediction,
    log_transform_rv_features,
    split_design_matrix_xy,
    standardize_train_test,
    transform_target,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TreeShapJobConfig:
    model_type: str
    feature_set: str
    target_horizon: int
    target_mode: Literal["point", "mean"] = "point"
    name: str | None = None
    split: Literal["test", "train"] = "test"
    include_features: list[str] | None = None
    top_n_features: int = 20
    dependence_features: list[str] | None = None
    dependence_top_n: int = 3
    waterfall_row: int | None = None
    max_background_samples: int = 200


@dataclass
class TreeShapConfig:
    jobs: list[TreeShapJobConfig]
    output_subdir: str = "shap"


@dataclass
class TreeShapJobResult:
    summary: pd.DataFrame
    shap_values: pd.DataFrame
    feature_data: pd.DataFrame
    base_values: pd.Series
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _TreeShapRunner:
    fit_model: Any
    model_kind: Literal["rf", "xgb"]


def resolve_rf_run_config_for_shap_job(
    *,
    base_run_cfg: RFRunConfig,
    model_info: dict[str, Any],
) -> RFRunConfig:
    wf_base = base_run_cfg.walk_forward or RFWalkForwardConfig()
    model_base = base_run_cfg.model or RFModelConfig()
    wf = replace(
        wf_base,
        window_type=str(model_info.get("window_type", wf_base.window_type)),
        initial_train_size=int(
            model_info.get("initial_train_size", wf_base.initial_train_size)
        ),
        test_size=int(model_info.get("test_size", wf_base.test_size)),
        step=int(model_info.get("step", wf_base.step)),
        rolling_window_size=model_info.get(
            "rolling_window_size", wf_base.rolling_window_size
        ),
        progress_bar=False,
    )
    model_cfg = replace(
        model_base,
        n_estimators=int(model_info.get("rf_n_estimators", model_base.n_estimators)),
        backend=str(model_info.get("rf_backend", model_base.backend)),
        device=str(model_info.get("rf_device", model_base.device)),
        criterion=str(model_info.get("rf_criterion", model_base.criterion)),
        max_depth=model_info.get("rf_max_depth", model_base.max_depth),
        min_samples_split=int(
            model_info.get("rf_min_samples_split", model_base.min_samples_split)
        ),
        min_samples_leaf=int(
            model_info.get("rf_min_samples_leaf", model_base.min_samples_leaf)
        ),
        max_features=model_info.get("rf_max_features", model_base.max_features),
        bootstrap=bool(model_info.get("rf_bootstrap", model_base.bootstrap)),
        random_state=model_info.get("rf_random_state", model_base.random_state),
        n_jobs=model_info.get("rf_n_jobs", model_base.n_jobs),
        standardize_features=bool(
            model_info.get("model_standardize_features", model_base.standardize_features)
        ),
        target_transform=str(
            model_info.get("model_target_transform", model_base.target_transform)
        ),
        prediction_floor=float(
            model_info.get("model_prediction_floor", model_base.prediction_floor)
        ),
        log_transform_rv_features=bool(
            model_info.get(
                "model_log_transform_rv_features", model_base.log_transform_rv_features
            )
        ),
        feature_floor=float(
            model_info.get("model_feature_floor", model_base.feature_floor)
        ),
    )
    if model_cfg.max_depth is not None:
        model_cfg = replace(model_cfg, max_depth=int(model_cfg.max_depth))
    return RFRunConfig(walk_forward=wf, model=model_cfg)


def resolve_xgb_run_config_for_shap_job(
    *,
    base_run_cfg: XGBRunConfig,
    model_info: dict[str, Any],
) -> XGBRunConfig:
    wf_base = base_run_cfg.walk_forward or XGBWalkForwardConfig()
    model_base = base_run_cfg.model or XGBModelConfig()
    wf = replace(
        wf_base,
        window_type=str(model_info.get("window_type", wf_base.window_type)),
        initial_train_size=int(
            model_info.get("initial_train_size", wf_base.initial_train_size)
        ),
        test_size=int(model_info.get("test_size", wf_base.test_size)),
        step=int(model_info.get("step", wf_base.step)),
        rolling_window_size=model_info.get(
            "rolling_window_size", wf_base.rolling_window_size
        ),
        progress_bar=False,
    )
    model_cfg = replace(
        model_base,
        n_estimators=int(model_info.get("xgb_n_estimators", model_base.n_estimators)),
        max_depth=int(model_info.get("xgb_max_depth", model_base.max_depth)),
        learning_rate=float(model_info.get("xgb_learning_rate", model_base.learning_rate)),
        subsample=float(model_info.get("xgb_subsample", model_base.subsample)),
        colsample_bytree=float(
            model_info.get("xgb_colsample_bytree", model_base.colsample_bytree)
        ),
        min_child_weight=float(
            model_info.get("xgb_min_child_weight", model_base.min_child_weight)
        ),
        reg_alpha=float(model_info.get("xgb_reg_alpha", model_base.reg_alpha)),
        reg_lambda=float(model_info.get("xgb_reg_lambda", model_base.reg_lambda)),
        objective=str(model_info.get("xgb_objective", model_base.objective)),
        random_state=model_info.get("xgb_random_state", model_base.random_state),
        n_jobs=model_info.get("xgb_n_jobs", model_base.n_jobs),
        standardize_features=bool(
            model_info.get("model_standardize_features", model_base.standardize_features)
        ),
        target_transform=str(
            model_info.get("model_target_transform", model_base.target_transform)
        ),
        prediction_floor=float(
            model_info.get("model_prediction_floor", model_base.prediction_floor)
        ),
        log_transform_rv_features=bool(
            model_info.get(
                "model_log_transform_rv_features", model_base.log_transform_rv_features
            )
        ),
        feature_floor=float(
            model_info.get("model_feature_floor", model_base.feature_floor)
        ),
    )
    return XGBRunConfig(walk_forward=wf, model=model_cfg)


def _pick_report_features(shap_values: pd.DataFrame, job: TreeShapJobConfig) -> list[str]:
    if shap_values.empty:
        return []
    if job.include_features:
        selected = [feat for feat in job.include_features if feat in shap_values.columns]
        if selected:
            return selected
    mean_abs = shap_values.abs().mean().sort_values(ascending=False)
    return cast("list[str]", mean_abs.head(max(job.top_n_features, 1)).index.tolist())


def _build_summary_frame(shap_values: pd.DataFrame) -> pd.DataFrame:
    if shap_values.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_shap"])

    all_features = shap_values.abs().mean().sort_values(ascending=False).index.tolist()
    return pd.DataFrame(
        {
            "feature": all_features,
            "mean_abs_shap": [float(shap_values[f].abs().mean()) for f in all_features],
            "mean_shap": [float(shap_values[f].mean()) for f in all_features],
        }
    )


def _fit_rf(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_cfg: RFModelConfig,
) -> Any:
    if model_cfg.backend == "sklearn":
        if model_cfg.device != "cpu":
            msg = (
                "sklearn RandomForestRegressor is CPU-only. "
                "Use backend='xgboost_rf' with device='cuda' for GPU RF runs."
            )
            raise ValueError(msg)
        model = RandomForestRegressor(
            n_estimators=model_cfg.n_estimators,
            criterion=model_cfg.criterion,
            max_depth=model_cfg.max_depth,
            min_samples_split=model_cfg.min_samples_split,
            min_samples_leaf=model_cfg.min_samples_leaf,
            max_features=model_cfg.max_features,  # type: ignore[arg-type]
            bootstrap=model_cfg.bootstrap,
            random_state=model_cfg.random_state,
            n_jobs=model_cfg.n_jobs,
        )
        model.fit(x_train, y_train)
        return model

    if model_cfg.backend != "xgboost_rf":
        msg = f"Unsupported RF backend: {model_cfg.backend}"
        raise ValueError(msg)

    n_features = x_train.shape[1]
    if model_cfg.max_features is None:
        colsample_bynode = 1.0
    elif isinstance(model_cfg.max_features, str):
        if model_cfg.max_features == "sqrt":
            colsample_bynode = min(
                max(math.sqrt(n_features) / n_features, 1.0 / n_features),
                1.0,
            )
        elif model_cfg.max_features == "log2":
            colsample_bynode = min(
                max(math.log2(max(n_features, 2)) / n_features, 1.0 / n_features),
                1.0,
            )
        else:
            msg = (
                f"Unsupported max_features string "
                f"for XGBRF backend: {model_cfg.max_features}"
            )
            raise ValueError(msg)
    elif isinstance(model_cfg.max_features, int):
        colsample_bynode = min(
            max(model_cfg.max_features / n_features, 1.0 / n_features), 1.0
        )
    else:
        colsample_bynode = min(max(float(model_cfg.max_features), 1.0 / n_features), 1.0)

    model = XGBRFRegressor(
        n_estimators=model_cfg.n_estimators,
        max_depth=6 if model_cfg.max_depth is None else int(model_cfg.max_depth),
        subsample=0.8 if model_cfg.bootstrap else 1.0,
        colsample_bynode=colsample_bynode,
        min_child_weight=float(model_cfg.min_samples_leaf),
        reg_lambda=0.0,
        learning_rate=1.0,
        objective="reg:squarederror",
        random_state=model_cfg.random_state,
        n_jobs=model_cfg.n_jobs,
        tree_method="hist",
        device=model_cfg.device,
    )
    model.fit(x_train, y_train)
    return model


def _fit_xgb(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_cfg: XGBModelConfig,
) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=model_cfg.n_estimators,
        max_depth=model_cfg.max_depth,
        learning_rate=model_cfg.learning_rate,
        subsample=model_cfg.subsample,
        colsample_bytree=model_cfg.colsample_bytree,
        min_child_weight=model_cfg.min_child_weight,
        reg_alpha=model_cfg.reg_alpha,
        reg_lambda=model_cfg.reg_lambda,
        objective=model_cfg.objective,
        random_state=model_cfg.random_state,
        n_jobs=model_cfg.n_jobs,
    )
    model.fit(x_train, y_train)
    return model


def _run_tree_shap_common(  # noqa: PLR0915
    *,
    data: pd.DataFrame,
    feature_cfg: RFFeatureConfig | XGBFeatureConfig,
    run_cfg: RFRunConfig | XGBRunConfig,
    job: TreeShapJobConfig,
    runner: _TreeShapRunner,
) -> TreeShapJobResult:
    model_cfg_any = run_cfg.model or (
        RFModelConfig() if runner.model_kind == "rf" else XGBModelConfig()
    )
    design, _, target_col = build_forecasting_design_matrix(
        data,
        feature_cfg,
        target_transform=model_cfg_any.target_transform,
    )
    x, y = split_design_matrix_xy(design, target_col)

    effective_model_cfg = model_cfg_any
    if feature_cfg.target_mode == "mean" and model_cfg_any.target_transform != "none":
        effective_model_cfg = replace(
            model_cfg_any,
            target_transform="none",
            prediction_floor=-1e12,
        )
    mean_log_target = (
        feature_cfg.target_mode == "mean" and model_cfg_any.target_transform == "log"
    )

    wf_cfg = run_cfg.walk_forward or (
        RFWalkForwardConfig(progress_bar=False)
        if runner.model_kind == "rf"
        else XGBWalkForwardConfig(progress_bar=False)
    )

    date_series = None
    if "Date" in data.columns:
        date_series = data.loc[x.index, "Date"].astype(str)

    transformed_feature_columns: list[str] = []
    if effective_model_cfg.log_transform_rv_features:
        x, transformed_feature_columns = log_transform_rv_features(
            x,
            floor=effective_model_cfg.feature_floor,
        )

    windows = build_walk_forward_windows(len(x), wf_cfg)
    shap_parts: list[pd.DataFrame] = []
    feature_parts: list[pd.DataFrame] = []
    pred_parts: list[pd.Series] = []
    pred_model_parts: list[pd.Series] = []
    base_parts: list[pd.Series] = []

    for train_start, train_end, test_start, test_end in windows:
        x_train = x.iloc[train_start:train_end].copy()
        y_train = y.iloc[train_start:train_end]
        x_test = x.iloc[test_start:test_end].copy()

        x_eval = x_test.copy() if job.split == "test" else x_train.copy()

        if effective_model_cfg.standardize_features:
            x_train_model, x_eval_model, _ = standardize_train_test(x_train, x_eval)
        else:
            x_train_model = x_train
            x_eval_model = x_eval

        y_train_model = transform_target(y_train, effective_model_cfg)
        fitted = runner.fit_model(x_train_model, y_train_model, effective_model_cfg)

        background = x_train_model
        if len(background) > job.max_background_samples:
            background = background.sample(
                n=job.max_background_samples,
                random_state=42,
            )

        explainer = shap.TreeExplainer(
            fitted,
            data=background,
            feature_perturbation="interventional",
        )
        explanation_raw = explainer(x_eval_model)
        explanation = (
            explanation_raw[0] if isinstance(explanation_raw, list) else explanation_raw
        )

        shap_window = pd.DataFrame(
            explanation.values,
            index=x_eval_model.index,
            columns=x_eval_model.columns,
        )
        base_values_np = np.asarray(explanation.base_values, dtype=float)
        if base_values_np.ndim == 0:
            base_values_np = np.repeat(base_values_np, len(x_eval_model))

        pred_model_np = fitted.predict(x_eval_model)
        pred_model_s = pd.Series(
            pred_model_np,
            index=x_eval_model.index,
            name="pred_transformed",
        )
        pred_raw = inverse_transform_prediction(
            pred_model_s.rename("y_pred"),
            model_cfg_any if mean_log_target else effective_model_cfg,
        )

        shap_parts.append(shap_window)
        feature_parts.append(x_eval_model)
        pred_model_parts.append(pred_model_s)
        pred_parts.append(pred_raw.rename("prediction"))
        base_parts.append(
            pd.Series(base_values_np, index=x_eval_model.index, name="base_value")
        )

    shap_df = cast(
        "pd.DataFrame",
        pd.concat(shap_parts, axis=0).groupby(level=0).mean().sort_index(),
    )
    feature_df = cast(
        "pd.DataFrame",
        pd.concat(feature_parts, axis=0).groupby(level=0).mean().sort_index(),
    )
    base_s = cast(
        "pd.Series",
        pd.concat(base_parts, axis=0).groupby(level=0).mean().sort_index(),
    )
    pred_model_s = cast(
        "pd.Series",
        pd.concat(pred_model_parts, axis=0).groupby(level=0).mean().sort_index(),
    )
    pred_s = cast(
        "pd.Series",
        pd.concat(pred_parts, axis=0).groupby(level=0).mean().sort_index(),
    )

    report_features = _pick_report_features(shap_df, job)
    shap_selected = (
        shap_df[report_features].copy()
        if report_features
        else pd.DataFrame(index=shap_df.index)
    )
    feature_selected = (
        feature_df[report_features].copy()
        if report_features
        else pd.DataFrame(index=feature_df.index)
    )

    summary = _build_summary_frame(shap_df)

    shap_out = shap_selected.copy()
    if date_series is not None:
        shap_out.insert(0, "Date", date_series.reindex(shap_out.index).to_numpy())
    shap_out.insert(
        1 if date_series is not None else 0,
        "base_value",
        base_s.reindex(shap_out.index).to_numpy(),
    )
    shap_out.insert(
        2 if date_series is not None else 1,
        "prediction_transformed",
        pred_model_s.reindex(shap_out.index).to_numpy(),
    )
    shap_out.insert(
        3 if date_series is not None else 2,
        "prediction",
        pred_s.reindex(shap_out.index).to_numpy(),
    )

    diagnostics = {
        "n_windows": len(windows),
        "n_obs_shap": len(shap_out),
        "target_horizon": feature_cfg.target_horizon,
        "model_type": job.model_type,
        "feature_set": job.feature_set,
        "split": job.split,
        "selected_features_report": report_features,
        "transformed_feature_columns": transformed_feature_columns,
        "max_background_samples": job.max_background_samples,
    }

    return TreeShapJobResult(
        summary=summary,
        shap_values=cast("pd.DataFrame", shap_out),
        feature_data=cast("pd.DataFrame", feature_selected),
        base_values=base_s.reindex(feature_selected.index),
        diagnostics=diagnostics,
    )


def run_rf_shap_for_job(
    *,
    data: pd.DataFrame,
    feature_cfg: RFFeatureConfig,
    run_cfg: RFRunConfig,
    job: TreeShapJobConfig,
) -> TreeShapJobResult:
    return _run_tree_shap_common(
        data=data,
        feature_cfg=feature_cfg,
        run_cfg=run_cfg,
        job=job,
        runner=_TreeShapRunner(fit_model=_fit_rf, model_kind="rf"),
    )


def run_xgb_shap_for_job(
    *,
    data: pd.DataFrame,
    feature_cfg: XGBFeatureConfig,
    run_cfg: XGBRunConfig,
    job: TreeShapJobConfig,
) -> TreeShapJobResult:
    return _run_tree_shap_common(
        data=data,
        feature_cfg=feature_cfg,
        run_cfg=run_cfg,
        job=job,
        runner=_TreeShapRunner(fit_model=_fit_xgb, model_kind="xgb"),
    )


def _sanitize_name(value: str) -> str:
    keep = [c if c.isalnum() or c in {"_", "-"} else "_" for c in value]
    return "".join(keep).strip("_")


def _save_summary_plot(
    shap_values: pd.DataFrame,
    feature_data: pd.DataFrame,
    out_path: Path,
) -> None:
    if shap_values.empty:
        return
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values.values,
        features=feature_data,
        feature_names=feature_data.columns.tolist(),
        show=False,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _save_dependence_plots(
    *,
    shap_values: pd.DataFrame,
    feature_data: pd.DataFrame,
    out_dir: Path,
    job: TreeShapJobConfig,
) -> list[Path]:
    if shap_values.empty:
        return []
    if job.dependence_features:
        feats = [f for f in job.dependence_features if f in shap_values.columns]
    else:
        mean_abs = shap_values.abs().mean().sort_values(ascending=False)
        feats = mean_abs.head(max(job.dependence_top_n, 1)).index.tolist()

    paths: list[Path] = []
    for feat in feats:
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            ind=feat,
            shap_values=shap_values.values,
            features=feature_data,
            feature_names=feature_data.columns.tolist(),
            show=False,
        )
        plt.tight_layout()
        out_path = out_dir / f"dependence_{_sanitize_name(feat)}.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        paths.append(out_path)
    return paths


def _save_waterfall_plot(
    *,
    shap_values: pd.DataFrame,
    feature_data: pd.DataFrame,
    base_values: pd.Series,
    out_path: Path,
    row_idx: int | None,
) -> None:
    if shap_values.empty:
        return
    row_pos = row_idx if row_idx is not None else len(shap_values) - 1
    row_pos = max(0, min(row_pos, len(shap_values) - 1))
    row_key = shap_values.index[row_pos]
    exp = shap.Explanation(
        values=shap_values.iloc[row_pos].to_numpy(dtype=float),
        base_values=float(base_values.loc[row_key]),
        data=feature_data.iloc[row_pos].to_numpy(dtype=float),
        feature_names=feature_data.columns.tolist(),
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, max_display=min(20, len(feature_data.columns)), show=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_tree_shap_job_outputs(
    *,
    result: TreeShapJobResult,
    job: TreeShapJobConfig,
    output_root: Path,
) -> dict[str, Path | list[Path]]:
    job_name = _sanitize_name(job.name or f"{job.model_type}_{job.feature_set}_{job.split}")
    job_dir = output_root / job_name
    plots_dir = job_dir / "plots"
    job_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = job_dir / "summary.csv"
    shap_values_path = job_dir / "shap_values.csv"
    feature_data_path = job_dir / "feature_data.csv"
    diagnostics_path = job_dir / "diagnostics.json"
    summary_plot_path = plots_dir / "summary_plot.png"
    waterfall_path = plots_dir / "waterfall_plot.png"

    result.summary.to_csv(summary_path, index=False)
    result.shap_values.to_csv(shap_values_path, index=True)
    result.feature_data.to_csv(feature_data_path, index=True)
    pd.Series(result.diagnostics, dtype=object).to_json(diagnostics_path, indent=2)

    shap_cols = result.feature_data.columns.tolist()
    shap_only = result.shap_values.reindex(columns=shap_cols).fillna(0.0)
    feature_only = result.feature_data.reindex(index=shap_only.index)
    base_only = result.base_values.reindex(index=shap_only.index)

    _save_summary_plot(shap_only, feature_only, summary_plot_path)
    dep_paths = _save_dependence_plots(
        shap_values=shap_only,
        feature_data=feature_only,
        out_dir=plots_dir,
        job=job,
    )
    _save_waterfall_plot(
        shap_values=shap_only,
        feature_data=feature_only,
        base_values=base_only,
        out_path=waterfall_path,
        row_idx=job.waterfall_row,
    )

    return {
        "dir": job_dir,
        "summary_csv": summary_path,
        "shap_values_csv": shap_values_path,
        "feature_data_csv": feature_data_path,
        "diagnostics_json": diagnostics_path,
        "summary_plot": summary_plot_path,
        "dependence_plots": dep_paths,
        "waterfall_plot": waterfall_path,
    }
