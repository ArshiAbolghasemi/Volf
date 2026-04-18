from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression

from src.model import (
    HARFeatureConfig,
    HARModelConfig,
    HARRunConfig,
    HARSelectionConfig,
    HARWalkForwardConfig,
)
from src.model.common.preprocessing import (
    build_forecasting_design_matrix,
    build_walk_forward_windows,
    inverse_transform_prediction,
    log_transform_rv_features,
    split_design_matrix_xy,
    transform_target,
)
from src.model.har.selection import select_har_features

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ShapJobConfig:
    model_type: str
    feature_set: str
    target_horizon: int
    name: str | None = None
    split: Literal["test", "train"] = "test"
    include_features: list[str] | None = None
    top_n_features: int = 20
    dependence_features: list[str] | None = None
    dependence_top_n: int = 3
    waterfall_row: int | None = None


@dataclass
class ShapConfig:
    jobs: list[ShapJobConfig]
    output_subdir: str = "shap"


@dataclass
class ShapJobResult:
    summary: pd.DataFrame
    shap_values: pd.DataFrame
    feature_data: pd.DataFrame
    base_values: pd.Series
    diagnostics: dict[str, Any]


def resolve_run_config_for_shap_job(
    *,
    base_run_cfg: HARRunConfig,
    model_info: dict[str, Any],
) -> HARRunConfig:
    wf_base = base_run_cfg.walk_forward or HARWalkForwardConfig()
    sel_base = base_run_cfg.selection or HARSelectionConfig()
    model_base = base_run_cfg.model or HARModelConfig()

    wf = replace(
        wf_base,
        window_type=str(model_info.get("walk_forward_window_type", wf_base.window_type)),
        initial_train_size=int(
            model_info.get("walk_forward_initial_train_size", wf_base.initial_train_size)
        ),
        test_size=int(model_info.get("walk_forward_test_size", wf_base.test_size)),
        step=int(model_info.get("walk_forward_step", wf_base.step)),
        rolling_window_size=model_info.get(
            "walk_forward_rolling_window_size", wf_base.rolling_window_size
        ),
        progress_bar=False,
    )
    model_cfg = replace(
        model_base,
        add_constant=bool(model_info.get("model_add_constant", model_base.add_constant)),
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
    return HARRunConfig(walk_forward=wf, selection=sel_base, model=model_cfg)


def _pick_report_features(shap_values: pd.DataFrame, job: ShapJobConfig) -> list[str]:
    if shap_values.empty:
        return []
    if job.include_features:
        selected = [feat for feat in job.include_features if feat in shap_values.columns]
        if selected:
            return selected

    mean_abs = shap_values.abs().mean().sort_values(ascending=False)
    return cast("list[str]", mean_abs.head(max(job.top_n_features, 1)).index.tolist())


def _fit_linear(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    add_constant: bool,
) -> LinearRegression:
    model = LinearRegression(fit_intercept=add_constant)
    model.fit(x_train, y_train)
    return model


def run_linear_shap_for_job(  # noqa: PLR0915
    *,
    data: pd.DataFrame,
    feature_cfg: HARFeatureConfig,
    core_columns: list[str],
    run_cfg: HARRunConfig,
    job: ShapJobConfig,
) -> ShapJobResult:
    model_cfg = run_cfg.model or HARModelConfig()
    design, _, target_col = build_forecasting_design_matrix(
        data,
        feature_cfg,
        target_transform=model_cfg.target_transform,
    )
    x, y = split_design_matrix_xy(design, target_col)

    effective_model_cfg = model_cfg
    if feature_cfg.target_mode == "mean" and model_cfg.target_transform != "none":
        effective_model_cfg = replace(
            model_cfg,
            target_transform="none",
            prediction_floor=-1e12,
        )
    mean_log_target = (
        feature_cfg.target_mode == "mean" and model_cfg.target_transform == "log"
    )

    selection_cfg = run_cfg.selection or HARSelectionConfig()
    wf_cfg = run_cfg.walk_forward or HARWalkForwardConfig(progress_bar=False)

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
    cached_selected_features: list[str] | None = None

    shap_parts: list[pd.DataFrame] = []
    feature_parts: list[pd.DataFrame] = []
    pred_parts: list[pd.Series] = []
    pred_model_parts: list[pd.Series] = []
    base_parts: list[pd.Series] = []

    for window_id, (train_start, train_end, test_start, test_end) in enumerate(windows):
        x_train = x.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        x_test = x.iloc[test_start:test_end]

        refit_every = max(selection_cfg.refit_every_windows, 1)
        should_refit = selection_cfg.method != "none" and (
            cached_selected_features is None or window_id % refit_every == 0
        )
        if should_refit:
            selected_features, _ = select_har_features(
                x_train=x_train,
                y_train=y_train,
                core_columns=core_columns,
                config=selection_cfg,
            )
            cached_selected_features = selected_features
        else:
            selected_features = cached_selected_features or x_train.columns.tolist()

        x_train_sel = x_train[selected_features].copy()
        x_eval_sel = (
            x_test[selected_features].copy()
            if job.split == "test"
            else x_train[selected_features].copy()
        )

        if effective_model_cfg.standardize_features:
            means = x_train_sel.mean()
            stds = x_train_sel.std(ddof=0).replace(0.0, 1.0)
            x_train_model = (x_train_sel - means) / stds
            x_eval_model = (x_eval_sel - means) / stds
        else:
            x_train_model = x_train_sel
            x_eval_model = x_eval_sel

        y_train_model = transform_target(y_train, effective_model_cfg)
        model = _fit_linear(
            x_train=x_train_model,
            y_train=y_train_model,
            add_constant=effective_model_cfg.add_constant,
        )

        explainer = shap.LinearExplainer(model, x_train_model)
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

        pred_model_np = model.predict(x_eval_model)
        pred_model_s = pd.Series(
            pred_model_np,
            index=x_eval_model.index,
            name="pred_transformed",
        )
        pred_raw = inverse_transform_prediction(
            pred_model_s.rename("y_pred"),
            model_cfg if mean_log_target else effective_model_cfg,
        )

        shap_parts.append(shap_window)
        feature_parts.append(x_eval_model)
        pred_model_parts.append(pred_model_s)
        pred_parts.append(pred_raw.rename("prediction"))
        base_parts.append(
            pd.Series(
                base_values_np,
                index=x_eval_model.index,
                name="base_value",
            )
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

    summary = pd.DataFrame(
        {
            "feature": report_features,
            "mean_abs_shap": [
                float(shap_selected[f].abs().mean()) for f in report_features
            ],
            "mean_shap": [float(shap_selected[f].mean()) for f in report_features],
        }
    ).sort_values("mean_abs_shap", ascending=False)

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
    }

    return ShapJobResult(
        summary=summary,
        shap_values=cast("pd.DataFrame", shap_out),
        feature_data=cast("pd.DataFrame", feature_selected),
        base_values=base_s.reindex(feature_selected.index),
        diagnostics=diagnostics,
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
    job: ShapJobConfig,
    job_name: str,
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
        prefix = f"{job_name}_" if job_name else ""
        out_path = out_dir / f"{prefix}dependence_{_sanitize_name(feat)}.png"
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


def save_shap_job_outputs(
    *,
    result: ShapJobResult,
    job: ShapJobConfig,
    output_root: Path,
) -> dict[str, Path | list[Path]]:
    job_name = _sanitize_name(job.name or f"{job.model_type}_{job.feature_set}_{job.split}")
    job_dir = output_root / job_name
    plots_dir = job_dir / "plots"
    job_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = job_dir / "summary.csv"
    diagnostics_path = job_dir / "diagnostics.json"

    summary_plot_path = plots_dir / "summary_plot.png"
    waterfall_path = plots_dir / "waterfall_plot.png"

    # Keep a single CSV artifact (shap_summary.csv); plots are saved separately.
    result.summary.to_csv(summary_path, index=False)
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
        job_name="",
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
        "diagnostics_json": diagnostics_path,
        "summary_plot": summary_plot_path,
        "dependence_plots": dep_paths,
        "waterfall_plot": waterfall_path,
    }
