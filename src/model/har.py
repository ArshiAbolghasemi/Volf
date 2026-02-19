from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from math import ceil
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from src.metrics import evaluate_statistical_metrics
from src.variable_selection import (
    BSRSelectionConfig,
    LassoSelectionConfig,
    backward_stepwise_feature_selection,
    lasso_time_series_feature_selection,
)

if TYPE_CHECKING:
    from statsmodels.regression.linear_model import RegressionResultsWrapper

logger = logging.getLogger(__name__)


@dataclass
class HARFeatureConfig:
    target_col: str
    core_columns: list[str]
    target_horizon: int = 1
    extra_feature_cols: list[str] | None = None
    target_col_name: str = "RV_target"


@dataclass
class HARSelectionConfig:
    method: Literal["lasso", "bsr", "none"] = "lasso"
    lasso: LassoSelectionConfig | None = None
    bsr: BSRSelectionConfig | None = None


@dataclass
class HARSplitConfig:
    val_size: float | int = 0.2
    test_size: float | int = 0.2


@dataclass
class HARModelConfig:
    add_constant: bool = True
    standardize_features: bool = False
    refit_on_train_val: bool = True


@dataclass
class HARGridConfig:
    feature_sets: dict[str, list[str]]
    base_feature_config: HARFeatureConfig


@dataclass
class HARRunConfig:
    split: HARSplitConfig | None = None
    selection: HARSelectionConfig | None = None
    model: HARModelConfig | None = None


@dataclass
class HARExperimentResult:
    selected_features: list[str]
    y_true_val: pd.Series
    y_pred_val: pd.Series
    y_true_test: pd.Series
    y_pred_test: pd.Series
    metrics: dict[str, dict[str, Any]]
    coefficients: pd.Series
    selection_info: dict[str, Any]
    model_info: dict[str, Any]


def _validate_feature_config(cfg: HARFeatureConfig, data: pd.DataFrame) -> None:
    logger.debug(
        (
            "Validating HAR feature config: target_col=%s, core_columns=%s, "
            "target_horizon=%s, extra_features=%s"
        ),
        cfg.target_col,
        cfg.core_columns,
        cfg.target_horizon,
        cfg.extra_feature_cols,
    )
    if cfg.target_col not in data.columns:
        msg = f"target_col '{cfg.target_col}' is not in dataframe columns."
        raise ValueError(msg)

    if not cfg.core_columns:
        msg = "core_columns must not be empty."
        raise ValueError(msg)

    missing_core = [col for col in cfg.core_columns if col not in data.columns]
    if missing_core:
        msg = f"core_columns missing from dataframe: {missing_core}"
        raise ValueError(msg)

    if cfg.extra_feature_cols:
        missing_extra = [col for col in cfg.extra_feature_cols if col not in data.columns]
        if missing_extra:
            msg = f"extra_feature_cols missing from dataframe: {missing_extra}"
            raise ValueError(msg)

    if cfg.target_horizon < 0:
        msg = "target_horizon must be >= 0."
        raise ValueError(msg)


def build_har_design_matrix(
    data: pd.DataFrame,
    config: HARFeatureConfig,
) -> tuple[pd.DataFrame, list[str], str]:
    logger.info("Building HAR design matrix")
    _validate_feature_config(config, data)

    feature_cols = config.core_columns.copy()
    if config.extra_feature_cols:
        feature_cols.extend(config.extra_feature_cols)

    design = cast("pd.DataFrame", data[feature_cols].copy())

    target_col = config.target_col_name
    if config.target_horizon == 0:
        design[target_col] = data[config.target_col]
    else:
        design[target_col] = data[config.target_col].shift(-config.target_horizon)

    logger.info(
        "HAR design matrix ready: rows=%d, n_features=%d, target=%s",
        len(design),
        len(design.columns) - 1,
        target_col,
    )
    return design, config.core_columns.copy(), target_col


def get_xy_from_har_design(
    design: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in design.columns:
        msg = f"target_col '{target_col}' not found in design dataframe"
        raise ValueError(msg)

    x = design.drop(columns=[target_col])
    y = design[target_col]

    clean = pd.concat([x, y.to_frame(name=target_col)], axis=1).dropna()
    x_clean = clean.drop(columns=[target_col])
    y_clean = cast("pd.Series", clean[target_col])
    logger.info(
        "Prepared X/y: n_rows=%d, n_features=%d",
        len(x_clean),
        len(x_clean.columns),
    )
    return x_clean, y_clean


def _resolve_size(size: float, n_obs: int, *, name: str) -> int:
    if isinstance(size, float):
        if not 0 < size < 1:
            msg = f"{name} as float must be in (0, 1)."
            raise ValueError(msg)
        n_size = ceil(n_obs * size)
    else:
        n_size = int(size)

    if n_size < 1:
        msg = f"{name} must be >= 1 observation."
        raise ValueError(msg)
    return n_size


def split_train_val_test(
    x: pd.DataFrame,
    y: pd.Series,
    split_config: HARSplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    n_obs = len(x)
    if n_obs != len(y):
        msg = "x and y must have same length"
        raise ValueError(msg)

    n_test = _resolve_size(split_config.test_size, n_obs, name="test_size")
    n_val = _resolve_size(split_config.val_size, n_obs, name="val_size")

    n_train = n_obs - n_val - n_test
    if n_train < 1:
        msg = (
            f"Not enough observations for train split. n_obs={n_obs}, "
            f"n_val={n_val}, n_test={n_test}"
        )
        raise ValueError(msg)

    train_end = n_train
    val_end = n_train + n_val

    x_train = x.iloc[:train_end]
    x_val = x.iloc[train_end:val_end]
    x_test = x.iloc[val_end:]

    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    y_test = y.iloc[val_end:]

    logger.info(
        "Split sizes -> train=%d, val=%d, test=%d",
        len(y_train),
        len(y_val),
        len(y_test),
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def _standardize_feature_splits(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    logger.info("Applying feature standardization using train split statistics")
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_scaled = pd.DataFrame(
        scaler.transform(x_train),
        index=x_train.index,
        columns=x_train.columns,
    )
    x_val_scaled = pd.DataFrame(
        scaler.transform(x_val),
        index=x_val.index,
        columns=x_val.columns,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(x_test),
        index=x_test.index,
        columns=x_test.columns,
    )
    return x_train_scaled, x_val_scaled, x_test_scaled, scaler


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
        logger.info("Feature selection method=none, selected=%d", len(selected))
        return selected, {"method": "none", "selected_features": selected}

    if config.method == "lasso":
        logger.info("Running feature selection with LASSO")
        lasso_cfg = config.lasso or LassoSelectionConfig()
        lasso_cfg = replace(lasso_cfg, core_columns=core_columns)
        lasso_result = lasso_time_series_feature_selection(
            x_train,
            y_train,
            config=lasso_cfg,
        )
        info = {"method": "lasso", **lasso_result.info}
        logger.info(
            "LASSO selected %d features (best_alpha=%s)",
            len(lasso_result.selected_features),
            info.get("best_alpha"),
        )
        return lasso_result.selected_features, info

    if config.method == "bsr":
        logger.info("Running feature selection with BSR")
        bsr_cfg = config.bsr or BSRSelectionConfig()
        bsr_cfg = replace(bsr_cfg, core_columns=tuple(core_columns))
        bsr_result = backward_stepwise_feature_selection(x_train, y_train, config=bsr_cfg)
        info = {"method": "bsr", **bsr_result.info}
        logger.info(
            "BSR selected %d features (successful_windows=%s)",
            len(bsr_result.selected_features),
            info.get("n_windows_successful"),
        )
        return bsr_result.selected_features, info

    msg = f"unsupported feature selection method: {config.method}"
    raise ValueError(msg)


def fit_har_ols(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    selected_features: list[str],
    *,
    add_constant: bool,
) -> RegressionResultsWrapper:
    logger.info(
        "Fitting HAR OLS model: n_rows=%d, n_features=%d, add_constant=%s",
        len(y_train),
        len(selected_features),
        add_constant,
    )
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


def run_har_experiment_from_xy(
    x: pd.DataFrame,
    y: pd.Series,
    core_columns: list[str],
    run_config: HARRunConfig | None = None,
) -> HARExperimentResult:
    logger.info("Starting HAR experiment from X/y")
    cfg = run_config or HARRunConfig()
    split_cfg = cfg.split or HARSplitConfig()
    selection_cfg = cfg.selection or HARSelectionConfig()
    model_cfg = cfg.model or HARModelConfig()

    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
        x=x,
        y=y,
        split_config=split_cfg,
    )

    selected_features, selection_info = select_har_features(
        x_train=x_train,
        y_train=y_train,
        core_columns=core_columns,
        config=selection_cfg,
    )
    logger.info("Selected features: %s", selected_features)

    x_train_sel = cast("pd.DataFrame", x_train[selected_features])
    x_val_sel = cast("pd.DataFrame", x_val[selected_features])
    x_test_sel = cast("pd.DataFrame", x_test[selected_features])

    scaler_fitted = False
    if model_cfg.standardize_features:
        x_train_sel, x_val_sel, x_test_sel, _ = _standardize_feature_splits(
            x_train_sel,
            x_val_sel,
            x_test_sel,
        )
        scaler_fitted = True

    model_train = fit_har_ols(
        x_train=x_train_sel,
        y_train=y_train,
        selected_features=selected_features,
        add_constant=model_cfg.add_constant,
    )

    y_pred_val = predict_har_ols(
        fitted_model=model_train,
        x=x_val_sel,
        selected_features=selected_features,
        add_constant=model_cfg.add_constant,
    )

    if model_cfg.refit_on_train_val:
        logger.info("Refitting model on train+validation data")
        x_train_val = pd.concat([x_train_sel, x_val_sel], axis=0)
        y_train_val = pd.concat([y_train, y_val], axis=0)
        model_final = fit_har_ols(
            x_train=x_train_val,
            y_train=y_train_val,
            selected_features=selected_features,
            add_constant=model_cfg.add_constant,
        )
        final_train_n = len(y_train_val)
    else:
        model_final = model_train
        final_train_n = len(y_train)

    y_pred_test = predict_har_ols(
        fitted_model=model_final,
        x=x_test_sel,
        selected_features=selected_features,
        add_constant=model_cfg.add_constant,
    )

    metrics = {
        "val": evaluate_statistical_metrics(y_val, y_pred_val),
        "test": evaluate_statistical_metrics(y_test, y_pred_test),
    }
    logger.info("Validation metrics: %s", metrics["val"])
    logger.info("Test metrics: %s", metrics["test"])

    coefficients = model_final.params.drop(labels=["const"], errors="ignore")
    coefficients.name = "coefficient"

    model_info = {
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "n_train_final_fit": final_train_n,
        "add_constant": model_cfg.add_constant,
        "standardize_features": model_cfg.standardize_features,
        "scaler_fitted": scaler_fitted,
        "refit_on_train_val": model_cfg.refit_on_train_val,
        "aic_final": float(model_final.aic),
        "bic_final": float(model_final.bic),
        "rsquared_final": float(model_final.rsquared),
    }
    logger.info(
        "Model summary: aic=%.4f, bic=%.4f, rsquared=%.4f",
        model_info["aic_final"],
        model_info["bic_final"],
        model_info["rsquared_final"],
    )

    return HARExperimentResult(
        selected_features=selected_features,
        y_true_val=y_val,
        y_pred_val=y_pred_val,
        y_true_test=y_test,
        y_pred_test=y_pred_test,
        metrics=metrics,
        coefficients=coefficients,
        selection_info=selection_info,
        model_info=model_info,
    )


def run_har_experiment_from_dataset(
    data: pd.DataFrame,
    *,
    feature_config: HARFeatureConfig,
    run_config: HARRunConfig | None = None,
) -> HARExperimentResult:
    logger.info("Starting HAR experiment from raw dataset")
    design, core_columns, target_col = build_har_design_matrix(data, feature_config)
    x, y = get_xy_from_har_design(design, target_col)

    result = run_har_experiment_from_xy(
        x=x,
        y=y,
        core_columns=core_columns,
        run_config=run_config,
    )
    cfg = run_config or HARRunConfig()
    split_cfg = cfg.split or HARSplitConfig()
    selection_cfg = cfg.selection or HARSelectionConfig()
    model_cfg = cfg.model or HARModelConfig()

    result.model_info.update(
        {
            "target_col_raw": feature_config.target_col,
            "target_col_model": target_col,
            "target_horizon": feature_config.target_horizon,
            "core_columns": feature_config.core_columns,
            "extra_feature_cols": feature_config.extra_feature_cols or [],
            "split_val_size": split_cfg.val_size,
            "split_test_size": split_cfg.test_size,
            "selection_method": selection_cfg.method,
            "lasso_config": (
                vars(selection_cfg.lasso) if selection_cfg.lasso is not None else None
            ),
            "bsr_config": (
                vars(selection_cfg.bsr) if selection_cfg.bsr is not None else None
            ),
            "model_add_constant": model_cfg.add_constant,
            "model_standardize_features": model_cfg.standardize_features,
            "model_refit_on_train_val": model_cfg.refit_on_train_val,
        }
    )
    return result


def run_har_feature_set_grid(
    data: pd.DataFrame,
    grid_config: HARGridConfig,
    *,
    run_config: HARRunConfig | None = None,
) -> dict[str, HARExperimentResult]:
    logger.info(
        "Running HAR feature-set grid: %d feature sets", len(grid_config.feature_sets)
    )
    base_cfg = grid_config.base_feature_config
    results: dict[str, HARExperimentResult] = {}

    for model_name, extra_cols in grid_config.feature_sets.items():
        logger.info(
            "Running feature set '%s' with %d additional features",
            model_name,
            len(extra_cols),
        )
        feature_cfg = replace(base_cfg, extra_feature_cols=extra_cols)
        results[model_name] = run_har_experiment_from_dataset(
            data,
            feature_config=feature_cfg,
            run_config=run_config,
        )
        logger.info("Completed feature set '%s'", model_name)

    return results
