from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path  # noqa: TC003
from typing import Any, cast

import pandas as pd

from src.model import XGBExperimentResult, XGBFeatureConfig, XGBRunConfig


def dataset_signature(data: pd.DataFrame) -> str:
    payload = data.to_csv(index=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, default=str)


def cache_key(
    *,
    model_name: str,
    feature_set_name: str,
    feature_cfg: XGBFeatureConfig,
    run_cfg: XGBRunConfig,
    data_signature_value: str,
) -> str:
    payload = {
        "cache_version": "xgb_v1",
        "model_name": model_name,
        "feature_set_name": feature_set_name,
        "feature_config": asdict(feature_cfg),
        "run_config": asdict(run_cfg),
        "data_signature": data_signature_value,
    }
    digest = hashlib.sha256(json_dumps(payload).encode("utf-8")).hexdigest()
    return f"{model_name}__{feature_set_name}__{digest[:16]}"


def cache_paths(cache_dir: Path, key: str) -> dict[str, Path]:
    base = cache_dir / key
    return {
        "base": base,
        "meta": base / "meta.parquet",
        "train_pred": base / "train_pred.parquet",
        "test_pred": base / "test_pred.parquet",
        "feature_importances": base / "feature_importances.parquet",
        "window_report": base / "window_report.parquet",
    }


def _serialize_model_info(model_info: dict[str, Any]) -> tuple[str, bool]:
    info_copy = dict(model_info)
    has_window_report = "window_report" in info_copy
    info_copy.pop("window_report", None)
    return json_dumps(info_copy), has_window_report


def save_result_cache(
    *,
    cache_dir: Path,
    key: str,
    result: XGBExperimentResult,
) -> None:
    paths = cache_paths(cache_dir, key)
    paths["base"].mkdir(parents=True, exist_ok=True)

    model_info_json, has_window_report = _serialize_model_info(result.model_info)
    meta = pd.DataFrame(
        [
            {
                "selected_features_json": json_dumps(result.selected_features),
                "metrics_json": json_dumps(result.metrics),
                "model_info_json": model_info_json,
                "has_window_report": has_window_report,
            }
        ]
    )
    meta.to_parquet(paths["meta"], index=False)

    pd.DataFrame(
        {"y_true": result.y_true_train, "y_pred": result.y_pred_train},
        index=result.y_true_train.index,
    ).to_parquet(paths["train_pred"], index=True)
    pd.DataFrame(
        {"y_true": result.y_true_test, "y_pred": result.y_pred_test},
        index=result.y_true_test.index,
    ).to_parquet(paths["test_pred"], index=True)
    result.feature_importances.to_frame(name="importance").to_parquet(
        paths["feature_importances"],
        index=True,
    )

    window_report = result.model_info.get("window_report")
    if isinstance(window_report, pd.DataFrame):
        window_report.to_parquet(paths["window_report"], index=False)


def load_result_cache(cache_dir: Path, key: str) -> XGBExperimentResult | None:
    paths = cache_paths(cache_dir, key)
    if not paths["meta"].exists():
        return None
    required = ["train_pred", "test_pred", "feature_importances"]
    if any(not paths[name].exists() for name in required):
        return None

    meta = pd.read_parquet(paths["meta"])
    if meta.empty:
        return None
    meta_row = meta.iloc[0]

    train_pred = pd.read_parquet(paths["train_pred"])
    test_pred = pd.read_parquet(paths["test_pred"])
    feature_importances = pd.read_parquet(paths["feature_importances"])

    y_true_train = train_pred["y_true"].copy()
    y_true_train.name = "y_true"
    y_pred_train = train_pred["y_pred"].copy()
    y_pred_train.name = "y_pred"
    y_true_test = test_pred["y_true"].copy()
    y_true_test.name = "y_true"
    y_pred_test = test_pred["y_pred"].copy()
    y_pred_test.name = "y_pred"

    importances = feature_importances["importance"].copy()
    importances.name = "importance"

    model_info = json.loads(str(meta_row["model_info_json"]))
    if bool(meta_row.get("has_window_report", False)) and paths["window_report"].exists():
        model_info["window_report"] = pd.read_parquet(paths["window_report"])

    return XGBExperimentResult(
        selected_features=list(json.loads(str(meta_row["selected_features_json"]))),
        y_true_train=cast("pd.Series", y_true_train),
        y_pred_train=cast("pd.Series", y_pred_train),
        y_true_test=cast("pd.Series", y_true_test),
        y_pred_test=cast("pd.Series", y_pred_test),
        metrics=dict(json.loads(str(meta_row["metrics_json"]))),
        feature_importances=cast("pd.Series", importances),
        model_info=model_info,
    )
