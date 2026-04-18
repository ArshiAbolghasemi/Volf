from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        out: Any = value
    elif isinstance(value, Path):
        out = str(value)
    elif isinstance(value, dict):
        out = {str(k): _jsonable(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple, set)):
        out = [_jsonable(v) for v in value]
    elif isinstance(value, pd.Series):
        out = {str(k): _jsonable(v) for k, v in value.to_dict().items()}
    elif isinstance(value, pd.DataFrame):
        out = value.to_dict(orient="records")
    elif hasattr(value, "item"):
        out = value.item()
    else:
        out = str(value)
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_predictions(path: Path, y_true: pd.Series, y_pred: pd.Series) -> None:
    frame = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        },
        index=y_true.index,
    )
    frame.to_csv(path, index=True)


def _write_series(path: Path, series: pd.Series, value_name: str) -> None:
    series.to_frame(name=value_name).to_csv(path, index=True)


def _write_checkpoint_for_result(  # noqa: PLR0913
    *,
    checkpoint_root: Path,
    model_family: str,
    target_mode: str,
    target_horizon: int,
    model_type: str,
    feature_set: str,
    result: Any,
) -> Path:
    checkpoint_dir = (
        checkpoint_root
        / f"target_horizon_{target_horizon}"
        / "checkpoints"
        / f"target_mode_{_safe_name(target_mode)}"
        / f"{_safe_name(model_type)}__{_safe_name(feature_set)}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_info = dict(getattr(result, "model_info", {}))
    window_report = model_info.pop("window_report", None)

    metadata = {
        "model_family": model_family,
        "target_mode": target_mode,
        "target_horizon": target_horizon,
        "model_type": model_type,
        "feature_set": feature_set,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_dir": str(checkpoint_dir),
    }
    _write_json(checkpoint_dir / "metadata.json", metadata)
    _write_json(checkpoint_dir / "metrics.json", dict(getattr(result, "metrics", {})))
    _write_json(checkpoint_dir / "model_info.json", model_info)

    selection_info = getattr(result, "selection_info", None)
    if isinstance(selection_info, dict):
        _write_json(checkpoint_dir / "selection_info.json", selection_info)

    selected_features = list(getattr(result, "selected_features", []))
    (checkpoint_dir / "selected_features.txt").write_text(
        "\n".join(str(col) for col in selected_features),
        encoding="utf-8",
    )

    _write_predictions(
        checkpoint_dir / "train_predictions.csv",
        result.y_true_train,
        result.y_pred_train,
    )
    _write_predictions(
        checkpoint_dir / "test_predictions.csv",
        result.y_true_test,
        result.y_pred_test,
    )

    if hasattr(result, "feature_importances"):
        _write_series(
            checkpoint_dir / "feature_importances.csv",
            result.feature_importances,
            "importance",
        )
    if hasattr(result, "coefficients"):
        _write_series(
            checkpoint_dir / "coefficients.csv",
            result.coefficients,
            "coefficient",
        )
    if isinstance(window_report, pd.DataFrame):
        window_report.to_csv(checkpoint_dir / "window_report.csv", index=False)

    return checkpoint_dir


def save_best_result_checkpoints(
    *,
    checkpoint_root: Path,
    model_family: str,
    target_mode: str,
    results_by_horizon: dict[int, dict[str, dict[str, Any]]],
) -> list[Path]:
    saved_dirs: list[Path] = []
    for target_horizon, model_results in sorted(results_by_horizon.items()):
        for model_type, feature_results in sorted(model_results.items()):
            for feature_set, result in sorted(feature_results.items()):
                saved_dirs.append(
                    _write_checkpoint_for_result(
                        checkpoint_root=checkpoint_root,
                        model_family=model_family,
                        target_mode=target_mode,
                        target_horizon=int(target_horizon),
                        model_type=str(model_type),
                        feature_set=str(feature_set),
                        result=result,
                    )
                )
    return saved_dirs
