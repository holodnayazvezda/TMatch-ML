from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.data_config import Paths
from ml.services.recommendation_feature_snapshot import build_feature_frame_from_snapshots
from ml.services.recommendation_scoring import (
    EnsembleWeights,
    SERVING_PREDICTION_COLUMN_BY_TARGET,
    SERVING_PROBABILITY_TARGETS,
    compute_weighted_utility_score_frame,
)

MODEL_FAMILY_STAGE_23 = "stage_2_3"
MODEL_FAMILY_STAGE_4 = "stage_4"


def resolve_model_family(served_stage: str) -> str | None:
    normalized_stage = str(served_stage).strip().lower()
    if normalized_stage in {"stage_2", "stage_3"}:
        return MODEL_FAMILY_STAGE_23
    if normalized_stage == "stage_4":
        return MODEL_FAMILY_STAGE_4
    return None


def score_candidate_frame(
    candidate_df: pd.DataFrame,
    *,
    family: str,
    weights: EnsembleWeights,
    model_dir: str | Path | None = None,
) -> pd.DataFrame:
    result = candidate_df.copy()
    for target in SERVING_PROBABILITY_TARGETS:
        result[SERVING_PREDICTION_COLUMN_BY_TARGET[target]] = np.nan
    if result.empty or family not in {MODEL_FAMILY_STAGE_23, MODEL_FAMILY_STAGE_4}:
        result["rerank_score"] = np.nan
        return result

    metadata = _load_metadata(family=family, model_dir=model_dir)
    if metadata is None:
        result["rerank_score"] = np.nan
        return result

    feature_frame = build_feature_frame_from_snapshots(
        result["feature_snapshot"].tolist(),
        family=family,
    )
    aligned_frame = _align_feature_frame(
        feature_frame,
        feature_columns=metadata.get("feature_columns", []),
        categorical_columns=metadata.get("categorical_columns", []),
    )

    has_any_prediction = False
    for target in SERVING_PROBABILITY_TARGETS:
        model = _load_model(family=family, target=target, model_dir=model_dir)
        if model is None:
            continue
        try:
            probabilities = model.predict_proba(aligned_frame)
        except Exception:
            continue
        if getattr(probabilities, "ndim", 1) == 2 and probabilities.shape[1] > 1:
            result[SERVING_PREDICTION_COLUMN_BY_TARGET[target]] = probabilities[:, 1]
        else:
            result[SERVING_PREDICTION_COLUMN_BY_TARGET[target]] = probabilities.reshape(-1)
        has_any_prediction = True

    if not has_any_prediction:
        result["rerank_score"] = np.nan
        return result

    for target in SERVING_PROBABILITY_TARGETS:
        result[SERVING_PREDICTION_COLUMN_BY_TARGET[target]] = pd.to_numeric(
            result[SERVING_PREDICTION_COLUMN_BY_TARGET[target]],
            errors="coerce",
        )
    result["rerank_score"] = compute_weighted_utility_score_frame(result, weights=weights)
    return result


def save_model_metadata(
    *,
    family: str,
    feature_columns: list[str],
    categorical_columns: list[str],
    feature_builder_version: str,
    model_dir: str | Path | None = None,
) -> Path:
    metadata_path = _resolve_metadata_path(family=family, model_dir=model_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": family,
        "feature_columns": list(feature_columns),
        "categorical_columns": list(categorical_columns),
        "feature_builder_version": feature_builder_version,
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return metadata_path


def _align_feature_frame(
    feature_frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    aligned = feature_frame.reindex(columns=feature_columns, fill_value=0.0).copy()
    categorical_set = set(categorical_columns)
    for column in aligned.columns:
        if column in categorical_set:
            aligned[column] = aligned[column].astype("string").fillna("")
        else:
            aligned[column] = pd.to_numeric(aligned[column], errors="coerce").fillna(0.0)
    return aligned


def _load_metadata(*, family: str, model_dir: str | Path | None) -> dict[str, Any] | None:
    metadata_path = _resolve_metadata_path(family=family, model_dir=model_dir)
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_model(*, family: str, target: str, model_dir: str | Path | None):
    model_path = _resolve_model_path(family=family, target=target, model_dir=model_dir)
    if not model_path.exists():
        return None
    try:
        from catboost import CatBoostClassifier
    except Exception:
        return None

    model = CatBoostClassifier()
    try:
        model.load_model(str(model_path))
    except Exception:
        return None
    return model


def _resolve_model_path(*, family: str, target: str, model_dir: str | Path | None) -> Path:
    base_dir = Path(model_dir) if model_dir is not None else _default_model_dir(family)
    return base_dir / f"{target}.cbm"


def _resolve_metadata_path(*, family: str, model_dir: str | Path | None) -> Path:
    base_dir = Path(model_dir) if model_dir is not None else _default_model_dir(family)
    return base_dir / "metadata.json"


def _default_model_dir(family: str) -> Path:
    if family == MODEL_FAMILY_STAGE_23:
        return Paths.stage_23_model_dir
    return Paths.stage_4_model_dir
