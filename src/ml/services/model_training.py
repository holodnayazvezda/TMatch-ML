from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.data_config import Paths
from ml.services.recommendation_feature_snapshot import build_feature_frame_from_snapshots
from ml.services.recommendation_ranking import (
    MODEL_FAMILY_STAGE_23,
    MODEL_FAMILY_STAGE_4,
    save_model_metadata,
)
from ml.services.recommendation_store import get_train_rows

TARGETS = ("like", "dislike", "match", "block")
FEATURE_BUILDER_VERSION = "v1"


def train_model_family(
    family: str,
    *,
    limit: int | None = None,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    train_rows = _load_family_rows(family=family, limit=limit)
    if not train_rows:
        return {
            "family": family,
            "row_count": 0,
            "trained_targets": [],
            "skipped_targets": list(TARGETS),
        }

    feature_snapshots = [
        row.get("features_json") or {}
        for row in train_rows
    ]
    feature_frame = build_feature_frame_from_snapshots(feature_snapshots, family=family)
    if feature_frame.empty:
        return {
            "family": family,
            "row_count": len(train_rows),
            "trained_targets": [],
            "skipped_targets": list(TARGETS),
        }

    categorical_columns = [
        column
        for column in feature_frame.columns
        if not pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    save_model_metadata(
        family=family,
        feature_columns=list(feature_frame.columns),
        categorical_columns=categorical_columns,
        feature_builder_version=FEATURE_BUILDER_VERSION,
        model_dir=model_dir,
    )

    trained_targets: list[str] = []
    skipped_targets: list[str] = []
    for target in TARGETS:
        target_column = f"label_true_{target}"
        labels = pd.Series([row.get(target_column) for row in train_rows], dtype="boolean")
        valid_mask = labels.notna()
        if valid_mask.sum() < 2 or labels.loc[valid_mask].nunique(dropna=True) < 2:
            skipped_targets.append(target)
            continue

        model = _train_binary_catboost(
            feature_frame=feature_frame.loc[valid_mask].reset_index(drop=True),
            labels=labels.loc[valid_mask].astype(int).reset_index(drop=True),
            categorical_columns=categorical_columns,
        )
        if model is None:
            skipped_targets.append(target)
            continue

        model_path = _resolve_model_path(family=family, target=target, model_dir=model_dir)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
        trained_targets.append(target)

    return {
        "family": family,
        "row_count": len(train_rows),
        "trained_targets": trained_targets,
        "skipped_targets": skipped_targets,
    }


def train_all_model_families(limit: int | None = None) -> dict[str, Any]:
    return {
        MODEL_FAMILY_STAGE_23: train_model_family(MODEL_FAMILY_STAGE_23, limit=limit),
        MODEL_FAMILY_STAGE_4: train_model_family(MODEL_FAMILY_STAGE_4, limit=limit),
    }


def run() -> None:
    print(train_all_model_families())


def _load_family_rows(*, family: str, limit: int | None) -> list[dict[str, Any]]:
    if family == MODEL_FAMILY_STAGE_23:
        return get_train_rows(limit=limit, served_stages=("stage_2", "stage_3"))
    if family == MODEL_FAMILY_STAGE_4:
        return get_train_rows(limit=limit, served_stages=("stage_4",))
    return []


def _train_binary_catboost(
    *,
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    categorical_columns: list[str],
):
    try:
        from catboost import CatBoostClassifier
    except Exception:
        return None

    train_frame = feature_frame.copy()
    categorical_indices: list[int] = []
    for column in train_frame.columns:
        if column in categorical_columns:
            categorical_indices.append(train_frame.columns.get_loc(column))
            train_frame[column] = train_frame[column].astype("string").fillna("")
        else:
            train_frame[column] = pd.to_numeric(train_frame[column], errors="coerce").fillna(0.0)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=200,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
    )
    model.fit(train_frame, labels, cat_features=categorical_indices)
    return model


def _resolve_model_path(*, family: str, target: str, model_dir: str | Path | None) -> Path:
    if model_dir is not None:
        return Path(model_dir) / f"{target}.cbm"
    if family == MODEL_FAMILY_STAGE_23:
        return Paths.stage_23_model_dir / f"{target}.cbm"
    return Paths.stage_4_model_dir / f"{target}.cbm"
