from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from core.data_config import Paths
from ml.services.model_versioning import (
    MlflowRunLogger,
    build_file_descriptor,
    fingerprint_rows,
    load_json_file,
    now_utc_iso,
    resolve_git_branch,
    resolve_git_commit,
    resolve_git_is_dirty,
    sha256_file,
    sha256_text,
    write_json_file,
)
from ml.services.recommendation_feature_snapshot import build_feature_frame_from_snapshots
from ml.services.recommendation_ranking import (
    MODEL_FAMILY_STAGE_23,
    MODEL_FAMILY_STAGE_4,
    save_model_metadata,
)
from ml.services.recommendation_store import get_train_rows

TARGETS = ("like", "dislike", "match", "block")
FEATURE_BUILDER_VERSION = "v1"
DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 200,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": 42,
    "verbose": False,
}
DEFAULT_TRAINING_CONFIG_PATH = Paths.project_root / "mlops" / "training_config.json"
DEFAULT_MLFLOW_TRACKING_URI = (Paths.out_data_path / "mlruns").resolve().as_uri()
DEFAULT_MLFLOW_EXPERIMENT_NAME = "recommendation-training"
_VERSION_FINGERPRINT_COLUMNS = [
    "ml_recommendation_id",
    "actor_user_id",
    "candidate_user_id",
    "shown_at",
    "last_interaction_at",
    "final_action",
    "served_stage",
    "label_true_like",
    "label_true_dislike",
    "label_true_match",
    "label_true_block",
]


def train_model_family(
    family: str,
    *,
    limit: int | None = None,
    model_dir: str | Path | None = None,
    targets: Iterable[str] = TARGETS,
    feature_builder_version: str = FEATURE_BUILDER_VERSION,
    catboost_params: dict[str, Any] | None = None,
    served_stages: tuple[str, ...] | None = None,
    mlflow_config: dict[str, Any] | None = None,
    version_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_targets = _normalize_targets(targets)
    catboost_config = _resolve_catboost_config(catboost_params)
    train_rows = _load_family_rows(
        family=family,
        limit=limit,
        served_stages=served_stages,
    )
    row_summary = _build_row_summary(train_rows)
    version_context = dict(version_context or {})

    mlflow_settings = _resolve_mlflow_config(mlflow_config)
    run_name = f"train_{family}_{now_utc_iso()}"
    run_tags = {
        "model_family": family,
        "data_fingerprint": row_summary["fingerprint"],
        "feature_builder_version": feature_builder_version,
    }
    for optional_tag in ("git_commit", "git_branch"):
        if optional_tag in version_context and version_context[optional_tag] is not None:
            run_tags[optional_tag] = version_context[optional_tag]
    if version_context.get("training_config_sha256"):
        run_tags["training_config_sha256"] = version_context["training_config_sha256"]
    if version_context.get("dvc_lock_sha256"):
        run_tags["dvc_lock_sha256"] = version_context["dvc_lock_sha256"]

    family_summary: dict[str, Any]
    with MlflowRunLogger(
        enabled=mlflow_settings["enabled"],
        tracking_uri=mlflow_settings["tracking_uri"],
        experiment_name=mlflow_settings["experiment_name"],
        run_name=run_name,
        tags=run_tags,
    ) as run_logger:
        run_logger.log_params(
            {
                "family": family,
                "limit": "none" if limit is None else int(limit),
                "feature_builder_version": feature_builder_version,
                "target_count": len(normalized_targets),
                "train_row_count": row_summary["row_count"],
                "train_row_fingerprint": row_summary["fingerprint"],
            }
        )
        run_logger.log_params({f"cb_{key}": value for key, value in catboost_config.items()})
        run_logger.log_metrics(
            {
                "train_row_count": row_summary["row_count"],
                "train_actor_count": row_summary["unique_actor_count"],
                "train_candidate_count": row_summary["unique_candidate_count"],
            }
        )

        family_summary = _train_family_models(
            family=family,
            train_rows=train_rows,
            model_dir=model_dir,
            targets=normalized_targets,
            feature_builder_version=feature_builder_version,
            catboost_params=catboost_config,
            run_logger=run_logger,
            row_summary=row_summary,
        )
        family_summary["mlflow"] = run_logger.to_dict()

    return family_summary


def _train_family_models(
    *,
    family: str,
    train_rows: list[dict[str, Any]],
    model_dir: str | Path | None,
    targets: tuple[str, ...],
    feature_builder_version: str,
    catboost_params: dict[str, Any],
    run_logger: MlflowRunLogger,
    row_summary: dict[str, Any],
) -> dict[str, Any]:
    if not train_rows:
        summary = {
            "family": family,
            "row_count": 0,
            "row_summary": row_summary,
            "trained_targets": [],
            "skipped_targets": list(targets),
            "target_metrics": {},
            "metadata": None,
            "model_artifacts": {},
        }
        run_logger.log_param("family_status", "no_rows")
        return summary

    feature_snapshots = [
        row.get("features_json") or {}
        for row in train_rows
    ]
    feature_frame = build_feature_frame_from_snapshots(feature_snapshots, family=family)
    if feature_frame.empty:
        summary = {
            "family": family,
            "row_count": len(train_rows),
            "row_summary": row_summary,
            "trained_targets": [],
            "skipped_targets": list(targets),
            "target_metrics": {},
            "metadata": None,
            "model_artifacts": {},
        }
        run_logger.log_param("family_status", "empty_features")
        return summary

    categorical_columns = [
        column
        for column in feature_frame.columns
        if not pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    save_model_metadata(
        family=family,
        feature_columns=list(feature_frame.columns),
        categorical_columns=categorical_columns,
        feature_builder_version=feature_builder_version,
        model_dir=model_dir,
    )
    metadata_path = _resolve_metadata_path(family=family, model_dir=model_dir)
    metadata_descriptor = build_file_descriptor(metadata_path)
    if metadata_path.exists():
        run_logger.log_artifact(metadata_path, artifact_path=f"models/{family}")

    trained_targets: list[str] = []
    skipped_targets: list[str] = []
    target_metrics: dict[str, dict[str, Any]] = {}
    model_artifacts: dict[str, dict[str, Any]] = {}
    for target in targets:
        target_column = f"label_true_{target}"
        labels = pd.Series([row.get(target_column) for row in train_rows], dtype="boolean")
        valid_mask = labels.notna()
        valid_labels = labels.loc[valid_mask].astype(int).reset_index(drop=True)
        sample_count = int(valid_labels.shape[0])
        positive_count = int(valid_labels.sum()) if sample_count > 0 else 0
        positive_rate = float((positive_count / sample_count) if sample_count > 0 else 0.0)

        if sample_count < 2 or valid_labels.nunique(dropna=True) < 2:
            skipped_targets.append(target)
            target_metrics[target] = {
                "status": "skipped_insufficient_labels",
                "sample_count": sample_count,
                "positive_count": positive_count,
                "positive_rate": round(positive_rate, 6),
            }
            run_logger.log_metric(f"{target}_sample_count", sample_count)
            run_logger.log_metric(f"{target}_positive_rate", round(positive_rate, 6))
            continue

        train_subset = feature_frame.loc[valid_mask].reset_index(drop=True)
        prepared_train_frame, categorical_indices = _prepare_catboost_frame(
            train_subset,
            categorical_columns=categorical_columns,
        )
        model = _train_binary_catboost(
            train_frame=prepared_train_frame,
            labels=valid_labels,
            categorical_indices=categorical_indices,
            catboost_params=catboost_params,
        )
        if model is None:
            skipped_targets.append(target)
            target_metrics[target] = {
                "status": "skipped_catboost_unavailable",
                "sample_count": sample_count,
                "positive_count": positive_count,
                "positive_rate": round(positive_rate, 6),
            }
            continue

        model_path = _resolve_model_path(family=family, target=target, model_dir=model_dir)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
        trained_targets.append(target)
        model_artifacts[target] = build_file_descriptor(model_path)
        run_logger.log_artifact(model_path, artifact_path=f"models/{family}")

        train_probabilities = _predict_binary_probabilities(model, prepared_train_frame)
        quality_metrics = _evaluate_binary_quality(valid_labels, train_probabilities)
        target_metrics[target] = {
            "status": "trained",
            "sample_count": sample_count,
            "positive_count": positive_count,
            "positive_rate": round(positive_rate, 6),
            **quality_metrics,
        }
        run_logger.log_metric(f"{target}_sample_count", sample_count)
        run_logger.log_metric(f"{target}_positive_rate", round(positive_rate, 6))
        for metric_name, metric_value in quality_metrics.items():
            if metric_value is None:
                continue
            run_logger.log_metric(f"{target}_{metric_name}", metric_value)

    family_summary = {
        "family": family,
        "row_count": len(train_rows),
        "row_summary": row_summary,
        "trained_targets": trained_targets,
        "skipped_targets": skipped_targets,
        "target_metrics": target_metrics,
        "metadata": metadata_descriptor,
        "model_artifacts": model_artifacts,
    }
    run_logger.log_param("family_status", "trained" if trained_targets else "no_models")
    run_logger.log_metric("trained_target_count", len(trained_targets))
    run_logger.log_metric("skipped_target_count", len(skipped_targets))
    return family_summary


def train_all_model_families(
    limit: int | None = None,
    *,
    config_path: str | Path | None = None,
    emit_dvc_metrics_path: str | Path | None = None,
    emit_version_manifest_path: str | Path | None = None,
    mlflow_enabled: bool | None = None,
) -> dict[str, Any]:
    resolved_config_path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else DEFAULT_TRAINING_CONFIG_PATH
    )
    training_config = _load_training_config(resolved_config_path)
    if limit is not None and int(limit) > 0:
        training_config["limit"] = int(limit)

    if mlflow_enabled is not None:
        training_config.setdefault("mlflow", {})
        training_config["mlflow"]["enabled"] = bool(mlflow_enabled)

    version_context = _build_version_context(
        config_path=resolved_config_path,
        training_config=training_config,
    )

    family_summaries: dict[str, Any] = {}
    for family, default_stages in (
        (MODEL_FAMILY_STAGE_23, ("stage_2", "stage_3")),
        (MODEL_FAMILY_STAGE_4, ("stage_4",)),
    ):
        family_config = dict(training_config.get("families", {}).get(family, {}))
        family_limit = family_config.get("limit", training_config.get("limit"))
        family_summaries[family] = train_model_family(
            family=family,
            limit=_coerce_positive_int(family_limit),
            targets=training_config.get("targets", TARGETS),
            feature_builder_version=str(training_config.get("feature_builder_version", FEATURE_BUILDER_VERSION)),
            catboost_params=training_config.get("catboost", {}),
            served_stages=tuple(
                _normalize_non_empty_strings(
                    family_config.get("served_stages", default_stages)
                )
            ) or default_stages,
            mlflow_config=training_config.get("mlflow", {}),
            version_context=version_context,
        )

    training_summary = {
        "generated_at": now_utc_iso(),
        "config": {
            "path": str(resolved_config_path),
            "exists": bool(resolved_config_path.exists()),
            "sha256": version_context.get("training_config_sha256"),
        },
        "git": {
            "branch": version_context.get("git_branch"),
            "commit": version_context.get("git_commit"),
            "is_dirty": version_context.get("git_is_dirty"),
        },
        "dvc": version_context.get("dvc", {}),
        "version_tracking": version_context.get("tracked_inputs", []),
        "families": family_summaries,
    }
    metrics_payload = _build_dvc_metrics_payload(training_summary)

    if emit_dvc_metrics_path is not None:
        write_json_file(
            Path(emit_dvc_metrics_path).expanduser().resolve(),
            metrics_payload,
        )
    if emit_version_manifest_path is not None:
        write_json_file(
            Path(emit_version_manifest_path).expanduser().resolve(),
            training_summary,
        )

    return training_summary


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train recommendation model families.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for training set.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_TRAINING_CONFIG_PATH),
        help="Path to JSON config with training/versioning settings.",
    )
    parser.add_argument(
        "--emit-dvc-metrics",
        default=None,
        help="Write numeric metrics JSON for DVC metrics tracking.",
    )
    parser.add_argument(
        "--emit-version-manifest",
        default=None,
        help="Write detailed training/version manifest JSON.",
    )
    parser.add_argument(
        "--mlflow",
        dest="mlflow_enabled",
        action="store_true",
        help="Force-enable MLflow logging for this run.",
    )
    parser.add_argument(
        "--no-mlflow",
        dest="mlflow_enabled",
        action="store_false",
        help="Force-disable MLflow logging for this run.",
    )
    parser.set_defaults(mlflow_enabled=None)
    args = parser.parse_args(argv)

    summary = train_all_model_families(
        limit=args.limit,
        config_path=args.config,
        emit_dvc_metrics_path=args.emit_dvc_metrics,
        emit_version_manifest_path=args.emit_version_manifest,
        mlflow_enabled=args.mlflow_enabled,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _load_family_rows(
    *,
    family: str,
    limit: int | None,
    served_stages: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    if served_stages is not None and len(served_stages) > 0:
        return get_train_rows(limit=limit, served_stages=served_stages)
    if family == MODEL_FAMILY_STAGE_23:
        return get_train_rows(limit=limit, served_stages=("stage_2", "stage_3"))
    if family == MODEL_FAMILY_STAGE_4:
        return get_train_rows(limit=limit, served_stages=("stage_4",))
    return []


def _train_binary_catboost(
    *,
    train_frame: pd.DataFrame,
    labels: pd.Series,
    categorical_indices: list[int],
    catboost_params: dict[str, Any],
):
    try:
        from catboost import CatBoostClassifier
    except Exception:
        return None

    model = CatBoostClassifier(**catboost_params)
    model.fit(train_frame, labels, cat_features=categorical_indices)
    return model


def _resolve_model_path(*, family: str, target: str, model_dir: str | Path | None) -> Path:
    if model_dir is not None:
        return Path(model_dir) / f"{target}.cbm"
    if family == MODEL_FAMILY_STAGE_23:
        return Paths.stage_23_model_dir / f"{target}.cbm"
    return Paths.stage_4_model_dir / f"{target}.cbm"


def _resolve_metadata_path(*, family: str, model_dir: str | Path | None) -> Path:
    if model_dir is not None:
        return Path(model_dir) / "metadata.json"
    if family == MODEL_FAMILY_STAGE_23:
        return Paths.stage_23_model_metadata_path
    return Paths.stage_4_model_metadata_path


def _prepare_catboost_frame(
    feature_frame: pd.DataFrame,
    *,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, list[int]]:
    prepared_frame = feature_frame.copy()
    categorical_indices: list[int] = []
    categorical_set = set(categorical_columns)
    for column in prepared_frame.columns:
        if column in categorical_set:
            categorical_indices.append(prepared_frame.columns.get_loc(column))
            prepared_frame[column] = prepared_frame[column].astype("string").fillna("")
        else:
            prepared_frame[column] = pd.to_numeric(prepared_frame[column], errors="coerce").fillna(0.0)
    return prepared_frame, categorical_indices


def _predict_binary_probabilities(model: Any, train_frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(train_frame)
    if getattr(probabilities, "ndim", 1) == 2 and probabilities.shape[1] > 1:
        return probabilities[:, 1]
    return probabilities.reshape(-1)


def _evaluate_binary_quality(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    labels_np = labels.to_numpy(dtype=np.float64)
    probabilities_np = np.asarray(probabilities, dtype=np.float64)
    probabilities_np = np.clip(probabilities_np, 1e-7, 1.0 - 1e-7)
    try:
        result["train_auc"] = round(float(roc_auc_score(labels_np, probabilities_np)), 6)
    except Exception:
        result["train_auc"] = None
    try:
        result["train_logloss"] = round(float(log_loss(labels_np, probabilities_np)), 6)
    except Exception:
        result["train_logloss"] = None
    return result


def _build_row_summary(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counter = Counter()
    shown_at_values: list[str] = []
    actor_ids: set[str] = set()
    candidate_ids: set[str] = set()

    label_positive_counts = {target: 0 for target in TARGETS}
    label_observed_counts = {target: 0 for target in TARGETS}
    for row in train_rows:
        stage_counter[str(row.get("served_stage") or "unknown")] += 1
        shown_at = row.get("shown_at")
        if shown_at:
            shown_at_values.append(str(shown_at))
        actor_user_id = row.get("actor_user_id")
        candidate_user_id = row.get("candidate_user_id")
        if actor_user_id:
            actor_ids.add(str(actor_user_id))
        if candidate_user_id:
            candidate_ids.add(str(candidate_user_id))
        for target in TARGETS:
            label_value = row.get(f"label_true_{target}")
            if label_value is None:
                continue
            label_observed_counts[target] += 1
            if bool(label_value):
                label_positive_counts[target] += 1

    shown_at_min = min(shown_at_values) if shown_at_values else None
    shown_at_max = max(shown_at_values) if shown_at_values else None
    return {
        "row_count": len(train_rows),
        "shown_at_min": shown_at_min,
        "shown_at_max": shown_at_max,
        "served_stage_counts": dict(stage_counter),
        "unique_actor_count": len(actor_ids),
        "unique_candidate_count": len(candidate_ids),
        "label_positive_counts": label_positive_counts,
        "label_observed_counts": label_observed_counts,
        "fingerprint": fingerprint_rows(
            train_rows,
            stable_columns=_VERSION_FINGERPRINT_COLUMNS,
        ),
    }


def _normalize_targets(raw_targets: Iterable[str]) -> tuple[str, ...]:
    normalized_targets: list[str] = []
    seen_targets: set[str] = set()
    for raw_target in raw_targets:
        normalized_target = str(raw_target).strip().lower()
        if not normalized_target:
            continue
        if normalized_target in seen_targets:
            continue
        seen_targets.add(normalized_target)
        normalized_targets.append(normalized_target)
    if not normalized_targets:
        return TARGETS
    return tuple(normalized_targets)


def _normalize_non_empty_strings(raw_values: Iterable[Any]) -> list[str]:
    normalized_values: list[str] = []
    seen_values: set[str] = set()
    for raw_value in raw_values:
        normalized_value = str(raw_value).strip()
        if not normalized_value or normalized_value in seen_values:
            continue
        normalized_values.append(normalized_value)
        seen_values.add(normalized_value)
    return normalized_values


def _resolve_catboost_config(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    merged_config = dict(DEFAULT_CATBOOST_PARAMS)
    for key, value in dict(raw_config or {}).items():
        merged_config[str(key)] = value
    if "verbose" not in merged_config:
        merged_config["verbose"] = False
    return merged_config


def _resolve_mlflow_config(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    mlflow_config = dict(raw_config or {})
    enabled_raw = mlflow_config.get("enabled", True)
    env_enabled = os.getenv("MLFLOW_ENABLED")
    if env_enabled is not None:
        enabled_raw = str(env_enabled).strip().lower() in {"1", "true", "yes", "on"}
    tracking_uri_raw = os.getenv(
        "MLFLOW_TRACKING_URI",
        mlflow_config.get("tracking_uri", DEFAULT_MLFLOW_TRACKING_URI),
    )
    experiment_name_raw = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        mlflow_config.get("experiment_name", DEFAULT_MLFLOW_EXPERIMENT_NAME),
    )
    return {
        "enabled": bool(enabled_raw),
        "tracking_uri": str(tracking_uri_raw),
        "experiment_name": str(experiment_name_raw),
    }


def _load_training_config(config_path: Path) -> dict[str, Any]:
    defaults = _default_training_config()
    if not config_path.exists():
        return defaults
    loaded_payload = load_json_file(config_path)
    if not isinstance(loaded_payload, dict):
        return defaults
    return _deep_merge_dict(defaults, loaded_payload)


def _default_training_config() -> dict[str, Any]:
    return {
        "limit": None,
        "feature_builder_version": FEATURE_BUILDER_VERSION,
        "targets": list(TARGETS),
        "catboost": dict(DEFAULT_CATBOOST_PARAMS),
        "mlflow": {
            "enabled": True,
            "tracking_uri": DEFAULT_MLFLOW_TRACKING_URI,
            "experiment_name": DEFAULT_MLFLOW_EXPERIMENT_NAME,
        },
        "families": {
            MODEL_FAMILY_STAGE_23: {
                "served_stages": ["stage_2", "stage_3"],
            },
            MODEL_FAMILY_STAGE_4: {
                "served_stages": ["stage_4"],
            },
        },
        "version_tracking": {
            "tracked_paths": [
                str(Paths.transaction_user_features_parquet_path),
                str(Paths.merged_user_features_parquet_path),
                str(Paths.popularity_snapshot_path),
                str(Paths.offline_artifacts_manifest_path),
            ],
        },
    }


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, override_value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(override_value, dict):
            result[key] = _deep_merge_dict(result[key], override_value)
            continue
        result[key] = override_value
    return result


def _build_version_context(*, config_path: Path, training_config: dict[str, Any]) -> dict[str, Any]:
    project_root = Paths.project_root
    dvc_yaml_path = project_root / "dvc.yaml"
    dvc_lock_path = project_root / "dvc.lock"

    tracked_paths = []
    raw_tracked_paths = (
        training_config.get("version_tracking", {}).get("tracked_paths", [])
    )
    for raw_path in raw_tracked_paths:
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = (project_root / path).resolve()
        tracked_paths.append(path)

    training_config_sha = (
        sha256_file(config_path)
        if config_path.exists()
        else sha256_text(json.dumps(training_config, ensure_ascii=True, sort_keys=True, default=str))
    )
    return {
        "git_branch": resolve_git_branch(project_root),
        "git_commit": resolve_git_commit(project_root),
        "git_is_dirty": resolve_git_is_dirty(project_root),
        "training_config_sha256": training_config_sha,
        "dvc_lock_sha256": sha256_file(dvc_lock_path) if dvc_lock_path.exists() else None,
        "dvc": {
            "dvc_yaml": build_file_descriptor(dvc_yaml_path),
            "dvc_lock": build_file_descriptor(dvc_lock_path),
        },
        "tracked_inputs": [build_file_descriptor(path) for path in tracked_paths],
    }


def _coerce_positive_int(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _build_dvc_metrics_payload(training_summary: dict[str, Any]) -> dict[str, Any]:
    families = training_summary.get("families", {})
    payload = {
        "generated_at": training_summary.get("generated_at"),
        "families": {},
    }
    for family_name, family_summary in families.items():
        target_metrics = family_summary.get("target_metrics", {})
        numeric_target_metrics: dict[str, dict[str, float | int]] = {}
        for target_name, metrics in target_metrics.items():
            if not isinstance(metrics, dict):
                continue
            numeric_target_metrics[target_name] = {}
            for metric_key in ("sample_count", "positive_count", "positive_rate", "train_auc", "train_logloss"):
                metric_value = metrics.get(metric_key)
                if isinstance(metric_value, bool):
                    continue
                if isinstance(metric_value, (int, float)):
                    numeric_target_metrics[target_name][metric_key] = metric_value
        payload["families"][family_name] = {
            "row_count": int(family_summary.get("row_count", 0)),
            "trained_target_count": int(len(family_summary.get("trained_targets", []))),
            "skipped_target_count": int(len(family_summary.get("skipped_targets", []))),
            "targets": numeric_target_metrics,
        }
    return payload


if __name__ == "__main__":
    run()
