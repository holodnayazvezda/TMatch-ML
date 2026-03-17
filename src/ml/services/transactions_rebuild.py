from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import uuid4

import structlog

from core.data_config import Paths
from ml.candidates import transactions_candidates
from ml.featurize import transactions_features
from ml.services.offline_artifacts import build_transaction_artifacts

log: structlog.BoundLogger = structlog.get_logger(__name__)

_REBUILD_LOCK = Lock()
_PATHS = Paths()


class TransactionsRebuildInProgressError(RuntimeError):
    pass


@dataclass(frozen=True)
class _PublishStep:
    target_path: Path
    backup_path: Path | None
    existed_before_publish: bool


def rebuild_transactions_data_and_artifacts() -> dict[str, Any]:
    log.info("transactions_rebuild_requested")
    if not _REBUILD_LOCK.acquire(blocking=False):
        raise TransactionsRebuildInProgressError("Transactions rebuild is already in progress.")

    overall_started_at = perf_counter()
    current_stage = "init"
    staging_run_dir = _PATHS.out_data_path / ".staging" / f"transactions-rebuild-{uuid4().hex}"
    try:
        log.info(
            "transactions_rebuild_started",
            staging_run_dir=str(staging_run_dir),
        )
        staging_run_dir.mkdir(parents=True, exist_ok=True)

        intermediate_dir = staging_run_dir / "intermediate"
        features_dir = intermediate_dir / "user_feature_stages"
        staged_tx_layer_path = intermediate_dir / transactions_features.TX_LAYER_PATH.name
        staged_base_features_path = features_dir / transactions_features.BASE_FEATURES_PATH.name
        staged_daily_features_path = features_dir / transactions_features.DAILY_FEATURES_PATH.name
        staged_category_features_path = features_dir / transactions_features.CATEGORY_FEATURES_PATH.name
        staged_mcc_features_path = features_dir / transactions_features.MCC_FEATURES_PATH.name
        staged_merchant_features_path = features_dir / transactions_features.MERCHANT_FEATURES_PATH.name
        staged_features_parquet_path = staging_run_dir / _PATHS.transaction_user_features_parquet_path.name
        staged_features_csv_path = staging_run_dir / _PATHS.transaction_user_features_csv_path.name
        staged_user_ids_csv_path = staging_run_dir / _PATHS.transaction_user_ids_csv_path.name

        current_stage = "build_transaction_features"
        stage_started_at = perf_counter()
        user_features_df = transactions_features.run_pipeline(
            intermediate_dir=intermediate_dir,
            features_dir=features_dir,
            tx_layer_path=staged_tx_layer_path,
            base_features_path=staged_base_features_path,
            daily_features_path=staged_daily_features_path,
            category_features_path=staged_category_features_path,
            mcc_features_path=staged_mcc_features_path,
            merchant_features_path=staged_merchant_features_path,
            final_parquet_path=staged_features_parquet_path,
            final_csv_path=staged_features_csv_path,
            rebuild_tx_layer=True,
            rebuild_feature_stages=True,
            rebuild_final_dataset=True,
        )
        _log_stage_finished(
            stage=current_stage,
            started_at=stage_started_at,
            row_count=int(user_features_df.height),
        )

        current_stage = "build_unique_user_ids"
        stage_started_at = perf_counter()
        user_ids_df = transactions_features.build_unique_user_ids_frame(user_features_df)
        staged_user_ids_csv_path.parent.mkdir(parents=True, exist_ok=True)
        user_ids_df.write_csv(staged_user_ids_csv_path)
        _log_stage_finished(
            stage=current_stage,
            started_at=stage_started_at,
            row_count=int(user_ids_df.height),
        )

        current_stage = "publish_datasets"
        stage_started_at = perf_counter()
        _publish_files_with_rollback(
            publish_mapping={
                staged_features_parquet_path: _PATHS.transaction_user_features_parquet_path,
                staged_features_csv_path: _PATHS.transaction_user_features_csv_path,
                staged_user_ids_csv_path: _PATHS.transaction_user_ids_csv_path,
            }
        )
        _log_stage_finished(
            stage=current_stage,
            started_at=stage_started_at,
            published_features_parquet_path=str(_PATHS.transaction_user_features_parquet_path),
            published_features_csv_path=str(_PATHS.transaction_user_features_csv_path),
            published_user_ids_csv_path=str(_PATHS.transaction_user_ids_csv_path),
        )

        current_stage = "rebuild_transaction_artifacts"
        stage_started_at = perf_counter()
        artifacts_summary = build_transaction_artifacts(force=True)
        _log_stage_finished(
            stage=current_stage,
            started_at=stage_started_at,
            status=artifacts_summary.get("status"),
            row_count=artifacts_summary.get("row_count"),
        )

        current_stage = "reload_transaction_bundle"
        stage_started_at = perf_counter()
        reload_status = transactions_candidates.reload_prebuilt_search_bundle()
        _log_stage_finished(
            stage=current_stage,
            started_at=stage_started_at,
            status=reload_status.get("status"),
            in_memory_ready=reload_status.get("in_memory_ready"),
        )

        summary = {
            "status": "ok",
            "elapsed_ms": round((perf_counter() - overall_started_at) * 1000, 2),
            "row_count": int(user_features_df.height),
            "user_id_count": int(user_ids_df.height),
            "paths": {
                "transaction_user_features_parquet_path": str(_PATHS.transaction_user_features_parquet_path),
                "transaction_user_features_csv_path": str(_PATHS.transaction_user_features_csv_path),
                "transaction_user_ids_csv_path": str(_PATHS.transaction_user_ids_csv_path),
                "transaction_search_bundle_path": str(_PATHS.transaction_search_bundle_path),
                "transaction_hnsw_index_path": str(_PATHS.transaction_hnsw_index_path),
            },
            "artifact_summary": artifacts_summary,
            "reload_status": reload_status,
        }
        log.info("transactions_rebuild_succeeded", **summary)
        return summary
    except Exception as exc:
        log.exception(
            "transactions_rebuild_failed",
            stage=current_stage,
            error=str(exc),
        )
        raise
    finally:
        try:
            shutil.rmtree(staging_run_dir, ignore_errors=True)
        finally:
            _REBUILD_LOCK.release()


def _log_stage_finished(*, stage: str, started_at: float, **payload: Any) -> None:
    log.info(
        "transactions_rebuild_stage_finished",
        stage=stage,
        elapsed_ms=round((perf_counter() - started_at) * 1000, 2),
        **payload,
    )


def _publish_files_with_rollback(*, publish_mapping: dict[Path, Path]) -> None:
    publish_steps: list[_PublishStep] = []
    try:
        for source_path, target_path in publish_mapping.items():
            if not source_path.exists():
                raise FileNotFoundError(f"Source file for publish does not exist: {source_path}")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            existed_before_publish = target_path.exists()
            backup_path = None
            if existed_before_publish:
                backup_path = target_path.with_name(f".{target_path.name}.bak.{uuid4().hex}")
                target_path.replace(backup_path)

            publish_steps.append(
                _PublishStep(
                    target_path=target_path,
                    backup_path=backup_path,
                    existed_before_publish=existed_before_publish,
                )
            )
            source_path.replace(target_path)
    except Exception:
        for publish_step in reversed(publish_steps):
            if publish_step.target_path.exists():
                publish_step.target_path.unlink()
            if publish_step.existed_before_publish and publish_step.backup_path and publish_step.backup_path.exists():
                publish_step.backup_path.replace(publish_step.target_path)
        raise
    else:
        for publish_step in publish_steps:
            if publish_step.backup_path and publish_step.backup_path.exists():
                publish_step.backup_path.unlink()
