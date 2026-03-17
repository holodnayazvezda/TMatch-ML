from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
import numpy as np
import pandas as pd
import structlog
from sqlalchemy import String, case, cast, func, select

from core.data_config import InteractionWeightsMapper, Paths
from ml.candidates import transactions_candidates
from ml.infra.db.models import Interaction
from ml.infra.db.session import SessionLocal
from ml.services.merged_user_feature_snapshot import build_merged_user_feature_snapshot
from ml.services.user_id import normalize_user_id

log: structlog.BoundLogger = structlog.get_logger(__name__)

_PATHS = Paths()


def build_transaction_artifacts(*, force: bool = False) -> dict[str, Any]:
    started_at = perf_counter()
    dataset_path = transactions_candidates.resolve_dataset_path()
    bundle_path = _PATHS.transaction_search_bundle_path
    index_path = _PATHS.transaction_hnsw_index_path

    if not force and _is_artifact_fresh(artifact_paths=(bundle_path, index_path), source_path=dataset_path):
        summary = _base_summary("transactions", "up_to_date", started_at)
        summary["dataset_path"] = str(dataset_path)
        return _record_summary(summary)

    df = transactions_candidates.load_dataset(dataset_path)
    if df.empty:
        summary = _base_summary("transactions", "empty_source", started_at)
        summary["dataset_path"] = str(dataset_path)
        return _record_summary(summary)

    config = transactions_candidates.CandidateConfig()
    bundle = transactions_candidates.prepare_search_artifacts(df)
    bundle.update(transactions_candidates.build_cluster_artifacts(bundle["scaled_matrix"], config=config))
    index = transactions_candidates.build_faiss_index(bundle["scaled_matrix"], config)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    _save_search_bundle(
        bundle_path,
        bundle,
        dataset_path=dataset_path,
        source_name="transactions",
    )

    summary = _base_summary("transactions", "built", started_at)
    summary.update(
        {
            "dataset_path": str(dataset_path),
            "row_count": int(bundle["output_df"].shape[0]),
            "feature_count": int(bundle["scaled_matrix"].shape[1]) if bundle["scaled_matrix"].ndim == 2 else 0,
            "index_size": int(index.ntotal),
        }
    )
    return _record_summary(summary)


def build_tx_interaction_artifacts(*, force: bool = False) -> dict[str, Any]:
    started_at = perf_counter()
    merged_path = build_merged_user_feature_snapshot(force=force)
    bundle_path = _PATHS.tx_interaction_search_bundle_path
    index_path = _PATHS.tx_interaction_hnsw_index_path

    if not merged_path.exists():
        summary = _base_summary("tx_interaction", "missing_source", started_at)
        summary["dataset_path"] = str(merged_path)
        return _record_summary(summary)

    if not force and _is_artifact_fresh(artifact_paths=(bundle_path, index_path), source_path=merged_path):
        summary = _base_summary("tx_interaction", "up_to_date", started_at)
        summary["dataset_path"] = str(merged_path)
        return _record_summary(summary)

    df = pd.read_parquet(merged_path)
    df[transactions_candidates.ID_COLUMN] = (
        df[transactions_candidates.ID_COLUMN]
        .map(lambda raw_id: normalize_user_id(raw_id, allow_hash=True))
        .astype("string")
    )
    df = df[df[transactions_candidates.ID_COLUMN].notna()].copy()
    if df.empty:
        summary = _base_summary("tx_interaction", "empty_source", started_at)
        summary["dataset_path"] = str(merged_path)
        return _record_summary(summary)

    date_columns = [
        column
        for column in df.columns
        if column != transactions_candidates.ID_COLUMN
        and (
            pd.api.types.is_datetime64_any_dtype(df[column])
            or str(column).endswith(("_at", "_dttm"))
        )
    ]
    bundle = transactions_candidates.prepare_search_artifacts(df, date_columns=date_columns)
    config = transactions_candidates.CandidateConfig()
    index_config = replace(
        config,
        topk_trans_sim=max(int(config.topk_inter_sim), int(config.topk_trans_sim)),
    )
    index = transactions_candidates.build_faiss_index(bundle["scaled_matrix"], index_config)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    _save_search_bundle(
        bundle_path,
        bundle,
        dataset_path=merged_path,
        source_name="tx_interaction",
    )

    summary = _base_summary("tx_interaction", "built", started_at)
    summary.update(
        {
            "dataset_path": str(merged_path),
            "row_count": int(bundle["output_df"].shape[0]),
            "feature_count": int(bundle["scaled_matrix"].shape[1]) if bundle["scaled_matrix"].ndim == 2 else 0,
            "index_size": int(index.ntotal),
        }
    )
    return _record_summary(summary)


def build_popularity_snapshot(*, force: bool = False, max_candidates: int = 200_000) -> dict[str, Any]:
    started_at = perf_counter()
    snapshot_path = _PATHS.popularity_snapshot_path
    _ = force

    action_weights = {
        str(action).strip().lower(): float(weight)
        for action, weight in dict(InteractionWeightsMapper().dict_to_map).items()
    }
    normalized_action_expression = func.lower(func.trim(cast(Interaction.action, String)))
    action_weight_expression = case(
        *(
            (
                normalized_action_expression == action,
                float(weight),
            )
            for action, weight in action_weights.items()
        ),
        else_=0.0,
    )
    with SessionLocal() as session:
        stmt = (
            select(
                Interaction.candidate_user_id,
                func.sum(action_weight_expression).label("popularity_score"),
            )
            .where(Interaction.deleted_at.is_(None))
            .group_by(Interaction.candidate_user_id)
            .order_by(
                func.sum(action_weight_expression).desc(),
                Interaction.candidate_user_id.asc(),
            )
            .limit(max(1, int(max_candidates)))
        )
        rows = list(session.execute(stmt).all())

    prepared_rows: list[dict[str, Any]] = []
    for candidate_user_id, popularity_score in rows:
        normalized_candidate_user_id = normalize_user_id(candidate_user_id, allow_hash=True)
        if normalized_candidate_user_id is None:
            continue
        prepared_rows.append(
            {
                "candidate_id": normalized_candidate_user_id,
                "popularity_score": float(popularity_score),
            }
        )
    popularity_df = pd.DataFrame(prepared_rows)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if popularity_df.empty:
        empty_snapshot_df = pd.DataFrame(
            {
                "candidate_id": pd.Series(dtype="string"),
                "popularity_score": pd.Series(dtype="float64"),
                "global_rank": pd.Series(dtype="int64"),
                "built_at": pd.Series(dtype="string"),
            }
        )
        empty_snapshot_df.to_parquet(snapshot_path, index=False)
        summary = _base_summary("popularity", "built_empty", started_at)
        summary.update(
            {
                "snapshot_path": str(snapshot_path),
                "row_count": 0,
            }
        )
        return _record_summary(summary)

    popularity_df = popularity_df.sort_values(
        ["popularity_score", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    popularity_df["global_rank"] = np.arange(1, len(popularity_df) + 1, dtype=np.int64)
    popularity_df["built_at"] = _now_utc_iso()
    popularity_df.to_parquet(snapshot_path, index=False)

    summary = _base_summary("popularity", "built", started_at)
    summary.update(
        {
            "snapshot_path": str(snapshot_path),
            "row_count": int(popularity_df.shape[0]),
        }
    )
    return _record_summary(summary)


def build_all_artifacts(*, force: bool = False) -> dict[str, dict[str, Any]]:
    return {
        "transactions": build_transaction_artifacts(force=force),
        "tx_interaction": build_tx_interaction_artifacts(force=force),
        "popularity": build_popularity_snapshot(force=force),
    }


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build offline recommendation artifacts.")
    parser.add_argument("--all", action="store_true", help="Build all offline artifacts.")
    parser.add_argument("--transactions", action="store_true", help="Build transaction artifacts.")
    parser.add_argument("--tx-interaction", action="store_true", help="Build tx_interaction artifacts.")
    parser.add_argument("--popularity", action="store_true", help="Build popularity snapshot.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if artifact looks fresh.")
    args = parser.parse_args(argv)

    run_all = bool(args.all or not (args.transactions or args.tx_interaction or args.popularity))
    payload: dict[str, Any] = {}
    if run_all or args.transactions:
        payload["transactions"] = build_transaction_artifacts(force=bool(args.force))
    if run_all or args.tx_interaction:
        payload["tx_interaction"] = build_tx_interaction_artifacts(force=bool(args.force))
    if run_all or args.popularity:
        payload["popularity"] = build_popularity_snapshot(force=bool(args.force))
    print(json.dumps(payload, ensure_ascii=True, indent=2))


def _save_search_bundle(
    bundle_path: Path,
    bundle: dict[str, Any],
    *,
    dataset_path: Path,
    source_name: str,
) -> None:
    serializable_bundle = {
        "output_df": bundle["output_df"],
        "feature_names": bundle["feature_names"],
        "scaled_matrix": bundle["scaled_matrix"],
        "user_ids": bundle["user_ids"],
        "id_index": bundle["id_index"],
        "cluster_count": bundle.get("cluster_count"),
        "cluster_assignments": bundle.get("cluster_assignments"),
        "cluster_member_indices": bundle.get("cluster_member_indices"),
        "cluster_order": bundle.get("cluster_order"),
        "cluster_centroids": bundle.get("cluster_centroids"),
        "dataset_path": str(dataset_path),
        "dataset_mtime": float(dataset_path.stat().st_mtime),
        "source_name": source_name,
        "built_at": _now_utc_iso(),
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with bundle_path.open("wb") as output_stream:
        pickle.dump(serializable_bundle, output_stream, protocol=pickle.HIGHEST_PROTOCOL)


def _is_artifact_fresh(*, artifact_paths: tuple[Path, ...], source_path: Path) -> bool:
    if not source_path.exists():
        return False
    source_mtime = float(source_path.stat().st_mtime)
    for artifact_path in artifact_paths:
        if not artifact_path.exists():
            return False
        if float(artifact_path.stat().st_mtime) < source_mtime:
            return False
    return True


def _record_summary(summary: dict[str, Any]) -> dict[str, Any]:
    log.info("offline_artifact_build_finished", **summary)
    manifest_path = _PATHS.offline_artifacts_manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_manifest(manifest_path)
    payload[str(summary["source"])] = dict(summary)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _base_summary(source: str, status: str, started_at: float) -> dict[str, Any]:
    return {
        "source": source,
        "status": status,
        "built_at": _now_utc_iso(),
        "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
    }


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
