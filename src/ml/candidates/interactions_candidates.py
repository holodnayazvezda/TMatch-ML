from __future__ import annotations

import pickle
from pathlib import Path
from threading import Lock

import faiss
import numpy as np
import pandas as pd
import structlog

from core.data_config import Paths
from ml.candidates import transactions_candidates
from ml.services.serving_policy import log_lazy_rebuild_blocked

DATA_PATH = Paths.merged_user_features_parquet_path
INDEX_PATH = Paths.tx_interaction_hnsw_index_path
SEARCH_BUNDLE_PATH = Paths.tx_interaction_search_bundle_path
ID_COLUMN = transactions_candidates.ID_COLUMN
EMPTY_RESULT_COLUMNS = ["rank", "distance_euclidean", "query_party_rk", "impactful_features", ID_COLUMN]

_SEARCH_LOCK = Lock()
_SEARCH_BUNDLE = None

log: structlog.BoundLogger = structlog.get_logger(__name__)


def warmup_search_bundle(config=None):
    _ = transactions_candidates._to_candidate_config(config)
    bundle = _get_search_bundle(search_bundle=None, config=config)
    if bundle is None:
        status_payload = get_search_bundle_artifact_status()
        log.warning("tx_interaction_candidates_warmup_degraded", **status_payload)
        return None
    return bundle


def get_interaction_candidates(user_id, config=transactions_candidates.CandidateConfig(), search_bundle=None):
    request_object = user_id
    resolved_user_id = transactions_candidates._resolve_user_id(request_object)
    effective_config = transactions_candidates._to_candidate_config(
        request_object if hasattr(request_object, "user_id") else config
    )

    search_bundle = _get_search_bundle(search_bundle=search_bundle, config=effective_config)
    if not search_bundle or search_bundle["output_df"].empty:
        return _empty_candidate_frame()

    id_index = search_bundle["id_index"]
    if resolved_user_id not in id_index:
        log.info(
            "tx_interaction_candidates_user_not_found",
            user_id=resolved_user_id,
            dataset_path=str(search_bundle.get("dataset_path", DATA_PATH)),
        )
        return _empty_candidate_frame()

    user_ids = search_bundle["user_ids"]
    scaled_matrix = search_bundle["scaled_matrix"]
    feature_names = search_bundle["feature_names"]

    if len(scaled_matrix) <= 1:
        return _empty_candidate_frame()

    query_position = int(id_index.get_loc(resolved_user_id))
    query_vector = scaled_matrix[query_position : query_position + 1]
    result_limit = min(
        len(scaled_matrix) - 1,
        max(
            int(getattr(effective_config, "topk_inter_sim", 0) or 0),
            150,
        ),
    )
    if result_limit <= 0:
        return _empty_candidate_frame()

    candidate_count = min(
        len(scaled_matrix),
        max(
            result_limit,
            int(getattr(effective_config, "n_min_candidates", 0) or 0),
            result_limit + 1,
        ),
    )

    index = search_bundle["faiss_index"]
    _, raw_candidate_indices = index.search(query_vector, candidate_count)
    raw_candidate_indices = raw_candidate_indices[0]

    unique_candidate_indices = []
    seen_indices = set()
    for candidate_index in raw_candidate_indices:
        candidate_index = int(candidate_index)
        if candidate_index < 0 or candidate_index == query_position:
            continue
        if candidate_index in seen_indices:
            continue
        seen_indices.add(candidate_index)
        unique_candidate_indices.append(candidate_index)

    if not unique_candidate_indices:
        return _empty_candidate_frame()

    candidate_indices = np.asarray(unique_candidate_indices, dtype=np.int32)
    exact_distances = np.linalg.norm(
        scaled_matrix[candidate_indices] - query_vector[0],
        axis=1,
    )

    filtered_indices, filtered_distances = transactions_candidates._filter_blocked_candidates(
        actor_user_id=resolved_user_id,
        candidate_indices=candidate_indices,
        exact_distances=exact_distances,
        search_bundle=search_bundle,
    )
    if filtered_indices.size == 0:
        return _empty_candidate_frame()

    top_indices = filtered_indices[:result_limit]
    top_distances = filtered_distances[:result_limit]
    impactful_features = [
        transactions_candidates._extract_impactful_features(
            query_vector=scaled_matrix[query_position],
            candidate_vector=scaled_matrix[candidate_index],
            feature_names=feature_names,
            top_k=effective_config.an_impactful_features,
        )
        for candidate_index in top_indices
    ]

    result_df = search_bundle["output_df"].iloc[top_indices].copy().reset_index(drop=True)
    result_df.insert(0, "rank", np.arange(1, len(result_df) + 1))
    result_df.insert(1, "distance_euclidean", top_distances.astype(float))
    result_df.insert(2, "query_party_rk", resolved_user_id)
    result_df.insert(3, "impactful_features", impactful_features)
    result_df["impactful_features_origin"] = "vector_distance_contribution"
    result_df["impactful_features_seed_basis"] = None
    return result_df


def _get_search_bundle(search_bundle=None, config=None):
    if search_bundle is not None:
        if "faiss_index" not in search_bundle:
            log_lazy_rebuild_blocked(
                source="tx_interaction",
                reason="search_bundle_missing_faiss_index",
                artifact_path=SEARCH_BUNDLE_PATH,
            )
            return None
        return search_bundle

    global _SEARCH_BUNDLE
    if _SEARCH_BUNDLE is not None:
        return _SEARCH_BUNDLE

    with _SEARCH_LOCK:
        if _SEARCH_BUNDLE is not None:
            return _SEARCH_BUNDLE
        _SEARCH_BUNDLE = _load_prebuilt_search_bundle()
    return _SEARCH_BUNDLE


def get_search_bundle_artifact_status() -> dict[str, object]:
    bundle_path = Path(SEARCH_BUNDLE_PATH)
    index_path = Path(INDEX_PATH)
    if not bundle_path.exists() or not index_path.exists():
        return {
            "source": "tx_interaction",
            "status": "missing",
            "bundle_path": str(bundle_path),
            "index_path": str(index_path),
        }

    try:
        with bundle_path.open("rb") as input_stream:
            payload = pickle.load(input_stream)
    except Exception as exc:
        return {
            "source": "tx_interaction",
            "status": "invalid",
            "bundle_path": str(bundle_path),
            "index_path": str(index_path),
            "error": str(exc),
        }

    row_count = int(payload.get("output_df", pd.DataFrame()).shape[0])
    latest_mtime = max(float(bundle_path.stat().st_mtime), float(index_path.stat().st_mtime))
    artifact_age_seconds = max(0.0, pd.Timestamp.utcnow().timestamp() - latest_mtime)
    return {
        "source": "tx_interaction",
        "status": "ready",
        "bundle_path": str(bundle_path),
        "index_path": str(index_path),
        "row_count": row_count,
        "artifact_age_seconds": round(artifact_age_seconds, 3),
    }


def _load_prebuilt_search_bundle() -> dict[str, object] | None:
    bundle_path = Path(SEARCH_BUNDLE_PATH)
    index_path = Path(INDEX_PATH)
    if not bundle_path.exists():
        log_lazy_rebuild_blocked(
            source="tx_interaction",
            reason="prebuilt_bundle_missing",
            artifact_path=bundle_path,
        )
        return None
    if not index_path.exists():
        log_lazy_rebuild_blocked(
            source="tx_interaction",
            reason="prebuilt_index_missing",
            artifact_path=index_path,
        )
        return None

    try:
        with bundle_path.open("rb") as input_stream:
            bundle = pickle.load(input_stream)
    except Exception as exc:
        log.warning(
            "tx_interaction_candidates_bundle_load_failed",
            bundle_path=str(bundle_path),
            error=str(exc),
        )
        return None

    required_fields = {"output_df", "feature_names", "scaled_matrix", "user_ids", "id_index"}
    if not required_fields.issubset(bundle):
        missing_fields = sorted(required_fields.difference(bundle))
        log.warning(
            "tx_interaction_candidates_bundle_invalid",
            bundle_path=str(bundle_path),
            missing_fields=missing_fields,
        )
        return None

    try:
        index = faiss.read_index(str(index_path))
    except Exception as exc:
        log.warning(
            "tx_interaction_candidates_index_load_failed",
            index_path=str(index_path),
            error=str(exc),
        )
        return None

    scaled_matrix = np.asarray(bundle["scaled_matrix"])
    feature_count = int(scaled_matrix.shape[1]) if scaled_matrix.ndim == 2 else 0
    if feature_count <= 0:
        return None
    if int(index.d) != feature_count or int(index.ntotal) != int(scaled_matrix.shape[0]):
        log.warning(
            "tx_interaction_candidates_index_shape_mismatch",
            index_path=str(index_path),
            index_dim=int(index.d),
            feature_count=feature_count,
            index_size=int(index.ntotal),
            row_count=int(scaled_matrix.shape[0]),
        )
        return None

    bundle["faiss_index"] = index
    return bundle


def _empty_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_RESULT_COLUMNS)
