from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Lock

import faiss
import numpy as np
import pandas as pd
import structlog

from core.data_config import Paths
from ml.candidates import transactions_candidates
from ml.services.merged_user_feature_snapshot import ensure_merged_user_feature_snapshot
from ml.services.user_id import normalize_user_id

DATA_PATH = Paths.merged_user_features_parquet_path
INDEX_PATH = Paths.tx_interaction_hnsw_index_path
ID_COLUMN = transactions_candidates.ID_COLUMN
EMPTY_RESULT_COLUMNS = ["rank", "distance_euclidean", "query_party_rk", "impactful_features", ID_COLUMN]

_SEARCH_LOCK = Lock()
_SEARCH_BUNDLE = None

log: structlog.BoundLogger = structlog.get_logger(__name__)


def warmup_search_bundle(config=None):
    effective_config = transactions_candidates._to_candidate_config(config)
    return _get_search_bundle(search_bundle=None, config=effective_config)


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
    support_weights = transactions_candidates._compute_support_weights_from_scores(
        top_distances,
        lower_is_better=True,
    )
    impactful_features = transactions_candidates._extract_impactful_features_batch(
        query_vector=scaled_matrix[query_position],
        candidate_matrix=scaled_matrix[top_indices],
        feature_names=feature_names,
        top_k=effective_config.an_impactful_features,
        support_weights=support_weights,
        tau=transactions_candidates.IMPACTFUL_FEATURE_SCORE_TAU,
    )

    result_df = search_bundle["output_df"].iloc[top_indices].copy().reset_index(drop=True)
    result_df.insert(0, "rank", np.arange(1, len(result_df) + 1))
    result_df.insert(1, "distance_euclidean", top_distances.astype(float))
    result_df.insert(2, "query_party_rk", resolved_user_id)
    result_df.insert(3, "impactful_features", impactful_features)
    return result_df


def _get_search_bundle(search_bundle=None, config=None):
    if search_bundle is not None:
        if "faiss_index" not in search_bundle and search_bundle.get("scaled_matrix") is not None:
            search_bundle["faiss_index"] = _load_or_build_faiss_index(
                scaled_matrix=search_bundle["scaled_matrix"],
                config=config or transactions_candidates.CandidateConfig(),
                dataset_path=Path(search_bundle.get("dataset_path", DATA_PATH)),
            )
        return search_bundle

    merged_path = ensure_merged_user_feature_snapshot()
    if not merged_path.exists():
        return None

    dataset_mtime = merged_path.stat().st_mtime

    global _SEARCH_BUNDLE
    if _SEARCH_BUNDLE is not None and _SEARCH_BUNDLE.get("dataset_mtime") == dataset_mtime:
        return _SEARCH_BUNDLE

    with _SEARCH_LOCK:
        if _SEARCH_BUNDLE is not None and _SEARCH_BUNDLE.get("dataset_mtime") == dataset_mtime:
            return _SEARCH_BUNDLE

        df = _load_merged_dataset(merged_path)
        if df.empty:
            _SEARCH_BUNDLE = {
                "output_df": pd.DataFrame(columns=[ID_COLUMN]),
                "scaled_matrix": np.empty((0, 0), dtype=np.float32),
                "feature_names": np.asarray([], dtype=object),
                "user_ids": np.asarray([], dtype=object),
                "id_index": pd.Index([], name=ID_COLUMN, dtype="string"),
                "dataset_path": str(merged_path),
                "dataset_mtime": dataset_mtime,
            }
            return _SEARCH_BUNDLE

        date_columns = _infer_merged_date_columns(df)
        bundle = transactions_candidates.prepare_search_artifacts(df, date_columns=date_columns)
        bundle["dataset_path"] = str(merged_path)
        bundle["dataset_mtime"] = dataset_mtime
        bundle["faiss_index"] = _load_or_build_faiss_index(
            scaled_matrix=bundle["scaled_matrix"],
            config=config or transactions_candidates.CandidateConfig(),
            dataset_path=merged_path,
        )
        _SEARCH_BUNDLE = bundle

    return _SEARCH_BUNDLE


def _load_merged_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    df[ID_COLUMN] = df[ID_COLUMN].map(lambda raw_id: normalize_user_id(raw_id, allow_hash=True)).astype("string")
    df = df[df[ID_COLUMN].notna()].copy()
    return df


def _infer_merged_date_columns(df: pd.DataFrame) -> list[str]:
    date_columns = []
    for column in df.columns:
        if column == ID_COLUMN:
            continue
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            date_columns.append(column)
            continue
        if not str(column).endswith(("_at", "_dttm")):
            continue
        coerced_series = pd.to_datetime(series, errors="coerce", utc=False)
        if coerced_series.notna().any():
            date_columns.append(column)
    return date_columns


def _load_or_build_faiss_index(*, scaled_matrix, config, dataset_path: Path):
    feature_count = int(scaled_matrix.shape[1]) if scaled_matrix.ndim == 2 else 0
    if feature_count <= 0:
        raise ValueError("Merged snapshot does not contain any usable feature columns for FAISS.")

    index_path = Path(INDEX_PATH)
    if index_path.exists() and index_path.stat().st_mtime >= dataset_path.stat().st_mtime:
        try:
            index = faiss.read_index(str(index_path))
            if int(index.d) == feature_count and int(index.ntotal) == int(scaled_matrix.shape[0]):
                return index
            log.warning(
                "tx_interaction_candidates_index_stale_shape_mismatch",
                index_path=str(index_path),
                index_dim=int(index.d),
                feature_count=feature_count,
                index_size=int(index.ntotal),
                row_count=int(scaled_matrix.shape[0]),
            )
        except Exception as exc:
            log.warning(
                "tx_interaction_candidates_index_load_failed",
                index_path=str(index_path),
                error=str(exc),
            )

    index_config = replace(
        transactions_candidates._to_candidate_config(config),
        topk_trans_sim=max(
            int(getattr(config, "topk_inter_sim", 0) or 0),
            int(getattr(config, "topk_trans_sim", 0) or 0),
        ),
    )
    index = transactions_candidates.build_faiss_index(scaled_matrix, index_config)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    return index


def _empty_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_RESULT_COLUMNS)
