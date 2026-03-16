from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ml.candidates import transactions_candidates
from ml.infra.db.interaction_feature_tables import user_interaction_features_table
from ml.infra.db.session import SessionLocal
from ml.services.recommendation_scoring import RecommendationStage
from ml.services.user_id import normalize_user_id

_TRANSACTION_SOURCE = "transactions"
_INTERACTION_SOURCE = "tx_interaction"
_EXOTIC_SOURCE = "exotic"
_POPULARITY_SOURCE = "popularity"


def attach_feature_snapshots(
    candidate_df: pd.DataFrame,
    *,
    actor_user_id: str,
    stage: RecommendationStage,
    session: Session | None = None,
) -> pd.DataFrame:
    result = candidate_df.copy()
    if result.empty:
        result["feature_snapshot"] = pd.Series(dtype=object)
        return result

    normalized_actor_user_id = normalize_user_id(actor_user_id, allow_hash=True)
    if normalized_actor_user_id is None:
        result["feature_snapshot"] = pd.Series([{} for _ in range(len(result))], dtype=object)
        return result

    rows = result.to_dict(orient="records")
    candidate_user_ids = [
        _normalize_candidate_id(row)
        for row in rows
    ]
    try:
        transaction_lookup = _load_transaction_feature_lookup([normalized_actor_user_id, *candidate_user_ids])
    except Exception:
        transaction_lookup = {}
    interaction_lookup: dict[str, dict[str, Any]] = {}
    if stage in {RecommendationStage.STAGE_3, RecommendationStage.STAGE_4}:
        try:
            interaction_lookup = _load_interaction_feature_lookup(
                [normalized_actor_user_id, *candidate_user_ids],
                session=session,
            )
        except Exception:
            interaction_lookup = {}

    snapshots: list[dict[str, Any]] = []
    for row in rows:
        candidate_user_id = _normalize_candidate_id(row)
        if candidate_user_id is None:
            snapshots.append({})
            continue

        snapshot = {
            "served_stage": stage.value,
            "actor_user_id": normalized_actor_user_id,
            "candidate_user_id": candidate_user_id,
        }
        snapshot.update(
            _flatten_prefixed_dict(
                transaction_lookup.get(normalized_actor_user_id),
                prefix="actor_transaction_",
                exclude_keys={"party_rk"},
            )
        )
        snapshot.update(
            _flatten_prefixed_dict(
                transaction_lookup.get(candidate_user_id),
                prefix="candidate_transaction_",
                exclude_keys={"party_rk"},
            )
        )
        if stage in {RecommendationStage.STAGE_3, RecommendationStage.STAGE_4}:
            snapshot.update(
                _flatten_prefixed_dict(
                    interaction_lookup.get(normalized_actor_user_id),
                    prefix="actor_interaction_",
                    exclude_keys={"user_id", "snapshot_at"},
                )
            )
            snapshot.update(
                _flatten_prefixed_dict(
                    interaction_lookup.get(candidate_user_id),
                    prefix="candidate_interaction_",
                    exclude_keys={"user_id", "snapshot_at"},
                )
            )
        snapshot.update(_extract_source_provenance(row))
        snapshots.append(snapshot)

    result["feature_snapshot"] = pd.Series(snapshots, dtype=object)
    return result


def build_feature_frame_from_snapshots(
    rows_or_frame: pd.DataFrame | Iterable[dict[str, Any]],
    *,
    family: str,
) -> pd.DataFrame:
    if isinstance(rows_or_frame, pd.DataFrame):
        frame = rows_or_frame.copy()
    else:
        frame = pd.DataFrame(list(rows_or_frame))
    if frame.empty:
        return frame

    excluded_columns = {
        "actor_user_id",
        "candidate_user_id",
        "served_stage",
    }
    if family == "stage_2_3":
        excluded_columns.update(
            {
                column
                for column in frame.columns
                if column.startswith("actor_interaction_") or column.startswith("candidate_interaction_")
            }
        )

    feature_columns = [column for column in frame.columns if column not in excluded_columns]
    if not feature_columns:
        return pd.DataFrame(index=frame.index)

    result = frame[feature_columns].copy()
    for column in result.columns:
        series = result[column]
        if pd.api.types.is_bool_dtype(series):
            result[column] = series.fillna(False).astype("int8")
        elif pd.api.types.is_numeric_dtype(series):
            result[column] = pd.to_numeric(series, errors="coerce")
        else:
            result[column] = series.astype("string").fillna("")
    return result


def _normalize_candidate_id(row: dict[str, Any]) -> str | None:
    for column in ("candidate_id", "party_rk", "user_id", "target_user_id", "id"):
        normalized_candidate_id = normalize_user_id(row.get(column), allow_hash=True)
        if normalized_candidate_id is not None:
            return normalized_candidate_id
    return None


def _load_transaction_feature_lookup(user_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    normalized_user_ids = _normalize_user_ids(user_ids)
    if not normalized_user_ids:
        return {}

    bundle = transactions_candidates.warmup_search_bundle()
    output_df = bundle["output_df"]
    id_index = bundle["id_index"]
    lookup: dict[str, dict[str, Any]] = {}
    for user_id in normalized_user_ids:
        if user_id not in id_index:
            continue
        row_index = int(id_index.get_loc(user_id))
        row = output_df.iloc[row_index].to_dict()
        row["party_rk"] = user_id
        lookup[user_id] = _to_json_safe(row)
    return lookup


def _load_interaction_feature_lookup(
    user_ids: Iterable[str],
    *,
    session: Session | None = None,
) -> dict[str, dict[str, Any]]:
    normalized_user_ids = _normalize_user_ids(user_ids)
    if not normalized_user_ids:
        return {}

    def _load(local_session: Session) -> dict[str, dict[str, Any]]:
        stmt = select(user_interaction_features_table).where(
            user_interaction_features_table.c.user_id.in_(normalized_user_ids)
        )
        result: dict[str, dict[str, Any]] = {}
        for row in local_session.execute(stmt).mappings():
            normalized_user_id = normalize_user_id(row.get("user_id"), allow_hash=True)
            if normalized_user_id is None:
                continue
            result[normalized_user_id] = _to_json_safe(dict(row))
        return result

    if session is not None:
        return _load(session)

    with SessionLocal() as local_session:
        return _load(local_session)


def _extract_source_provenance(source_row: dict[str, Any]) -> dict[str, Any]:
    has_retrieval_columns = any(str(column).startswith("retrieval_") for column in source_row)
    inferred_primary_source = _infer_primary_source(source_row)

    from_transaction = _resolve_source_membership(
        source_row,
        column_name="retrieval_transactions_is_candidate",
        inferred_primary_source=inferred_primary_source,
        source_name=_TRANSACTION_SOURCE,
    )
    from_interaction = _resolve_source_membership(
        source_row,
        column_name="retrieval_tx_interaction_is_candidate",
        inferred_primary_source=inferred_primary_source,
        source_name=_INTERACTION_SOURCE,
    )
    from_exotic = _resolve_source_membership(
        source_row,
        column_name="retrieval_exotic_is_candidate",
        inferred_primary_source=inferred_primary_source,
        source_name=_EXOTIC_SOURCE,
    )
    from_popularity = _resolve_source_membership(
        source_row,
        column_name="retrieval_popularity_is_candidate",
        inferred_primary_source=inferred_primary_source,
        source_name=_POPULARITY_SOURCE,
    )

    transaction_score = _first_float(
        source_row.get("retrieval_transactions_score"),
        source_row.get("distance_euclidean") if from_transaction and not has_retrieval_columns else None,
    )
    interaction_score = _first_float(source_row.get("retrieval_tx_interaction_score"))
    exotic_score = _first_float(
        source_row.get("retrieval_exotic_score"),
        source_row.get("distance_euclidean") if from_exotic and not has_retrieval_columns else None,
    )
    popularity_score = _first_float(
        source_row.get("retrieval_popularity_score"),
        source_row.get("popularity_score") if from_popularity and not has_retrieval_columns else None,
    )

    return {
        "from_transaction": bool(from_transaction),
        "from_interaction": bool(from_interaction),
        "from_exotic": bool(from_exotic),
        "from_popularity": bool(from_popularity),
        "retrieval_primary_source": source_row.get("retrieval_primary_source") or inferred_primary_source,
        "retrieval_sources_count": _first_int(source_row.get("retrieval_sources_count")),
        "retrieval_sources": source_row.get("retrieval_sources"),
        "transaction_rank": _first_int(
            source_row.get("retrieval_transactions_rank"),
            source_row.get("rank") if from_transaction and not has_retrieval_columns else None,
        ),
        "interaction_rank": _first_int(source_row.get("retrieval_tx_interaction_rank")),
        "exotic_rank": _first_int(
            source_row.get("retrieval_exotic_rank"),
            source_row.get("rank") if from_exotic and not has_retrieval_columns else None,
        ),
        "popularity_rank": _first_int(
            source_row.get("retrieval_popularity_rank"),
            source_row.get("rank") if from_popularity and not has_retrieval_columns else None,
        ),
        "transaction_score": transaction_score,
        "interaction_score": interaction_score,
        "exotic_score": exotic_score,
        "popularity_score": popularity_score,
        "retrieval_best_rank": _first_float(source_row.get("retrieval_best_rank")),
        "retrieval_mean_rank": _first_float(source_row.get("retrieval_mean_rank")),
        "retrieval_best_score": _first_float(source_row.get("retrieval_best_score")),
        "retrieval_mean_score": _first_float(source_row.get("retrieval_mean_score")),
        "pre_rerank_similarity": _first_float(
            source_row.get("distance_euclidean"),
            source_row.get("similarity"),
            transaction_score,
            interaction_score,
            exotic_score,
            popularity_score,
        ),
    }


def _resolve_source_membership(
    source_row: dict[str, Any],
    *,
    column_name: str,
    inferred_primary_source: str | None,
    source_name: str,
) -> bool:
    raw_value = source_row.get(column_name)
    if raw_value is not None:
        if isinstance(raw_value, str):
            return raw_value.strip().lower() in {"1", "true", "yes"}
        return bool(raw_value)
    return inferred_primary_source == source_name


def _infer_primary_source(source_row: dict[str, Any]) -> str | None:
    raw_primary_source = source_row.get("retrieval_primary_source")
    if raw_primary_source is not None:
        return str(raw_primary_source).strip()
    if "cluster_id" in source_row:
        return _EXOTIC_SOURCE
    if "popularity_score" in source_row:
        return _POPULARITY_SOURCE
    if "query_party_rk" in source_row and "distance_euclidean" in source_row:
        return _TRANSACTION_SOURCE
    return None


def _flatten_prefixed_dict(
    raw_mapping: dict[str, Any] | None,
    *,
    prefix: str,
    exclude_keys: set[str],
) -> dict[str, Any]:
    if not raw_mapping:
        return {}
    return {
        f"{prefix}{key}": _to_json_safe(value)
        for key, value in raw_mapping.items()
        if key not in exclude_keys
    }


def _normalize_user_ids(user_ids: Iterable[str]) -> list[str]:
    normalized: set[str] = set()
    for user_id in user_ids:
        normalized_user_id = normalize_user_id(user_id, allow_hash=True)
        if normalized_user_id is not None:
            normalized.add(normalized_user_id)
    return sorted(normalized)


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed != parsed:
            continue
        return parsed
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if str(value) == "<NA>":
        return None
    if isinstance(value, float) and value != value:
        return None
    return value
