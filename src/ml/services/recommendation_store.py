from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from ml.infra.db.init_db import init_db
from ml.infra.db.models import (
    Interaction,
    MLRecommendationGroup,
    MLRecommendationImpression,
)
from ml.infra.db.session import SessionLocal, engine
from ml.services.interaction_ingestion import TRAINING_INTERACTION_ACTIONS
from ml.services.recommendation_scoring import PREDICTION_COLUMN_BY_TARGET
from ml.services.user_id import normalize_user_id

GROUP_STATUS_PENDING = "pending"
GROUP_STATUS_FINALIZED = "finalized"
GROUP_STATUS_DISCARDED = "discarded"

ROW_STATUS_PENDING = "pending"
ROW_STATUS_FINALIZED = "finalized"
ROW_STATUS_DISCARDED = "discarded"


def save_recommendation_group(
    recommendation_id: str,
    actor_user_id: str,
    recommendation_rows: Sequence[dict[str, Any]],
    *,
    shown_at: datetime | None = None,
    served_stage: str = "stage_4",
) -> None:
    init_db()
    with SessionLocal() as session:
        with session.begin():
            save_recommendation_group_in_session(
                session,
                recommendation_id=recommendation_id,
                actor_user_id=actor_user_id,
                recommendation_rows=recommendation_rows,
                shown_at=shown_at,
                served_stage=served_stage,
            )


def save_recommendation_group_in_session(
    session: Session,
    *,
    recommendation_id: str,
    actor_user_id: str,
    recommendation_rows: Sequence[dict[str, Any]],
    shown_at: datetime | None = None,
    served_stage: str = "stage_4",
) -> None:
    if not recommendation_rows:
        return

    shown_timestamp = _to_aware_utc(shown_at or datetime.now(timezone.utc))
    if shown_timestamp is None:
        raise ValueError("shown_at must be a valid datetime")

    normalized_actor_user_id = _normalize_required_user_id(
        actor_user_id,
        field_name="actor_user_id",
    )
    normalized_recommendation_id = _normalize_required_string(
        recommendation_id,
        field_name="ml_recommendation_id",
    )
    normalized_served_stage = _normalize_required_string(
        served_stage,
        field_name="served_stage",
    )

    group_payload = {
        "ml_recommendation_id": normalized_recommendation_id,
        "actor_user_id": normalized_actor_user_id,
        "shown_at": shown_timestamp,
        "served_stage": normalized_served_stage,
        "group_status": GROUP_STATUS_PENDING,
        "resolved_at": None,
    }
    session.execute(
        pg_insert(MLRecommendationGroup.__table__)
        .values(group_payload)
        .on_conflict_do_update(
            index_elements=[MLRecommendationGroup.__table__.c.ml_recommendation_id],
            set_={
                "actor_user_id": pg_insert(MLRecommendationGroup.__table__).excluded.actor_user_id,
                "shown_at": pg_insert(MLRecommendationGroup.__table__).excluded.shown_at,
                "served_stage": pg_insert(MLRecommendationGroup.__table__).excluded.served_stage,
                "group_status": pg_insert(MLRecommendationGroup.__table__).excluded.group_status,
                "resolved_at": pg_insert(MLRecommendationGroup.__table__).excluded.resolved_at,
            },
        )
    )

    impression_payloads = []
    for recommendation_row in recommendation_rows:
        candidate_user_id = _normalize_required_user_id(
            recommendation_row.get("candidate_user_id"),
            field_name="candidate_user_id",
        )
        source_row = recommendation_row.get("source_row")
        if not isinstance(source_row, dict):
            raise ValueError("Each recommendation row must include source_row dict.")

        serialized_source_row = _to_json_safe(source_row)
        impression_payloads.append(
            {
                "ml_recommendation_id": normalized_recommendation_id,
                "actor_user_id": normalized_actor_user_id,
                "candidate_user_id": candidate_user_id,
                "shown_at": shown_timestamp,
                "last_interaction_at": None,
                "final_action": None,
                "row_status": ROW_STATUS_PENDING,
                "label_true_like": None,
                "label_true_dislike": None,
                "label_true_match": None,
                "label_true_block": None,
                "label_pred_like": _extract_prediction_value(source_row, "like"),
                "label_pred_dislike": _extract_prediction_value(source_row, "dislike"),
                "label_pred_match": _extract_prediction_value(source_row, "match"),
                "label_pred_block": _extract_prediction_value(source_row, "block"),
                "features_json": _build_feature_snapshot(serialized_source_row),
                "source_row_json": serialized_source_row,
            }
        )

    impression_table = MLRecommendationImpression.__table__
    session.execute(
        pg_insert(impression_table)
        .values(impression_payloads)
        .on_conflict_do_update(
            index_elements=[
                impression_table.c.ml_recommendation_id,
                impression_table.c.actor_user_id,
                impression_table.c.candidate_user_id,
            ],
            set_={
                "shown_at": pg_insert(impression_table).excluded.shown_at,
                "last_interaction_at": None,
                "final_action": None,
                "row_status": ROW_STATUS_PENDING,
                "label_true_like": None,
                "label_true_dislike": None,
                "label_true_match": None,
                "label_true_block": None,
                "label_pred_like": pg_insert(impression_table).excluded.label_pred_like,
                "label_pred_dislike": pg_insert(impression_table).excluded.label_pred_dislike,
                "label_pred_match": pg_insert(impression_table).excluded.label_pred_match,
                "label_pred_block": pg_insert(impression_table).excluded.label_pred_block,
                "features_json": pg_insert(impression_table).excluded.features_json,
                "source_row_json": pg_insert(impression_table).excluded.source_row_json,
            },
        )
    )


def run_with_user_recommendation_lock(
    actor_user_id: str,
    *,
    poll_wait_seconds: float,
    callback: Callable[[Session], Any],
) -> Any:
    normalized_actor_user_id = _normalize_required_user_id(
        actor_user_id,
        field_name="actor_user_id",
    )
    init_db()

    with engine.connect() as lock_connection:
        _acquire_user_recommendation_lock(lock_connection, normalized_actor_user_id)
        try:
            with SessionLocal() as session:
                with session.begin():
                    pending_group_ids = _list_pending_group_ids_for_user_in_session(session, normalized_actor_user_id)

                if pending_group_ids:
                    time.sleep(max(float(poll_wait_seconds), 0.0))
                    with session.begin():
                        unresolved_group_ids = _list_pending_group_ids_for_user_in_session(
                            session,
                            normalized_actor_user_id,
                        )
                        if unresolved_group_ids:
                            discard_groups_in_session(session, unresolved_group_ids)

                return callback(session)
        finally:
            _release_user_recommendation_lock(lock_connection, normalized_actor_user_id)


def get_pending_group_for_user(actor_user_id: str) -> dict[str, Any] | None:
    init_db()
    with SessionLocal() as session:
        with session.begin():
            return get_pending_group_for_user_in_session(session, actor_user_id)


def get_pending_group_for_user_in_session(session: Session, actor_user_id: str) -> dict[str, Any] | None:
    normalized_actor_user_id = _normalize_required_user_id(
        actor_user_id,
        field_name="actor_user_id",
    )
    stmt = (
        select(MLRecommendationGroup)
        .where(
            MLRecommendationGroup.actor_user_id == normalized_actor_user_id,
            MLRecommendationGroup.group_status == GROUP_STATUS_PENDING,
        )
        .order_by(MLRecommendationGroup.shown_at.desc(), MLRecommendationGroup.ml_recommendation_id.desc())
        .limit(1)
    )
    group = session.execute(stmt).scalar_one_or_none()
    if group is None:
        return None
    return _serialize_group(group)


def discard_group(ml_recommendation_id: str) -> bool:
    init_db()
    with SessionLocal() as session:
        with session.begin():
            return discard_group_in_session(session, ml_recommendation_id)


def discard_group_in_session(session: Session, ml_recommendation_id: str) -> bool:
    normalized_recommendation_id = _normalize_required_string(
        ml_recommendation_id,
        field_name="ml_recommendation_id",
    )
    resolved_at = datetime.now(timezone.utc)

    group_result = session.execute(
        update(MLRecommendationGroup)
        .where(
            MLRecommendationGroup.ml_recommendation_id == normalized_recommendation_id,
            MLRecommendationGroup.group_status == GROUP_STATUS_PENDING,
        )
        .values(
            group_status=GROUP_STATUS_DISCARDED,
            resolved_at=resolved_at,
        )
    )
    if int(group_result.rowcount or 0) <= 0:
        return False

    session.execute(
        update(MLRecommendationImpression)
        .where(
            MLRecommendationImpression.ml_recommendation_id == normalized_recommendation_id,
        )
        .values(
            row_status=ROW_STATUS_DISCARDED,
            final_action=None,
            last_interaction_at=None,
            label_true_like=None,
            label_true_dislike=None,
            label_true_match=None,
            label_true_block=None,
        )
    )
    return True


def discard_groups_in_session(session: Session, ml_recommendation_ids: Sequence[str]) -> int:
    normalized_ids = _normalize_non_empty_strings(ml_recommendation_ids)
    discarded_count = 0
    for recommendation_id in normalized_ids:
        if discard_group_in_session(session, recommendation_id):
            discarded_count += 1
    return discarded_count


def finalize_groups_from_interactions(
    ml_recommendation_ids: Iterable[str],
    *,
    session: Session | None = None,
) -> dict[str, int]:
    normalized_ids = _normalize_non_empty_strings(ml_recommendation_ids)
    if not normalized_ids:
        return _empty_finalize_summary()

    if session is not None:
        return finalize_groups_from_interactions_in_session(session, normalized_ids)

    init_db()
    with SessionLocal() as local_session:
        with local_session.begin():
            return finalize_groups_from_interactions_in_session(local_session, normalized_ids)


def finalize_groups_from_interactions_in_session(
    session: Session,
    ml_recommendation_ids: Sequence[str],
) -> dict[str, int]:
    normalized_ids = _normalize_non_empty_strings(ml_recommendation_ids)
    if not normalized_ids:
        return _empty_finalize_summary()

    pending_groups = session.execute(
        select(MLRecommendationGroup)
        .where(
            MLRecommendationGroup.ml_recommendation_id.in_(normalized_ids),
            MLRecommendationGroup.group_status == GROUP_STATUS_PENDING,
        )
        .order_by(MLRecommendationGroup.shown_at.asc(), MLRecommendationGroup.ml_recommendation_id.asc())
    ).scalars().all()
    if not pending_groups:
        return _empty_finalize_summary()

    pending_group_ids = [group.ml_recommendation_id for group in pending_groups]
    impression_rows = session.execute(
        select(MLRecommendationImpression)
        .where(MLRecommendationImpression.ml_recommendation_id.in_(pending_group_ids))
        .order_by(
            MLRecommendationImpression.ml_recommendation_id.asc(),
            MLRecommendationImpression.actor_user_id.asc(),
            MLRecommendationImpression.candidate_user_id.asc(),
        )
    ).scalars().all()

    latest_training_events: dict[tuple[str, str, str], dict[str, Any]] = {}
    ignored_non_training_action_count = 0
    interaction_rows = session.execute(
        select(
            Interaction.ml_recommendation_id,
            Interaction.actor_user_id,
            Interaction.candidate_user_id,
            Interaction.action,
            Interaction.created_at,
            Interaction.id,
        )
        .where(
            Interaction.deleted_at.is_(None),
            Interaction.ml_recommendation_id.in_(pending_group_ids),
        )
        .order_by(
            Interaction.ml_recommendation_id.asc(),
            Interaction.actor_user_id.asc(),
            Interaction.candidate_user_id.asc(),
            Interaction.created_at.desc(),
            Interaction.id.desc(),
        )
    ).all()
    for recommendation_id, actor_user_id, candidate_user_id, action, created_at, event_id in interaction_rows:
        normalized_action = str(action).strip().lower()
        if normalized_action not in TRAINING_INTERACTION_ACTIONS:
            ignored_non_training_action_count += 1
            continue

        key = (str(recommendation_id), str(actor_user_id), str(candidate_user_id))
        if key in latest_training_events:
            continue
        latest_training_events[key] = {
            "final_action": normalized_action,
            "last_interaction_at": created_at,
            "event_id": str(event_id),
        }

    now_utc = datetime.now(timezone.utc)
    rows_by_group: dict[str, list[MLRecommendationImpression]] = {}
    for impression_row in impression_rows:
        rows_by_group.setdefault(impression_row.ml_recommendation_id, []).append(impression_row)

    finalized_row_count = 0
    pending_row_count = 0
    finalized_group_count = 0
    pending_group_count = 0

    for recommendation_group in pending_groups:
        group_rows = rows_by_group.get(recommendation_group.ml_recommendation_id, [])
        group_is_finalized = True
        for impression_row in group_rows:
            key = (
                impression_row.ml_recommendation_id,
                impression_row.actor_user_id,
                impression_row.candidate_user_id,
            )
            training_event = latest_training_events.get(key)
            if training_event is None:
                impression_row.row_status = ROW_STATUS_PENDING
                impression_row.last_interaction_at = None
                impression_row.final_action = None
                impression_row.label_true_like = None
                impression_row.label_true_dislike = None
                impression_row.label_true_match = None
                impression_row.label_true_block = None
                pending_row_count += 1
                group_is_finalized = False
                continue

            labels = _labels_from_terminal_action(training_event["final_action"])
            impression_row.last_interaction_at = training_event["last_interaction_at"]
            impression_row.final_action = training_event["final_action"]
            impression_row.row_status = ROW_STATUS_FINALIZED
            impression_row.label_true_like = labels["label_true_like"]
            impression_row.label_true_dislike = labels["label_true_dislike"]
            impression_row.label_true_match = labels["label_true_match"]
            impression_row.label_true_block = labels["label_true_block"]
            finalized_row_count += 1

        if group_is_finalized and group_rows:
            recommendation_group.group_status = GROUP_STATUS_FINALIZED
            recommendation_group.resolved_at = now_utc
            finalized_group_count += 1
        else:
            recommendation_group.group_status = GROUP_STATUS_PENDING
            recommendation_group.resolved_at = None
            pending_group_count += 1

    return {
        "touched_group_count": len(pending_groups),
        "finalized_group_count": finalized_group_count,
        "pending_group_count": pending_group_count,
        "finalized_row_count": finalized_row_count,
        "pending_row_count": pending_row_count,
        "ignored_non_training_action_count": ignored_non_training_action_count,
    }


def get_train_rows(
    limit: int | None = None,
    *,
    served_stages: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    init_db()
    stmt = (
        select(MLRecommendationImpression, MLRecommendationGroup)
        .join(
            MLRecommendationGroup,
            MLRecommendationGroup.ml_recommendation_id == MLRecommendationImpression.ml_recommendation_id,
        )
        .where(
            MLRecommendationGroup.group_status == GROUP_STATUS_FINALIZED,
            MLRecommendationImpression.row_status == ROW_STATUS_FINALIZED,
        )
        .order_by(MLRecommendationImpression.shown_at.asc())
    )
    normalized_served_stages = _normalize_non_empty_strings(served_stages or [])
    if normalized_served_stages:
        stmt = stmt.where(MLRecommendationGroup.served_stage.in_(normalized_served_stages))
    if limit is not None and limit > 0:
        stmt = stmt.limit(int(limit))

    with SessionLocal() as session:
        rows = session.execute(stmt).all()
        return [
            _serialize_impression(impression_row, group_row)
            for impression_row, group_row in rows
        ]


def get_shown_candidate_ids(
    actor_user_id: str,
    *,
    candidate_user_ids: Sequence[str] | None = None,
) -> set[str]:
    init_db()

    normalized_actor_user_id = _normalize_required_user_id(
        actor_user_id,
        field_name="actor_user_id",
    )
    normalized_candidate_ids = _normalize_non_empty_user_ids(candidate_user_ids or [])
    stmt = select(MLRecommendationImpression.candidate_user_id).where(
        MLRecommendationImpression.actor_user_id == normalized_actor_user_id,
    )
    if normalized_candidate_ids:
        stmt = stmt.where(MLRecommendationImpression.candidate_user_id.in_(normalized_candidate_ids))

    with SessionLocal() as session:
        return set(session.execute(stmt).scalars().all())


def _list_pending_group_ids_for_user_in_session(session: Session, actor_user_id: str) -> list[str]:
    stmt = (
        select(MLRecommendationGroup.ml_recommendation_id)
        .where(
            MLRecommendationGroup.actor_user_id == actor_user_id,
            MLRecommendationGroup.group_status == GROUP_STATUS_PENDING,
        )
        .order_by(MLRecommendationGroup.shown_at.asc(), MLRecommendationGroup.ml_recommendation_id.asc())
    )
    return list(session.execute(stmt).scalars().all())


def _extract_prediction_value(source_row: dict[str, Any], target: str) -> float | None:
    prediction_column = PREDICTION_COLUMN_BY_TARGET[target]
    return _to_nullable_float(source_row.get(prediction_column))


def _build_feature_snapshot(source_row_json: dict[str, Any]) -> dict[str, Any]:
    feature_snapshot = source_row_json.get("feature_snapshot")
    if isinstance(feature_snapshot, dict) and feature_snapshot:
        return _to_json_safe(feature_snapshot)

    excluded_columns = {
        "impactful_features",
        "rerank_score",
        "feature_snapshot",
        "reasons",
        "already_shown",
        *PREDICTION_COLUMN_BY_TARGET.values(),
    }
    result = {
        key: value
        for key, value in source_row_json.items()
        if key not in excluded_columns
    }
    result.update(
        {
            "from_transaction": bool(source_row_json.get("retrieval_transactions_is_candidate", 0)),
            "from_interaction": bool(source_row_json.get("retrieval_tx_interaction_is_candidate", 0)),
            "from_exotic": bool(source_row_json.get("retrieval_exotic_is_candidate", 0)),
            "from_popularity": bool(source_row_json.get("retrieval_popularity_is_candidate", 0)),
            "impactful_features_origin": source_row_json.get("impactful_features_origin"),
            "impactful_features_seed_basis": source_row_json.get("impactful_features_seed_basis"),
            "retrieval_primary_source": source_row_json.get("retrieval_primary_source"),
            "retrieval_sources_count": source_row_json.get("retrieval_sources_count"),
            "retrieval_sources": source_row_json.get("retrieval_sources"),
            "transaction_rank": source_row_json.get("retrieval_transactions_rank"),
            "interaction_rank": source_row_json.get("retrieval_tx_interaction_rank"),
            "exotic_rank": source_row_json.get("retrieval_exotic_rank"),
            "popularity_rank": source_row_json.get("retrieval_popularity_rank"),
            "transaction_score": source_row_json.get("retrieval_transactions_score"),
            "interaction_score": source_row_json.get("retrieval_tx_interaction_score"),
            "exotic_score": source_row_json.get("retrieval_exotic_score"),
            "popularity_score": source_row_json.get("retrieval_popularity_score"),
            "retrieval_best_rank": source_row_json.get("retrieval_best_rank"),
            "retrieval_mean_rank": source_row_json.get("retrieval_mean_rank"),
            "retrieval_best_score": source_row_json.get("retrieval_best_score"),
            "retrieval_mean_score": source_row_json.get("retrieval_mean_score"),
            "pre_rerank_similarity": source_row_json.get("distance_euclidean"),
        }
    )
    return _to_json_safe(result)


def _labels_from_terminal_action(action: str) -> dict[str, bool]:
    return {
        "label_true_like": action == "like",
        "label_true_dislike": action == "dislike",
        "label_true_match": action == "match",
        "label_true_block": action == "block",
    }


def _empty_finalize_summary() -> dict[str, int]:
    return {
        "touched_group_count": 0,
        "finalized_group_count": 0,
        "pending_group_count": 0,
        "finalized_row_count": 0,
        "pending_row_count": 0,
        "ignored_non_training_action_count": 0,
    }


def _serialize_group(group: MLRecommendationGroup) -> dict[str, Any]:
    return {
        "ml_recommendation_id": group.ml_recommendation_id,
        "actor_user_id": group.actor_user_id,
        "shown_at": group.shown_at.isoformat() if group.shown_at else None,
        "served_stage": group.served_stage,
        "group_status": group.group_status,
        "resolved_at": group.resolved_at.isoformat() if group.resolved_at else None,
    }


def _serialize_impression(
    impression_row: MLRecommendationImpression,
    group_row: MLRecommendationGroup | None = None,
) -> dict[str, Any]:
    return {
        "ml_recommendation_id": impression_row.ml_recommendation_id,
        "actor_user_id": impression_row.actor_user_id,
        "candidate_user_id": impression_row.candidate_user_id,
        "shown_at": impression_row.shown_at.isoformat() if impression_row.shown_at else None,
        "last_interaction_at": (
            impression_row.last_interaction_at.isoformat() if impression_row.last_interaction_at else None
        ),
        "final_action": impression_row.final_action,
        "row_status": impression_row.row_status,
        "served_stage": group_row.served_stage if group_row is not None else None,
        "group_status": group_row.group_status if group_row is not None else None,
        "label_true_like": impression_row.label_true_like,
        "label_true_dislike": impression_row.label_true_dislike,
        "label_true_match": impression_row.label_true_match,
        "label_true_block": impression_row.label_true_block,
        "label_pred_like": impression_row.label_pred_like,
        "label_pred_dislike": impression_row.label_pred_dislike,
        "label_pred_match": impression_row.label_pred_match,
        "label_pred_block": impression_row.label_pred_block,
        "features_json": _to_json_safe(impression_row.features_json),
        "source_row_json": _to_json_safe(impression_row.source_row_json),
    }


def _to_nullable_float(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _to_aware_utc(raw_value: Any) -> datetime | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, datetime):
        date_time = raw_value
    else:
        text_value = str(raw_value).strip()
        if not text_value:
            return None
        if text_value.endswith("Z"):
            text_value = text_value[:-1] + "+00:00"
        date_time = datetime.fromisoformat(text_value)

    if date_time.tzinfo is None:
        return date_time.replace(tzinfo=timezone.utc)
    return date_time.astimezone(timezone.utc)


def _normalize_required_string(raw_value: Any, *, field_name: str) -> str:
    if raw_value is None:
        raise ValueError(f"{field_name} must not be empty.")
    normalized_value = str(raw_value).strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized_value


def _normalize_required_user_id(raw_value: Any, *, field_name: str) -> str:
    normalized_value = normalize_user_id(raw_value, allow_hash=True)
    if normalized_value is None:
        raise ValueError(f"{field_name} must be a valid UUID.")
    return normalized_value


def _normalize_non_empty_strings(values: Iterable[Any]) -> list[str]:
    normalized_values: list[str] = []
    seen_values: set[str] = set()
    for raw_value in values:
        if raw_value is None:
            continue
        normalized_value = str(raw_value).strip()
        if not normalized_value or normalized_value in seen_values:
            continue
        seen_values.add(normalized_value)
        normalized_values.append(normalized_value)
    return normalized_values


def _normalize_non_empty_user_ids(values: Iterable[Any]) -> list[str]:
    normalized_values: list[str] = []
    seen_values: set[str] = set()
    for raw_value in values:
        normalized_value = normalize_user_id(raw_value, allow_hash=True)
        if normalized_value is None or normalized_value in seen_values:
            continue
        seen_values.add(normalized_value)
        normalized_values.append(normalized_value)
    return normalized_values


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
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


def _acquire_user_recommendation_lock(connection: Connection, actor_user_id: str) -> None:
    lock_key = _build_recommendation_lock_key(actor_user_id)
    connection.execute(
        text("SELECT pg_advisory_lock(:lock_key)"),
        {"lock_key": lock_key},
    )
    connection.commit()


def _release_user_recommendation_lock(connection: Connection, actor_user_id: str) -> None:
    lock_key = _build_recommendation_lock_key(actor_user_id)
    try:
        connection.execute(
            text("SELECT pg_advisory_unlock(:lock_key)"),
            {"lock_key": lock_key},
        )
        connection.commit()
    except Exception:
        # unlock failure is non-fatal here because connection close will rollback and release resources
        connection.rollback()


def _build_recommendation_lock_key(actor_user_id: str) -> int:
    digest = hashlib.blake2b(actor_user_id.encode("utf-8"), digest_size=8).digest()
    unsigned_value = int.from_bytes(digest, byteorder="big", signed=False)
    if unsigned_value >= 2**63:
        return unsigned_value - 2**64
    return unsigned_value
