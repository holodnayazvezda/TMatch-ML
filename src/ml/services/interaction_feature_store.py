from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy import delete, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ml.featurize.interactions_features import (
    ACTIONS,
    INPUT_PATH,
    build_base_events_df_from_records,
    build_feature_frame_from_base_events,
    to_naive_utc,
)
from ml.infra.db.init_db import init_db
from ml.infra.db.interaction_feature_tables import user_interaction_features_table
from ml.infra.db.models import Interaction
from ml.infra.db.session import SessionLocal

ADVISORY_LOCK_KEY = 902_451
RAW_INTERACTION_BATCH_SIZE = 5_000
FEATURE_ROW_BATCH_SIZE = 500
USER_REFRESH_CHUNK_SIZE = 1_000

SUPPORTED_INTERACTION_ACTIONS = set(ACTIONS)


def process_interaction_batch(
    events: Sequence[dict[str, Any]],
    *,
    refresh_features: bool = True,
) -> dict[str, int]:
    summary, _ = _process_interaction_batch_internal(
        events,
        refresh_features=refresh_features,
    )
    return summary


def refresh_user_features(user_ids: Iterable[str]) -> dict[str, int]:
    normalized_user_ids = _normalize_user_ids(user_ids)
    if not normalized_user_ids:
        return {
            "refreshed_user_count": 0,
            "upserted_feature_row_count": 0,
            "deleted_feature_row_count": 0,
        }

    init_db()
    with SessionLocal() as session:
        with session.begin():
            _acquire_advisory_lock(session)
            return _refresh_user_features_in_session(session, normalized_user_ids)


def backfill_jsonl_to_database(
    jsonl_path: Path = INPUT_PATH,
    *,
    interaction_batch_size: int = RAW_INTERACTION_BATCH_SIZE,
    user_refresh_chunk_size: int = USER_REFRESH_CHUNK_SIZE,
) -> dict[str, int]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Interaction dataset not found: {jsonl_path}")

    total_lines = 0
    total_stored_interactions = 0
    all_affected_user_ids: set[str] = set()
    current_batch: list[dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as input_stream:
        for line in input_stream:
            line = line.strip()
            if not line:
                continue

            current_batch.append(json.loads(line))
            total_lines += 1

            if len(current_batch) >= interaction_batch_size:
                batch_summary, affected_user_ids = _process_interaction_batch_internal(
                    current_batch,
                    refresh_features=False,
                )
                total_stored_interactions += batch_summary["stored_interaction_count"]
                all_affected_user_ids.update(affected_user_ids)
                current_batch.clear()

    if current_batch:
        batch_summary, affected_user_ids = _process_interaction_batch_internal(
            current_batch,
            refresh_features=False,
        )
        total_stored_interactions += batch_summary["stored_interaction_count"]
        all_affected_user_ids.update(affected_user_ids)

    total_refreshed_users = 0
    total_deleted_feature_rows = 0
    normalized_user_ids = sorted(all_affected_user_ids)
    for user_id_chunk in _chunked(normalized_user_ids, user_refresh_chunk_size):
        refresh_summary = refresh_user_features(user_id_chunk)
        total_refreshed_users += refresh_summary["upserted_feature_row_count"]
        total_deleted_feature_rows += refresh_summary["deleted_feature_row_count"]

    return {
        "jsonl_line_count": total_lines,
        "stored_interaction_count": total_stored_interactions,
        "affected_user_count": len(all_affected_user_ids),
        "upserted_feature_row_count": total_refreshed_users,
        "deleted_feature_row_count": total_deleted_feature_rows,
    }


def _process_interaction_batch_internal(
    events: Sequence[dict[str, Any]],
    *,
    refresh_features: bool,
) -> tuple[dict[str, int], list[str]]:
    normalized_events = _normalize_storage_events(events)
    affected_user_ids = _collect_affected_user_ids(normalized_events)

    if not normalized_events:
        return (
            {
                "stored_interaction_count": 0,
                "affected_user_count": 0,
                "refreshed_user_count": 0,
                "upserted_feature_row_count": 0,
                "deleted_feature_row_count": 0,
            },
            [],
        )

    init_db()
    with SessionLocal() as session:
        with session.begin():
            _acquire_advisory_lock(session)
            _upsert_interactions(session, normalized_events)

            refresh_summary = {
                "refreshed_user_count": 0,
                "upserted_feature_row_count": 0,
                "deleted_feature_row_count": 0,
            }
            if refresh_features and affected_user_ids:
                refresh_summary = _refresh_user_features_in_session(session, affected_user_ids)

    return (
        {
            "stored_interaction_count": len(normalized_events),
            "affected_user_count": len(affected_user_ids),
            **refresh_summary,
        },
        affected_user_ids,
    )


def _refresh_user_features_in_session(session: Session, user_ids: list[str]) -> dict[str, int]:
    if not user_ids:
        return {
            "refreshed_user_count": 0,
            "upserted_feature_row_count": 0,
            "deleted_feature_row_count": 0,
        }

    feature_frame = _build_feature_frame_for_users(session, user_ids)
    upserted_user_ids = set(feature_frame["user_id"].to_list()) if feature_frame.height > 0 else set()
    deleted_user_ids = [user_id for user_id in user_ids if user_id not in upserted_user_ids]

    _upsert_feature_rows(session, feature_frame)
    deleted_feature_row_count = _delete_feature_rows(session, deleted_user_ids)

    return {
        "refreshed_user_count": len(user_ids),
        "upserted_feature_row_count": len(upserted_user_ids),
        "deleted_feature_row_count": deleted_feature_row_count,
    }


def _build_feature_frame_for_users(session: Session, user_ids: list[str]) -> pl.DataFrame:
    stmt = (
        select(
            Interaction.id.label("event_id"),
            Interaction.created_at.label("occurred_at"),
            Interaction.actor_user_id.label("actor_user_id"),
            Interaction.candidate_user_id.label("target_user_id"),
            Interaction.action.label("action"),
        )
        .where(
            Interaction.deleted_at.is_(None),
            or_(
                Interaction.actor_user_id.in_(user_ids),
                Interaction.candidate_user_id.in_(user_ids),
            ),
        )
    )

    records = []
    for row in session.execute(stmt).mappings():
        records.append(
            {
                "event_id": str(row["event_id"]),
                "event_type": "interaction.created",
                "occurred_at": to_naive_utc(row["occurred_at"]),
                "actor_user_id": str(row["actor_user_id"]),
                "target_user_id": str(row["target_user_id"]),
                "action": str(row["action"]).strip().lower(),
            }
        )

    base_events_df = build_base_events_df_from_records(records)
    return build_feature_frame_from_base_events(
        base_events_df,
        all_user_ids=user_ids,
    )


def _upsert_interactions(session: Session, events: Sequence[dict[str, Any]]) -> None:
    interaction_table = Interaction.__table__
    update_columns = {
        "created_at": pg_insert(interaction_table).excluded.created_at,
        "deleted_at": pg_insert(interaction_table).excluded.deleted_at,
        "actor_user_id": pg_insert(interaction_table).excluded.actor_user_id,
        "candidate_user_id": pg_insert(interaction_table).excluded.candidate_user_id,
        "action": pg_insert(interaction_table).excluded.action,
        "ml_recommendation_id": pg_insert(interaction_table).excluded.ml_recommendation_id,
    }

    for chunk in _chunked(list(events), RAW_INTERACTION_BATCH_SIZE):
        stmt = (
            pg_insert(interaction_table)
            .values(chunk)
            .on_conflict_do_update(
                index_elements=[interaction_table.c.id],
                set_=update_columns,
            )
        )
        session.execute(stmt)


def _upsert_feature_rows(session: Session, feature_frame: pl.DataFrame) -> None:
    if feature_frame.height == 0:
        return

    feature_table = user_interaction_features_table
    update_columns = {
        column_name: pg_insert(feature_table).excluded[column_name]
        for column_name in feature_frame.columns
        if column_name != "user_id"
    }

    for feature_slice in feature_frame.iter_slices(n_rows=FEATURE_ROW_BATCH_SIZE):
        stmt = (
            pg_insert(feature_table)
            .values(feature_slice.to_dicts())
            .on_conflict_do_update(
                index_elements=[feature_table.c.user_id],
                set_=update_columns,
            )
        )
        session.execute(stmt)


def _delete_feature_rows(session: Session, user_ids: Sequence[str]) -> int:
    normalized_user_ids = _normalize_user_ids(user_ids)
    if not normalized_user_ids:
        return 0

    stmt = delete(user_interaction_features_table).where(
        user_interaction_features_table.c.user_id.in_(normalized_user_ids)
    )
    result = session.execute(stmt)
    return int(result.rowcount or 0)


def _normalize_storage_events(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_by_id: dict[str, dict[str, Any]] = {}
    for raw_event in events:
        normalized_event = _normalize_storage_event(raw_event)
        if normalized_event is None:
            continue
        normalized_by_id[normalized_event["id"]] = normalized_event
    return list(normalized_by_id.values())


def _normalize_storage_event(raw_event: dict[str, Any]) -> dict[str, Any] | None:
    event_id = _normalize_string(
        raw_event.get("event_id")
        or raw_event.get("audit_event_id")
        or raw_event.get("id")
    )
    actor_user_id = _normalize_string(raw_event.get("actor_user_id"))
    target_user_id = _normalize_string(
        raw_event.get("target_user_id") or raw_event.get("candidate_user_id")
    )

    action = raw_event.get("action")
    if action is None:
        payload = raw_event.get("payload")
        if isinstance(payload, dict):
            action = payload.get("action")
    normalized_action = _normalize_action(action)

    created_at = to_naive_utc(raw_event.get("created_at") or raw_event.get("occurred_at"))
    deleted_at = to_naive_utc(raw_event.get("deleted_at"))
    ml_recommendation_id = _normalize_string(raw_event.get("ml_recommendation_id"))

    if (
        event_id is None
        or actor_user_id is None
        or target_user_id is None
        or normalized_action is None
        or created_at is None
    ):
        return None

    return {
        "id": event_id,
        "created_at": created_at,
        "deleted_at": deleted_at,
        "actor_user_id": actor_user_id,
        "candidate_user_id": target_user_id,
        "action": normalized_action,
        "ml_recommendation_id": ml_recommendation_id,
    }


def _collect_affected_user_ids(events: Sequence[dict[str, Any]]) -> list[str]:
    affected_user_ids = {
        user_id
        for event in events
        for user_id in (event["actor_user_id"], event["candidate_user_id"])
        if user_id
    }
    return sorted(affected_user_ids)


def _normalize_user_ids(user_ids: Iterable[str]) -> list[str]:
    normalized: set[str] = set()
    for user_id in user_ids:
        normalized_user_id = _normalize_string(user_id)
        if normalized_user_id is not None:
            normalized.add(normalized_user_id)
    return sorted(normalized)


def _normalize_action(raw_action: Any) -> str | None:
    if raw_action is None:
        return None

    action = str(raw_action).strip().lower()
    if action not in SUPPORTED_INTERACTION_ACTIONS:
        return None
    return action


def _normalize_string(raw_value: Any) -> str | None:
    if raw_value is None:
        return None

    value = str(raw_value).strip()
    if not value:
        return None
    return value


def _chunked(values: Sequence[Any], chunk_size: int) -> Iterable[list[Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    for start in range(0, len(values), chunk_size):
        yield list(values[start:start + chunk_size])


def _acquire_advisory_lock(session: Session) -> None:
    session.execute(
        text("SELECT pg_advisory_xact_lock(:lock_key)"),
        {"lock_key": ADVISORY_LOCK_KEY},
    )
