from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from threading import Lock
from typing import Any

SUPPORTED_INTERACTION_ACTIONS = {"block", "skip", "unblock"}
INTERACTION_ACTION_ALIASES = {"ublock": "unblock"}

_INGESTION_LOCK = Lock()
_LAST_GROUPED_DATASET: dict[str, list[dict[str, Any]]] = {}


def ingest_audit_events_batch(audit_events: list[dict[str, Any]]) -> dict[str, int]:
    grouped_by_user, summary = _split_audit_events(audit_events)
    incremental_user_update_stub(grouped_by_user)

    return summary


def incremental_user_update_stub(grouped_by_user: dict[str, list[dict[str, Any]]]) -> None:
    """
    Placeholder for future incremental user state updates.
    TODO: persist/update per-user aggregates in DB/feature store.
    """
    with _INGESTION_LOCK:
        _LAST_GROUPED_DATASET.clear()
        _LAST_GROUPED_DATASET.update(deepcopy(grouped_by_user))


def get_last_ingestion_debug_snapshot() -> dict[str, Any]:
    with _INGESTION_LOCK:
        return {
            "grouped_by_user": deepcopy(_LAST_GROUPED_DATASET),
        }


def _split_audit_events(
    audit_events: list[dict[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, int],
]:
    grouped_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)

    summary = {
        "received_total": len(audit_events),
        "processed_total": 0,
        "ignored_unknown_action_count": 0,
        "ignored_non_interaction_count": 0,
        "ignored_sent_to_ml_true_count": 0,
        "invalid_rows_count": 0,
        "grouped_users_count": 0,
    }

    for event in audit_events:
        if bool(event.get("sent_to_ml", False)):
            summary["ignored_sent_to_ml_true_count"] += 1
            continue

        if str(event.get("event_type", "")).strip() != "interaction.created":
            summary["ignored_non_interaction_count"] += 1
            continue

        actor_user_id = _normalize_id(event.get("actor_user_id"))
        target_user_id = _normalize_id(event.get("target_user_id"))
        if actor_user_id is None or target_user_id is None or actor_user_id == target_user_id:
            summary["invalid_rows_count"] += 1
            continue

        payload = event.get("payload")
        if payload is None:
            summary["invalid_rows_count"] += 1
            continue
        if not isinstance(payload, Mapping):
            summary["invalid_rows_count"] += 1
            continue

        raw_action = payload.get("action")
        if raw_action is None:
            summary["invalid_rows_count"] += 1
            continue

        action = _normalize_action(raw_action)
        if action is None:
            summary["ignored_unknown_action_count"] += 1
            continue

        created_at = event.get("created_at")
        created_at_iso = _to_iso_datetime(created_at)
        if created_at_iso is None:
            summary["invalid_rows_count"] += 1
            continue

        normalized_row = {
            "audit_event_id": str(event.get("id", "")),
            "created_at": created_at_iso,
            "actor_user_id": actor_user_id,
            "target_user_id": target_user_id,
            "action": action,
            "sent_to_ml": False,
        }
        grouped_by_user[actor_user_id].append(normalized_row)

        summary["processed_total"] += 1

    summary["grouped_users_count"] = len(grouped_by_user)
    return dict(grouped_by_user), summary


def _normalize_action(raw_action: Any) -> str | None:
    if raw_action is None:
        return None

    action = str(raw_action).strip().lower()
    action = INTERACTION_ACTION_ALIASES.get(action, action)
    if action not in SUPPORTED_INTERACTION_ACTIONS:
        return None
    return action


def _normalize_id(raw_id: Any) -> str | None:
    if raw_id is None:
        return None
    value = str(raw_id).strip()
    if not value:
        return None
    return value


def _to_iso_datetime(raw_value: Any) -> str | None:
    if isinstance(raw_value, datetime):
        return raw_value.isoformat()

    if raw_value is None:
        return None

    text = str(raw_value).strip()
    if not text:
        return None
    return text
