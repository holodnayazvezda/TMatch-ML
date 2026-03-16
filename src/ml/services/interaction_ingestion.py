from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from ml.services.user_id import normalize_user_id, normalize_uuid

SUPPORTED_INTERACTION_ACTIONS = {
    "like",
    "dislike",
    "skip",
    "match",
    "chat_started",
    "block",
    "unblock",
}
TRAINING_INTERACTION_ACTIONS = {
    "like",
    "dislike",
    "match",
    "block",
}
INTERACTION_ACTION_ALIASES = {"ublock": "unblock"}


def normalize_audit_event_rows(audit_events: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    normalized_events: list[dict[str, Any]] = []
    summary = {
        "received_total": len(audit_events),
        "processed_total": 0,
        "ignored_unknown_action_count": 0,
        "ignored_non_interaction_count": 0,
        "invalid_rows_count": 0,
    }

    for event in audit_events:
        normalized_event = normalize_audit_event_row(event)
        if normalized_event is None:
            if str(event.get("event_type", "")).strip() != "interaction.created":
                summary["ignored_non_interaction_count"] += 1
                continue

            raw_action = _extract_action(event)
            if raw_action is not None and _normalize_action(raw_action) is None:
                summary["ignored_unknown_action_count"] += 1
                continue

            summary["invalid_rows_count"] += 1
            continue

        normalized_events.append(normalized_event)
        summary["processed_total"] += 1

    return normalized_events, summary


def normalize_audit_event_row(event: Mapping[str, Any]) -> dict[str, Any] | None:
    if str(event.get("event_type", "")).strip() != "interaction.created":
        return None

    actor_user_id = _normalize_id(event.get("actor_user_id"))
    candidate_user_id = _normalize_id(
        event.get("candidate_user_id") or event.get("target_user_id")
    )
    if actor_user_id is None or candidate_user_id is None or actor_user_id == candidate_user_id:
        return None

    event_id = normalize_uuid(event.get("id") or event.get("event_id"))
    created_at_iso = _to_iso_datetime(event.get("created_at"))
    if event_id is None or created_at_iso is None:
        return None

    action = _normalize_action(_extract_action(event))
    if action is None:
        return None

    return {
        "event_id": event_id,
        "created_at": created_at_iso,
        "deleted_at": _to_iso_datetime(event.get("deleted_at")),
        "actor_user_id": actor_user_id,
        "candidate_user_id": candidate_user_id,
        "action": action,
        "ml_recommendation_id": _normalize_string(event.get("ml_recommendation_id")),
    }


def extract_training_action(raw_action: Any) -> str | None:
    action = _normalize_action(raw_action)
    if action not in TRAINING_INTERACTION_ACTIONS:
        return None
    return action


def _extract_action(event: Mapping[str, Any]) -> Any:
    if event.get("action") is not None:
        return event.get("action")

    payload = event.get("payload")
    if isinstance(payload, Mapping):
        return payload.get("action")
    return None


def _normalize_action(raw_action: Any) -> str | None:
    if raw_action is None:
        return None

    action = str(raw_action).strip().lower()
    action = INTERACTION_ACTION_ALIASES.get(action, action)
    if action not in SUPPORTED_INTERACTION_ACTIONS:
        return None
    return action


def _normalize_id(raw_id: Any) -> str | None:
    return normalize_user_id(raw_id, allow_hash=False)


def _normalize_string(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
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
