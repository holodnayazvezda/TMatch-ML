from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any

_STORE_LOCK = Lock()
_RECOMMENDATION_ROW_STORE: dict[str, dict[str, Any]] = {}


def save_recommendation_row(
    recommendation_id: str,
    user_id: str,
    candidate_user_id: str,
    source_row: dict[str, Any],
) -> None:
    """
    In-memory stub.
    TODO: persist to DB/event bus for reranker training labels.
    """
    record = {
        "recommendation_id": recommendation_id,
        "user_id": user_id,
        "candidate_user_id": candidate_user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_row": source_row,
    }

    with _STORE_LOCK:
        _RECOMMENDATION_ROW_STORE[recommendation_id] = record


def get_recommendation_row(recommendation_id: str) -> dict[str, Any] | None:
    with _STORE_LOCK:
        record = _RECOMMENDATION_ROW_STORE.get(recommendation_id)
        if record is None:
            return None
        return dict(record)
