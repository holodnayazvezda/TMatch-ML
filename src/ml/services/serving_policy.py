from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

log: structlog.BoundLogger = structlog.get_logger(__name__)

STRICT_OFFLINE_SERVING = True


def is_lazy_rebuild_allowed_in_serving() -> bool:
    """Serving path must never rebuild heavy artifacts on demand."""
    return not STRICT_OFFLINE_SERVING


def log_lazy_rebuild_blocked(
    *,
    source: str,
    reason: str,
    artifact_path: str | Path | None = None,
    actor_user_id: str | None = None,
    **extra: Any,
) -> None:
    payload: dict[str, Any] = {
        "source": source,
        "reason": reason,
        "strict_offline_serving": STRICT_OFFLINE_SERVING,
    }
    if artifact_path is not None:
        payload["artifact_path"] = str(artifact_path)
    if actor_user_id is not None:
        payload["actor_user_id"] = str(actor_user_id)
    payload.update(extra)
    log.warning("recommendation_serving_lazy_rebuild_blocked", **payload)
