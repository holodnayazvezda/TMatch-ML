from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import structlog
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from ml.config import settings
from ml.infra.db.init_db import init_db
from ml.infra.db.session import SessionLocal
from ml.services.interaction_feature_store import process_interaction_batch_in_session
from ml.services.interaction_ingestion import normalize_audit_event_rows
from ml.services.recommendation_store import finalize_groups_from_interactions_in_session

log: structlog.BoundLogger = structlog.get_logger(__name__)


class InteractionEventPoller:
    def __init__(
        self,
        *,
        poll_interval_seconds: float | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.poll_interval_seconds = (
            float(poll_interval_seconds)
            if poll_interval_seconds is not None
            else float(settings.interaction_poll_interval_seconds)
        )
        self.batch_size = (
            int(batch_size)
            if batch_size is not None
            else int(settings.interaction_poll_batch_size)
        )
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return

        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_forever(), name="interaction-event-poller")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is None:
            return
        await self._task
        self._task = None

    async def _run_forever(self) -> None:
        log.info(
            "interaction_event_poller_started",
            poll_interval_seconds=self.poll_interval_seconds,
            batch_size=self.batch_size,
        )
        while not self._stop_event.is_set():
            try:
                summary = await asyncio.to_thread(poll_audit_events_once, self.batch_size)
                if summary.get("selected_total", 0) > 0:
                    log.info("interaction_event_poller_cycle_finished", **summary)
            except Exception as exc:
                log.warning("interaction_event_poller_cycle_failed", error=str(exc))

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=max(self.poll_interval_seconds, 0.0),
                )
            except TimeoutError:
                continue
        log.info("interaction_event_poller_stopped")


def poll_audit_events_once(batch_size: int | None = None) -> dict[str, int]:
    effective_batch_size = int(batch_size or settings.interaction_poll_batch_size)
    init_db()

    with SessionLocal() as session:
        with session.begin():
            audit_events = _select_pending_audit_events(session, effective_batch_size)
            if not audit_events:
                return _empty_poll_summary()

            event_ids = [str(event["id"]) for event in audit_events]
            normalized_events, normalization_summary = normalize_audit_event_rows(audit_events)
            interaction_summary = process_interaction_batch_in_session(
                session,
                normalized_events,
                refresh_features=True,
            )

            finalized_summary = finalize_groups_from_interactions_in_session(
                session,
                [
                    event["ml_recommendation_id"]
                    for event in normalized_events
                    if event.get("ml_recommendation_id") is not None
                ],
            )
            acked_total = _mark_audit_events_sent_to_ml(session, event_ids)

            return {
                "selected_total": len(audit_events),
                "acked_total": acked_total,
                **normalization_summary,
                **interaction_summary,
                **finalized_summary,
            }


def _select_pending_audit_events(session: Session, batch_size: int) -> list[dict[str, Any]]:
    stmt = text(
        """
        SELECT
            id::text AS id,
            created_at AS created_at,
            deleted_at AS deleted_at,
            actor_user_id::text AS actor_user_id,
            target_user_id::text AS target_user_id,
            payload AS payload,
            ml_recommendation_id::text AS ml_recommendation_id,
            event_type::text AS event_type,
            sent_to_ml AS sent_to_ml
        FROM audit_events
        WHERE sent_to_ml = false
          AND deleted_at IS NULL
          AND event_type::text = 'interaction.created'
        ORDER BY created_at ASC, id ASC
        LIMIT :batch_size
        FOR UPDATE SKIP LOCKED
        """
    )
    rows = session.execute(stmt, {"batch_size": max(batch_size, 1)}).mappings().all()
    return [dict(row) for row in rows]


def _mark_audit_events_sent_to_ml(session: Session, event_ids: Sequence[str]) -> int:
    if not event_ids:
        return 0

    stmt = (
        text(
            """
            UPDATE audit_events
            SET sent_to_ml = true
            WHERE id::text IN :event_ids
            """
        )
        .bindparams(bindparam("event_ids", expanding=True))
    )
    result = session.execute(stmt, {"event_ids": list(event_ids)})
    return int(result.rowcount or 0)


def _empty_poll_summary() -> dict[str, int]:
    return {
        "selected_total": 0,
        "acked_total": 0,
        "received_total": 0,
        "processed_total": 0,
        "ignored_unknown_action_count": 0,
        "ignored_non_interaction_count": 0,
        "invalid_rows_count": 0,
        "stored_interaction_count": 0,
        "affected_user_count": 0,
        "refreshed_user_count": 0,
        "upserted_feature_row_count": 0,
        "deleted_feature_row_count": 0,
        "touched_group_count": 0,
        "finalized_group_count": 0,
        "pending_group_count": 0,
        "finalized_row_count": 0,
        "pending_row_count": 0,
        "ignored_non_training_action_count": 0,
    }
