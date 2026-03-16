from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, or_, select

from core.config import CONFIG
from ml.services.user_id import normalize_user_id

from .db.models import Interaction
from .db.session import SessionLocal


class InteractionBlocklistService:
    def is_blocked(self, actor_user_id: str, candidate_user_id: str) -> bool:
        return candidate_user_id in self.get_blocked_targets(
            actor_user_id=actor_user_id,
            candidate_user_ids=[candidate_user_id],
        )

    def get_blocked_targets(
        self,
        actor_user_id: str,
        candidate_user_ids: Iterable[str],
    ) -> set[str]:
        normalized_actor_user_id = _normalize_id(actor_user_id)
        if normalized_actor_user_id is None:
            return set()

        candidate_mapping = _build_candidate_mapping(candidate_user_ids)
        if not candidate_mapping:
            return set()

        latest_interactions = self._get_latest_interactions(
            actor_user_id=normalized_actor_user_id,
            candidate_user_ids=list(candidate_mapping),
        )
        now = datetime.now(timezone.utc)

        blocked_targets: set[str] = set()
        for source_user_id, target_user_id, action, created_at in latest_interactions:
            normalized_source_user_id = _normalize_id(source_user_id)
            normalized_target_user_id = _normalize_id(target_user_id)
            if normalized_source_user_id is None or normalized_target_user_id is None:
                continue

            if normalized_source_user_id == normalized_actor_user_id:
                if _is_actor_side_excluded(action=action, created_at=created_at, now=now):
                    blocked_targets.update(candidate_mapping.get(normalized_target_user_id, ()))
                continue

            if normalized_target_user_id == normalized_actor_user_id:
                if _is_reverse_blocked(action=action):
                    blocked_targets.update(candidate_mapping.get(normalized_source_user_id, ()))

        return blocked_targets

    def _get_latest_interactions(
        self,
        actor_user_id: str,
        candidate_user_ids: list[str],
    ) -> list[tuple[str, str, str, datetime]]:
        latest_interactions = (
            select(
                Interaction.actor_user_id.label("actor_user_id"),
                Interaction.candidate_user_id.label("candidate_user_id"),
                Interaction.action.label("action"),
                Interaction.created_at.label("created_at"),
                func.row_number()
                .over(
                    partition_by=(Interaction.actor_user_id, Interaction.candidate_user_id),
                    order_by=(Interaction.created_at.desc(), Interaction.id.desc()),
                )
                .label("row_number"),
            )
            .where(
                Interaction.deleted_at.is_(None),
                or_(
                    (
                        (Interaction.actor_user_id == actor_user_id)
                        & Interaction.candidate_user_id.in_(candidate_user_ids)
                    ),
                    (
                        (Interaction.candidate_user_id == actor_user_id)
                        & Interaction.actor_user_id.in_(candidate_user_ids)
                    ),
                ),
            )
            .subquery()
        )

        stmt = select(
            latest_interactions.c.actor_user_id,
            latest_interactions.c.candidate_user_id,
            latest_interactions.c.action,
            latest_interactions.c.created_at,
        ).where(latest_interactions.c.row_number == 1)

        with SessionLocal() as session:
            return list(session.execute(stmt).all())


def _build_candidate_mapping(candidate_user_ids: Iterable[str]) -> dict[str, set[str]]:
    candidate_mapping: dict[str, set[str]] = {}
    for candidate_user_id in candidate_user_ids:
        normalized_candidate_id = _normalize_id(candidate_user_id)
        if normalized_candidate_id is None:
            continue
        candidate_mapping.setdefault(normalized_candidate_id, set()).add(normalized_candidate_id)
    return candidate_mapping


def _is_actor_side_excluded(action: str, created_at: datetime, now: datetime) -> bool:
    normalized_action = str(action).strip().lower()
    if normalized_action in {"block", "dislike"}:
        return True
    if normalized_action != "skip":
        return False

    created_at_utc = created_at
    if created_at_utc.tzinfo is None:
        created_at_utc = created_at_utc.replace(tzinfo=timezone.utc)

    cooldown_deadline = created_at_utc + timedelta(days=CONFIG.days_cooldown)
    return cooldown_deadline > now


def _is_reverse_blocked(action: str) -> bool:
    return str(action).strip().lower() == "block"


def _normalize_id(value: str | None) -> str | None:
    return normalize_user_id(value, allow_hash=False)
