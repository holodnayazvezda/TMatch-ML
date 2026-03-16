from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import select

from ml.candidates import transactions_candidates
from ml.infra.blocked_service import InteractionBlocklistService
from ml.infra.db.models import Interaction
from ml.infra.db.session import SessionLocal
from ml.services.user_id import normalize_user_id

_BLOCKLIST_SERVICE = InteractionBlocklistService()


def get_popular_candidates(conf) -> pd.DataFrame:
    actor_user_id = normalize_user_id(getattr(conf, "user_id", None), allow_hash=True)
    top_n = max(int(getattr(conf, "top_n", 0) or 0), 0)
    if actor_user_id is None or top_n <= 0:
        return pd.DataFrame()

    weights = {
        str(action).strip().lower(): float(weight)
        for action, weight in dict(getattr(conf, "popularity_action_weights", {}) or {}).items()
    }
    if not weights:
        weights = {
            "like": 3.0,
            "match": 8.0,
            "chat_started": 4.0,
            "skip": -1.0,
            "dislike": -6.0,
            "block": -15.0,
        }

    try:
        with SessionLocal() as session:
            stmt = (
                select(
                    Interaction.candidate_user_id,
                    Interaction.action,
                )
                .where(Interaction.deleted_at.is_(None))
            )
            rows = list(session.execute(stmt).all())
    except Exception:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    popularity_by_candidate: dict[str, float] = {}
    for candidate_user_id, action in rows:
        normalized_candidate_user_id = normalize_user_id(candidate_user_id, allow_hash=True)
        normalized_action = str(action).strip().lower()
        if normalized_candidate_user_id is None:
            continue
        popularity_by_candidate[normalized_candidate_user_id] = (
            popularity_by_candidate.get(normalized_candidate_user_id, 0.0)
            + float(weights.get(normalized_action, 0.0))
        )

    if not popularity_by_candidate:
        return pd.DataFrame()

    try:
        blocked_candidate_ids = _BLOCKLIST_SERVICE.get_blocked_targets(
            actor_user_id=actor_user_id,
            candidate_user_ids=list(popularity_by_candidate),
        )
    except Exception:
        blocked_candidate_ids = set()
    rows_for_frame: list[dict[str, Any]] = []
    impactful_feature_count = int(
        getattr(
            conf,
            "an_impactful_features",
            transactions_candidates.CandidateConfig.an_impactful_features,
        )
        or 0
    )
    for candidate_user_id, popularity_score in sorted(
        popularity_by_candidate.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        if candidate_user_id == actor_user_id:
            continue
        if candidate_user_id in blocked_candidate_ids:
            continue
        rows_for_frame.append(
            {
                "rank": len(rows_for_frame) + 1,
                "query_party_rk": actor_user_id,
                "party_rk": candidate_user_id,
                "popularity_score": float(popularity_score),
                "impactful_features": transactions_candidates._extract_seeded_impactful_features(
                    seed_value=f"{actor_user_id}:{candidate_user_id}:popularity",
                    top_k=impactful_feature_count,
                ),
            }
        )
        if len(rows_for_frame) >= top_n:
            break

    return pd.DataFrame(rows_for_frame)
