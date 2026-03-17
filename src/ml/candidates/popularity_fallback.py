from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import select

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
                "popularity_score": float(candidate_row.get("popularity_score", 0.0)),
                "impactful_features_seed_basis": seed_basis,
            }
        )
        if len(rows_for_frame) >= top_n:
            break

    result_df = pd.DataFrame(rows_for_frame)
    if result_df.empty:
        return result_df

    support_weights = transactions_candidates._compute_support_weights_from_scores(
        result_df["popularity_score"].to_numpy(dtype=float),
        lower_is_better=False,
    )
    seed_basis_values = result_df["impactful_features_seed_basis"].astype("string").tolist()
    impactful_features, impactful_origins, impactful_seed_basis = (
        transactions_candidates.build_impactful_features_for_candidate_ids(
            query_user_id=actor_user_id,
            candidate_user_ids=result_df["party_rk"].astype("string").tolist(),
            top_k=impactful_feature_count,
            support_weights=support_weights,
            seed_bases=seed_basis_values,
        )
    )
    result_df["impactful_features"] = impactful_features
    result_df["impactful_features_origin"] = impactful_origins
    result_df["impactful_features_seed_basis"] = impactful_seed_basis
    return result_df


def _get_popularity_snapshot(*, force_reload: bool = False) -> pd.DataFrame:
    global _CACHED_SNAPSHOT_DF, _CACHED_SNAPSHOT_MTIME
    snapshot_path = _SNAPSHOT_PATH
    if not snapshot_path.exists():
        log_lazy_rebuild_blocked(
            source="popularity",
            reason="precomputed_snapshot_missing",
            artifact_path=snapshot_path,
        )
        _CACHED_SNAPSHOT_DF = pd.DataFrame(columns=["candidate_id", "popularity_score"])
        _CACHED_SNAPSHOT_MTIME = None
        return _CACHED_SNAPSHOT_DF

    snapshot_mtime = float(snapshot_path.stat().st_mtime)
    with _CACHE_LOCK:
        if (
            not force_reload
            and _CACHED_SNAPSHOT_DF is not None
            and _CACHED_SNAPSHOT_MTIME is not None
            and _CACHED_SNAPSHOT_MTIME == snapshot_mtime
        ):
            return _CACHED_SNAPSHOT_DF

        try:
            snapshot_df = pd.read_parquet(snapshot_path)
        except Exception as exc:
            log.warning(
                "popularity_snapshot_load_failed",
                snapshot_path=str(snapshot_path),
                error=str(exc),
            )
            _CACHED_SNAPSHOT_DF = pd.DataFrame(columns=["candidate_id", "popularity_score"])
            _CACHED_SNAPSHOT_MTIME = snapshot_mtime
            return _CACHED_SNAPSHOT_DF

        candidate_column = "candidate_id"
        if candidate_column not in snapshot_df.columns:
            if "candidate_user_id" in snapshot_df.columns:
                snapshot_df = snapshot_df.rename(columns={"candidate_user_id": "candidate_id"})
            else:
                snapshot_df["candidate_id"] = pd.Series(dtype="string")
        if "popularity_score" not in snapshot_df.columns:
            snapshot_df["popularity_score"] = 0.0

        snapshot_df["candidate_id"] = (
            snapshot_df["candidate_id"]
            .map(lambda raw_id: normalize_user_id(raw_id, allow_hash=True))
            .astype("string")
        )
        snapshot_df["popularity_score"] = pd.to_numeric(snapshot_df["popularity_score"], errors="coerce").fillna(0.0)
        snapshot_df = snapshot_df[
            snapshot_df["candidate_id"].notna() & (snapshot_df["candidate_id"] != "")
        ].copy()
        snapshot_df = snapshot_df.sort_values(
            ["popularity_score", "candidate_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

        _CACHED_SNAPSHOT_DF = snapshot_df
        _CACHED_SNAPSHOT_MTIME = snapshot_mtime
        return _CACHED_SNAPSHOT_DF
