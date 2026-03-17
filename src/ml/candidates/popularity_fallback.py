from __future__ import annotations

from threading import Lock
from typing import Any

import pandas as pd
import structlog

from core.data_config import Paths
from ml.candidates import transactions_candidates
from ml.infra.blocked_service import InteractionBlocklistService
from ml.services.serving_policy import log_lazy_rebuild_blocked
from ml.services.user_id import normalize_user_id

_BLOCKLIST_SERVICE = InteractionBlocklistService()
_SNAPSHOT_PATH = Paths.popularity_snapshot_path
_CACHE_LOCK = Lock()
_CACHED_SNAPSHOT_DF: pd.DataFrame | None = None
_CACHED_SNAPSHOT_MTIME: float | None = None

log: structlog.BoundLogger = structlog.get_logger(__name__)


def warmup_popularity_snapshot(*, force_reload: bool = False) -> dict[str, object]:
    snapshot_df = _get_popularity_snapshot(force_reload=force_reload)
    status_payload = get_popularity_snapshot_status()
    status_payload["row_count"] = int(snapshot_df.shape[0])
    return status_payload


def get_popularity_snapshot_status() -> dict[str, object]:
    snapshot_path = _SNAPSHOT_PATH
    if not snapshot_path.exists():
        return {
            "source": "popularity",
            "status": "missing",
            "snapshot_path": str(snapshot_path),
        }

    try:
        snapshot_df = _get_popularity_snapshot(force_reload=False)
    except Exception as exc:
        return {
            "source": "popularity",
            "status": "invalid",
            "snapshot_path": str(snapshot_path),
            "error": str(exc),
        }

    latest_mtime = float(snapshot_path.stat().st_mtime)
    artifact_age_seconds = max(0.0, pd.Timestamp.utcnow().timestamp() - latest_mtime)
    return {
        "source": "popularity",
        "status": "ready",
        "snapshot_path": str(snapshot_path),
        "row_count": int(snapshot_df.shape[0]),
        "artifact_age_seconds": round(artifact_age_seconds, 3),
    }


def get_popular_candidates(conf) -> pd.DataFrame:
    actor_user_id = normalize_user_id(getattr(conf, "user_id", None), allow_hash=True)
    top_n = max(int(getattr(conf, "top_n", 0) or 0), 0)
    if actor_user_id is None or top_n <= 0:
        return pd.DataFrame()

    snapshot_df = _get_popularity_snapshot()
    if snapshot_df.empty:
        return pd.DataFrame()

    query_limit = max(top_n * 20, top_n)
    candidates_df = snapshot_df.head(query_limit).copy()
    if candidates_df.empty:
        return pd.DataFrame()

    candidate_ids = candidates_df["candidate_id"].astype("string").tolist()
    try:
        blocked_candidate_ids = _BLOCKLIST_SERVICE.get_blocked_targets(
            actor_user_id=actor_user_id,
            candidate_user_ids=candidate_ids,
        )
    except Exception:
        blocked_candidate_ids = set()

    impactful_feature_count = int(
        getattr(
            conf,
            "an_impactful_features",
            transactions_candidates.CandidateConfig.an_impactful_features,
        )
        or 0
    )
    rows_for_frame: list[dict[str, Any]] = []
    for candidate_row in candidates_df.to_dict(orient="records"):
        candidate_user_id = normalize_user_id(candidate_row.get("candidate_id"), allow_hash=True)
        if candidate_user_id is None:
            continue
        if candidate_user_id == actor_user_id:
            continue
        if candidate_user_id in blocked_candidate_ids:
            continue
        seed_basis = f"{actor_user_id}:{candidate_user_id}:popularity"
        rows_for_frame.append(
            {
                "rank": len(rows_for_frame) + 1,
                "query_party_rk": actor_user_id,
                "party_rk": candidate_user_id,
                "popularity_score": float(candidate_row.get("popularity_score", 0.0)),
                "impactful_features": transactions_candidates._extract_seeded_impactful_features(
                    seed_value=seed_basis,
                    top_k=impactful_feature_count,
                ),
                "impactful_features_origin": "seeded_fallback",
                "impactful_features_seed_basis": seed_basis,
            }
        )
        if len(rows_for_frame) >= top_n:
            break
    return pd.DataFrame(rows_for_frame)


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
