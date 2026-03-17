from __future__ import annotations

import json
import uuid
from time import perf_counter
from typing import Any, Iterator

import polars as pl
import structlog
from sqlalchemy import bindparam, text

from ml.featurize import transactions_features
from ml.infra.db.session import SessionLocal
from ml.services.user_id import normalize_user_id

log: structlog.BoundLogger = structlog.get_logger(__name__)

_DATING_PROFILE_TABLE = "dating_profiles"
_DATING_PROFILE_TRAIT_TABLE = "dating_profile_traits"

_TRAIT_TAGS_INTERNAL_COLUMN = "trait_tags_internal"
_DATING_PROFILE_BATCH_SIZE = 1000


def sync_dating_profile_traits(user_features_df: pl.DataFrame) -> dict[str, Any]:
    payload_by_user = _build_traits_payload_by_user(user_features_df)
    user_ids = sorted(payload_by_user)

    summary = {
        "status": "ok",
        "users_in_payload": len(user_ids),
        "profiles_found": 0,
        "profiles_without_row": 0,
        "existing_traits_deleted": 0,
        "inserted_traits_count": 0,
        "elapsed_ms": 0.0,
    }
    if not user_ids:
        log.info("dating_profile_traits_sync_skipped_empty_payload", **summary)
        return summary

    started_at = perf_counter()
    profile_map: dict[str, str] = {}
    deleted_count = 0
    inserted_count = 0
    try:
        with SessionLocal() as session:
            with session.begin():
                for user_id_chunk in _iter_chunks(user_ids, chunk_size=_DATING_PROFILE_BATCH_SIZE):
                    chunk_profile_map = _load_dating_profile_ids(session, user_ids=user_id_chunk)
                    profile_map.update(chunk_profile_map)

                profile_ids = sorted(set(profile_map.values()))
                if profile_ids:
                    for profile_id_chunk in _iter_chunks(profile_ids, chunk_size=_DATING_PROFILE_BATCH_SIZE):
                        deleted_count += _delete_existing_traits(session, dating_profile_ids=profile_id_chunk)

                trait_rows = _build_trait_rows(payload_by_user=payload_by_user, profile_map=profile_map)
                if trait_rows:
                    for trait_row_chunk in _iter_chunks(trait_rows, chunk_size=_DATING_PROFILE_BATCH_SIZE):
                        inserted_count += _insert_traits(session, rows=trait_row_chunk)

        summary["profiles_found"] = len(profile_map)
        summary["profiles_without_row"] = max(0, len(user_ids) - len(profile_map))
        summary["existing_traits_deleted"] = int(deleted_count)
        summary["inserted_traits_count"] = int(inserted_count)
        summary["elapsed_ms"] = round((perf_counter() - started_at) * 1000, 2)
        log.info("dating_profile_traits_sync_succeeded", **summary)
        return summary
    except Exception as exc:
        summary["status"] = "failed"
        summary["profiles_found"] = len(profile_map)
        summary["profiles_without_row"] = max(0, len(user_ids) - len(profile_map))
        summary["existing_traits_deleted"] = int(deleted_count)
        summary["inserted_traits_count"] = int(inserted_count)
        summary["elapsed_ms"] = round((perf_counter() - started_at) * 1000, 2)
        log.exception("dating_profile_traits_sync_failed", error=str(exc), **summary)
        raise


def _build_traits_payload_by_user(user_features_df: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    required_columns = [
        "party_rk",
        _TRAIT_TAGS_INTERNAL_COLUMN,
        *transactions_features.TRAIT_SCORE_COLUMN_BY_CODE.values(),
    ]
    working_df = user_features_df
    missing_columns = [column_name for column_name in required_columns if column_name not in working_df.columns]
    if missing_columns:
        working_df = working_df.with_columns([pl.lit(None).alias(column_name) for column_name in missing_columns])

    payload_by_user: dict[str, list[dict[str, Any]]] = {}
    for row in working_df.select(required_columns).iter_rows(named=True):
        normalized_user_id = normalize_user_id(row.get("party_rk"), allow_hash=True)
        if normalized_user_id is None:
            continue

        trait_codes = _deserialize_trait_codes(row.get(_TRAIT_TAGS_INTERNAL_COLUMN))
        traits_payload = []
        for trait_code in trait_codes:
            score_column = transactions_features.TRAIT_SCORE_COLUMN_BY_CODE.get(trait_code)
            if score_column is None:
                continue
            score_value = _coerce_score(row.get(score_column))
            if score_value <= 0.0:
                continue
            traits_payload.append(
                {
                    "trait_code": trait_code,
                    "score": score_value,
                    "is_hidden": False,
                }
            )

        payload_by_user[normalized_user_id] = traits_payload
    return payload_by_user


def _load_dating_profile_ids(session, *, user_ids: list[str]) -> dict[str, str]:
    if not user_ids:
        return {}
    stmt = (
        text(
            f"""
            SELECT id::text AS profile_id, user_id::text AS user_id
            FROM {_DATING_PROFILE_TABLE}
            WHERE user_id::text IN :user_ids
            """
        )
        .bindparams(bindparam("user_ids", expanding=True))
    )
    rows = session.execute(stmt, {"user_ids": user_ids}).mappings().all()
    return {
        str(row["user_id"]): str(row["profile_id"])
        for row in rows
        if row.get("user_id") is not None and row.get("profile_id") is not None
    }


def _delete_existing_traits(session, *, dating_profile_ids: list[str]) -> int:
    if not dating_profile_ids:
        return 0
    stmt = (
        text(
            f"""
            DELETE FROM {_DATING_PROFILE_TRAIT_TABLE}
            WHERE dating_profile_id::text IN :dating_profile_ids
            """
        )
        .bindparams(bindparam("dating_profile_ids", expanding=True))
    )
    result = session.execute(stmt, {"dating_profile_ids": dating_profile_ids})
    return int(result.rowcount or 0)


def _build_trait_rows(
    *,
    payload_by_user: dict[str, list[dict[str, Any]]],
    profile_map: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for user_id in sorted(payload_by_user):
        dating_profile_id = profile_map.get(user_id)
        if dating_profile_id is None:
            continue
        for trait_payload in payload_by_user[user_id]:
            rows.append(
                {
                    "id": uuid.uuid4(),
                    "dating_profile_id": uuid.UUID(dating_profile_id),
                    "trait_code": str(trait_payload["trait_code"]),
                    "score": float(trait_payload["score"]),
                    "is_hidden": bool(trait_payload.get("is_hidden", False)),
                }
            )
    return rows


def _insert_traits(session, *, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    stmt = text(
        f"""
        INSERT INTO {_DATING_PROFILE_TRAIT_TABLE}
            (id, dating_profile_id, trait_code, score, is_hidden)
        VALUES
            (:id, :dating_profile_id, :trait_code, :score, :is_hidden)
        """
    )
    result = session.execute(stmt, rows)
    return int(result.rowcount or 0)


def _deserialize_trait_codes(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []

    if isinstance(raw_value, list):
        parsed = raw_value
    else:
        text_value = str(raw_value).strip()
        if not text_value:
            return []
        try:
            parsed = json.loads(text_value)
        except json.JSONDecodeError:
            return []

    if not isinstance(parsed, list):
        return []

    ordered_codes: list[str] = []
    seen_codes: set[str] = set()
    for trait_code in parsed:
        normalized_code = str(trait_code).strip()
        if not normalized_code:
            continue
        if normalized_code not in transactions_features.TRAIT_SCORE_COLUMN_BY_CODE:
            continue
        if normalized_code in seen_codes:
            continue
        seen_codes.add(normalized_code)
        ordered_codes.append(normalized_code)
    return ordered_codes


def _coerce_score(raw_value: Any) -> float:
    try:
        score_value = float(raw_value)
    except (TypeError, ValueError):
        return 0.0
    if score_value <= 0.0:
        return 0.0
    if score_value >= 1.0:
        return 1.0
    return round(score_value, 6)


def _iter_chunks(items: list[Any], *, chunk_size: int) -> Iterator[list[Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]
