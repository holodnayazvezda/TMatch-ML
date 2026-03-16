from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import bindparam, text

from ml.infra.db.session import SessionLocal
from ml.services.user_id import normalize_user_id


@dataclass(frozen=True)
class UserTargetingProfile:
    user_id: str
    gender: str
    age: int
    gender_target: str
    age_from: int
    age_to: int


def filter_candidate_frame_by_targeting(
    candidate_df: pd.DataFrame,
    *,
    actor_user_id: str,
    as_of: date | None = None,
) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    result = candidate_df.copy()
    if "candidate_id" not in result.columns:
        return result.iloc[0:0].copy()

    normalized_actor_user_id = normalize_user_id(actor_user_id, allow_hash=True)
    if normalized_actor_user_id is None:
        return result.iloc[0:0].copy()

    candidate_ids = _normalize_user_ids(result["candidate_id"].tolist())
    constraint_lookup = load_user_targeting_profiles(
        [normalized_actor_user_id, *candidate_ids],
        as_of=as_of,
    )
    actor_profile = constraint_lookup.get(normalized_actor_user_id)
    if actor_profile is None:
        return result.iloc[0:0].copy()

    allowed_candidate_ids = {
        candidate_id
        for candidate_id in candidate_ids
        if (
            candidate_profile := constraint_lookup.get(candidate_id)
        ) is not None
        and is_mutually_targeting_compatible(actor_profile, candidate_profile)
    }
    if not allowed_candidate_ids:
        return result.iloc[0:0].copy()

    filtered = result[result["candidate_id"].isin(allowed_candidate_ids)].copy()
    return filtered.reset_index(drop=True)


def load_user_targeting_profiles(
    user_ids: Iterable[str],
    *,
    as_of: date | None = None,
) -> dict[str, UserTargetingProfile]:
    normalized_user_ids = _normalize_user_ids(user_ids)
    if not normalized_user_ids:
        return {}

    stmt = (
        text(
            """
            SELECT
                profiles.user_id::text AS user_id,
                profiles.gender::text AS gender,
                profiles.birth_date AS birth_date,
                targeting.gender_target::text AS gender_target,
                targeting.age_from AS age_from,
                targeting.age_to AS age_to
            FROM profiles
            LEFT JOIN targeting
                ON targeting.user_id = profiles.user_id
            WHERE profiles.user_id::text IN :user_ids
            """
        )
        .bindparams(bindparam("user_ids", expanding=True))
    )

    lookup: dict[str, UserTargetingProfile] = {}
    reference_date = as_of or date.today()
    with SessionLocal() as session:
        rows = session.execute(stmt, {"user_ids": normalized_user_ids}).mappings().all()

    for row in rows:
        profile = _build_user_targeting_profile(dict(row), as_of=reference_date)
        if profile is None:
            continue
        lookup[profile.user_id] = profile
    return lookup


def is_mutually_targeting_compatible(
    actor_profile: UserTargetingProfile,
    candidate_profile: UserTargetingProfile,
) -> bool:
    return _matches_targeting(actor_profile, candidate_profile) and _matches_targeting(
        candidate_profile,
        actor_profile,
    )


def _matches_targeting(
    source_profile: UserTargetingProfile,
    candidate_profile: UserTargetingProfile,
) -> bool:
    if not _gender_target_matches(
        gender_target=source_profile.gender_target,
        candidate_gender=candidate_profile.gender,
    ):
        return False

    return source_profile.age_from <= candidate_profile.age <= source_profile.age_to


def _build_user_targeting_profile(
    raw_row: dict[str, Any],
    *,
    as_of: date,
) -> UserTargetingProfile | None:
    user_id = normalize_user_id(raw_row.get("user_id"), allow_hash=True)
    gender = _normalize_gender(raw_row.get("gender"))
    age = _resolve_age(raw_row.get("birth_date"), as_of=as_of)
    gender_target = _normalize_gender_target(raw_row.get("gender_target"))
    age_range = _normalize_age_range(raw_row.get("age_from"), raw_row.get("age_to"))

    if user_id is None or gender is None or age is None or gender_target is None or age_range is None:
        return None

    return UserTargetingProfile(
        user_id=user_id,
        gender=gender,
        age=age,
        gender_target=gender_target,
        age_from=age_range[0],
        age_to=age_range[1],
    )


def _normalize_user_ids(user_ids: Iterable[str]) -> list[str]:
    normalized_user_ids: set[str] = set()
    for user_id in user_ids:
        normalized_user_id = normalize_user_id(user_id, allow_hash=True)
        if normalized_user_id is None:
            continue
        normalized_user_ids.add(normalized_user_id)
    return sorted(normalized_user_ids)


def _normalize_gender(raw_value: Any) -> str | None:
    normalized_text = _normalize_text(raw_value)
    if normalized_text is None:
        return None

    if normalized_text in {"male", "man", "men", "m"}:
        return "male"
    if normalized_text in {"female", "woman", "women", "f"}:
        return "female"
    if normalized_text in {"other", "nonbinary", "non_binary"}:
        return "other"
    return None


def _normalize_gender_target(raw_value: Any) -> str | None:
    normalized_text = _normalize_text(raw_value)
    if normalized_text is None:
        return None

    if normalized_text in {"all", "any", "both", "everyone", "anyone"}:
        return "any"
    if normalized_text in {"male", "man", "men", "m"}:
        return "male"
    if normalized_text in {"female", "woman", "women", "f"}:
        return "female"
    if ("male" in normalized_text and "female" in normalized_text) or (
        "man" in normalized_text and "woman" in normalized_text
    ):
        return "any"
    return None


def _normalize_age_range(raw_from: Any, raw_to: Any) -> tuple[int, int] | None:
    try:
        age_from = int(raw_from)
        age_to = int(raw_to)
    except (TypeError, ValueError):
        return None

    if age_from < 0 or age_to < 0:
        return None
    if age_from > age_to:
        age_from, age_to = age_to, age_from
    return age_from, age_to


def _resolve_age(raw_birth_date: Any, *, as_of: date) -> int | None:
    birth_date = _coerce_date(raw_birth_date)
    if birth_date is None or birth_date > as_of:
        return None

    age = as_of.year - birth_date.year
    if (as_of.month, as_of.day) < (birth_date.month, birth_date.day):
        age -= 1
    return age if age >= 0 else None


def _coerce_date(raw_value: Any) -> date | None:
    if isinstance(raw_value, datetime):
        return raw_value.date()
    if isinstance(raw_value, date):
        return raw_value
    if raw_value is None:
        return None

    text_value = str(raw_value).strip()
    if not text_value:
        return None

    try:
        return datetime.fromisoformat(text_value).date()
    except ValueError:
        return None


def _gender_target_matches(*, gender_target: str, candidate_gender: str) -> bool:
    if gender_target == "any":
        return True
    return gender_target == candidate_gender


def _normalize_text(raw_value: Any) -> str | None:
    if raw_value is None:
        return None

    text_value = str(raw_value).strip().lower()
    if not text_value:
        return None

    normalized_text = text_value.rsplit(".", 1)[-1]
    return normalized_text.replace("-", "_").replace(" ", "_")
