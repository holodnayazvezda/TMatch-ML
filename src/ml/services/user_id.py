from __future__ import annotations

import re
import uuid
from typing import Any


USER_HASH_UUID_NAMESPACE = uuid.UUID("8c1153c4-a5bc-4f16-bd2f-3b6f62b4f6b1")

_HASH_64_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def is_uuid(value: Any) -> bool:
    return normalize_uuid(value) is not None


def normalize_uuid(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return str(uuid.UUID(text))
    except (AttributeError, TypeError, ValueError):
        return None


def is_hash64(value: Any) -> bool:
    return _normalize_hash(value) is not None


def hash_to_uuid(value: Any) -> str | None:
    normalized_hash = _normalize_hash(value)
    if normalized_hash is None:
        return None
    return str(uuid.uuid5(USER_HASH_UUID_NAMESPACE, normalized_hash))


def normalize_user_id(value: Any, *, allow_hash: bool = False) -> str | None:
    normalized_uuid = normalize_uuid(value)
    if normalized_uuid is not None:
        return normalized_uuid
    if not allow_hash:
        return None
    return hash_to_uuid(value)


def _normalize_hash(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None
    if _HASH_64_PATTERN.fullmatch(text) is None:
        return None
    return text
