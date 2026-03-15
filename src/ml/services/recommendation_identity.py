import secrets
import time

_CROCKFORD_BASE32_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_TIMESTAMP_LENGTH = 10
_ULID_RANDOM_LENGTH = 16
_ULID_RANDOM_BITS = 80


def generate_recommendation_id() -> str:
    """
    Placeholder entrypoint for recommendation_id generation.
    TODO: replace with production strategy (e.g., centralized ID service).
    """
    return _generate_ulid()


def _generate_ulid() -> str:
    timestamp_ms = int(time.time() * 1000)
    randomness = secrets.randbits(_ULID_RANDOM_BITS)
    return (
        _encode_crockford_base32(timestamp_ms, _ULID_TIMESTAMP_LENGTH)
        + _encode_crockford_base32(randomness, _ULID_RANDOM_LENGTH)
    )


def _encode_crockford_base32(value: int, length: int) -> str:
    if value < 0:
        raise ValueError("value must be non-negative")

    encoded = []
    current = value
    for _ in range(length):
        encoded.append(_CROCKFORD_BASE32_ALPHABET[current & 31])
        current >>= 5
    encoded.reverse()
    return "".join(encoded)
