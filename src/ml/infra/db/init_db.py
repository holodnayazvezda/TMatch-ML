from __future__ import annotations

from threading import Lock

from sqlalchemy import text

from .base import Base
from .session import engine

_SCHEMA_INIT_LOCK = Lock()
_SCHEMA_READY = False


def init_db(force: bool = False) -> None:
    global _SCHEMA_READY

    if _SCHEMA_READY and not force:
        return

    with _SCHEMA_INIT_LOCK:
        if _SCHEMA_READY and not force:
            return

        # Import models/tables lazily so they are registered in Base.metadata.
        from . import interaction_feature_tables, models  # noqa: F401

        Base.metadata.create_all(bind=engine)

        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_interactions_actor_user_id
                    ON interactions (actor_user_id)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_interactions_candidate_user_id
                    ON interactions (candidate_user_id)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_interactions_actor_created_at
                    ON interactions (actor_user_id, created_at)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_interactions_candidate_created_at
                    ON interactions (candidate_user_id, created_at)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_interactions_actor_candidate_created_at
                    ON interactions (actor_user_id, candidate_user_id, created_at)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_user_interaction_features_snapshot_at
                    ON user_interaction_features (snapshot_at)
                    """
                )
            )

        _SCHEMA_READY = True


if __name__ == "__main__":
    init_db(force=True)
    print("DB schema initialized")
