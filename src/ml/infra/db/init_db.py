from __future__ import annotations

from threading import Lock

from sqlalchemy import text

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
        from . import interaction_feature_tables, models

        managed_tables = (
            interaction_feature_tables.user_interaction_features_table,
            models.MLRecommendationGroup.__table__,
            models.MLRecommendationImpression.__table__,
            models.MLRecommendationCycleState.__table__,
            models.MLRecommendationSample.__table__,
        )
        for table in managed_tables:
            table.create(bind=engine, checkfirst=True)

        with engine.begin() as connection:
            for table_name, column_name in (
                ("user_interaction_features", "user_id"),
                ("ml_recommendation_groups", "actor_user_id"),
                ("ml_recommendation_impressions", "actor_user_id"),
                ("ml_recommendation_impressions", "candidate_user_id"),
                ("ml_recommendation_cycle_state", "actor_user_id"),
                ("ml_recommendation_samples", "user_id"),
                ("ml_recommendation_samples", "candidate_user_id"),
            ):
                _ensure_uuid_column(connection, table_name=table_name, column_name=column_name)

            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_user_interaction_features_snapshot_at
                    ON user_interaction_features (snapshot_at)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_ml_recommendation_samples_label_status
                    ON ml_recommendation_samples (label_status)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_ml_recommendation_samples_user_candidate
                    ON ml_recommendation_samples (user_id, candidate_user_id)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_ml_recommendation_samples_created_at
                    ON ml_recommendation_samples (created_at)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    ALTER TABLE ml_recommendation_groups
                    ADD COLUMN IF NOT EXISTS served_stage VARCHAR(16) NOT NULL DEFAULT 'stage_4'
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_ml_recommendation_groups_actor_status_shown_at
                    ON ml_recommendation_groups (actor_user_id, group_status, shown_at)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_ml_recommendation_groups_served_stage
                    ON ml_recommendation_groups (served_stage)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_ml_recommendation_groups_shown_at
                    ON ml_recommendation_groups (shown_at)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_ml_recommendation_cycle_state_started_at
                    ON ml_recommendation_cycle_state (cycle_started_at)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_ml_recommendation_impressions_group_id
                    ON ml_recommendation_impressions (ml_recommendation_id)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_ml_recommendation_impressions_row_status_shown_at
                    ON ml_recommendation_impressions (row_status, shown_at)
                    """
                )
            )
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS ix_ml_recommendation_impressions_actor_candidate
                    ON ml_recommendation_impressions (actor_user_id, candidate_user_id)
                    """
                )
            )
        _SCHEMA_READY = True


def _ensure_uuid_column(connection, *, table_name: str, column_name: str) -> None:
    connection.execute(
        text(
            f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = '{table_name}'
                      AND column_name = '{column_name}'
                      AND data_type <> 'uuid'
                ) THEN
                    EXECUTE 'ALTER TABLE {table_name} ALTER COLUMN {column_name} TYPE UUID USING {column_name}::uuid';
                END IF;
            END
            $$;
            """
        )
    )


if __name__ == "__main__":
    init_db(force=True)
    print("DB schema initialized")
