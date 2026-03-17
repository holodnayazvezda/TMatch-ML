from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = (
        Index("ix_interactions_actor_user_id", "actor_user_id"),
        Index("ix_interactions_candidate_user_id", "candidate_user_id"),
        Index("ix_interactions_ml_recommendation_id", "ml_recommendation_id"),
        Index("ix_interactions_actor_created_at", "actor_user_id", "created_at"),
        Index("ix_interactions_candidate_created_at", "candidate_user_id", "created_at"),
        Index(
            "ix_interactions_actor_candidate_created_at",
            "actor_user_id",
            "candidate_user_id",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    actor_user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), nullable=False)
    candidate_user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    ml_recommendation_id: Mapped[str | None] = mapped_column(String, nullable=True)


class MLRecommendationGroup(Base):
    __tablename__ = "ml_recommendation_groups"
    __table_args__ = (
        Index(
            "ix_ml_recommendation_groups_actor_status_shown_at",
            "actor_user_id",
            "group_status",
            "shown_at",
        ),
        Index("ix_ml_recommendation_groups_shown_at", "shown_at"),
    )

    ml_recommendation_id: Mapped[str] = mapped_column(String, primary_key=True)
    actor_user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), nullable=False)
    shown_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    served_stage: Mapped[str] = mapped_column(String(16), nullable=False, default="stage_4")
    group_status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MLRecommendationImpression(Base):
    __tablename__ = "ml_recommendation_impressions"
    __table_args__ = (
        Index("ix_ml_recommendation_impressions_group_id", "ml_recommendation_id"),
        Index("ix_ml_recommendation_impressions_row_status_shown_at", "row_status", "shown_at"),
        Index(
            "ix_ml_recommendation_impressions_actor_candidate",
            "actor_user_id",
            "candidate_user_id",
        ),
    )

    ml_recommendation_id: Mapped[str] = mapped_column(String, primary_key=True)
    actor_user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True)
    candidate_user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True)
    shown_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_interaction_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    final_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    row_status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")

    label_true_like: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    label_true_dislike: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    label_true_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    label_true_block: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    label_pred_like: Mapped[float | None] = mapped_column(Float, nullable=True)
    label_pred_dislike: Mapped[float | None] = mapped_column(Float, nullable=True)
    label_pred_match: Mapped[float | None] = mapped_column(Float, nullable=True)
    label_pred_block: Mapped[float | None] = mapped_column(Float, nullable=True)

    features_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    source_row_json: Mapped[dict] = mapped_column(JSONB, nullable=False)


class MLRecommendationCycleState(Base):
    __tablename__ = "ml_recommendation_cycle_state"
    __table_args__ = (
        Index("ix_ml_recommendation_cycle_state_started_at", "cycle_started_at"),
    )

    actor_user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), primary_key=True)
    cycle_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MLRecommendationSample(Base):
    __tablename__ = "ml_recommendation_samples"
    __table_args__ = (
        Index("ix_ml_recommendation_samples_label_status", "label_status"),
        Index(
            "ix_ml_recommendation_samples_user_candidate",
            "user_id",
            "candidate_user_id",
        ),
        Index("ix_ml_recommendation_samples_created_at", "created_at"),
    )

    ml_recommendation_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), nullable=False)
    candidate_user_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), nullable=False)
    rerank_mode: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    features_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    source_row_json: Mapped[dict] = mapped_column(JSONB, nullable=False)

    y_pred_block: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_pred_dislike: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_pred_skip: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_pred_like: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_pred_match: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_pred_chat_started: Mapped[float | None] = mapped_column(Float, nullable=True)

    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_weights_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    score_weights_version: Mapped[str] = mapped_column(String(32), nullable=False)

    y_true_block: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    y_true_dislike: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    y_true_skip: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    y_true_like: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    y_true_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    y_true_chat_started: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    label_status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    terminal_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    finalized_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    label_rule_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
