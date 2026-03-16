from datetime import datetime

from sqlalchemy import DateTime, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = (
        Index("ix_interactions_actor_user_id", "actor_user_id"),
        Index("ix_interactions_candidate_user_id", "candidate_user_id"),
        Index("ix_interactions_actor_created_at", "actor_user_id", "created_at"),
        Index("ix_interactions_candidate_created_at", "candidate_user_id", "created_at"),
        Index(
            "ix_interactions_actor_candidate_created_at",
            "actor_user_id",
            "candidate_user_id",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    actor_user_id: Mapped[str] = mapped_column(String, nullable=False)
    candidate_user_id: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    ml_recommendation_id: Mapped[str | None] = mapped_column(String, nullable=True)
