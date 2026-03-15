from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import CONFIG


engine = create_engine(
    CONFIG.postgres_dsn,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)
