from dataclasses import dataclass


@dataclass(frozen=True)
class GlobalConfig:
    n_max_candidates: int | None = 300
    n_min_candidates: int | None = 100
    
    n_rec_users: int = 5
    postgres_dsn: str = "postgresql+psycopg://postgres:postgres@host.docker.internal:5432/ml_db"



    days_cooldown: int = 7

CONFIG = GlobalConfig()