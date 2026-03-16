from dataclasses import dataclass
from os import getenv
from urllib.parse import quote_plus


@dataclass(frozen=True)
class GlobalConfig:
    n_max_candidates: int | None = 300
    n_min_candidates: int | None = 100

    n_rec_users: int = 5
    days_cooldown: int = 7

    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ml_db"

    @property
    def postgres_dsn(self) -> str:
        password = quote_plus(self.postgres_password)
        return f"postgresql+psycopg://{self.postgres_user}:{password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


CONFIG = GlobalConfig(
    postgres_user=getenv("POSTGRES_USER", "postgres"),
    postgres_password=getenv("POSTGRES_PASSWORD", "postgres"),
    postgres_host=getenv("POSTGRES_HOST", "localhost"),
    postgres_port=int(getenv("POSTGRES_PORT", "5432")),
    postgres_db=getenv("POSTGRES_DB", "ml_db"),
)
