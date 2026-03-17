from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    server_host: str = "0.0.0.0"
    server_port: int = 8080
    root_path: str = ""
    warmup_faiss_on_startup: bool = True
    interaction_poll_interval_seconds: float = 0.5
    interaction_poll_batch_size: int = 500


settings = Settings()
