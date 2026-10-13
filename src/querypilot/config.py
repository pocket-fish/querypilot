from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QUERYPILOT_", env_file=".env", extra="ignore")

    ollama_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"
    max_iterations: int = 3
    data_db_path: str = "data/demo.db"
    history_db_path: str = "data/history.db"
    checkpoint_db_path: str = "data/checkpoints.db"
    max_rows: int = 100
    log_level: str = "INFO"


def get_settings() -> Settings:
    return Settings()
