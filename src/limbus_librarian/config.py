from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LIMBUS_",
        extra="ignore",
        populate_by_name=True,
    )

    data_dir: Path = Path("./data")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "limbus_chunks"
    embedding_model: str = "text-embedding-3-small"
    generate_model: str = "gpt-5.6-terra"
    utility_model: str = "gpt-5.6-luna"
    default_config: str = "hybrid"
    cors_origins: str = "http://localhost:5173"
    user_agent: str = "LimbusLibrarian/0.1 (fan research tool; contact: local-dev)"
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "LIMBUS_OPENAI_API_KEY"),
    )
    embedding_dims: int = 1536
    mediawiki_api: str = "https://limbuscompany.wiki.gg/api.php"

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def configs_dir(self) -> Path:
        return self.repo_root / "configs"

    @property
    def fixtures_dir(self) -> Path:
        return self.repo_root / "data" / "fixtures"

    @property
    def gold_path(self) -> Path:
        return self.repo_root / "data" / "eval" / "gold" / "v1.jsonl"

    @property
    def catalog_path(self) -> Path:
        return Path(self.data_dir) / "processed" / "catalog.sqlite"

    @property
    def documents_path(self) -> Path:
        return Path(self.data_dir) / "processed" / "documents.jsonl"

    @property
    def chunks_path(self) -> Path:
        return Path(self.data_dir) / "processed" / "chunks.jsonl"

    @property
    def bm25_path(self) -> Path:
        return Path(self.data_dir) / "indexes" / "bm25"

    @property
    def dense_path(self) -> Path:
        return Path(self.data_dir) / "indexes" / "dense.npz"

    @property
    def dense_manifest_path(self) -> Path:
        return Path(self.data_dir) / "indexes" / "manifest.json"

    def cors_origin_list(self) -> list[str]:
        return [part.strip() for part in self.cors_origins.split(",") if part.strip()]


def get_settings() -> Settings:
    return Settings()
