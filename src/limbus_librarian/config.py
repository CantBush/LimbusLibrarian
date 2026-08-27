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
    default_config: str = "vector_only"
    cors_origins: str = "http://localhost:5173"
    user_agent: str = "LimbusLibrarian/0.1 (fan research tool; contact: local-dev)"
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "LIMBUS_OPENAI_API_KEY"),
    )
    embedding_dims: int = 1536
    mediawiki_api: str = "https://limbuscompany.wiki.gg/api.php"
    wiki_categories: str = (
        "Characters,Sinners,Story,Cantos,Factions,Abnormalities,Locations,Lore"
    )
    wiki_category_depth: int = 2
    wiki_batch_size: int = 50

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
    def gold_dir(self) -> Path:
        return self.repo_root / "data" / "eval" / "gold"

    @property
    def gold_path(self) -> Path:
        return self.gold_dir / "v1.jsonl"

    def gold_path_for(self, name: str) -> Path:
        candidate = Path(name)
        if candidate.name != name or candidate.suffix not in {"", ".jsonl"}:
            raise ValueError("Gold set must be a name such as 'wiki_v1'")
        return self.gold_dir / (candidate.name if candidate.suffix else f"{candidate.name}.jsonl")

    @property
    def catalog_path(self) -> Path:
        return Path(self.data_dir) / "processed" / "catalog.sqlite"

    @property
    def ingest_state_path(self) -> Path:
        return Path(self.data_dir) / "raw" / "ingest_state.json"

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

    def wiki_category_list(self) -> tuple[str, ...]:
        return tuple(part.strip() for part in self.wiki_categories.split(",") if part.strip())


def get_settings() -> Settings:
    return Settings()
