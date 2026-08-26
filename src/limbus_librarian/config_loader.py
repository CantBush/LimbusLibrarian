from __future__ import annotations

from pathlib import Path

import yaml

from limbus_librarian.models import RetrievalConfig


def load_retrieval_config(path: Path) -> RetrievalConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RetrievalConfig.model_validate(data)


def load_named_config(configs_dir: Path, name: str) -> RetrievalConfig:
    path = configs_dir / "retrieval" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown retrieval config: {name}")
    return load_retrieval_config(path)


def list_config_ids(configs_dir: Path) -> list[str]:
    folder = configs_dir / "retrieval"
    return sorted(p.stem for p in folder.glob("*.yaml"))
