from pathlib import Path

from limbus_librarian.config import Settings


def test_settings_load():
    settings = Settings()
    assert settings.embedding_model
    assert settings.generate_model
    assert (Path(__file__).resolve().parents[1] / "NOTICE.md").exists()
