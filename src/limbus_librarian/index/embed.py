from __future__ import annotations

import hashlib

import numpy as np


def hashed_embed(texts: list[str], dims: int = 1536) -> np.ndarray:
    matrix = np.zeros((len(texts), dims), dtype=np.float32)
    for i, text in enumerate(texts):
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dims).astype(np.float32)
        n = np.linalg.norm(vec)
        if n:
            vec /= n
        matrix[i] = vec
    return matrix


class Embedder:
    def __init__(
        self,
        model: str,
        api_key: str = "",
        dims: int = 1536,
        adapter=None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.dims = dims
        self._adapter = adapter

    @property
    def provider(self) -> str:
        return "openai" if self.api_key else "hash"

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._adapter is None:
            from limbus_librarian.llm import LLMAdapter

            self._adapter = LLMAdapter(
                api_key=self.api_key,
                embedding_model=self.model,
                dims=self.dims,
            )
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        if self.provider != "openai":
            return self._adapter.embed(texts)
        batches = [
            self._adapter.embed(texts[start : start + 128])
            for start in range(0, len(texts), 128)
        ]
        return np.concatenate(batches, axis=0)
