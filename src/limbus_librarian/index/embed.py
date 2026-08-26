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
    def __init__(self, model: str, api_key: str = "", dims: int = 1536) -> None:
        self.model = model
        self.api_key = api_key
        self.dims = dims

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        if self.api_key:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.embeddings.create(model=self.model, input=texts)
            vecs = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            return np.array(vecs, dtype=np.float32)
        return hashed_embed(texts, self.dims)
