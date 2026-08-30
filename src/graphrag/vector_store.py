from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorStore:
    """Cosine-similarity index over normalized embeddings.

    Swap this for a real Milvus collection in production; add/search stay
    the same shape.
    """

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.metadata: dict[str, dict] = {}
        self._vectors: np.ndarray | None = None

    def add(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]) -> None:
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self.ids.extend(ids)
        for chunk_id, meta in zip(ids, metadatas):
            self.metadata[chunk_id] = meta

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        if self._vectors is None or len(self.ids) == 0:
            return []
        scores = self._vectors @ query_vector
        top_k = min(top_k, len(self.ids))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i])) for i in top_indices]

    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(dir_path / "vectors.npy", self._vectors)
        (dir_path / "meta.json").write_text(
            json.dumps({"ids": self.ids, "metadata": self.metadata}), encoding="utf-8"
        )

    @classmethod
    def load(cls, dir_path: str | Path) -> "VectorStore":
        dir_path = Path(dir_path)
        store = cls()
        store._vectors = np.load(dir_path / "vectors.npy")
        meta = json.loads((dir_path / "meta.json").read_text(encoding="utf-8"))
        store.ids = meta["ids"]
        store.metadata = meta["metadata"]
        return store
