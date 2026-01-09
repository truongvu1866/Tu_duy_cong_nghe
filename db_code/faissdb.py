import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple


class FaceFaissDB:
    def __init__(
        self,
            dim: int = 512,
            index_type: str = "hnsw",
            ef_search: int = 64,
            ef_construction: int = 200
    ) -> None:
        self.dim = dim
        self.index_type = index_type
        self.ef_search = ef_search
        self.ef_construction = ef_construction

        self.index = self._create_index()
        self.id_map = []  # index_id -> user_id

    # =========================
    # Index creation
    # =========================
    def _create_index(self):
        if self.index_type == "flat":
            index = faiss.IndexFlatIP(self.dim)

        elif self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, 32)
            index.hnsw.efSearch = self.ef_search
            index.hnsw.efConstruction = self.ef_construction

        else:
            raise ValueError("Unsupported index type")

        return index

    # =========================
    # Normalize embedding
    # =========================
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        if v.ndim == 1:
            norm = np.linalg.norm(v)
            return v if norm < 1e-6 else v / norm
        else:
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            return np.divide(v, norms, where=norms > 1e-6)

    # =========================
    # Add user
    # =========================
    def add_user(self, user_id, embedding):
        embedding = np.asarray(embedding, dtype=np.float32)
        embedding = self._normalize(embedding)

        self.index.add(embedding)
        self.id_map.extend(user_id)

    # =========================
    # Bulk add
    # =========================
    def add_users(self, user_ids: List[int], embeddings: np.ndarray):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        embeddings = self._normalize(embeddings).reshape(1, -1)

        self.index.add(embeddings)
        self.id_map.extend(user_ids)

    # =========================
    # Search
    # =========================
    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 1
    ) -> List[Tuple[int, float]]:
        if self.index.ntotal == 0:
            return []

        embedding = np.asarray(embedding, dtype=np.float32)
        embedding = self._normalize(embedding).reshape(1, -1)

        scores, indices = self.index.search(embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            user_id = self.id_map[idx]
            results.append((user_id, float(score)))

        return results

    # =========================
    # Remove user (rebuild)
    # =========================
    def remove_user(self, user_id: int):
        if user_id not in self.id_map:
            return

        keep_indices = [i for i, uid in enumerate(self.id_map) if uid != user_id]

        if not keep_indices:
            self.index = self._create_index()
            self.id_map = []
            return

        embeddings = np.vstack([
            self.index.reconstruct(i) for i in keep_indices
        ])
        embeddings = self._normalize(embeddings)

        self.index = self._create_index()
        self.index.add(embeddings)
        self.id_map = new_id_map

    # =========================
    # Save
    # =========================
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "faiss.index"))

        with open(os.path.join(path, "id_map.pkl"), "wb") as f:
            pickle.dump(self.id_map, f)

    # =========================
    # Load
    # =========================
    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))

        with open(os.path.join(path, "id_map.pkl"), "rb") as f:
            self.id_map = pickle.load(f)

        # restore params
        if isinstance(self.index, faiss.IndexHNSW):
            self.index.hnsw.efSearch = self.ef_search
