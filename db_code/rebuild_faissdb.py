"""
Rebuild FAISS database
- Normalize toàn bộ embedding
- Build lại index sạch
- Chỉ cần chạy 1 lần, KHÔNG chạy trong main
"""

import os
import pickle
import numpy as np

from db_code.faissdb import FaceFaissDB


# =========================
# CONFIG
# =========================
OLD_DATA_PATH = "data/id_map.pkl"   # file chứa embedding gốc
FAISS_DB_PATH = "faiss_db"              # thư mục output
DIM = 512


# =========================
# LOAD OLD DATA
# =========================
def load_old_embeddings(path):
    """
    Expect format:
    [
        (user_id, embedding_np),
        (user_id, embedding_np),
        ...
    ]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"[INFO] Loaded {len(data)} embeddings")
    return data


# =========================
# REBUILD FAISS
# =========================
def rebuild_faiss():
    data = load_old_embeddings(OLD_DATA_PATH)

    db = FaceFaissDB(
        dim=DIM,
        index_type="hnsw",      # hoặc "flat"
        ef_search=64,
        ef_construction=200
    )

    count = 0
    for user_id, emb in data:
        emb = np.asarray(emb, dtype=np.float32)

        if emb.shape[0] != DIM:
            print(f"[WARN] Skip user {user_id}, invalid shape {emb.shape}")
            continue

        db.add_user(user_id, emb)
        count += 1

    print(f"[INFO] Added {count} vectors to FAISS")

    # save
    db.save(FAISS_DB_PATH)
    print(f"[DONE] FAISS DB rebuilt and saved to '{FAISS_DB_PATH}'")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    rebuild_faiss()
