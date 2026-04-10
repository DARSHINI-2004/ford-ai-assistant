# vectorstore/faiss_store.py
"""
FAISS vector store wrapper.

Explanations:
- FAISS is a library for efficient similarity search over vectors.
- We store normalized embeddings so that cosine similarity is equivalent to dot product.
- This module builds an index from text chunks and allows querying by embedding.
"""

import faiss
import numpy as np
import os
import json
from typing import List, Tuple

INDEX_DIR = os.getenv("INDEX_DIR", "vectorstore")
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_FILE = os.path.join(INDEX_DIR, "meta.json")


class FaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = None
        self.metadatas = []  # list of dicts for each vector
        os.makedirs(INDEX_DIR, exist_ok=True)

    def build_index(self, embeddings: np.ndarray, metadatas: List[dict]):
        """
        Build a flat (exact) index. For production, you might use IVF or HNSW.
        """
        assert embeddings.shape[1] == self.dim
        # Use IndexFlatIP because we store normalized vectors; inner product = cosine
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))
        self.metadatas = metadatas
        # Save index and metadata
        faiss.write_index(self.index, INDEX_FILE)
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)

    def load_index(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "r", encoding="utf-8") as f:
                self.metadatas = json.load(f)
            return True
        return False

    def search(self, query_emb: np.ndarray, top_k: int = 3) -> List[Tuple[dict, float]]:
        """
        Search the index with a normalized query embedding.
        Returns list of (metadata, score) sorted by score desc.
        Score is cosine similarity (since vectors are normalized).
        """
        if self.index is None:
            raise RuntimeError("Index not loaded")
        # query_emb shape: (1, dim)
        D, I = self.index.search(query_emb.astype(np.float32), top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            results.append((self.metadatas[idx], float(score)))
        return results
