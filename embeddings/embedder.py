# embeddings/embedder.py
"""
Embedding utilities.

Explanations (in comments):
- Embeddings are numeric vector representations of text that capture semantic meaning.
- We use sentence-transformers 'all-MiniLM-L6-v2' because:
  * It's lightweight and fast (good for a mini project / fresher assessment).
  * It produces high-quality sentence embeddings suitable for semantic search.
  * It balances performance and compute cost for CPU environments.
- We normalize embeddings before storing in FAISS so cosine similarity can be computed
  via inner product (dot product) on normalized vectors.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import os

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        # Load the sentence-transformers model
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        """
        Compute embeddings for a list of texts.
        Returns L2-normalized numpy array (shape: [n, dim]) for cosine similarity.
        """
        if isinstance(texts, str):
            texts = [texts]
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Normalize to unit vectors for cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        embs = embs / norms
        return embs
