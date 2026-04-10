# api/routes.py
"""
FastAPI routes:
- GET /search?query= : semantic search over owner manual chunks (top 3)
- POST /ask : RAG endpoint that returns grounded answers using retrieved context
- GET /recommend?query= : rule-based vehicle recommendation
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List
import os
import json
import numpy as np

from embeddings.embedder import Embedder
from vectorstore.faiss_store import FaissStore
from rag.rag import generate_grounded_answer
from recommend.recommender import Recommender

router = APIRouter()

# Load metadata and index on startup
EMBED_DIM = 384  # all-MiniLM-L6-v2 embedding dimension
embedder = Embedder()
faiss_store = FaissStore(dim=EMBED_DIM)
if not faiss_store.load_index():
    # Index will be built by main startup script if missing
    pass

recommender = Recommender()


class AskRequest(BaseModel):
    query: str


@router.get("/search")
def search(query: str = Query(..., min_length=1)):
    """
    Semantic search endpoint.
    Returns top 3 relevant chunks and similarity scores.
    """
    # Compute embedding
    q_emb = embedder.embed(query)
    try:
        results = faiss_store.search(q_emb, top_k=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Format response
    response = []
    for meta, score in results:
        response.append({
            "model": meta.get("model"),
            "chunk_type": meta.get("chunk_type"),
            "text": meta.get("text"),
            "score": score
        })
    return {"query": query, "results": response}


@router.post("/ask")
def ask(req: AskRequest):
    """
    RAG endpoint:
    - Retrieve top-k chunks
    - Inject into prompt
    - Generate grounded answer
    Strict rule: If no relevant data found, return "Data not available"
    """
    query = req.query
    q_emb = embedder.embed(query)
    try:
        results = faiss_store.search(q_emb, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    contexts = [r[0]["text"] for r in results if r[1] > 0.05]  # threshold to filter very low similarity
    # If no contexts, return Data not available
    if not contexts:
        return {"answer": "Data not available", "source_chunks": []}

    answer = generate_grounded_answer(contexts, query)
    return {"answer": answer, "source_chunks": [{"model": r[0]["model"], "chunk_type": r[0]["chunk_type"], "text": r[0]["text"], "score": r[1]} for r in results]}


@router.get("/recommend")
def recommend(query: str = Query(..., min_length=1)):
    """
    Vehicle recommendation endpoint.
    Returns top 2 vehicles with explanation.
    """
    recs = recommender.recommend(query, top_n=2)
    return {"query": query, "recommendations": recs}
