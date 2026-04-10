# main.py
"""
Application entrypoint.

Responsibilities:
- Build FAISS index from data if not present.
- Start FastAPI app.
"""

import uvicorn
import json
import os
from fastapi import FastAPI
from api.routes import router
from embeddings.embedder import Embedder
from vectorstore.faiss_store import FaissStore
from typing import List

DATA_FILE = os.path.join("data", "ford_vehicles.json")
INDEX_DIR = os.getenv("INDEX_DIR", "vectorstore")
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_FILE = os.path.join(INDEX_DIR, "meta.json")

app = FastAPI(title="Automotive AI Assistant - Ford Vehicle Intelligence System")
app.include_router(router)

# Text chunking utility
def chunk_manuals(data: dict) -> List[dict]:
    """
    Convert owner manual snippets into small chunks for embeddings.
    Each chunk metadata includes: model, chunk_type, text
    """
    chunks = []
    for v in data.get("vehicles", []):
        model = v["model"]
        # Add structured fields as searchable text too
        structured_text = f"{model} | Engine: {v.get('engine')} | Transmission: {v.get('transmission')} | Fuel: {v.get('fuel_type')} | Seating: {v.get('seating_capacity')}"
        chunks.append({"model": model, "chunk_type": "specs", "text": structured_text})
        # Service data as chunks
        service = v.get("service_data", {})
        for k, val in service.items():
            chunks.append({"model": model, "chunk_type": f"service_{k}", "text": f"{model} {k.replace('_',' ')}: {val}"})
        # Owner manual snippets: split into short sentences (already short in dataset)
        snippets = v.get("owner_manual_snippets", {})
        for stype, sents in snippets.items():
            for s in sents:
                chunks.append({"model": model, "chunk_type": stype, "text": s})
    return chunks


def build_or_load_index():
    """
    Build FAISS index if not present. This runs at startup.
    """
    embedder = Embedder()
    # Determine embedding dimension by embedding a sample
    sample_emb = embedder.embed("sample")
    dim = sample_emb.shape[1]
    store = FaissStore(dim=dim)
    if store.load_index():
        print("Loaded existing FAISS index.")
        return store
    # Load data and chunk
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = chunk_manuals(data)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embs = embedder.embed(texts)
    # Build index
    store.build_index(embs, chunks)
    print("FAISS index built and saved.")
    return store


@app.on_event("startup")
def startup_event():
    # Build or load index and set the global faiss_store used by routes
    global router, embedder, faiss_store
    from api import routes as rmod
    # Build/load index
    store = build_or_load_index()
    # Assign to module-level objects used in routes
    rmod.faiss_store = store
    rmod.embedder = Embedder()
    print("Startup complete. API ready.")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
