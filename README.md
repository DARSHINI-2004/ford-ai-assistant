# Automotive AI Assistant – Ford Vehicle Intelligence System

## Overview
This project is a mini AI-powered automotive knowledge assistant built with **FastAPI**.  
It answers queries about Ford vehicles, including:
- Vehicle models & specifications
- Features & safety systems
- Service schedules & maintenance
- Owner manual snippets
- Common troubleshooting issues

It demonstrates **semantic search**, **RAG (Retrieval-Augmented Generation)**, and **recommendation logic**.

---

##  Features
- **/search** → Semantic search using FAISS + sentence-transformers
- **/ask** → RAG assistant with context injection and hallucination mitigation
- **/recommend** → Simple vehicle recommendation logic based on attributes

---

## Project Structure

rag/

├── api/

│   └── routes.py          # FastAPI routes

├── embeddings/

│   └── embedder.py        # Embedding pipeline

├── recommend/

│   └── recommender.py     # Recommendation logic

├── data/

│   └── ford_dataset.json  # Synthetic dataset

├── main.py                # FastAPI entrypoint

├── requirements.txt       # Dependencies

├── Dockerfile             # Containerization

└── README.md              # Documentation

# ford-ai-assistant

graph TD
    User[User / Client] -->|HTTP Request| API[FastAPI /api/routes.py]

    subgraph "Core Services"
        API -->|Query| RAG[RAG Logic /rag/rag.py]
        API -->|Attributes| REC[Recommender /recommend/recommender.py]
        API -->|Search Term| SEARCH[Semantic Search]
    end

    subgraph "Vector Engine"
        RAG -->|Context Retrieval| FAISS[FAISS Vector Store]
        SEARCH -->|Vector Matching| FAISS
        EMB[Sentence-Transformers] -->|Generates Embeddings| FAISS
    end

    subgraph "Data Layer"
        FAISS -.->|Metadata Lookup| META[meta.json]
        DATA[ford_vehicles.json] -->|Chunking| EMB
    end

    RAG -->|Augmented Prompt| LLM[LLM / Hallucination Mitigation]
    LLM -->|Final Answer| API
    REC -->|Recommendations| API
    API -->|JSON Response| User
