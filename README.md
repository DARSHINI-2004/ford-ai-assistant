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

                +----------------------+
                |       User           |
                | (Query Input)        |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |    API Layer         |
                | (Flask / FastAPI)   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |  Query Processing    |
                | (NLP / Preprocess)   |
                +----------+-----------+
                           |
                           v
        +----------------------------------------+
        |  Retrieval / Knowledge Layer (RAG)     |
        |  - Vehicle Dataset                     |
        |  - Specs / Features / Manuals          |
        +----------+-----------------------------+
                   |
                   v
        +-----------------------------+
        |  Response Generation Layer  |
        | (Rule-based / LLM Logic)    |
        +-------------+---------------+
                      |
                      v
                +----------------------+
                |   Final Response     |
                | (Answer to User)     |
                +----------------------+
