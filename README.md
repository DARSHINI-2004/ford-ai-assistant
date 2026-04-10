# Automotive AI Assistant – Ford Vehicle Intelligence System

## Overview
This project is a mini AI-powered automotive knowledge assistant built using FastAPI. It answers queries related to Ford vehicles, including specifications, features, service schedules, and common issues.

The system uses Retrieval-Augmented Generation (RAG) with semantic search to provide accurate and context-aware responses.

---

## Features
- `/search` → Semantic search using FAISS and sentence-transformers  
- `/ask` → RAG-based Q&A with context-aware responses  
- `/recommend` → Rule-based vehicle recommendation system  

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/DARSHINI-2004/ford-ai-assistant.git
cd ford-ai-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
uvicorn main:app --reload
```

### 5. Access API
Open in browser:
```
http://127.0.0.1:8001/docs
```

---

## Architecture Explanation
The system follows a modular pipeline:

1. User sends a query through API endpoints  
2. FastAPI handles request routing  
3. Query is processed using NLP preprocessing  
4. FAISS retrieves relevant data from the dataset  
5. Retrieved context is passed into the RAG pipeline  
6. Response is generated using rule-based or LLM logic  
7. Final response is returned to the user  

---

## Architecture Diagram
```
User → API → NLP Processing → FAISS Retrieval → RAG → Response
```

---

## Design Decisions
- FastAPI chosen for its speed and automatic API documentation  
- FAISS used for efficient similarity search  
- sentence-transformers used for semantic understanding  
- RAG approach reduces hallucination and improves accuracy  
- Modular structure ensures scalability  
- Synthetic dataset simulates real-world automotive data  

---

## Project Structure
```
rag/
├── api/
│   └── routes.py
├── embeddings/
│   └── embedder.py
├── recommend/
│   └── recommender.py
├── data/
│   └── ford_dataset.json
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Docker Support

### Build image
```bash
docker build -t ford-ai-assistant .
```

### Run container
```bash
docker run -p 8000:8000 ford-ai-assistant
```

---
