# Automotive AI Assistant 

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
http://127.0.0.1:8000/docs
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
                +----------------------+
                |       User           |
                | (Query Input)        |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |    API Layer         |
                | (FastAPI)            |
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


---


