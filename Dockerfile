# Dockerfile
FROM python:3.11-slim

# Install system deps for faiss and sentence-transformers
RUN apt-get update && apt-get install -y build-essential git wget libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . /app

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
