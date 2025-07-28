FROM python:3.9

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies (prefer binary wheels)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Pre-download Sentence Transformers model for offline use
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Set offline mode for HuggingFace libraries
ENV TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

# Copy application code
COPY intelligent_document_analyzer.py .

# Create directories for input/output
RUN mkdir -p /app/input /app/output /app/documents

# Set environment variables to ensure CPU-only execution
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=4

# Entry point
CMD ["python", "intelligent_document_analyzer.py"]
