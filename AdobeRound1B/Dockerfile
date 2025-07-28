# Adobe Hackathon Round 1B - Intelligent Document Analyzer
# Optimized Docker container for CPU-only operation

FROM --platform=linux/amd64 python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create input/output directories
RUN mkdir -p /app/input /app/output /app/cache

# Copy application code
COPY intelligent_document_analyzer.py .
COPY cache/ ./cache/

# Set environment variables for optimal performance
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/app/cache

# Expose volumes for input/output
VOLUME ["/app/input", "/app/output"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the intelligent document analyzer
CMD ["python", "intelligent_document_analyzer.py"]
