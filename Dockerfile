# =============================================================================
# News Topic Classification — Docker Configuration
# =============================================================================
# Multi-stage build for production API deployment
#
# Build:
#   docker build -t news-classifier .
#
# Run:
#   docker run -p 8000:8000 -v ./saved_models:/app/saved_models news-classifier
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM python:3.12.3-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data'); nltk.download('wordnet', download_dir='/usr/local/nltk_data'); nltk.download('punkt_tab', download_dir='/usr/local/nltk_data')"

# ---------------------------------------------------------------------------
# Stage 2: Production
# ---------------------------------------------------------------------------
FROM python:3.12.3-slim AS production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/local/nltk_data /usr/local/nltk_data

# Ensure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV NLTK_DATA=/usr/local/nltk_data

# Copy application code
COPY configs/ configs/
COPY src/ src/
COPY api/ api/
COPY main.py .

# Create directories
RUN mkdir -p saved_models results logs data

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUTF8=1
ENV PYTHONHASHSEED=0

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
