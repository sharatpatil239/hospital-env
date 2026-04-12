# ──────────────────────────────────────────────────────────────────────────
# Hospital Decision-Making OpenEnv
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Hospital OpenEnv"
LABEL org.opencontainers.image.description="OpenEnv benchmark: hospital triage and resource allocation"
LABEL org.opencontainers.image.version="1.1.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir openenv

# Copy all project files
COPY . .

# Fix permissions
RUN chown -R appuser:appuser /app

USER appuser

# HF uses port 7860
EXPOSE 7860

# Health check (important for HF)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Debug log on start
ENV PYTHONUNBUFFERED=1

# Start server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]