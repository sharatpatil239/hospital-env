# ──────────────────────────────────────────────────────────────────────────
# Hospital Decision-Making OpenEnv
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Hospital OpenEnv"
LABEL org.opencontainers.image.description="OpenEnv benchmark: hospital triage and resource allocation"
LABEL org.opencontainers.image.version="1.1.0"


# Install only necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app


# Copy requirements first (better caching)
COPY requirements.txt .


# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# Copy project files
COPY . .


# Set permissions
RUN chown -R appuser:appuser /app

USER appuser


EXPOSE 7860


HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1


ENV PYTHONUNBUFFERED=1


CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]