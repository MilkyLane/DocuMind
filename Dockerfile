# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — dependency builder
# Install Python deps into a separate layer so they are cached independently
# of the application code.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps required to compile some Python packages (e.g. asyncpg, Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# OCR + PDF runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source
COPY src/       ./src/
COPY api/       ./api/
COPY models/    ./models/
COPY frontend/  ./frontend/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Use shell form so $PORT is expanded at runtime (Railway injects PORT env var)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
