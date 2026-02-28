# ── Base image ──────────────────────────────────────────────────
# Python 3.11 slim = small image, faster builds
FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────
# Only curl needed for Docker health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ──────────────────────────────────
# Copy requirements first so Docker caches this layer
# (only reinstalls if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project files ───────────────────────────────────────────
COPY . .

# ── Expose ports ─────────────────────────────────────────────────
# 8000 = FastAPI, 8501 = Streamlit
EXPOSE 8000 8501

# ── Default command ──────────────────────────────────────────────
# docker-compose overrides this per service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]