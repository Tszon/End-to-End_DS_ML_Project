# ---------- Stage 1: Builder ----------
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for scientific libs
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install deps into a target folder (not system-wide)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only the necessary project folders
COPY scripts/ scripts/
COPY src/ src/
COPY models/ models/
COPY reports_app/ reports_app/
COPY data/ data/
COPY config.json config.json

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "scripts/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
