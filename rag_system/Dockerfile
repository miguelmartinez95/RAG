FROM python:3.11-slim

# ---- Environment ----
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TRANSFORMERS_CACHE=/models/hf_cache \
    HF_HOME=/models/hf_cache \
    MODEL_DIR=/models \
    MLFLOW_TRACKING_URI=http://mlflow-server:5000

WORKDIR /app

# ---- Create directories for models and vector DB ----
RUN mkdir -p /models/hf_cache /models/generation /models/evaluator /models/reranker /models/embeddings /vector_db

# ---- System dependencies ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy requirements and install ----
COPY requirements/prod.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Copy project files ----
COPY . /app

RUN mkdir -p /vector_db /rag/data/documents


# ---- Copy entrypoint script ----
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ---- Expose ports ----
EXPOSE 8080
EXPOSE 5000

# ---- Set entrypoint ----
ENTRYPOINT ["/app/entrypoint.sh"]
