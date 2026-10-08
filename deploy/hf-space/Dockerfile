# All-in-one image for a Hugging Face Docker Space: Postgres+pgvector, Redis, SeaweedFS (S3),
# the FastAPI API, the arq worker and the built React UI, supervised in one container.
# Build context = a staging dir laid out by build.sh (api/, web-dist/, deploy files).
FROM pgvector/pgvector:pg16

ARG TARGETARCH=amd64
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:/usr/lib/postgresql/16/bin:$PATH \
    HOME=/home/user HF_HOME=/home/user/.cache/huggingface FASTEMBED_CACHE_PATH=/opt/fastembed

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv redis-server curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://github.com/seaweedfs/seaweedfs/releases/latest/download/linux_${TARGETARCH}.tar.gz" \
       | tar -xz -C /usr/local/bin weed

# HF Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user || true
WORKDIR /app

COPY api/requirements.txt /app/requirements.txt
RUN python3 -m venv /opt/venv && pip install --no-cache-dir -r requirements.txt supervisor

# Bake the embedding model into the image so the first semantic search is fast.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" \
    && chmod -R a+rX /opt/fastembed

COPY api/ /app/
COPY web-dist/ /app/web-dist/
COPY serve.py seed.py supervisord.conf entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh && rm -rf /home/user/.cache && chown -R 1000:1000 /app /home/user

USER 1000
EXPOSE 7860
CMD ["/app/entrypoint.sh"]
