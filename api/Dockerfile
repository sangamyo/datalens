FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/hf-cache

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 5 -r requirements.txt

COPY alembic.ini start-api.sh ./
COPY alembic ./alembic
COPY app ./app
RUN chmod +x start-api.sh

EXPOSE 8000
# The API container applies migrations before serving; the worker overrides this command
# (and waits for the API to be healthy, so it never races the migration).
CMD ["./start-api.sh"]
