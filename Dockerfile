FROM python:3.11-slim

# System deps for lxml / asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default entrypoint – overridden per-service in docker-compose.yml
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
