# Media Enrichment Pipeline

Async, distributed pipeline to enrich 50 k – 200 k media outlet records from
a bare `title` to a fully-structured `media_items` row including a pgvector
semantic embedding.

---

## Architecture

```
producer.py  (or POST /enrich/start)
     │
     │  enrich_item task × N
     ▼
┌─────────────────────────────────────┐   queue: enrichment
│  Worker Step 1 – Domain Discovery   │   services/search.py
│  • Slug inference (DNS HEAD check)  │
│  • Falls back to name-only if 404   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Worker Step 2 – Scraping           │   services/scraper.py
│  • httpx async + BeautifulSoup      │
│  • Extracts og:description, meta,   │
│    body excerpt (≤ 3 000 chars)     │
│  • Tenacity retry (3× exp. backoff) │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Worker Step 3 – LLM Structuring    │   services/llm.py
│  • gpt-4o-mini, JSON mode           │
│  • Outputs: description, category,  │
│    tags, audience, metrics          │
│  • Validates & coerces output       │
└──────────────┬──────────────────────┘
               │  DB upsert (standard fields)
               │  + RPUSH embedding:pending
               ▼
          PostgreSQL
               │
               │  when queue ≥ batch_embed_size
               │  OR Celery beat fires (every 30 s)
               ▼
┌─────────────────────────────────────┐   queue: embeddings
│  Worker Step 4 – Batch Embedding    │   tasks/batch_embed.py
│  • Pop 100 IDs from Redis list      │
│  • Fetch text from DB               │
│  • Single OpenAI /embeddings call   │
│  • Raw SQL: SET embedding=$1::vector│
│    (Hibernate @Transient constraint)│
└─────────────────────────────────────┘
```

---

## Quick start

```bash
cp .env.example .env
# edit .env — set DATABASE_URL, REDIS_URL, OPENAI_API_KEY
pip install -r requirements.txt

# Terminal 1 – enrichment workers
make worker-enrich

# Terminal 2 – embedding batch worker
make worker-embed

# Terminal 3 – Celery beat (periodic flush)
make beat

# Terminal 4 – optional FastAPI control plane
make api

# Enqueue everything
make producer

# Or cap at 500 for a test run
make enrich-100
```

### Docker (recommended for production)

```bash
cp .env.example .env
docker compose up --build -d

# Scale enrichment workers
docker compose up --scale worker-enrich=4 -d

# Flower dashboard → http://localhost:5555
# API             → http://localhost:8001/docs
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | localhost/advertising_db | asyncpg DSN |
| `REDIS_URL` | redis://localhost:6379/0 | broker + queue |
| `OPENAI_API_KEY` | _(empty)_ | required when mocks off |
| `USE_MOCK_LLM` | `true` | `false` → real OpenAI calls |
| `USE_MOCK_SEARCH` | `true` | `false` → real DNS probes |
| `BATCH_EMBED_SIZE` | `100` | items per embedding API call |
| `BATCH_EMBED_INTERVAL` | `30` | beat fallback interval (s) |
| `SCRAPE_TIMEOUT` | `5.0` | httpx timeout per URL (s) |
| `MAX_RETRIES` | `3` | Tenacity max attempts |
| `LLM_RATE_LIMIT_DELAY` | `0.5` | courtesy sleep between LLM calls |

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/enrich/start?limit=N` | Queue all unenriched items |
| `GET`  | `/enrich/status` | DB counts + Redis queue depth |
| `POST` | `/enrich/embed-batch` | Manually trigger embedding flush |
| `POST` | `/enrich/item/{id}?title=...` | Re-enrich a single item |
| `GET`  | `/health` | Liveness probe |

---

## Cost estimate (real API, 200 k items)

| Stage | Model | Cost |
|---|---|---|
| LLM structuring | gpt-4o-mini (~500 tok/item) | ~$15 |
| Embeddings | text-embedding-3-small | ~$6 |
| **Total** | | **~$21** |

With 20 `worker-enrich` containers the full run completes in **~3-4 hours**.

---

## Module layout

```
enrichment/
├── config.py               # Pydantic Settings (all env vars)
├── celery_app.py           # Celery factory + beat schedule
├── db.py                   # asyncpg helpers (read + write + raw vector SQL)
├── producer.py             # CLI: query DB → dispatch tasks
├── api.py                  # FastAPI control plane
├── services/
│   ├── search.py           # Stage 1: domain discovery
│   ├── scraper.py          # Stage 2: async scraping
│   ├── llm.py              # Stage 3: LLM structuring
│   └── embeddings.py       # Stage 4: batch embedding
├── tasks/
│   ├── enrich.py           # Celery task: stages 1-3
│   └── batch_embed.py      # Celery task: stage 4
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```
