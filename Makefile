.PHONY: install worker-enrich worker-embed beat api producer flower dry-run

install:
	pip install -r requirements.txt

# ── Local dev (without Docker) ────────────────────────────────────────────────

worker-enrich:
	celery -A celery_app worker --queues=enrichment --concurrency=10 --loglevel=info

worker-embed:
	celery -A celery_app worker --queues=embeddings --concurrency=2 --loglevel=info

beat:
	celery -A celery_app beat --loglevel=info

api:
	uvicorn api:app --reload --port 8001

flower:
	celery -A celery_app flower

# ── Producer shortcuts ────────────────────────────────────────────────────────

dry-run:
	python producer.py --dry-run

producer:
	python producer.py

enrich-100:
	python producer.py --limit 100

enrich-1000:
	python producer.py --limit 1000

# ── Docker shortcuts ──────────────────────────────────────────────────────────

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f worker-enrich worker-embed beat

scale-workers:
	docker compose up --scale worker-enrich=4 -d
