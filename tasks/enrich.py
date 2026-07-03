"""
Stage 1-3 Celery task: Search → Scrape → LLM → DB upsert → queue for embedding.

One Celery task = one media_item.  Tasks fan out from producer.py
and land in the `enrichment` queue consumed by -c 10 workers.

Retry policy:
  - 3 attempts with exponential back-off (10s, 20s, 40s)
  - acks_late + reject_on_worker_lost: safe re-queue if worker dies
"""

import asyncio
import logging

from celery import Task

from celery_app import celery_app
from config import settings
from db import update_media_item
from services.discovery import discover
from services.scraper import scrape
from services.llm import structure_media_item

import redis as _redis_sync

logger = logging.getLogger(__name__)

# Redis key for the pending-embedding queue (FIFO list)
EMBED_QUEUE_KEY = "embedding:pending"


def _redis():
    return _redis_sync.from_url(settings.redis_url, decode_responses=True)


# ─── Main task ────────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="tasks.enrich.enrich_item",
    max_retries=settings.max_retries,
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=120,  # seconds – raises SoftTimeLimitExceeded
    time_limit=150,
)
def enrich_item(self: Task, item_id: str, title: str) -> dict:
    """
    Full per-item enrichment pipeline.

    Parameters
    ----------
    item_id : UUID string of the media_items row
    title   : outlet name (the only field we start with)
    """
    try:
        return asyncio.run(_pipeline(item_id, title))
    except Exception as exc:
        countdown = 10 * (2 ** self.request.retries)
        logger.warning(
            "enrich_item failed [attempt %d] for '%s': %s — retry in %ds",
            self.request.retries + 1, title, exc, countdown,
        )
        raise self.retry(exc=exc, countdown=countdown)


# ─── Async pipeline ───────────────────────────────────────────────────────────

async def _pipeline(item_id: str, title: str) -> dict:
    # ── Step 1: multi-source discovery (Wikidata / Wikipedia / slug probe) ───
    discovery = await discover(title)
    logger.info("[%s] domain=%s source=%s", title, discovery.url or "not found", discovery.source)

    # ── Step 2: scrape / extract context ────────────────────────────────────
    context_text = await scrape(
        name=title,
        url=discovery.url,
        wikipedia_extract=discovery.wikipedia_extract,
        source=discovery.source,
    )
    logger.debug("[%s] context_text length=%d", title, len(context_text))

    # ── Step 3: LLM structuring ──────────────────────────────────────────────
    structured = await structure_media_item(title, context_text)
    logger.info("[%s] category=%s tier=%s", title,
                structured["category"], structured["metrics"]["reach_tier"])

    # ── Step 4: persist standard fields ─────────────────────────────────────
    await update_media_item(
        item_id=item_id,
        description=structured["description"],
        category=structured["category"],
        tags=structured.get("tags", []),
        audience=structured.get("audience", {}),
        metrics=structured.get("metrics", {"reach_tier": "unknown", "data_source": "llm_estimate"}),
    )
    logger.info("[%s] saved to DB", title)

    # ── Step 5: push to embedding queue; trigger batch if threshold reached ──
    _enqueue_for_embedding(item_id)

    return {
        "item_id":  item_id,
        "title":    title,
        "category": structured["category"],
        "tier":     structured["metrics"]["reach_tier"],
        "status":   "enriched",
    }


def _enqueue_for_embedding(item_id: str) -> None:
    r = _redis()
    r.rpush(EMBED_QUEUE_KEY, item_id)
    depth = r.llen(EMBED_QUEUE_KEY)
    logger.debug("Embedding queue depth: %d", depth)

    # Proactively fire a batch task if threshold is met —
    # Celery beat is only a fallback for quiet periods.
    if depth >= settings.batch_embed_size:
        from tasks.batch_embed import process_embedding_batch
        process_embedding_batch.apply_async(queue="embeddings")
        logger.info("Triggered embedding batch (queue depth=%d)", depth)
