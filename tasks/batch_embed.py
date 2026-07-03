"""
Stage 4 Celery task: Batch Embedding → raw-SQL DB upsert.

Trigger sources (two paths, same handler):
  A. Proactive  – enrich_item calls .apply_async() when queue ≥ batch_embed_size
  B. Periodic   – Celery beat fires every `batch_embed_interval` seconds
                  as a safety-net for quiet periods

A distributed Redis lock prevents two workers from processing the same items
concurrently.  Lock TTL = 120s; well above the expected ~5s batch runtime.

The embedding column is @Transient in Hibernate, so we MUST use a native SQL
update — never rely on JPA/Hibernate to write this field.
"""

import asyncio
import logging

from celery_app import celery_app
from config import settings
from db import fetch_items_for_embedding, update_embedding
from services.embeddings import create_embeddings_batch

import redis as _redis_sync

logger = logging.getLogger(__name__)

EMBED_QUEUE_KEY  = "embedding:pending"
EMBED_LOCK_KEY   = "embedding:lock"
LOCK_TTL_SECONDS = 120


def _redis():
    return _redis_sync.from_url(settings.redis_url, decode_responses=True)


# ─── Celery task ─────────────────────────────────────────────────────────────

@celery_app.task(
    name="tasks.batch_embed.process_embedding_batch",
    max_retries=3,
    acks_late=True,
    soft_time_limit=90,
    time_limit=120,
)
def process_embedding_batch() -> dict:
    """
    1. Acquire distributed lock
    2. Atomically pop up to batch_embed_size IDs from Redis list
    3. Fetch their text from PostgreSQL
    4. Single batch API call → N embeddings
    5. For each: raw SQL  UPDATE media_items SET embedding = $1::vector WHERE id = $2::uuid
    """
    r = _redis()

    acquired = r.set(EMBED_LOCK_KEY, "1", nx=True, ex=LOCK_TTL_SECONDS)
    if not acquired:
        logger.info("Embedding batch skipped — another worker holds the lock")
        return {"status": "skipped", "reason": "lock_held"}

    try:
        return asyncio.run(_batch_async(r))
    except Exception as exc:
        logger.error("Embedding batch failed: %s", exc, exc_info=True)
        return {"status": "error", "message": str(exc)}
    finally:
        r.delete(EMBED_LOCK_KEY)


# ─── Async implementation ─────────────────────────────────────────────────────

async def _batch_async(r: "_redis_sync.Redis") -> dict:  # type: ignore[name-defined]
    queue_depth = r.llen(EMBED_QUEUE_KEY)
    if queue_depth == 0:
        logger.debug("Embedding queue is empty")
        return {"status": "empty", "processed": 0}

    batch_size = min(queue_depth, settings.batch_embed_size)
    logger.info("Processing embedding batch: %d items (queue depth: %d)", batch_size, queue_depth)

    # Atomic pop: read slice then trim — both in one pipeline round-trip
    pipe = r.pipeline()
    pipe.lrange(EMBED_QUEUE_KEY, 0, batch_size - 1)
    pipe.ltrim(EMBED_QUEUE_KEY, batch_size, -1)
    item_ids, _ = pipe.execute()

    if not item_ids:
        return {"status": "empty", "processed": 0}

    # Fetch text from DB (only items that were fully enriched)
    items = await fetch_items_for_embedding(item_ids)
    if not items:
        logger.warning("No enriched items found for embedding batch (IDs may not exist yet)")
        # Re-push IDs so they're retried on the next beat tick
        if item_ids:
            r.rpush(EMBED_QUEUE_KEY, *item_ids)
        return {"status": "no_items", "processed": 0}

    # Single API call for the entire batch
    pairs = await create_embeddings_batch(items)

    # Persist each vector via native SQL (Hibernate @Transient constraint)
    success, failed = 0, 0
    for item_id, vector in pairs:
        try:
            await update_embedding(item_id, vector)
            success += 1
        except Exception as exc:
            logger.error("Failed to write embedding for %s: %s", item_id, exc)
            failed += 1

    logger.info(
        "Embedding batch complete: %d/%d written, %d failed",
        success, len(pairs), failed,
    )
    return {
        "status":    "ok",
        "processed": success,
        "failed":    failed,
        "total":     len(item_ids),
    }
