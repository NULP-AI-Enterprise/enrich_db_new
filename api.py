"""
FastAPI – pipeline control plane.

Endpoints:
  POST /enrich/start          – kick off bulk enrichment (background)
  GET  /enrich/status         – live queue depths + DB counts
  POST /enrich/embed-batch    – manually fire a batch-embed job
  POST /enrich/item/{id}      – re-enrich a single item by ID
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from config import settings
from db import count_unenriched, fetch_unenriched
from tasks.enrich import EMBED_QUEUE_KEY, enrich_item

logger = logging.getLogger(__name__)


# ─── App lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
    logger.info("Redis connection established")
    yield
    await app.state.redis.aclose()
    logger.info("Redis connection closed")


app = FastAPI(
    title="Media Enrichment Pipeline",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Response models ──────────────────────────────────────────────────────────

class StartResponse(BaseModel):
    status: str
    queued_estimate: int
    total_unenriched: int


class StatusResponse(BaseModel):
    unenriched_in_db: int
    embedding_queue_depth: int
    pipeline_healthy: bool


class BatchResponse(BaseModel):
    task_id: str
    status: str


# ─── Background dispatch helper ───────────────────────────────────────────────

async def _dispatch_all(limit: int | None) -> int:
    offset, dispatched = 0, 0
    while True:
        remaining = (limit - dispatched) if limit else settings.producer_page_size
        page = await fetch_unenriched(
            limit=min(settings.producer_page_size, remaining),
            offset=offset,
        )
        if not page:
            break
        for item in page:
            countdown = dispatched // settings.task_stagger_per
            enrich_item.apply_async(
                args=[item["id"], item["title"]],
                queue="enrichment",
                countdown=countdown,
            )
            dispatched += 1
        offset += len(page)
        if limit and dispatched >= limit:
            break
    logger.info("Dispatched %d enrichment tasks", dispatched)
    return dispatched


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post("/enrich/start", response_model=StartResponse)
async def start_enrichment(
    background_tasks: BackgroundTasks,
    limit: Annotated[int | None, Query(description="Cap on number of items to enqueue")] = None,
):
    """
    Dispatch Celery tasks for every media_item where description IS NULL.
    Returns immediately; actual dispatch runs in a FastAPI background task.
    """
    total = await count_unenriched()
    queued_estimate = min(total, limit) if limit else total
    background_tasks.add_task(_dispatch_all, limit)
    return StartResponse(
        status="dispatching",
        queued_estimate=queued_estimate,
        total_unenriched=total,
    )


@app.get("/enrich/status", response_model=StatusResponse)
async def pipeline_status(request: Request):
    """Return live DB count and Redis queue depth."""
    r: aioredis.Redis = request.app.state.redis
    unenriched, embed_depth = await asyncio.gather(
        count_unenriched(),
        r.llen(EMBED_QUEUE_KEY),
    )
    return StatusResponse(
        unenriched_in_db=unenriched,
        embedding_queue_depth=embed_depth,
        pipeline_healthy=True,
    )


@app.post("/enrich/embed-batch", response_model=BatchResponse)
async def trigger_embed_batch():
    """Manually fire a batch-embedding job (useful after a large import)."""
    from tasks.batch_embed import process_embedding_batch
    task = process_embedding_batch.apply_async(queue="embeddings")
    return BatchResponse(task_id=task.id, status="queued")


@app.post("/enrich/item/{item_id}", response_model=dict)
async def reenrich_item(item_id: str, title: str = Query(...)):
    """Re-enrich a single item by its UUID (e.g. after a failed run)."""
    task = enrich_item.apply_async(
        args=[item_id, title],
        queue="enrichment",
    )
    return {"task_id": task.id, "item_id": item_id, "status": "queued"}


@app.get("/health")
async def health():
    return {"status": "ok"}
