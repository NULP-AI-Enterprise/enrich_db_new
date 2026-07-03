"""
Database layer using asyncpg directly.

Design note:
  Celery tasks call asyncio.run() which spawns a fresh event loop per task.
  A module-level asyncpg pool would be bound to the loop that created it and
  break on subsequent loops.  We therefore create short-lived connections per
  operation.  The overhead (~5-15 ms) is negligible compared to scraping or
  LLM latency.

  For the FastAPI process a real pool is initialised in lifespan() in api.py.
"""

import json
import logging
from typing import Any

import asyncpg
from config import settings

logger = logging.getLogger(__name__)


# ─── Connection helper ────────────────────────────────────────────────────────

async def _connect() -> asyncpg.Connection:
    return await asyncpg.connect(settings.database_url)


# ─── Read ─────────────────────────────────────────────────────────────────────

async def count_unenriched() -> int:
    conn = await _connect()
    try:
        return await conn.fetchval(
            "SELECT COUNT(*) FROM media_items WHERE description IS NULL"
        )
    finally:
        await conn.close()


async def fetch_unenriched(limit: int = 500, offset: int = 0) -> list[dict]:
    conn = await _connect()
    try:
        rows = await conn.fetch(
            """
            SELECT id::text, title
            FROM media_items
            WHERE description IS NULL
            ORDER BY created_at ASC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
        return [{"id": r["id"], "title": r["title"]} for r in rows]
    finally:
        await conn.close()


async def fetch_items_for_embedding(item_ids: list[str]) -> list[dict]:
    """Return title/description/category/tags for building embedding text."""
    conn = await _connect()
    try:
        rows = await conn.fetch(
            """
            SELECT id::text, title, description, category, tags
            FROM media_items
            WHERE id = ANY($1::uuid[])
              AND description IS NOT NULL
            """,
            item_ids,
        )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


# ─── Write ────────────────────────────────────────────────────────────────────

async def update_media_item(
    item_id: str,
    description: str,
    category: str,
    tags: list[str],
    audience: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Update all standard (non-vector) fields via ORM-compatible asyncpg."""
    conn = await _connect()
    try:
        await conn.execute(
            """
            UPDATE media_items SET
                description = $1,
                category    = $2,
                tags        = $3,
                audience    = $4::jsonb,
                metrics     = $5::jsonb,
                updated_at  = NOW()
            WHERE id = $6::uuid
            """,
            description,
            category,
            tags,
            json.dumps(audience),
            json.dumps(metrics),
            item_id,
        )
    finally:
        await conn.close()


async def update_embedding(item_id: str, embedding: list[float]) -> None:
    """
    Raw native SQL update for the vector column.

    The embedding field is declared @Transient in the Spring/Hibernate entity
    (see db-schema.json notes) meaning Hibernate never touches it.  All writes
    must therefore go through a native query — exactly what we do here.

    pgvector expects the string literal '[0.123, 0.456, ...]'.
    """
    conn = await _connect()
    try:
        vec_literal = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
        await conn.execute(
            "UPDATE media_items SET embedding = $1::vector WHERE id = $2::uuid",
            vec_literal,
            item_id,
        )
    finally:
        await conn.close()
