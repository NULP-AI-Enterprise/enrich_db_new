"""
Database layer using asyncpg with a per-event-loop connection pool.

Pool strategy
─────────────
• FastAPI process: one event loop for the lifetime of the process → pool is
  created once on first use and reused for all requests.
• Celery tasks: each task calls asyncio.run(), which creates a fresh event loop.
  The pool keyed to the previous loop is gone, so a new pool (min=1, max=5) is
  created for that task's loop and released when asyncio.run() returns.

This avoids both the "pool bound to wrong loop" error AND the per-call
connection churn that caused connection exhaustion under concurrent Celery load.
"""

import asyncio
import contextlib
import json
import logging
from typing import Any

import asyncpg
from config import settings

logger = logging.getLogger(__name__)

# ─── Pool registry (loop-id → pool) ──────────────────────────────────────────

_pools: dict[int, asyncpg.Pool] = {}


async def _get_pool() -> asyncpg.Pool:
    """Return the pool for the running event loop, creating it if necessary."""
    loop_id = id(asyncio.get_running_loop())
    pool = _pools.get(loop_id)
    if pool is None or pool._closed:
        pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=1,
            max_size=5,          # cap concurrent DB connections per event loop
            command_timeout=30,
            statement_cache_size=0,   # required when behind pgBouncer
        )
        _pools[loop_id] = pool
        logger.debug("DB pool created for loop %d (max_size=5)", loop_id)
    return pool


@contextlib.asynccontextmanager
async def _conn():
    """Acquire a connection from the pool and release it on exit."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        yield conn


# ─── Read ─────────────────────────────────────────────────────────────────────

async def count_unenriched() -> int:
    async with _conn() as conn:
        return await conn.fetchval(
            "SELECT COUNT(*) FROM media_items WHERE description IS NULL"
        )


async def fetch_unenriched(limit: int = 500, offset: int = 0) -> list[dict]:
    async with _conn() as conn:
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


async def fetch_items_for_embedding(item_ids: list[str]) -> list[dict]:
    """Return fields needed to build embedding text, including structured columns."""
    async with _conn() as conn:
        rows = await conn.fetch(
            """
            SELECT id::text, title, description, category, tags,
                   format_type, language
            FROM media_items
            WHERE id = ANY($1::uuid[])
              AND description IS NOT NULL
            """,
            item_ids,
        )
        return [dict(r) for r in rows]


# ─── Write ────────────────────────────────────────────────────────────────────

async def update_media_item(
    item_id: str,
    description: str,
    category: str,
    tags: list[str],
    audience: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Update LLM-generated fields (non-vector). Structured CSV columns are untouched."""
    async with _conn() as conn:
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


async def update_embedding(item_id: str, embedding: list[float]) -> None:
    """
    Native SQL update for the vector column.
    The embedding field is @Transient in Hibernate — all writes must use native SQL.
    pgvector expects the string literal '[0.123, 0.456, ...]'.
    """
    async with _conn() as conn:
        vec_literal = "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
        await conn.execute(
            "UPDATE media_items SET embedding = $1::vector WHERE id = $2::uuid",
            vec_literal,
            item_id,
        )
