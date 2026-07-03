"""
Stage 4 – Batch Embedding Generation.

Key design: NEVER call the embedding API one item at a time.
The batch_embed task accumulates 100+ items in Redis, then issues a single
API request here.  OpenAI's /embeddings endpoint accepts up to 2048 inputs
per call, so even 200 k items require only ~2000 API calls total.

The mock path generates unit-normalised pseudo-random 1536-d vectors seeded
by content hash — stable across restarts, no API key needed.
"""

import hashlib
import logging
import math
import random

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config import settings

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 1536
_MAX_CHARS_PER_ITEM = 8000  # ~6k tokens; stays under model limits


# ─── Text preparation ─────────────────────────────────────────────────────────

def build_embedding_text(item: dict) -> str:
    """Concatenate available fields into a single embedding-friendly string."""
    parts = [item.get("title") or ""]
    if item.get("description"):
        parts.append(item["description"])
    if item.get("category"):
        parts.append(f"Category: {item['category']}")
    if item.get("tags"):
        raw = item["tags"]
        tag_list = list(raw) if isinstance(raw, (list, tuple)) else []
        if tag_list:
            parts.append(f"Tags: {', '.join(tag_list)}")
    return ". ".join(filter(None, parts))[:_MAX_CHARS_PER_ITEM]


# ─── Mock embeddings ──────────────────────────────────────────────────────────

def _mock_vector(text: str) -> list[float]:
    """Deterministic unit-normalised 1536-d vector seeded by text hash."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    vec = [rng.gauss(0.0, 1.0) for _ in range(_EMBEDDING_DIM)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


# ─── Real OpenAI path ─────────────────────────────────────────────────────────

class _RateLimitError(Exception):
    pass


@retry(
    stop=stop_after_attempt(settings.max_retries),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    retry=retry_if_exception_type(_RateLimitError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _call_openai_batch(texts: list[str]) -> list[list[float]]:
    try:
        from openai import AsyncOpenAI, RateLimitError
    except ImportError:
        raise RuntimeError("openai package not installed")

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    try:
        resp = await client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
            encoding_format="float",
        )
        # API guarantees same order as input
        ordered = sorted(resp.data, key=lambda x: x.index)
        return [item.embedding for item in ordered]
    except RateLimitError as e:
        raise _RateLimitError(str(e)) from e


# ─── Public API ───────────────────────────────────────────────────────────────

async def create_embeddings_batch(
    items: list[dict],
) -> list[tuple[str, list[float]]]:
    """
    Single entry point for the batch embedding worker.

    Args:
        items: list of dicts with keys: id, title, description, category, tags

    Returns:
        List of (item_id, embedding_vector) pairs in the same order as input.
    """
    if not items:
        return []

    texts = [build_embedding_text(item) for item in items]
    ids   = [item["id"] for item in items]

    logger.info("Embedding batch: %d items, mock=%s", len(items), settings.use_mock_llm)

    if settings.use_mock_llm:
        import asyncio
        await asyncio.sleep(0.05)
        vectors = [_mock_vector(t) for t in texts]
    else:
        vectors = await _call_openai_batch(texts)

    return list(zip(ids, vectors))
