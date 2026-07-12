"""
Stage 4 – Batch Embedding Generation.

Key design: NEVER call the embedding API one item at a time.
The batch_embed task accumulates 100+ items in Redis, then issues a single
API request here.

build_embedding_text now includes format_type, language, audience interests,
and geographic coverage to improve semantic matching quality.
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
_MAX_CHARS_PER_ITEM = 8000


# ─── Text preparation ─────────────────────────────────────────────────────────

def build_embedding_text(item: dict) -> str:
    """Build a semantic string for embedding from available item fields."""
    parts = [item.get("title") or ""]

    if item.get("description"):
        parts.append(item["description"])

    if item.get("category"):
        parts.append(f"Category: {item['category']}")

    if item.get("format_type"):
        parts.append(f"Format: {item['format_type']}")

    if item.get("language"):
        parts.append(f"Language: {item['language']}")

    if item.get("tags"):
        raw = item["tags"]
        tag_list = list(raw) if isinstance(raw, (list, tuple)) else []
        if tag_list:
            parts.append(f"Tags: {', '.join(tag_list)}")

    # Audience interests — critical for matching queries like "targeting AI engineers"
    audience = item.get("audience") or {}
    interests = audience.get("interests")
    if isinstance(interests, list) and interests:
        parts.append(f"Audience interests: {', '.join(str(i) for i in interests)}")

    # Geographic coverage — critical for city/region matching
    metrics = item.get("metrics") or {}
    geo_coverage = metrics.get("geographic_coverage")
    if isinstance(geo_coverage, list) and geo_coverage:
        parts.append(f"Geographic coverage: {', '.join(str(g) for g in geo_coverage)}")

    # Content topics — specific editorial focus areas (populated after re-enrichment)
    content_topics = metrics.get("content_topics")
    if isinstance(content_topics, list) and content_topics:
        parts.append(f"Content topics: {', '.join(str(t) for t in content_topics)}")

    # Intentionally omit cost_usd, DR, DA — those are filterable numbers,
    # not semantic signals. Embedding captures WHAT the outlet is, not how much.
    return ". ".join(filter(None, parts))[:_MAX_CHARS_PER_ITEM]


# ─── Mock embeddings ──────────────────────────────────────────────────────────

def _mock_vector(text: str) -> list[float]:
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
        ordered = sorted(resp.data, key=lambda x: x.index)
        return [item.embedding for item in ordered]
    except RateLimitError as e:
        raise _RateLimitError(str(e)) from e


# ─── Public API ───────────────────────────────────────────────────────────────

async def create_embeddings_batch(
    items: list[dict],
) -> list[tuple[str, list[float]]]:
    """
    Args:
        items: list of dicts with keys: id, title, description, category,
               tags, format_type, language  (new fields included)
    Returns:
        List of (item_id, embedding_vector) pairs.
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
