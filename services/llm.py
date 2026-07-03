"""
Stage 3 – LLM Structuring.

Sends scraped context to an LLM and enforces strict JSON output that matches
the media_items schema exactly.  The mock path is deterministic (seeded by
title hash) so tests are reproducible without any API key.

OpenAI `response_format={"type": "json_object"}` guarantees valid JSON output.
Tenacity retries on rate-limit (429) and server errors (5xx).
"""

import asyncio
import json
import logging
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

# ─── Schema contract ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior media analyst specialising in Ukrainian and Eastern European media.

Given information about a news/media outlet, return ONLY valid JSON — no markdown,
no commentary, no code fences — that conforms EXACTLY to this schema:

{
  "description":  "<3-4 sentences in English. State: what the outlet covers, who founded/runs it if known, its geographic focus, and its editorial angle>",
  "category":     "<exactly one of: News|Business|Technology|Sports|Fashion|Agriculture|Video|Entertainment|Science|Politics>",
  "tags":         ["<6-8 precise English keyword tags: topic beats, geography, format>"],
  "audience": {
    "age_range":  "<realistic age range, e.g. 18-35 or 30-55>",
    "interests":  ["<3-5 specific audience interests>"],
    "demographics": {
      "primary_language": "<ISO-639-1, e.g. uk / en / ru>",
      "geo":              "<specific city, region or country — NOT just 'Ukraine' if regional>",
      "gender_split":     "<realistic estimate, e.g. 60M/40F>"
    }
  },
  "metrics": {
    "reach_tier":  "<national=country-wide audience | regional=one oblast/city | local=neighbourhood/district | niche=specialist topic>",
    "data_source": "llm_estimate"
  }
}

Rules:
- Base ALL facts on the provided context. Do NOT invent founding years, owners, or URLs.
- If the outlet is clearly regional (covers one city/oblast), set reach_tier="regional" or "local".
- tags must be specific (e.g. "Kharkiv", "investigative journalism") not generic ("news", "media").
- description must reflect the actual editorial focus visible in the context."""

# Allowed values for validation
_VALID_CATEGORIES = {
    "News", "Business", "Technology", "Sports", "Fashion",
    "Agriculture", "Video", "Entertainment", "Science", "Politics",
}
_VALID_TIERS = {"national", "regional", "local", "niche"}


# ─── Output validation ────────────────────────────────────────────────────────

def _validate(data: dict, title: str) -> dict:
    """Clamp and coerce LLM output to exactly the required schema."""
    if data.get("category") not in _VALID_CATEGORIES:
        data["category"] = "News"
    if data.get("metrics", {}).get("reach_tier") not in _VALID_TIERS:
        data.setdefault("metrics", {})["reach_tier"] = "regional"
    data.setdefault("metrics", {})["data_source"] = "llm_estimate"

    tags = data.get("tags") or []
    data["tags"] = [str(t) for t in tags[:50]]

    data.setdefault("audience", {
        "age_range": "25-44",
        "interests": [],
        "demographics": {"primary_language": "uk", "geo": "Ukraine", "gender_split": "50M/50F"},
    })
    return data


# ─── Mock path ────────────────────────────────────────────────────────────────

_MOCK_CATEGORIES = list(_VALID_CATEGORIES)
_MOCK_TIERS = list(_VALID_TIERS)


def _mock(title: str) -> dict:
    seed = sum(ord(c) for c in title)
    rng = random.Random(seed)
    first_word = title.lower().split()[0] if title else "media"
    return {
        "description": (
            f"{title} is a digital media outlet delivering news and analysis to "
            f"a broad online audience. It covers current events, society, and culture."
        ),
        "category": rng.choice(_MOCK_CATEGORIES),
        "tags": ["news", "media", "journalism", "online", first_word],
        "audience": {
            "age_range": "25-44",
            "interests": ["news", "current events", "society"],
            "demographics": {
                "primary_language": "uk",
                "geo": "Ukraine",
                "gender_split": "50M/50F",
            },
        },
        "metrics": {
            "reach_tier": rng.choice(_MOCK_TIERS),
            "data_source": "llm_estimate",
        },
    }


# ─── Real OpenAI path ─────────────────────────────────────────────────────────

class _RateLimitError(Exception):
    pass


@retry(
    stop=stop_after_attempt(settings.max_retries),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(_RateLimitError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _call_openai(context_text: str) -> dict:
    try:
        from openai import AsyncOpenAI, RateLimitError, APIStatusError
    except ImportError:
        raise RuntimeError("openai package not installed")

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    try:
        resp = await client.chat.completions.create(
            model=settings.openai_chat_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": context_text},
            ],
            temperature=0.15,
            max_tokens=600,
        )
        return json.loads(resp.choices[0].message.content)
    except RateLimitError as e:
        raise _RateLimitError(str(e)) from e
    except APIStatusError as e:
        if e.status_code >= 500:
            raise _RateLimitError(str(e)) from e
        raise


# ─── Public API ───────────────────────────────────────────────────────────────

async def structure_media_item(title: str, context_text: str) -> dict:
    """
    Returns a validated dict matching the media_items schema.
    Gracefully falls back to mock on any LLM failure.
    """
    if settings.use_mock_llm:
        await asyncio.sleep(0.02)
        return _validate(_mock(title), title)

    await asyncio.sleep(settings.llm_rate_limit_delay)
    try:
        result = await _call_openai(context_text)
        return _validate(result, title)
    except Exception as exc:
        logger.error("LLM failed for '%s' (%s) — using mock fallback", title, exc)
        return _validate(_mock(title), title)
