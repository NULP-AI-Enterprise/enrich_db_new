"""
Stage 3 – LLM Structuring.

Generates description, category, tags, audience, and reduced metrics for a
media outlet.  Pricing/reach metrics are NOT generated here — they come from
real PRNEW CSV data stored in structured columns (cost_usd, similarweb_visits,
ahrefs_dr, etc.).  LLM is only asked for things it can infer from editorial
context: geographic coverage, publishing frequency, ad formats, social presence.
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
You are a senior media analyst with global expertise across all markets and regions.

Given information about a news/media outlet, return ONLY valid JSON — no markdown,
no commentary, no code fences — that conforms EXACTLY to this schema:

{
  "description":  "<3-4 sentences in English. State: what the outlet covers, who founded/runs it if known, its geographic focus, and its editorial angle>",
  "category":     "<exactly one of: News|Business|Technology|Sports|Fashion|Agriculture|Video|Entertainment|Science|Health|Politics>",
  "tags":         ["<6-8 precise English keyword tags: topic beats, geography, format>"],
  "audience": {
    "age_range":  "<realistic age range, e.g. 18-35 or 30-55>",
    "interests":  ["<3-5 specific audience interests>"],
    "demographics": {
      "primary_language": "<ISO-639-1, e.g. uk / en / ru>",
      "geo":              "<specific city, region or country — be as specific as the context allows>",
      "gender_split":     "<realistic estimate, e.g. 60M/40F>"
    }
  },
  "metrics": {
    "geographic_coverage":   ["<1-5 specific city or oblast names this outlet meaningfully covers>"],
    "publishing_frequency":  "<daily|weekly|monthly|breaking_news>",
    "ad_formats_available":  ["<from: Banner, Native, Pre-roll, Sponsored Post, Newsletter, Social, Podcast>"],
    "social_media_presence": "<strong=active large following | moderate=regular posts | weak=minimal | none>"
  }
}

Rules:
- Base ALL facts on the provided context. Do NOT invent founding years, owners, or URLs.
- If the outlet is clearly regional, geographic_coverage must list specific place names.
- tags must be specific (e.g. "Kharkiv", "investigative journalism") not generic ("news", "media").
- ad_formats_available: infer from outlet type — news sites support Banner+Native+Sponsored Post;
  video platforms support Pre-roll; newsletters support Newsletter.
- description must reflect the actual editorial focus visible in the context.
- DO NOT include pricing_tier or reach_tier — those come from real traffic data, not estimates."""

_VALID_CATEGORIES = {
    "News", "Business", "Technology", "Sports", "Fashion",
    "Agriculture", "Video", "Entertainment", "Science", "Health", "Politics",
}
_VALID_FREQUENCIES = {"daily", "weekly", "monthly", "breaking_news"}
_VALID_SOCIAL = {"strong", "moderate", "weak", "none"}


# ─── Output validation ────────────────────────────────────────────────────────

def _validate(data: dict, title: str) -> dict:
    if data.get("category") not in _VALID_CATEGORIES:
        data["category"] = "News"

    metrics = data.setdefault("metrics", {})
    if metrics.get("publishing_frequency") not in _VALID_FREQUENCIES:
        metrics["publishing_frequency"] = "daily"
    if metrics.get("social_media_presence") not in _VALID_SOCIAL:
        metrics["social_media_presence"] = "moderate"
    if not isinstance(metrics.get("geographic_coverage"), list):
        metrics["geographic_coverage"] = []
    if not isinstance(metrics.get("ad_formats_available"), list):
        metrics["ad_formats_available"] = ["Banner", "Native"]

    # Remove any stale pricing/reach fields the LLM might still emit
    metrics.pop("reach_tier", None)
    metrics.pop("pricing_tier", None)
    metrics.pop("data_source", None)

    tags = data.get("tags") or []
    data["tags"] = [str(t) for t in tags[:50]]

    data.setdefault("audience", {
        "age_range": "25-44",
        "interests": [],
        "demographics": {"primary_language": "unknown", "geo": "unknown", "gender_split": "50M/50F"},
    })
    return data


# ─── Mock path ────────────────────────────────────────────────────────────────

_MOCK_CATEGORIES = list(_VALID_CATEGORIES)


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
        "tags": ["news", "media", "journalism", "online", first_word, "ukraine"],
        "audience": {
            "age_range": "25-44",
            "interests": ["news", "current events", "society"],
            "demographics": {
                "primary_language": "en",
                "geo": "unknown",
                "gender_split": "50M/50F",
            },
        },
        "metrics": {
            "geographic_coverage": [],
            "publishing_frequency": "daily",
            "ad_formats_available": ["Banner", "Native", "Sponsored Post"],
            "social_media_presence": "moderate",
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
            max_tokens=900,
        )
        return json.loads(resp.choices[0].message.content)
    except RateLimitError as e:
        raise _RateLimitError(str(e)) from e
    except APIStatusError as e:
        if e.status_code >= 500:
            raise _RateLimitError(str(e)) from e
        raise
    finally:
        await client.aclose()


# ─── Public API ───────────────────────────────────────────────────────────────

async def structure_media_item(title: str, context_text: str) -> dict:
    if settings.use_mock_llm:
        await asyncio.sleep(0.02)
        return _validate(_mock(title), title)

    await asyncio.sleep(settings.llm_rate_limit_delay)
    result = await _call_openai(context_text)
    return _validate(result, title)
