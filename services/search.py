"""
Stage 1 – Domain Discovery.

Strategy (cheapest first, most expensive last):
  1. Slug inference  – generate URL candidates from the title and HEAD-check them
  2. Mock path       – deterministic fake URL (for dev / testing)

In production set USE_MOCK_SEARCH=false and optionally wire in a real
Google Custom Search API call as a third fallback.
"""

import asyncio
import logging
import re

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, reraise

from config import settings

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (compatible; MediaEnrichBot/1.0)"
_TLDS = [".com", ".net", ".org", ".ua", ".info"]

# ─── Transliteration (Ukrainian → Latin slug) ─────────────────────────────────

_TRANSLIT: dict[str, str] = {
    "а": "a",  "б": "b",  "в": "v",  "г": "h",  "д": "d",  "е": "e",
    "є": "ie", "ж": "zh", "з": "z",  "и": "y",  "і": "i",  "ї": "i",
    "й": "y",  "к": "k",  "л": "l",  "м": "m",  "н": "n",  "о": "o",
    "п": "p",  "р": "r",  "с": "s",  "т": "t",  "у": "u",  "ф": "f",
    "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch","ь": "",
    "ю": "yu", "я": "ya",
}

_STOP_WORDS = {
    "the", "a", "an", "of", "and", "or", "in", "at", "by",
    "на", "для", "та", "і",  "в",  "з",  "до",
}


def _to_slug(text: str) -> str:
    result = []
    for ch in text.lower():
        if ch in _TRANSLIT:
            result.append(_TRANSLIT[ch])
        elif ch.isascii() and (ch.isalnum() or ch == "-"):
            result.append(ch)
        elif ch == " ":
            result.append("-")
    slug = re.sub(r"-{2,}", "-", "".join(result)).strip("-")
    return slug[:40]


def _candidates(title: str) -> list[str]:
    """Generate domain candidates from title; no network call."""
    words = [w for w in re.split(r"\s+", title.lower()) if w not in _STOP_WORDS]
    compact = re.sub(r"[^a-z0-9]", "", "".join(words[:3]))
    slug = _to_slug(title)

    bases: list[str] = []
    if compact:
        bases.append(compact)
    if slug and slug != compact:
        bases.append(slug)

    urls = []
    for base in bases:
        for tld in _TLDS:
            urls.append(f"https://{base}{tld}")
    return urls


# ─── URL reachability probe ───────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
    reraise=False,
)
async def _head(client: httpx.AsyncClient, url: str) -> bool:
    try:
        r = await client.head(url, timeout=3.0, follow_redirects=True)
        return r.status_code < 400
    except Exception:
        return False


# ─── Public API ───────────────────────────────────────────────────────────────

async def find_domain_real(title: str) -> str | None:
    candidates = _candidates(title)
    async with httpx.AsyncClient(headers={"User-Agent": _UA}) as client:
        for url in candidates:
            if await _head(client, url):
                logger.debug("Domain found: %s → %s", title, url)
                return url
    return None


async def find_domain_mock(title: str) -> str | None:
    """Returns the first slug candidate without any network calls."""
    await asyncio.sleep(0.01)
    candidates = _candidates(title)
    return candidates[0] if candidates else None


async def find_domain(title: str) -> str | None:
    if settings.use_mock_search:
        return await find_domain_mock(title)
    return await find_domain_real(title)
