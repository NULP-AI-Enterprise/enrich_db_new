"""
Stage 2 – Lightweight Web Scraping.

Extracts SEO signals (title, og:description, meta description, body excerpt)
from the outlet's homepage.  Falls back gracefully when a URL is unavailable
or the scrape fails after retries.

httpx is used in async mode; BeautifulSoup parses HTML on CPU.
Tenacity handles transient network errors with exponential backoff.
"""

import logging

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from config import settings

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "uk,en-US;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
}

# CSS selectors tried in priority order for body text
_BODY_SELECTORS = ["main p", "article p", "[role=main] p", ".content p", "body p"]
_MAX_BODY_CHARS = 3000
_MAX_TOTAL_CHARS = 6000


# ─── HTML parsing ─────────────────────────────────────────────────────────────

def _parse(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    def _meta(name: str | None = None, prop: str | None = None) -> str:
        tag = (
            soup.find("meta", attrs={"name": name}) if name
            else soup.find("meta", property=prop)
        )
        return (tag.get("content") or "").strip() if tag else ""  # type: ignore[union-attr]

    title_tag = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else ""

    body_text = ""
    for sel in _BODY_SELECTORS:
        parts = [p.get_text(" ", strip=True) for p in soup.select(sel) if len(p.get_text(strip=True)) > 40]
        if parts:
            body_text = " ".join(parts)[:_MAX_BODY_CHARS]
            break

    return {
        "url":        url,
        "page_title": page_title,
        "og_title":   _meta(prop="og:title"),
        "og_desc":    _meta(prop="og:description"),
        "meta_desc":  _meta(name="description"),
        "body_text":  body_text,
    }


def _flatten(signals: dict, outlet_name: str) -> str:
    """Produce a single text blob to feed into the LLM."""
    lines = [f"Outlet name: {outlet_name}"]
    if signals.get("page_title"):
        lines.append(f"Page title: {signals['page_title']}")
    desc = signals.get("og_desc") or signals.get("meta_desc")
    if desc:
        lines.append(f"Description: {desc}")
    if signals.get("body_text"):
        lines.append(f"Content: {signals['body_text']}")
    return "\n".join(lines)[:_MAX_TOTAL_CHARS]


# ─── HTTP fetch with retry ────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(settings.max_retries),
    wait=wait_exponential(multiplier=1, min=2, max=16),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=False,
)
async def _fetch(url: str) -> dict | None:
    async with httpx.AsyncClient(
        headers=_HEADERS,
        timeout=httpx.Timeout(settings.scrape_timeout),
        follow_redirects=True,
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return _parse(resp.text, url)


# ─── Public API ───────────────────────────────────────────────────────────────

async def scrape_outlet(title: str, url: str | None) -> str:
    """
    Returns a text blob for the LLM.
    Gracefully degrades: URL missing → title only; scrape error → title only.
    """
    if not url:
        logger.debug("No URL for '%s' — title-only enrichment", title)
        return f"Outlet name: {title}"

    signals = await _fetch(url)
    if not signals:
        logger.warning("Scrape failed for '%s' (%s) after retries", title, url)
        return f"Outlet name: {title}"

    text = _flatten(signals, title)
    logger.debug("Scraped %d chars for '%s'", len(text), title)
    return text
