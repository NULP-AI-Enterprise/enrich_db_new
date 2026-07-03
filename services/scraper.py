"""
Stage 2 – Content extraction.

Handles three source types:
  A. Real website      – homepage + /about + /про-нас, JSON-LD, RSS
  B. Wikipedia page    – uses the Wikipedia extracts API (no HTML parsing)
  C. No URL            – title-only fallback

Returns a plain-text context blob for the LLM (≤ 6000 chars).
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET

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

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _UA,
    "Accept-Language": "uk,en-US;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_MAX_BODY   = 3000
_MAX_TOTAL  = 6000
_ABOUT_SLUGS = ["/about", "/about-us", "/про-нас", "/pro-nas", "/contacts", "/about/us"]
_RSS_SLUGS   = ["/feed", "/rss", "/feed.xml", "/rss.xml", "/atom.xml"]


# ─── JSON-LD extraction ───────────────────────────────────────────────────────

def _jsonld(soup: BeautifulSoup) -> dict:
    out: dict = {}
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            if isinstance(data, list):
                data = data[0]
            t = data.get("@type", "")
            if any(x in t for x in ("NewsMediaOrganization", "Organization", "WebSite", "NewsArticle")):
                out["name"]        = data.get("name", "")
                out["description"] = data.get("description", "")
                out["url"]         = data.get("url", "")
                break
        except Exception:
            pass
    return out


# ─── Open Graph / meta extraction ────────────────────────────────────────────

def _meta(soup: BeautifulSoup, name: str | None = None, prop: str | None = None) -> str:
    tag = (
        soup.find("meta", attrs={"name": name}) if name
        else soup.find("meta", property=prop)
    )
    return (tag.get("content") or "").strip() if tag else ""  # type: ignore[union-attr]


# ─── RSS feed parsing ─────────────────────────────────────────────────────────

def _rss_categories(xml_text: str) -> list[str]:
    cats: set[str] = set()
    try:
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for item in root.iter("item"):
            for cat in item.findall("category"):
                if cat.text:
                    cats.add(cat.text.strip())
        if not cats:
            for entry in root.findall("atom:entry", ns):
                for cat in entry.findall("atom:category", ns):
                    cats.add(cat.get("term", ""))
    except ET.ParseError:
        pass
    return list(cats)[:15]


# ─── HTML body text ───────────────────────────────────────────────────────────

_SELECTORS = ["main p", "article p", "[role=main] p", ".content p", "#content p", "section p", "body p"]

def _body_text(soup: BeautifulSoup) -> str:
    for sel in _SELECTORS:
        parts = [
            p.get_text(" ", strip=True)
            for p in soup.select(sel)
            if len(p.get_text(strip=True)) > 40
        ]
        if parts:
            return " ".join(parts)[:_MAX_BODY]
    return ""


# ─── HTTP fetch with retry ────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(settings.max_retries),
    wait=wait_exponential(multiplier=1, min=2, max=16),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=False,
)
async def _get(client: httpx.AsyncClient, url: str) -> str | None:
    try:
        r = await client.get(url, timeout=httpx.Timeout(settings.scrape_timeout))
        if r.status_code < 400:
            return r.text
    except (httpx.HTTPStatusError, httpx.RequestError):
        pass
    return None


# ─── Scrape real website ──────────────────────────────────────────────────────

async def _scrape_site(client: httpx.AsyncClient, base_url: str) -> dict:
    signals: dict = {"url": base_url}

    # Homepage
    html = await _get(client, base_url)
    if not html:
        return signals

    soup = BeautifulSoup(html, "lxml")

    # Structured data (highest priority)
    ld = _jsonld(soup)
    signals.update({k: v for k, v in ld.items() if v})

    # Open Graph / meta
    signals.setdefault("og_title",  _meta(soup, prop="og:title"))
    signals.setdefault("og_desc",   _meta(soup, prop="og:description"))
    signals.setdefault("meta_desc", _meta(soup, name="description"))
    title_tag = soup.find("title")
    signals.setdefault("page_title", title_tag.get_text(strip=True) if title_tag else "")
    signals.setdefault("body_text",  _body_text(soup))

    # /about page – richer description
    for slug in _ABOUT_SLUGS:
        about_html = await _get(client, base_url.rstrip("/") + slug)
        if about_html:
            a_soup = BeautifulSoup(about_html, "lxml")
            about_text = _body_text(a_soup)
            if len(about_text) > len(signals.get("body_text", "")):
                signals["about_text"] = about_text[:_MAX_BODY]
            break

    # RSS feed – category signals
    for slug in _RSS_SLUGS:
        rss = await _get(client, base_url.rstrip("/") + slug)
        if rss and "<?xml" in rss[:200]:
            signals["rss_categories"] = _rss_categories(rss)
            break

    return signals


def _flatten_site(signals: dict, name: str) -> str:
    parts = [f"Outlet name: {name}"]
    if signals.get("name") and signals["name"].lower() != name.lower():
        parts.append(f"Official name: {signals['name']}")
    if signals.get("og_desc") or signals.get("meta_desc") or signals.get("description"):
        parts.append("Description: " + (
            signals.get("description") or signals.get("og_desc") or signals.get("meta_desc")
        ))
    if signals.get("about_text"):
        parts.append(f"About: {signals['about_text']}")
    elif signals.get("body_text"):
        parts.append(f"Content: {signals['body_text']}")
    if signals.get("rss_categories"):
        parts.append(f"Content categories: {', '.join(signals['rss_categories'])}")
    return "\n".join(parts)[:_MAX_TOTAL]


# ─── Wikipedia extract (no HTML scraping needed) ─────────────────────────────

def _flatten_wikipedia(extract: str, name: str) -> str:
    return f"Outlet name: {name}\nWikipedia article:\n{extract[:_MAX_TOTAL - 100]}"


# ─── Public API ───────────────────────────────────────────────────────────────

async def scrape(
    name: str,
    url: str | None,
    wikipedia_extract: str | None,
    source: str = "none",
) -> str:
    """
    Returns a text blob for the LLM.

    Strategy:
      - If we have a real website (source != wikipedia_*): scrape the site;
        if scraping fails but WP extract exists, fall back to it.
      - If source is wikipedia_*: use the extract directly (no scraping).
      - If nothing: name-only stub.
    """
    is_wikipedia_url = source.startswith("wikipedia")

    # Use Wikipedia extract when it's the primary source
    if is_wikipedia_url and wikipedia_extract:
        logger.debug("Using Wikipedia extract for '%s'", name)
        return _flatten_wikipedia(wikipedia_extract, name)

    # Scrape real website
    if url and not is_wikipedia_url:
        async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True) as client:
            signals = await _scrape_site(client, url)

        # Merge in Wikipedia extract as additional context
        text = _flatten_site(signals, name)
        if wikipedia_extract and len(text) < 500:
            # Scraping gave little content (bot-blocked) – prepend WP extract
            text = _flatten_wikipedia(wikipedia_extract, name) + "\n\n" + text

        logger.debug("Scraped '%s': %d chars (source=%s)", name, len(text), source)
        return text

    # Wikipedia extract available but URL was an official site
    if wikipedia_extract:
        return _flatten_wikipedia(wikipedia_extract, name)

    # Last resort: name only
    logger.warning("No content for '%s' — name-only enrichment", name)
    return f"Outlet name: {name}"
