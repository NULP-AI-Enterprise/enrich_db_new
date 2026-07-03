"""
Multi-source domain & metadata discovery.

Priority pipeline (cheapest / most reliable first):
  1. Name contains explicit TLD  ("Район.in.ua" → rayon.in.ua)
  2. Wikipedia Ukrainian API     → Wikidata P856 (official website)
  3. Wikipedia English API       → Wikidata P856
  4. Google Custom Search API    (optional, needs GOOGLE_API_KEY + GOOGLE_CX)
  5. SerpAPI                     (optional, needs SERP_API_KEY)
  6. Slug inference + HEAD probe (last resort)

Returns DiscoveryResult with url, wikipedia_extract, and source tag so
the scraper knows whether to scrape the real site or use the WP extract.
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (compatible; MediaEnrichBot/1.0; +https://thesis-i.com)"
_TIMEOUT = httpx.Timeout(8.0)


# ─── Result type ─────────────────────────────────────────────────────────────

@dataclass
class DiscoveryResult:
    url: str | None = None
    wikipedia_extract: str | None = None   # lead paragraph from Wikipedia
    wikidata_qid: str | None = None
    source: str = "none"                   # slug|wikidata|google|serp|known
    extra: dict = field(default_factory=dict)  # founding_year, country, lang …


# ─── Transliteration ─────────────────────────────────────────────────────────

_UA_TABLE: dict[str, str] = {
    "а":"a","б":"b","в":"v","г":"h","ґ":"g","д":"d","е":"e","є":"ie",
    "ж":"zh","з":"z","и":"y","і":"i","ї":"i","й":"y","к":"k","л":"l",
    "м":"m","н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u",
    "ф":"f","х":"kh","ц":"ts","ч":"ch","ш":"sh","щ":"shch","ь":"",
    "ю":"yu","я":"ya",
}
_STOP = {"the","a","an","of","and","or","in","at","by","на","для","та","і","в","з","до"}


def _latin(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch in _UA_TABLE:
            out.append(_UA_TABLE[ch])
        elif ch.isascii() and (ch.isalnum() or ch in "-_"):
            out.append(ch)
        elif ch in " \t":
            out.append("-")
    return re.sub(r"-{2,}", "-", "".join(out)).strip("-")


# ─── Source 1: name contains explicit domain ──────────────────────────────────

_KNOWN_TLDS = re.compile(
    r"\.("
    r"com\.ua|org\.ua|net\.ua|in\.ua|com|net|org|ua|info|media|online|news"
    r")$",
    re.I,
)

def _extract_tld_from_name(name: str) -> str | None:
    """
    If the outlet name itself contains a domain-like suffix, reconstruct the URL.
    e.g. "Район.in.ua" → "https://rayon.in.ua"
         "Чернівці.com" → "https://chernivtsi.com"
         "Мукачево.net" → "https://mukachevo.net"
    """
    m = _KNOWN_TLDS.search(name)
    if not m:
        return None
    tld = m.group(0)                      # e.g.  ".in.ua"
    prefix = name[: m.start()].strip()    # e.g.  "Район"
    slug = _latin(prefix)
    if not slug:
        return None
    return f"https://{slug}{tld}"


# ─── Wikipedia / Wikidata helpers ─────────────────────────────────────────────

async def _wiki_search(client: httpx.AsyncClient, name: str, lang: str) -> str | None:
    """Return the best-matching Wikipedia page title."""
    try:
        r = await client.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search",
                "srsearch": name, "srlimit": 5,
                "srnamespace": 0, "format": "json",
            },
        )
        hits = r.json().get("query", {}).get("search", [])
        # Prefer an exact title match
        name_lower = name.lower()
        for h in hits:
            if name_lower in h["title"].lower():
                return h["title"]
        return hits[0]["title"] if hits else None
    except Exception as e:
        logger.debug("Wikipedia search error (%s, %s): %s", lang, name, e)
        return None


async def _wiki_extract(client: httpx.AsyncClient, title: str, lang: str) -> dict:
    """Return {qid, extract, url, extra} for a Wikipedia page title."""
    result: dict = {}
    try:
        r = await client.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": title,
                "prop": "pageprops|extracts|info",
                "exintro": True,
                "explaintext": True,
                "inprop": "url",
                "format": "json",
            },
        )
        pages = r.json().get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if pid == "-1":
                continue
            result["extract"] = (page.get("extract") or "").strip()[:3000]
            result["qid"] = page.get("pageprops", {}).get("wikibase_item")
            result["wp_url"] = page.get("fullurl")
    except Exception as e:
        logger.debug("Wikipedia extract error (%s): %s", title, e)
    return result


async def _wikidata_website(client: httpx.AsyncClient, qid: str) -> dict:
    """
    Fetch official website (P856) and extra metadata from Wikidata.
    Returns {url, founding_year, country, language}.
    """
    out: dict = {}
    try:
        r = await client.get(
            f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
        )
        entity = r.json().get("entities", {}).get(qid, {})
        claims = entity.get("claims", {})

        def _first_str(prop: str) -> str | None:
            stmts = claims.get(prop, [])
            if not stmts:
                return None
            sv = stmts[0].get("mainsnak", {}).get("datavalue", {})
            if sv.get("type") == "string":
                return sv["value"]
            return None

        def _first_time(prop: str) -> str | None:
            stmts = claims.get(prop, [])
            if not stmts:
                return None
            sv = stmts[0].get("mainsnak", {}).get("datavalue", {})
            if sv.get("type") == "time":
                return sv["value"].get("time", "")[:5].lstrip("+")
            return None

        out["url"]           = _first_str("P856")
        out["founding_year"] = _first_time("P571")  # inception
        # P495 = country, P407 = language — we'd need label lookup for those
    except Exception as e:
        logger.debug("Wikidata error (%s): %s", qid, e)
    return out


# ─── Source 4: Google Custom Search API ──────────────────────────────────────

async def _google_search(client: httpx.AsyncClient, name: str) -> str | None:
    if not (settings.google_api_key and settings.google_cx):
        return None
    try:
        r = await client.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": settings.google_api_key,
                "cx":  settings.google_cx,
                "q":   f'"{name}" офіційний сайт новини',
                "num": 3,
            },
        )
        items = r.json().get("items", [])
        if items:
            return items[0]["link"]
    except Exception as e:
        logger.debug("Google CSE error: %s", e)
    return None


# ─── Source 5: SerpAPI ────────────────────────────────────────────────────────

async def _serp_search(client: httpx.AsyncClient, name: str) -> str | None:
    if not settings.serp_api_key:
        return None
    try:
        r = await client.get(
            "https://serpapi.com/search",
            params={
                "api_key": settings.serp_api_key,
                "engine":  "google",
                "q":       f'"{name}" site офіційний',
                "num":     3,
                "hl":      "uk",
                "gl":      "ua",
            },
        )
        results = r.json().get("organic_results", [])
        if results:
            return results[0]["link"]
    except Exception as e:
        logger.debug("SerpAPI error: %s", e)
    return None


# ─── Source 6: Slug inference + HEAD probe ───────────────────────────────────

_TLDS = [".com.ua", ".ua", ".com", ".net", ".org", ".media", ".online"]

def _slug_candidates(name: str) -> list[str]:
    # Remove any TLD-like suffix from the name before slugging
    clean = _KNOWN_TLDS.sub("", name).strip()
    words = [w for w in re.split(r"[\s\-_]+", clean) if w and w.lower() not in _STOP]
    if not words:
        return []
    base = _latin("".join(words[:3]))
    if not base:
        return []
    return [f"https://{base}{tld}" for tld in _TLDS]


@retry(stop=stop_after_attempt(1), wait=wait_exponential(min=0.5, max=2), reraise=False)
async def _head_ok(client: httpx.AsyncClient, url: str) -> bool:
    try:
        r = await client.head(url, timeout=4.0, follow_redirects=True)
        return r.status_code < 400
    except Exception:
        return False


async def _slug_probe(client: httpx.AsyncClient, name: str) -> str | None:
    for url in _slug_candidates(name):
        if await _head_ok(client, url):
            return url
    return None


# ─── Public entry point ───────────────────────────────────────────────────────

async def discover(name: str) -> DiscoveryResult:
    """
    Full multi-source discovery for a single outlet name.
    Returns DiscoveryResult with the best URL and any Wikipedia text found.
    """
    result = DiscoveryResult()

    async with httpx.AsyncClient(
        headers={"User-Agent": _UA},
        timeout=_TIMEOUT,
        follow_redirects=True,
    ) as client:

        # ── 1. Name contains explicit domain ─────────────────────────────────
        tld_url = _extract_tld_from_name(name)
        if tld_url:
            if await _head_ok(client, tld_url):
                result.url    = tld_url
                result.source = "tld_in_name"
                logger.info("  [discovery] TLD-in-name → %s", tld_url)
            # Even if the URL is wrong, still try Wikipedia for metadata

        # ── 2+3. Wikipedia (UK then EN) → Wikidata ────────────────────────────
        for lang in ("uk", "en"):
            title = await _wiki_search(client, name, lang)
            if not title:
                continue
            wp = await _wiki_extract(client, title, lang)
            if not wp:
                continue

            result.wikipedia_extract = wp.get("extract")
            result.wikidata_qid      = wp.get("qid")

            if wp.get("qid"):
                wd = await _wikidata_website(client, wp["qid"])
                if wd.get("url"):
                    result.url    = wd["url"].rstrip("/")
                    result.source = "wikidata"
                    result.extra.update({k: v for k, v in wd.items() if v and k != "url"})
                    logger.info("  [discovery] Wikidata P856 → %s", result.url)
                    break  # best possible source

            # Wikipedia page itself as fallback URL
            if not result.url and wp.get("wp_url"):
                result.url    = wp["wp_url"]
                result.source = f"wikipedia_{lang}"
                logger.info("  [discovery] Wikipedia page → %s", result.url)
            break  # only need one Wikipedia hit

        # ── 4. Google Custom Search ───────────────────────────────────────────
        if not result.url:
            g = await _google_search(client, name)
            if g:
                result.url    = g
                result.source = "google"
                logger.info("  [discovery] Google CSE → %s", g)

        # ── 5. SerpAPI ────────────────────────────────────────────────────────
        if not result.url:
            s = await _serp_search(client, name)
            if s:
                result.url    = s
                result.source = "serp"
                logger.info("  [discovery] SerpAPI → %s", s)

        # ── 6. Slug + HEAD probe ──────────────────────────────────────────────
        if not result.url:
            slug_url = await _slug_probe(client, name)
            if slug_url:
                result.url    = slug_url
                result.source = "slug"
                logger.info("  [discovery] Slug probe → %s", slug_url)

    if not result.url and not result.wikipedia_extract:
        logger.warning("  [discovery] no source found for '%s'", name)

    return result
