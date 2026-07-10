#!/usr/bin/env python3.11
"""
Production enrichment runner.

Features vs run_local.py:
  • Multi-source discovery: TLD-in-name → Wikidata/Wikipedia → Google/SerpAPI → slug
  • Improved scraper: JSON-LD, /about page, RSS categories, Wikipedia extract fallback
  • Concurrent processing with asyncio.Semaphore (CONCURRENCY items at once)
  • Per-item retry (up to MAX_RETRIES) with exponential back-off
  • Failed-item log for manual review
  • Live progress counter + final report table

Usage:
  python3.11 run_production.py [--limit N] [--dry-run] [--concurrency N]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()

from config import settings
from db import (
    fetch_unenriched,
    fetch_items_for_embedding,
    update_embedding,
    update_media_item,
    count_unenriched,
)
from services.discovery import discover, DiscoveryResult
from services.scraper import scrape
from services.llm import structure_media_item
from services.embeddings import create_embeddings_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-22s  %(message)s",
    stream=sys.stdout,
)
# Silence noisy httpx request logs – keep only WARNING+
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("run_production")


# ─── Per-item result ──────────────────────────────────────────────────────────

@dataclass
class ItemResult:
    id: str
    title: str
    status: str = "pending"   # ok | failed | skipped
    category: str = ""
    tier: str = ""
    source: str = ""          # discovery source
    context_chars: int = 0
    error: str = ""
    attempts: int = 0
    extra: dict = field(default_factory=dict)


# ─── Single-item pipeline ─────────────────────────────────────────────────────

async def _enrich_item(
    item: dict,
    sem: asyncio.Semaphore,
    counter: list[int],
    total: int,
) -> ItemResult:
    result = ItemResult(id=item["id"], title=item["title"])

    async with sem:
        for attempt in range(1, settings.max_retries + 1):
            result.attempts = attempt
            try:
                await _pipeline(item, result)
                result.status = "ok"
                break
            except Exception as exc:
                result.error = str(exc)
                if attempt < settings.max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "  ↻ retry %d/%d for '%s' in %ds: %s",
                        attempt, settings.max_retries, item["title"], wait, exc,
                    )
                    await asyncio.sleep(wait)
                else:
                    result.status = "failed"
                    logger.error("  ✗ gave up on '%s': %s", item["title"], exc)

    counter[0] += 1
    pct = counter[0] * 100 // total
    logger.info(
        "[%3d/%d %3d%%]  %-26s  %s  %-10s  src=%-12s  %d chars",
        counter[0], total, pct,
        item["title"][:26],
        "✓" if result.status == "ok" else "✗",
        result.category or "—",
        result.source or "—",
        result.context_chars,
    )
    return result


async def _pipeline(item: dict, result: ItemResult) -> None:
    title = item["title"]

    # ── Stage 1: multi-source discovery ─────────────────────────────────────
    discovery: DiscoveryResult = await discover(title)
    result.source = discovery.source

    # ── Stage 2: content extraction ─────────────────────────────────────────
    context = await scrape(
        name=title,
        url=discovery.url,
        wikipedia_extract=discovery.wikipedia_extract,
        source=discovery.source,
    )
    result.context_chars = len(context)

    # ── Stage 3: LLM structuring ─────────────────────────────────────────────
    structured = await structure_media_item(title, context)
    result.category = structured["category"]
    result.tier     = ", ".join(structured.get("metrics", {}).get("geographic_coverage", []) or ["?"])
    result.extra    = discovery.extra  # founding_year etc.

    # Merge Wikidata extras into metrics if available
    metrics = structured.get("metrics", {})
    if discovery.extra.get("founding_year"):
        metrics["founding_year"] = discovery.extra["founding_year"]
    structured["metrics"] = metrics

    # ── Stage 4: persist standard fields ────────────────────────────────────
    await update_media_item(
        item_id=item["id"],
        description=structured["description"],
        category=structured["category"],
        tags=structured.get("tags", []),
        audience=structured.get("audience", {}),
        metrics=structured["metrics"],
    )


# ─── Embedding batch ──────────────────────────────────────────────────────────

async def _embed_batch(results: list[ItemResult]) -> tuple[int, int]:
    """Fetch text, single API call, raw SQL vector write."""
    ok_ids = [r.id for r in results if r.status == "ok"]
    if not ok_ids:
        return 0, 0

    logger.info("\n━━ Batch embedding %d items…", len(ok_ids))
    db_items = await fetch_items_for_embedding(ok_ids)
    if not db_items:
        logger.warning("No items found for embedding batch")
        return 0, 0

    pairs = await create_embeddings_batch(db_items)
    success = failed = 0
    for item_id, vector in pairs:
        try:
            await update_embedding(item_id, vector)
            success += 1
        except Exception as exc:
            logger.error("  Embedding write failed %s: %s", item_id, exc)
            failed += 1

    logger.info("  ✓ Embeddings: %d written, %d failed", success, failed)
    return success, failed


# ─── Report ───────────────────────────────────────────────────────────────────

def _print_report(results: list[ItemResult], elapsed: float, embed_ok: int) -> None:
    ok      = [r for r in results if r.status == "ok"]
    failed  = [r for r in results if r.status == "failed"]

    # Source distribution
    sources: dict[str, int] = {}
    for r in ok:
        sources[r.source] = sources.get(r.source, 0) + 1

    logger.info("\n%s", "=" * 70)
    logger.info("ENRICHMENT REPORT")
    logger.info("  Total processed : %d", len(results))
    logger.info("  Successful      : %d", len(ok))
    logger.info("  Failed          : %d", len(failed))
    logger.info("  Embeddings      : %d", embed_ok)
    logger.info("  Elapsed         : %.1fs", elapsed)
    logger.info("  Discovery sources: %s",
                "  ".join(f"{k}={v}" for k, v in sorted(sources.items(), key=lambda x: -x[1])))
    logger.info("")
    logger.info("  %-28s  %-16s  %-10s  %-12s  chars", "Title", "Category", "Tier", "Source")
    logger.info("  " + "-" * 72)
    for r in sorted(ok, key=lambda x: x.source):
        logger.info(
            "  %-28s  %-16s  %-10s  %-12s  %d",
            r.title[:28], r.category, r.tier, r.source, r.context_chars,
        )
    if failed:
        logger.info("")
        logger.info("  FAILED items:")
        for r in failed:
            logger.info("  ✗ %-28s  %s", r.title[:28], r.error[:80])
    logger.info("=" * 70)


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main(limit: int | None, concurrency: int, dry_run: bool) -> None:
    total_in_db = await count_unenriched()
    items = await fetch_unenriched(limit=limit or 10_000)

    logger.info("Unenriched in DB : %d", total_in_db)
    logger.info("Will process     : %d  (concurrency=%d)", len(items), concurrency)
    logger.info("LLM mock         : %s  |  Search mock: %s",
                settings.use_mock_llm, settings.use_mock_search)

    if dry_run:
        logger.info("[DRY RUN] Exiting without changes.")
        return
    if not items:
        logger.info("Nothing to enrich.")
        return

    sem     = asyncio.Semaphore(concurrency)
    counter = [0]
    t0      = time.monotonic()

    tasks = [
        _enrich_item(item, sem, counter, len(items))
        for item in items
    ]
    results: list[ItemResult] = await asyncio.gather(*tasks)

    embed_ok, _ = await _embed_batch(results)
    elapsed = time.monotonic() - t0
    _print_report(results, elapsed, embed_ok)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Production enrichment runner")
    p.add_argument("--limit",       type=int, default=None, metavar="N",
                   help="Max items to process")
    p.add_argument("--concurrency", type=int, default=settings.concurrency, metavar="N",
                   help=f"Parallel items (default {settings.concurrency})")
    p.add_argument("--dry-run",     action="store_true",
                   help="Count only, no DB writes")
    return p.parse_args()


if __name__ == "__main__":
    args = _args()
    asyncio.run(main(
        limit=args.limit,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
    ))
