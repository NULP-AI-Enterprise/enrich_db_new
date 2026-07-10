"""
Standalone runner — bypasses Celery for small batches.
Runs the full 4-stage pipeline directly with asyncio.
Usage: python run_local.py
"""

import asyncio
import logging
import sys
import os

# make sure .env is loaded before any imports
from dotenv import load_dotenv
load_dotenv()

from db import fetch_unenriched, update_media_item, fetch_items_for_embedding, update_embedding
from services.search import find_domain
from services.scraper import scrape_outlet
from services.llm import structure_media_item
from services.embeddings import create_embeddings_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("run_local")


async def enrich_one(item: dict) -> dict:
    item_id, title = item["id"], item["title"]
    logger.info("━━ [%s] starting", title)

    url = await find_domain(title)
    logger.info("  domain    : %s", url or "not found")

    context = await scrape_outlet(title, url)
    logger.info("  context   : %d chars", len(context))

    structured = await structure_media_item(title, context)
    logger.info("  category  : %s  |  coverage: %s",
                structured["category"],
                structured.get("metrics", {}).get("geographic_coverage", []))
    logger.info("  tags      : %s", ", ".join(structured.get("tags", [])))

    await update_media_item(
        item_id=item_id,
        description=structured["description"],
        category=structured["category"],
        tags=structured.get("tags", []),
        audience=structured.get("audience", {}),
        metrics=structured.get("metrics", {}),
    )
    logger.info("  ✓ saved to DB")
    return {"id": item_id, "title": title, **structured}


async def main() -> None:
    items = await fetch_unenriched(limit=500)
    if not items:
        logger.info("Nothing to enrich — all items already have descriptions.")
        return

    logger.info("Found %d unenriched items\n", len(items))

    enriched = []
    for item in items:
        try:
            result = await enrich_one(item)
            enriched.append(result)
        except Exception as e:
            logger.error("Failed '%s': %s", item["title"], e)

    # ── Batch embedding ──────────────────────────────────────────────────────
    if not enriched:
        logger.warning("No items enriched successfully.")
        return

    logger.info("\n━━ Batch embedding %d items…", len(enriched))
    db_items = await fetch_items_for_embedding([e["id"] for e in enriched])
    pairs = await create_embeddings_batch(db_items)

    embed_ok = 0
    for item_id, vector in pairs:
        try:
            await update_embedding(item_id, vector)
            embed_ok += 1
        except Exception as e:
            logger.error("Embedding write failed for %s: %s", item_id, e)

    logger.info("  ✓ embeddings written: %d / %d", embed_ok, len(pairs))

    # ── Final summary ────────────────────────────────────────────────────────
    logger.info("\n%s", "=" * 60)
    logger.info("DONE — %d items enriched + %d embeddings stored", len(enriched), embed_ok)
    logger.info("%-30s  %-16s  %s", "Title", "Category", "Coverage")
    logger.info("-" * 60)
    for r in enriched:
        coverage = r.get("metrics", {}).get("geographic_coverage", [])
        logger.info("%-30s  %-16s  %s",
                    r["title"][:30],
                    r.get("category", "?"),
                    ", ".join(coverage) if coverage else "?")


if __name__ == "__main__":
    asyncio.run(main())
