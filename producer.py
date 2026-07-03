#!/usr/bin/env python
"""
Task Producer – CLI entry point.

Queries media_items for rows where description IS NULL and dispatches
one enrich_item Celery task per row.

Usage:
    python producer.py               # enqueue everything
    python producer.py --limit 500   # cap at 500 items
    python producer.py --dry-run     # count only, no tasks sent
"""

import argparse
import asyncio
import logging
import sys

from db import count_unenriched, fetch_unenriched
from tasks.enrich import enrich_item
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


async def _run(limit: int | None, dry_run: bool) -> None:
    total = await count_unenriched()
    effective = min(total, limit) if limit else total
    logger.info("Unenriched items in DB: %d  |  will enqueue: %d", total, effective)

    if dry_run:
        logger.info("[DRY RUN] No tasks sent.")
        return

    dispatched = 0
    offset = 0

    while True:
        remaining = (limit - dispatched) if limit else settings.producer_page_size
        page = await fetch_unenriched(
            limit=min(settings.producer_page_size, remaining),
            offset=offset,
        )
        if not page:
            break

        for item in page:
            # Stagger countdown to prevent thundering-herd on the DB/LLM layer.
            # Every `task_stagger_per` tasks we add 1 s of delay.
            countdown = dispatched // settings.task_stagger_per

            enrich_item.apply_async(
                args=[item["id"], item["title"]],
                queue="enrichment",
                countdown=countdown,
            )
            dispatched += 1

        offset += len(page)
        logger.info("Dispatched %d / %d …", dispatched, effective)

        if limit and dispatched >= limit:
            break

    logger.info("Done. %d tasks sent to the `enrichment` queue.", dispatched)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enqueue unenriched media_items for the enrichment pipeline.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of items to enqueue (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not enqueue any tasks.",
    )
    args = parser.parse_args()
    asyncio.run(_run(limit=args.limit, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
