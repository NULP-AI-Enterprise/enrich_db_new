from celery import Celery
from config import settings

celery_app = Celery(
    "enrichment",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["tasks.enrich", "tasks.batch_embed"],
)

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=3600,

    # Clock
    timezone="UTC",
    enable_utc=True,

    # Reliability
    task_track_started=True,
    task_acks_late=True,          # re-queue if worker crashes mid-task
    worker_prefetch_multiplier=1, # fair dispatch; long-running tasks don't starve others

    # Routing
    task_routes={
        "tasks.enrich.enrich_item":               {"queue": "enrichment"},
        "tasks.batch_embed.process_embedding_batch": {"queue": "embeddings"},
    },

    # Celery Beat – fallback trigger every N seconds even if queue never hits threshold
    beat_schedule={
        "embedding-batch-flush": {
            "task": "tasks.batch_embed.process_embedding_batch",
            "schedule": settings.batch_embed_interval,
        },
    },
    beat_scheduler="celery.beat:PersistentScheduler",
)
