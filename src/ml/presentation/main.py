import asyncio
import sys

import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from ml.config import settings
from ml.candidates import transactions_candidates
from ml.infra.logging.structlog import configure_logging
from ml.presentation.routes.health import router as health_router
from ml.presentation.routes.interactions import router as interactions_router
from ml.presentation.routes.recommendation import router as recommendation_router

log: structlog.BoundLogger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="ML",
        version="0.1.0",
        root_path=settings.root_path,
    )
    app.include_router(health_router)
    app.include_router(interactions_router)
    app.include_router(recommendation_router)

    @app.on_event("startup")
    async def warmup_faiss_cache() -> None:
        if not settings.warmup_faiss_on_startup:
            log.info("faiss_warmup_skipped")
            return

        log.info("faiss_warmup_started")
        await run_in_threadpool(transactions_candidates.warmup_search_bundle)
        log.info("faiss_warmup_finished")

    return app


def run() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    load_dotenv(override=False)

    configure_logging()
    log.info("booting", host=settings.server_host, port=settings.server_port)

    app = create_app()
    uvicorn.run(
        app,
        host=settings.server_host,
        port=settings.server_port,
        log_config=None,
        timeout_keep_alive=120,
    )


if __name__ == "__main__":
    run()
