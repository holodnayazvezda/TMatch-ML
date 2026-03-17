import asyncio
import sys

import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from ml.config import settings
from ml.candidates import interactions_candidates, transactions_candidates
from ml.infra.logging.structlog import configure_logging
from ml.presentation.routes.health import router as health_router
from ml.presentation.routes.monitoring import router as monitoring_router
from ml.presentation.routes.recommendation import router as recommendation_router
from ml.services.interaction_event_poller import InteractionEventPoller
from ml.services.recommendation_runtime_config import DEFAULT_RECOMMENDATION_RUNTIME_CONFIG
from ml.services.recommendation_scoring import RecommendationStage

log: structlog.BoundLogger = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="ML",
        version="0.1.0",
        root_path=settings.root_path,
    )
    app.include_router(health_router)
    app.include_router(admin_rebuild_router)
    app.include_router(monitoring_router)
    app.include_router(recommendation_router)
    app.state.interaction_event_poller = InteractionEventPoller()

    @app.on_event("startup")
    async def warmup_faiss_cache() -> None:
        if not settings.warmup_faiss_on_startup:
            log.info("faiss_warmup_skipped")
        else:
            log.info("faiss_warmup_started")
            await run_in_threadpool(transactions_candidates.warmup_search_bundle)
            if DEFAULT_RECOMMENDATION_RUNTIME_CONFIG.recommendation_stage == RecommendationStage.STAGE_4:
                await run_in_threadpool(interactions_candidates.warmup_search_bundle)
            log.info("faiss_warmup_finished")

        app.state.interaction_event_poller.start()

    @app.on_event("shutdown")
    async def stop_background_services() -> None:
        await app.state.interaction_event_poller.stop()

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
