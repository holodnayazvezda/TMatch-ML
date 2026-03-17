import asyncio
import sys
from time import perf_counter

import structlog
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from ml.config import settings
from ml.candidates import interactions_candidates, popularity_fallback, transactions_candidates
from ml.infra.logging.structlog import configure_logging
from ml.presentation.routes.admin_rebuild import router as admin_rebuild_router
from ml.presentation.routes.health import router as health_router
from ml.presentation.routes.recommendation import router as recommendation_router
from ml.presentation.routes.user_description import router as user_description_router
from ml.services.interaction_event_poller import InteractionEventPoller
from ml.services.recommendation_runtime_config import DEFAULT_RECOMMENDATION_RUNTIME_CONFIG
from ml.services.recommendation_scoring import RecommendationStage

log: structlog.BoundLogger = structlog.get_logger(__name__)


def _log_artifact_status(*, source: str, payload: dict[str, object], started_at: float) -> None:
    status_payload = dict(payload)
    status_payload.setdefault("source", source)
    status_payload["load_ms"] = round((perf_counter() - started_at) * 1000, 2)
    log.info("recommendation_artifact_warmup", **status_payload)


def create_app() -> FastAPI:
    app = FastAPI(
        title="ML",
        version="0.1.0",
        root_path=settings.root_path,
    )
    app.include_router(health_router)
    app.include_router(admin_rebuild_router)
    app.include_router(recommendation_router)
    app.include_router(user_description_router)
    app.state.interaction_event_poller = InteractionEventPoller()

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception(
            "unhandled_exception",
            path=request.url.path,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.on_event("startup")
    async def warmup_faiss_cache() -> None:
        if not settings.warmup_faiss_on_startup:
            log.info("faiss_warmup_skipped")
        else:
            log.info("faiss_warmup_started")
            started_at = perf_counter()
            await run_in_threadpool(transactions_candidates.warmup_search_bundle)
            _log_artifact_status(
                source="transactions",
                payload=transactions_candidates.get_search_bundle_artifact_status(),
                started_at=started_at,
            )
            if DEFAULT_RECOMMENDATION_RUNTIME_CONFIG.recommendation_stage == RecommendationStage.STAGE_4:
                started_at = perf_counter()
                await run_in_threadpool(interactions_candidates.warmup_search_bundle)
                _log_artifact_status(
                    source="tx_interaction",
                    payload=interactions_candidates.get_search_bundle_artifact_status(),
                    started_at=started_at,
                )
            started_at = perf_counter()
            popularity_payload = await run_in_threadpool(popularity_fallback.warmup_popularity_snapshot)
            _log_artifact_status(
                source="popularity",
                payload=popularity_payload,
                started_at=started_at,
            )
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
