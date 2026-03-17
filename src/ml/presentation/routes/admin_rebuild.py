from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from ml.config import settings
from ml.services.transactions_rebuild import (
    TransactionsRebuildInProgressError,
    rebuild_transactions_data_and_artifacts,
)

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/admin/rebuild", tags=["admin"])

_ADMIN_KEY_HEADER_NAME = "X-ML-Admin-Key"


@router.post("/transactions")
async def rebuild_transactions(request: Request) -> JSONResponse:
    configured_rebuild_key = str(settings.rebuild_api_key or "").strip()
    if not configured_rebuild_key:
        raise HTTPException(
            status_code=503,
            detail="Transactions rebuild endpoint is disabled. Configure REBUILD_API_KEY.",
        )

    provided_rebuild_key = str(request.headers.get(_ADMIN_KEY_HEADER_NAME, "") or "").strip()
    if provided_rebuild_key != configured_rebuild_key:
        raise HTTPException(status_code=403, detail="Invalid admin key.")

    try:
        payload = await run_in_threadpool(rebuild_transactions_data_and_artifacts)
    except TransactionsRebuildInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("transactions_rebuild_http_failed")
        raise HTTPException(status_code=500, detail=f"Transactions rebuild failed: {exc}") from exc

    return JSONResponse(status_code=200, content=payload)
