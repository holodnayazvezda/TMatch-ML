from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from ml.services.recommendation_quality import get_model_quality_report

log: structlog.BoundLogger = structlog.get_logger(__name__)
router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/model-quality")
async def model_quality(
    limit: int = Query(20_000, ge=1, le=500_000),
    include_pending: bool = Query(True),
    include_rows: bool = Query(False),
    rows_limit: int = Query(200, ge=1, le=5_000),
    served_stage: str | None = Query(
        default=None,
        description="Optional stage filter (for example: stage_4).",
    ),
) -> JSONResponse:
    served_stages = [served_stage] if served_stage else None
    try:
        payload = await run_in_threadpool(
            get_model_quality_report,
            limit=limit,
            served_stages=served_stages,
            include_pending=include_pending,
            include_rows=include_rows,
            rows_limit=rows_limit,
        )
    except Exception as exc:
        log.exception("monitoring_model_quality_failed")
        raise HTTPException(status_code=500, detail=f"Failed to compute model quality metrics: {exc}") from exc
    return JSONResponse(status_code=200, content=payload)
