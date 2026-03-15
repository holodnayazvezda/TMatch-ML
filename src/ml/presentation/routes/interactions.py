from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from ml.services.interaction_ingestion import ingest_audit_events_batch

router = APIRouter(prefix="/interactions", tags=["interactions"])


class AuditEventRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Any = None
    created_at: Any = None
    deleted_at: Any = None
    event_type: Any = None
    actor_user_id: Any = None
    target_user_id: Any = None
    payload: Any = None
    sent_to_ml: Any = None


@router.post("/batch")
async def interactions_batch(payload: list[AuditEventRequest]) -> JSONResponse:
    events = [item.model_dump(mode="python") for item in payload]
    summary = await run_in_threadpool(ingest_audit_events_batch, events)
    return JSONResponse(status_code=200, content=summary)
