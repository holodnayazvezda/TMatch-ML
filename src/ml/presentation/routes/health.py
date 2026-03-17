from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ml.services.artifact_readiness import get_artifact_readiness_payload

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
def health_ready() -> JSONResponse:
    payload = get_artifact_readiness_payload()
    if payload["ready"]:
        return JSONResponse(status_code=200, content=payload)
    return JSONResponse(status_code=503, content=payload)
