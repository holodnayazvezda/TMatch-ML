from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ml.services.transaction_traits import build_transaction_trait_reasons_response

router = APIRouter(prefix="/user-traits", tags=["user-traits"])


class UserDescriptionRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


@router.post("")
async def user_traits(payload: UserDescriptionRequest) -> JSONResponse:
    try:
        response_payload = await run_in_threadpool(
            build_transaction_trait_reasons_response,
            payload.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process request: {exc}") from exc

    return JSONResponse(status_code=200, content=response_payload)
