from collections.abc import Mapping

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ml.reranker import rerank
from ml.services.recommendation_identity import generate_recommendation_id
from ml.services.recommendation_runtime_config import (
    DEFAULT_RECOMMENDATION_RUNTIME_CONFIG,
)
from ml.services.recommendation_store import save_recommendation_row

router = APIRouter(prefix="/recommendation", tags=["recommendation"])


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


@router.post("")
async def recommendation(payload: RecommendationRequest) -> JSONResponse:
    conf, g_conf = DEFAULT_RECOMMENDATION_RUNTIME_CONFIG.build_configs(payload.user_id)

    try:
        result_df = await run_in_threadpool(rerank, conf, g_conf)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process request: {exc}") from exc

    recommendations = []
    for row in result_df.to_dict(orient="records"):
        candidate_user_id = _pick_first_not_none(
            row.get("candidate_id"),
            row.get("party_rk"),
            row.get("user_id"),
            row.get("target_user_id"),
            row.get("id"),
        )
        if candidate_user_id is None:
            continue

        reasons = _normalize_reasons(row.get("impactful_features"))
        recommendation_id = generate_recommendation_id()
        save_recommendation_row(
            recommendation_id=recommendation_id,
            user_id=str(payload.user_id),
            candidate_user_id=str(candidate_user_id),
            source_row=row,
        )

        recommendations.append(
            {
                "ml_recommendation_id": recommendation_id,
                "user_id": str(payload.user_id),
                "candidate_user_id": str(candidate_user_id),
                "reasons": reasons,
            }
        )

    return JSONResponse(status_code=200, content=recommendations)


def _pick_first_not_none(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and value != value:
            continue
        return value
    return None


def _normalize_reasons(raw_reasons) -> dict[str, float]:
    if raw_reasons is None:
        return {}

    if isinstance(raw_reasons, Mapping):
        normalized = {}
        for key, value in raw_reasons.items():
            try:
                normalized[str(key)] = float(value)
            except (TypeError, ValueError):
                normalized[str(key)] = 0.0
        return normalized

    if isinstance(raw_reasons, list):
        return {str(feature): 0.0 for feature in raw_reasons}

    return {str(raw_reasons): 0.0}
