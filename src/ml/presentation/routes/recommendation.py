from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import structlog
from ml.config import settings
from ml.services.recommendation_identity import generate_recommendation_id
from ml.services.recommendation_runtime_config import (
    DEFAULT_RECOMMENDATION_RUNTIME_CONFIG,
)
from ml.services.recommendation_serving import get_recommendation_cycle_state, get_recommendations_for_user
from ml.services.recommendation_store import (
    reset_recommendation_cycle,
    run_with_user_recommendation_lock,
    save_recommendation_group_in_session,
)
from ml.services.user_id import normalize_user_id

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/recommendation", tags=["recommendation"])


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


@router.post("")
async def recommendation(payload: RecommendationRequest) -> JSONResponse:
    try:
        recommendations = await run_in_threadpool(_generate_recommendations_response, payload.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("recommendation_request_failed")
        raise HTTPException(status_code=500, detail=f"Failed to process request: {exc}") from exc

    return JSONResponse(status_code=200, content=recommendations)


def _generate_recommendations_response(raw_user_id: str) -> list[dict[str, object]]:
    request_user_id = str(raw_user_id).strip()
    normalized_user_id = normalize_user_id(request_user_id, allow_hash=True)
    if normalized_user_id is None:
        raise ValueError("Invalid user_id.")

    conf, g_conf = DEFAULT_RECOMMENDATION_RUNTIME_CONFIG.build_configs(normalized_user_id)
    served_stage = DEFAULT_RECOMMENDATION_RUNTIME_CONFIG.recommendation_stage.value

    def _locked_flow(session: Session) -> list[dict[str, object]]:
        result_df = get_recommendations_for_user(
            user_id=normalized_user_id,
            runtime_config=DEFAULT_RECOMMENDATION_RUNTIME_CONFIG,
            conf=conf,
            g_conf=g_conf,
        )
        if result_df.empty:
            cycle_state = get_recommendation_cycle_state(
                actor_user_id=normalized_user_id,
                stage=DEFAULT_RECOMMENDATION_RUNTIME_CONFIG.recommendation_stage,
            )
            if bool(cycle_state.get("is_exhausted", False)):
                cycle_started_at = reset_recommendation_cycle(normalized_user_id)
                log.info(
                    "recommendation_cycle_reset",
                    actor_user_id=normalized_user_id,
                    cycle_started_at=cycle_started_at.isoformat(),
                    reason="eligible_pool_exhausted",
                    total_pool_count=int(cycle_state.get("total_pool_count", 0)),
                    eligible_pool_count=int(cycle_state.get("eligible_pool_count", 0)),
                    shown_eligible_count=int(cycle_state.get("shown_eligible_count", 0)),
                )
                result_df = get_recommendations_for_user(
                    user_id=normalized_user_id,
                    runtime_config=DEFAULT_RECOMMENDATION_RUNTIME_CONFIG,
                    conf=conf,
                    g_conf=g_conf,
                )

        recommendation_id = generate_recommendation_id()
        shown_at = datetime.now(timezone.utc)

        recommendations: list[dict[str, object]] = []
        recommendation_rows: list[dict[str, object]] = []
        seen_candidate_ids: set[str] = set()
        for row in result_df.to_dict(orient="records"):
            candidate_user_id = _pick_first_not_none(
                row.get("candidate_id"),
                row.get("party_rk"),
                row.get("user_id"),
                row.get("target_user_id"),
                row.get("id"),
            )
            normalized_candidate_id = normalize_user_id(candidate_user_id, allow_hash=True)
            if normalized_candidate_id is None:
                continue
            if normalized_candidate_id in seen_candidate_ids:
                continue
            seen_candidate_ids.add(normalized_candidate_id)

            reasons = row.get("reasons")
            if not isinstance(reasons, dict) or len(reasons) != 3:
                continue

            recommendation_rows.append(
                {
                    "candidate_user_id": normalized_candidate_id,
                    "source_row": row,
                }
            )
            recommendations.append(
                {
                    "ml_recommendation_id": recommendation_id,
                    "user_id": normalized_user_id,
                    "candidate_user_id": normalized_candidate_id,
                    "reasons": reasons,
                }
            )

        if recommendation_rows:
            with session.begin():
                save_recommendation_group_in_session(
                    session,
                    recommendation_id=recommendation_id,
                    actor_user_id=normalized_user_id,
                    recommendation_rows=recommendation_rows,
                    shown_at=shown_at,
                    served_stage=served_stage,
                )

        return recommendations

    return run_with_user_recommendation_lock(
        normalized_user_id,
        poll_wait_seconds=float(settings.interaction_poll_interval_seconds),
        callback=_locked_flow,
    )


def _pick_first_not_none(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and value != value:
            continue
        return value
    return None
