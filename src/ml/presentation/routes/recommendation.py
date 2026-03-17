from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ml.explanation import map_feature_impacts_to_labels
from ml.config import settings
from ml.services.recommendation_identity import generate_recommendation_id
from ml.services.recommendation_runtime_config import (
    DEFAULT_RECOMMENDATION_RUNTIME_CONFIG,
)
from ml.services.recommendation_scoring import EnsembleWeights
from ml.services.recommendation_serving import get_recommendation_cycle_state, get_recommendations_for_user
from ml.services.recommendation_store import (
    run_with_user_recommendation_lock,
    save_recommendation_group_in_session,
)
from ml.services.user_id import normalize_user_id

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

            recommendation_rows.append(
                {
                    "candidate_user_id": normalized_candidate_id,
                    "source_row": _build_quality_source_row_payload(
                        source_row=row,
                        request_id=recommendation_id,
                        viewer_id=normalized_user_id,
                        candidate_id=normalized_candidate_id,
                        served_stage=served_stage,
                        ensemble_weights=DEFAULT_RECOMMENDATION_RUNTIME_CONFIG.ensemble_weights,
                    ),
                }
            )
            recommendations.append(
                {
                    "ml_recommendation_id": recommendation_id,
                    "user_id": normalized_user_id,
                    "candidate_user_id": normalized_candidate_id,
                    "reasons": map_feature_impacts_to_labels(row.get("impactful_features")),
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


def _build_quality_source_row_payload(
    *,
    source_row: dict[str, Any],
    request_id: str,
    viewer_id: str,
    candidate_id: str,
    served_stage: str,
    ensemble_weights: EnsembleWeights,
) -> dict[str, Any]:
    payload = dict(source_row)
    payload["request_id"] = request_id
    payload["viewer_id"] = viewer_id
    payload["candidate_id"] = candidate_id
    payload["served_stage"] = served_stage

    payload["pred_proba_like"] = _to_nullable_float(payload.get("y_pred_like"))
    payload["pred_proba_skip"] = _to_nullable_float(payload.get("y_pred_dislike"))
    payload["pred_proba_block"] = _to_nullable_float(payload.get("y_pred_block"))
    payload["pred_proba_chat"] = _to_nullable_float(payload.get("y_pred_match"))

    model_version = str(ensemble_weights.version)
    payload.setdefault("model_version", model_version)
    payload.setdefault("reranker_version", model_version)
    payload.setdefault("score_weights_version", model_version)
    payload.setdefault(
        "score_weights_json",
        {
            "w_like": float(ensemble_weights.w_like),
            "w_skip": float(ensemble_weights.w_dislike),
            "w_block": float(ensemble_weights.w_block),
            "w_chat": float(ensemble_weights.w_match),
        },
    )

    rerank_score = _to_nullable_float(payload.get("rerank_score"))
    if rerank_score is None:
        rerank_score = _compute_pred_score_from_probabilities(payload, ensemble_weights=ensemble_weights)
    payload["pred_score"] = rerank_score
    return payload


def _compute_pred_score_from_probabilities(
    source_row: dict[str, Any],
    *,
    ensemble_weights: EnsembleWeights,
) -> float | None:
    pred_like = _to_nullable_float(source_row.get("y_pred_like"))
    pred_skip = _to_nullable_float(source_row.get("y_pred_dislike"))
    pred_block = _to_nullable_float(source_row.get("y_pred_block"))
    pred_chat = _to_nullable_float(source_row.get("y_pred_match"))
    if pred_like is None and pred_skip is None and pred_block is None and pred_chat is None:
        return None

    return float(
        (float(pred_like or 0.0) * float(ensemble_weights.w_like))
        + (float(pred_chat or 0.0) * float(ensemble_weights.w_match))
        + (float(pred_skip or 0.0) * float(ensemble_weights.w_dislike))
        + (float(pred_block or 0.0) * float(ensemble_weights.w_block))
    )


def _to_nullable_float(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed
