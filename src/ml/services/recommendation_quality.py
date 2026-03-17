from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from ml.services.recommendation_runtime_config import DEFAULT_RECOMMENDATION_RUNTIME_CONFIG
from ml.services.recommendation_store import get_recommendation_quality_rows

QUALITY_EVENTS: tuple[str, ...] = ("like", "skip", "block", "chat")
PRED_COLUMN_BY_EVENT = {
    "like": "pred_proba_like",
    "skip": "pred_proba_skip",
    "block": "pred_proba_block",
    "chat": "pred_proba_chat",
}
TRUE_COLUMN_BY_EVENT = {
    "like": "true_label_like",
    "skip": "true_label_skip",
    "block": "true_label_block",
    "chat": "true_label_chat",
}
DEFAULT_LOGLOSS_EPS = 1e-15
_UNKNOWN_VERSION = "unknown"
_OUTPUT_ROW_COLUMNS = [
    "request_id",
    "viewer_id",
    "candidate_id",
    "model_version",
    "reranker_version",
    "pred_proba_like",
    "pred_proba_skip",
    "pred_proba_block",
    "pred_proba_chat",
    "pred_score",
    "true_label_like",
    "true_label_skip",
    "true_label_block",
    "true_label_chat",
    "true_score",
    "label_matured",
    "shown_at",
    "served_stage",
]


def get_model_quality_report(
    *,
    limit: int = 20_000,
    served_stages: Iterable[str] | None = None,
    include_pending: bool = True,
    include_rows: bool = False,
    rows_limit: int = 200,
    eps: float = DEFAULT_LOGLOSS_EPS,
) -> dict[str, Any]:
    rows = get_recommendation_quality_rows(
        limit=max(int(limit), 1),
        served_stages=list(served_stages or []),
        include_pending=bool(include_pending),
    )
    quality_frame = build_model_quality_frame(rows=rows, eps=eps)
    aggregates = aggregate_model_quality_metrics(quality_frame)

    payload: dict[str, Any] = {
        "status": "ok",
        "input_row_count": int(len(rows)),
        "include_pending": bool(include_pending),
        "served_stages": list(served_stages or []),
        "head_aliases": {
            "skip": "dislike",
            "chat": "match",
        },
        "weights": _default_weight_map(),
        "metrics": aggregates,
    }
    if include_rows:
        row_count = max(int(rows_limit), 0)
        payload["rows"] = quality_frame.head(row_count)[_OUTPUT_ROW_COLUMNS].to_dict(orient="records")
    return payload


def build_model_quality_frame(
    *,
    rows: list[dict[str, Any]],
    eps: float = DEFAULT_LOGLOSS_EPS,
    weight_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame = _ensure_quality_columns(frame)
    frame = _normalize_quality_columns(frame)
    weights = _resolve_weight_map(weight_map)
    enabled_events = [event_name for event_name in QUALITY_EVENTS if event_name in weights]

    if frame.empty:
        return frame

    # Per-head logloss on matured rows only.
    matured_mask = frame["label_matured"].fillna(False).astype(bool)
    for event_name in QUALITY_EVENTS:
        pred_column = PRED_COLUMN_BY_EVENT[event_name]
        true_column = TRUE_COLUMN_BY_EVENT[event_name]
        pred_values = pd.to_numeric(frame[pred_column], errors="coerce")
        true_values = pd.to_numeric(frame[true_column], errors="coerce")
        valid_mask = matured_mask & pred_values.notna() & true_values.notna()
        frame[f"logloss_{event_name}"] = np.nan
        if bool(valid_mask.any()):
            clipped_proba = pred_values.clip(lower=float(eps), upper=float(1.0 - eps))
            losses = -(
                true_values * np.log(clipped_proba)
                + (1.0 - true_values) * np.log(1.0 - clipped_proba)
            )
            frame.loc[valid_mask, f"logloss_{event_name}"] = losses.loc[valid_mask]

    # Predicted score from probabilities using the same serving weights.
    pred_score = pd.Series(np.zeros(frame.shape[0], dtype=float), index=frame.index, dtype=float)
    for event_name in enabled_events:
        pred_column = PRED_COLUMN_BY_EVENT[event_name]
        pred_score = pred_score + (
            pd.to_numeric(frame[pred_column], errors="coerce").fillna(0.0) * float(weights[event_name])
        )

    existing_pred_score = pd.to_numeric(frame["pred_score"], errors="coerce")
    frame["pred_score"] = np.where(existing_pred_score.notna(), existing_pred_score, pred_score).astype(float)

    # True score from matured true labels with the same weights.
    frame["true_score"] = np.nan
    if enabled_events:
        true_parts = {
            event_name: pd.to_numeric(frame[TRUE_COLUMN_BY_EVENT[event_name]], errors="coerce")
            for event_name in enabled_events
        }
        true_ready_mask = matured_mask.copy()
        true_score = pd.Series(np.zeros(frame.shape[0], dtype=float), index=frame.index, dtype=float)
        for event_name in enabled_events:
            values = true_parts[event_name]
            true_ready_mask = true_ready_mask & values.notna()
            true_score = true_score + (values.fillna(0.0) * float(weights[event_name]))
        frame.loc[true_ready_mask, "true_score"] = true_score.loc[true_ready_mask]

    return frame


def aggregate_model_quality_metrics(quality_frame: pd.DataFrame) -> dict[str, Any]:
    if quality_frame.empty:
        return {
            "total_rows": 0,
            "matured_rows": 0,
            "logloss_like": None,
            "logloss_like_count": 0,
            "logloss_skip": None,
            "logloss_skip_count": 0,
            "logloss_block": None,
            "logloss_block_count": 0,
            "logloss_chat": None,
            "logloss_chat_count": 0,
            "mse_final_score": None,
            "mse_final_score_count": 0,
        }

    result = {
        "total_rows": int(quality_frame.shape[0]),
        "matured_rows": int(quality_frame["label_matured"].fillna(False).astype(bool).sum()),
    }
    for event_name in QUALITY_EVENTS:
        loss_column = f"logloss_{event_name}"
        values = pd.to_numeric(quality_frame[loss_column], errors="coerce")
        valid_values = values[np.isfinite(values.to_numpy(dtype=float, copy=False))]
        result[f"logloss_{event_name}_count"] = int(valid_values.shape[0])
        result[f"logloss_{event_name}"] = (
            round(float(valid_values.mean()), 8) if not valid_values.empty else None
        )

    pred_score = pd.to_numeric(quality_frame["pred_score"], errors="coerce")
    true_score = pd.to_numeric(quality_frame["true_score"], errors="coerce")
    matured_mask = quality_frame["label_matured"].fillna(False).astype(bool)
    mse_mask = matured_mask & pred_score.notna() & true_score.notna()
    mse_count = int(mse_mask.sum())
    result["mse_final_score_count"] = mse_count
    if mse_count <= 0:
        result["mse_final_score"] = None
        return result

    mse_value = float(np.mean((pred_score.loc[mse_mask] - true_score.loc[mse_mask]) ** 2))
    result["mse_final_score"] = round(mse_value, 8)
    return result


def _ensure_quality_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column_name in _OUTPUT_ROW_COLUMNS:
        if column_name not in result.columns:
            result[column_name] = np.nan
    for event_name in QUALITY_EVENTS:
        loss_column = f"logloss_{event_name}"
        if loss_column not in result.columns:
            result[loss_column] = np.nan
    return result


def _normalize_quality_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    text_columns = (
        "request_id",
        "viewer_id",
        "candidate_id",
        "model_version",
        "reranker_version",
        "shown_at",
        "served_stage",
    )
    for column_name in text_columns:
        result[column_name] = result[column_name].map(_normalize_optional_text)

    for event_name in QUALITY_EVENTS:
        result[PRED_COLUMN_BY_EVENT[event_name]] = pd.to_numeric(
            result[PRED_COLUMN_BY_EVENT[event_name]],
            errors="coerce",
        )
        result[TRUE_COLUMN_BY_EVENT[event_name]] = pd.to_numeric(
            result[TRUE_COLUMN_BY_EVENT[event_name]],
            errors="coerce",
        )

    result["pred_score"] = pd.to_numeric(result["pred_score"], errors="coerce")
    result["true_score"] = pd.to_numeric(result["true_score"], errors="coerce")
    result["label_matured"] = result["label_matured"].fillna(False).astype(bool)
    result["model_version"] = result["model_version"].fillna(_UNKNOWN_VERSION)
    result["reranker_version"] = result["reranker_version"].fillna(_UNKNOWN_VERSION)
    return result


def _resolve_weight_map(weight_map: dict[str, float] | None) -> dict[str, float]:
    if weight_map is not None:
        return {
            str(head_name).strip().lower(): float(weight_value)
            for head_name, weight_value in dict(weight_map).items()
            if str(head_name).strip()
        }
    return _default_weight_map()


def _default_weight_map() -> dict[str, float]:
    weights = DEFAULT_RECOMMENDATION_RUNTIME_CONFIG.ensemble_weights
    return {
        "like": float(getattr(weights, "w_like", 0.0)),
        "skip": float(getattr(weights, "w_dislike", 0.0)),
        "block": float(getattr(weights, "w_block", 0.0)),
        "chat": float(getattr(weights, "w_match", 0.0)),
    }


def _normalize_optional_text(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    text_value = str(raw_value).strip()
    if not text_value:
        return None
    return text_value
