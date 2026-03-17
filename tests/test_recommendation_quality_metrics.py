from __future__ import annotations

import math

from ml.services.recommendation_quality import (
    aggregate_model_quality_metrics,
    build_model_quality_frame,
)


def test_quality_metrics_are_computed_on_matured_rows() -> None:
    rows = [
        {
            "request_id": "r1",
            "viewer_id": "u1",
            "candidate_id": "c1",
            "model_version": "v1",
            "reranker_version": "v1",
            "pred_proba_like": 1.0,
            "pred_proba_skip": 0.0,
            "pred_proba_block": 0.0,
            "pred_proba_chat": 1.0,
            "true_label_like": 1,
            "true_label_skip": 0,
            "true_label_block": 0,
            "true_label_chat": 1,
            "label_matured": True,
        },
        {
            "request_id": "r2",
            "viewer_id": "u2",
            "candidate_id": "c2",
            "model_version": "v1",
            "reranker_version": "v1",
            "pred_proba_like": 0.2,
            "pred_proba_skip": 0.7,
            "pred_proba_block": 0.1,
            "pred_proba_chat": 0.4,
            "true_label_like": 0,
            "true_label_skip": 1,
            "true_label_block": 0,
            "true_label_chat": 0,
            "label_matured": True,
        },
    ]
    frame = build_model_quality_frame(rows=rows)
    metrics = aggregate_model_quality_metrics(frame)

    assert metrics["total_rows"] == 2
    assert metrics["matured_rows"] == 2
    assert metrics["logloss_like_count"] == 2
    assert metrics["logloss_skip_count"] == 2
    assert metrics["logloss_block_count"] == 2
    assert metrics["logloss_chat_count"] == 2
    assert metrics["mse_final_score_count"] == 2
    assert metrics["mse_final_score"] is not None

    assert frame["logloss_like"].notna().all()
    assert frame["logloss_skip"].notna().all()
    assert frame["logloss_block"].notna().all()
    assert frame["logloss_chat"].notna().all()
    assert math.isfinite(float(frame.loc[0, "logloss_like"]))
    assert math.isfinite(float(frame.loc[0, "logloss_skip"]))


def test_quality_metrics_handle_no_matured_rows() -> None:
    rows = [
        {
            "request_id": "r1",
            "viewer_id": "u1",
            "candidate_id": "c1",
            "pred_proba_like": 0.9,
            "pred_proba_skip": 0.1,
            "pred_proba_block": 0.2,
            "pred_proba_chat": 0.8,
            "label_matured": False,
        }
    ]
    frame = build_model_quality_frame(rows=rows)
    metrics = aggregate_model_quality_metrics(frame)

    assert metrics["total_rows"] == 1
    assert metrics["matured_rows"] == 0
    assert metrics["logloss_like"] is None
    assert metrics["logloss_skip"] is None
    assert metrics["logloss_block"] is None
    assert metrics["logloss_chat"] is None
    assert metrics["mse_final_score"] is None
    assert metrics["mse_final_score_count"] == 0


def test_quality_metrics_skip_rows_with_missing_true_labels() -> None:
    rows = [
        {
            "request_id": "r1",
            "viewer_id": "u1",
            "candidate_id": "c1",
            "pred_proba_like": 0.8,
            "pred_proba_skip": 0.2,
            "pred_proba_block": 0.1,
            "pred_proba_chat": 0.7,
            "true_label_like": 1,
            "true_label_skip": 0,
            "true_label_block": 0,
            "true_label_chat": 1,
            "label_matured": True,
        },
        {
            "request_id": "r2",
            "viewer_id": "u2",
            "candidate_id": "c2",
            "pred_proba_like": 0.3,
            "pred_proba_skip": 0.4,
            "pred_proba_block": 0.2,
            "pred_proba_chat": 0.2,
            "true_label_like": 0,
            "true_label_skip": 1,
            "true_label_block": 0,
            "true_label_chat": None,
            "label_matured": True,
        },
    ]
    frame = build_model_quality_frame(rows=rows)
    metrics = aggregate_model_quality_metrics(frame)

    assert metrics["logloss_like_count"] == 2
    assert metrics["logloss_skip_count"] == 2
    assert metrics["logloss_block_count"] == 2
    assert metrics["logloss_chat_count"] == 1
    assert metrics["mse_final_score_count"] == 1


def test_quality_metrics_handle_missing_heads_in_weight_config() -> None:
    rows = [
        {
            "request_id": "r1",
            "viewer_id": "u1",
            "candidate_id": "c1",
            "pred_proba_like": 0.8,
            "pred_proba_block": 0.1,
            "true_label_like": 1,
            "true_label_block": 0,
            "label_matured": True,
        }
    ]
    frame = build_model_quality_frame(
        rows=rows,
        weight_map={
            "like": 1.0,
            "block": -2.0,
        },
    )
    metrics = aggregate_model_quality_metrics(frame)

    assert metrics["mse_final_score_count"] == 1
    assert frame.loc[0, "true_score"] == 1.0
    assert abs(float(frame.loc[0, "pred_score"]) - 0.6) < 1e-9
