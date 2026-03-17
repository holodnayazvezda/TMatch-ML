from __future__ import annotations

import pandas as pd

from ml.ml_config import RerankConfig
from ml.services import recommendation_serving
from ml.services.recommendation_scoring import RecommendationStage

ACTOR_ID = "11111111-1111-1111-1111-111111111111"
U1 = "00000000-0000-0000-0000-000000000001"
U2 = "00000000-0000-0000-0000-000000000002"
U3 = "00000000-0000-0000-0000-000000000003"
U4 = "00000000-0000-0000-0000-000000000004"
U5 = "00000000-0000-0000-0000-000000000005"
U6 = "00000000-0000-0000-0000-000000000006"
P1 = "00000000-0000-0000-0000-000000000011"
P2 = "00000000-0000-0000-0000-000000000012"
P3 = "00000000-0000-0000-0000-000000000013"
P4 = "00000000-0000-0000-0000-000000000014"


def _build_conf() -> RerankConfig:
    return RerankConfig(
        user_id=ACTOR_ID,
        top_n=50,
        uncertainty_mean_score_threshold=0.0,
        uncertainty_gap_threshold=0.05,
    )


def test_uncertainty_s1_returns_standard_top_k() -> None:
    candidate_df = pd.DataFrame(
        [
            {"candidate_id": U1, "rerank_score": 0.80, "rank": 1, "transaction_cluster_id": "0"},
            {"candidate_id": U2, "rerank_score": 0.70, "rank": 2, "transaction_cluster_id": "0"},
            {"candidate_id": U3, "rerank_score": 0.60, "rank": 3, "transaction_cluster_id": "1"},
            {"candidate_id": U4, "rerank_score": 0.50, "rank": 4, "transaction_cluster_id": "2"},
            {"candidate_id": U5, "rerank_score": 0.40, "rank": 5, "transaction_cluster_id": "3"},
            {"candidate_id": U6, "rerank_score": 0.10, "rank": 6, "transaction_cluster_id": "4"},
        ]
    )

    result_df = recommendation_serving._apply_uncertainty_strategy(
        candidate_df=candidate_df,
        stage=RecommendationStage.STAGE_4,
        actor_user_id=ACTOR_ID,
        conf=_build_conf(),
        top_n=3,
    )
    assert result_df["candidate_id"].tolist() == [U1, U2, U3]


def test_uncertainty_s2_uses_top15_diversity_then_fill() -> None:
    candidate_df = pd.DataFrame(
        [
            {"candidate_id": U1, "rerank_score": 0.20, "rank": 1, "transaction_cluster_id": "0"},
            {"candidate_id": U2, "rerank_score": 0.20, "rank": 2, "transaction_cluster_id": "0"},
            {"candidate_id": U3, "rerank_score": 0.20, "rank": 3, "transaction_cluster_id": "0"},
            {"candidate_id": U4, "rerank_score": 0.19, "rank": 4, "transaction_cluster_id": "1"},
            {"candidate_id": U5, "rerank_score": 0.19, "rank": 5, "transaction_cluster_id": "2"},
            {"candidate_id": U6, "rerank_score": 0.18, "rank": 6, "transaction_cluster_id": "3"},
        ]
    )

    result_df = recommendation_serving._apply_uncertainty_strategy(
        candidate_df=candidate_df,
        stage=RecommendationStage.STAGE_4,
        actor_user_id=ACTOR_ID,
        conf=_build_conf(),
        top_n=4,
    )
    assert result_df["candidate_id"].tolist() == [U1, U4, U5, U6]


def test_uncertainty_s3_uses_global_diversity_then_fill() -> None:
    candidate_df = pd.DataFrame(
        [
            {"candidate_id": U1, "rerank_score": 0.90, "rank": 1, "transaction_cluster_id": "0"},
            {"candidate_id": U2, "rerank_score": 0.80, "rank": 2, "transaction_cluster_id": "0"},
            {"candidate_id": U3, "rerank_score": 0.70, "rank": 3, "transaction_cluster_id": "1"},
            {"candidate_id": U4, "rerank_score": 0.60, "rank": 4, "transaction_cluster_id": "2"},
            {"candidate_id": U5, "rerank_score": 0.50, "rank": 5, "transaction_cluster_id": "3"},
            {"candidate_id": U6, "rerank_score": -4.0, "rank": 6, "transaction_cluster_id": "4"},
        ]
    )

    result_df = recommendation_serving._apply_uncertainty_strategy(
        candidate_df=candidate_df,
        stage=RecommendationStage.STAGE_4,
        actor_user_id=ACTOR_ID,
        conf=_build_conf(),
        top_n=4,
    )
    assert result_df["candidate_id"].tolist() == [U1, U3, U4, U5]


def test_uncertainty_s4_uses_popularity_with_diversity(monkeypatch) -> None:
    candidate_df = pd.DataFrame(
        [
            {"candidate_id": U1, "rerank_score": -0.20, "rank": 1, "transaction_cluster_id": "0"},
            {"candidate_id": U2, "rerank_score": -0.20, "rank": 2, "transaction_cluster_id": "1"},
            {"candidate_id": U3, "rerank_score": -0.20, "rank": 3, "transaction_cluster_id": "2"},
            {"candidate_id": U4, "rerank_score": -0.20, "rank": 4, "transaction_cluster_id": "3"},
            {"candidate_id": U5, "rerank_score": -0.20, "rank": 5, "transaction_cluster_id": "4"},
            {"candidate_id": U6, "rerank_score": -0.20, "rank": 6, "transaction_cluster_id": "5"},
        ]
    )
    popularity_df = pd.DataFrame(
        [
            {"candidate_id": P1, "popularity_score": 10.0, "rank": 1, "transaction_cluster_id": "0"},
            {"candidate_id": P2, "popularity_score": 9.0, "rank": 2, "transaction_cluster_id": "0"},
            {"candidate_id": P3, "popularity_score": 8.0, "rank": 3, "transaction_cluster_id": "1"},
            {"candidate_id": P4, "popularity_score": 7.0, "rank": 4, "transaction_cluster_id": "2"},
        ]
    )

    monkeypatch.setattr(recommendation_serving, "_safe_fetch", lambda fetcher, conf: popularity_df.copy())
    monkeypatch.setattr(
        recommendation_serving,
        "attach_feature_snapshots",
        lambda source_df, actor_user_id, stage: source_df.copy(),
    )
    monkeypatch.setattr(
        recommendation_serving,
        "_apply_hard_filters",
        lambda source_df, actor_user_id, targeting_profile_cache=None: source_df.copy(),
    )

    result_df = recommendation_serving._apply_uncertainty_strategy(
        candidate_df=candidate_df,
        stage=RecommendationStage.STAGE_4,
        actor_user_id=ACTOR_ID,
        conf=_build_conf(),
        top_n=3,
    )
    assert result_df["candidate_id"].tolist() == [P1, P3, P4]
