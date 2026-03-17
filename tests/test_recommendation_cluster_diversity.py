from __future__ import annotations

import pandas as pd

from ml.services import recommendation_serving

U1 = "00000000-0000-0000-0000-000000000001"
U2 = "00000000-0000-0000-0000-000000000002"
U3 = "00000000-0000-0000-0000-000000000003"
U4 = "00000000-0000-0000-0000-000000000004"
U5 = "00000000-0000-0000-0000-000000000005"


def test_select_cluster_diverse_candidates_prefers_distinct_clusters_then_fills() -> None:
    candidate_df = pd.DataFrame(
        [
            {"candidate_id": U1, "rerank_score": 0.90, "transaction_cluster_id": "0"},
            {"candidate_id": U2, "rerank_score": 0.85, "transaction_cluster_id": "0"},
            {"candidate_id": U3, "rerank_score": 0.80, "transaction_cluster_id": "1"},
            {"candidate_id": U4, "rerank_score": 0.70, "transaction_cluster_id": "2"},
            {"candidate_id": U5, "rerank_score": 0.60, "transaction_cluster_id": "2"},
        ]
    )

    selected_df = recommendation_serving._select_cluster_diverse_candidates(candidate_df, top_k=4)
    assert selected_df["candidate_id"].tolist() == [U1, U3, U4, U2]


def test_select_cluster_diverse_candidates_uses_unknown_bucket_for_missing_cluster() -> None:
    candidate_df = pd.DataFrame(
        [
            {"candidate_id": U1, "rerank_score": 0.90, "transaction_cluster_id": None},
            {"candidate_id": U2, "rerank_score": 0.80, "transaction_cluster_id": None},
            {"candidate_id": U3, "rerank_score": 0.70, "transaction_cluster_id": "1"},
            {"candidate_id": U4, "rerank_score": 0.60, "transaction_cluster_id": "2"},
        ]
    )

    selected_df = recommendation_serving._select_cluster_diverse_candidates(candidate_df, top_k=3)
    assert selected_df["candidate_id"].tolist() == [U1, U3, U4]
    assert selected_df["transaction_cluster_id"].tolist()[0] == "unknown"


def test_select_cluster_diverse_candidates_handles_empty_and_top_k_zero() -> None:
    empty_df = pd.DataFrame(columns=["candidate_id", "rerank_score", "transaction_cluster_id"])
    assert recommendation_serving._select_cluster_diverse_candidates(empty_df, top_k=5).empty

    non_empty_df = pd.DataFrame([{"candidate_id": U1, "rerank_score": 0.5, "transaction_cluster_id": "0"}])
    selected_df = recommendation_serving._select_cluster_diverse_candidates(non_empty_df, top_k=0)
    assert selected_df.empty
