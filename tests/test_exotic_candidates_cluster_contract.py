from __future__ import annotations

import numpy as np
import pandas as pd

from ml.candidates.exotic_candidates import get_exotic_candidates


def test_exotic_candidates_return_candidate_id_and_cluster_id() -> None:
    user_ids = np.asarray(
        [
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
            "00000000-0000-0000-0000-000000000003",
            "00000000-0000-0000-0000-000000000004",
        ],
        dtype=object,
    )
    search_bundle = {
        "faiss_index": object(),
        "output_df": pd.DataFrame(
            {
                "party_rk": pd.Series(user_ids, dtype="string"),
                "feature_x": [0.1, 0.2, 0.3, 0.4],
            }
        ),
        "user_ids": user_ids,
        "cluster_member_indices": {
            0: np.asarray([0, 2], dtype=np.int32),
            1: np.asarray([1, 3], dtype=np.int32),
        },
        "cluster_order": np.asarray([0, 1], dtype=np.int32),
        "cluster_centroids": np.asarray(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "scaled_matrix": np.asarray(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [0.2, 0.2],
                [1.2, 1.2],
            ],
            dtype=np.float32,
        ),
        "id_index": pd.Index(user_ids, name="party_rk"),
        "cluster_count": 2,
        "cluster_assignments": np.asarray([0, 1, 0, 1], dtype=np.int32),
    }

    result_df = get_exotic_candidates(3, search_bundle=search_bundle)
    assert not result_df.empty
    assert "candidate_id" in result_df.columns
    assert "cluster_id" in result_df.columns
    assert result_df["candidate_id"].astype("string").notna().all()
    assert (result_df["candidate_id"].astype("string") != "").all()
