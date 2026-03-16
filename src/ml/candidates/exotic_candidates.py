from __future__ import annotations

import numpy as np
import pandas as pd

from ml.candidates import transactions_candidates
from ml.services.user_id import normalize_user_id


def get_exotic_candidates(request_or_n_people, search_bundle=None):
    requested_count, actor_user_id = _resolve_exotic_request(request_or_n_people)
    if requested_count <= 0:
        return pd.DataFrame()

    search_bundle = transactions_candidates._get_search_bundle(search_bundle=search_bundle)
    output_df = search_bundle["output_df"]
    if output_df.empty:
        return output_df.copy()

    user_ids = search_bundle["user_ids"]
    cluster_member_indices = search_bundle["cluster_member_indices"]
    cluster_order = search_bundle["cluster_order"]
    cluster_centroids = search_bundle["cluster_centroids"]
    scaled_matrix = search_bundle["scaled_matrix"]
    id_index = search_bundle["id_index"]

    blocked_candidate_ids = _get_blocked_candidate_ids(
        actor_user_id=actor_user_id,
        user_ids=user_ids,
    )
    excluded_user_id = actor_user_id if actor_user_id in id_index else None

    prepared_clusters = _prepare_cluster_candidates(
        cluster_order=cluster_order,
        cluster_member_indices=cluster_member_indices,
        cluster_centroids=cluster_centroids,
        user_ids=user_ids,
        blocked_candidate_ids=blocked_candidate_ids,
        excluded_user_id=excluded_user_id,
    )
    if not prepared_clusters:
        return pd.DataFrame()

    selected_indices, selected_cluster_ids, centroid_distances = _round_robin_collect(
        prepared_clusters=prepared_clusters,
        requested_count=requested_count,
        scaled_matrix=scaled_matrix,
        cluster_centroids=cluster_centroids,
    )
    if not selected_indices:
        return pd.DataFrame()

    result_df = output_df.iloc[selected_indices].copy().reset_index(drop=True)
    impactful_feature_count = int(
        getattr(
            request_or_n_people,
            "an_impactful_features",
            transactions_candidates.CandidateConfig.an_impactful_features,
        )
        or 0
    )
    impactful_features = [
        transactions_candidates._extract_seeded_impactful_features(
            seed_value=f"{actor_user_id or 'anonymous'}:{str(user_ids[int(candidate_index)])}:exotic",
            top_k=impactful_feature_count,
        )
        for candidate_index in selected_indices
    ]
    result_df.insert(0, "rank", np.arange(1, len(result_df) + 1, dtype=np.int32))
    result_df.insert(1, "cluster_id", np.asarray(selected_cluster_ids, dtype=np.int32))
    result_df.insert(2, "distance_euclidean", np.asarray(centroid_distances, dtype=float))
    result_df.insert(3, "impactful_features", impactful_features)
    return result_df
 
def _resolve_exotic_request(request_or_n_people):
    if isinstance(request_or_n_people, (int, np.integer)):
        return max(0, int(request_or_n_people)), None

    if isinstance(request_or_n_people, str):
        normalized_user_id = normalize_user_id(request_or_n_people, allow_hash=True)
        if normalized_user_id is not None:
            return transactions_candidates.CandidateConfig.n_rec_users, normalized_user_id
        return max(0, int(request_or_n_people)), None

    requested_count = int(
        getattr(
            request_or_n_people,
            "top_n",
            transactions_candidates.CandidateConfig.n_rec_users,
        )
        or 0
    )
    actor_user_id = normalize_user_id(getattr(request_or_n_people, "user_id", None), allow_hash=True)
    return max(0, requested_count), actor_user_id


def _get_blocked_candidate_ids(actor_user_id, user_ids):
    if actor_user_id is None:
        return set()

    candidate_user_ids = [str(candidate_user_id) for candidate_user_id in user_ids]
    try:
        blocked_candidate_ids = transactions_candidates._INTERACTION_BLOCKLIST_SERVICE.get_blocked_targets(
            actor_user_id=actor_user_id,
            candidate_user_ids=candidate_user_ids,
        )
    except Exception:
        return set()

    return set(blocked_candidate_ids or [])


def _prepare_cluster_candidates(
    *,
    cluster_order,
    cluster_member_indices,
    cluster_centroids,
    user_ids,
    blocked_candidate_ids,
    excluded_user_id,
):
    prepared_clusters = []
    for cluster_id in cluster_order:
        cluster_id = int(cluster_id)
        member_indices = cluster_member_indices.get(cluster_id)
        if member_indices is None or len(member_indices) == 0:
            continue

        eligible_member_indices = [
            int(candidate_index)
            for candidate_index in member_indices
            if _is_candidate_allowed(
                candidate_user_id=str(user_ids[int(candidate_index)]),
                blocked_candidate_ids=blocked_candidate_ids,
                excluded_user_id=excluded_user_id,
            )
        ]
        if not eligible_member_indices:
            continue

        prepared_clusters.append(
            {
                "cluster_id": cluster_id,
                "candidate_indices": np.asarray(eligible_member_indices, dtype=np.int32),
                "cluster_centroid": cluster_centroids[cluster_id],
            }
        )

    return prepared_clusters


def _is_candidate_allowed(*, candidate_user_id, blocked_candidate_ids, excluded_user_id):
    if excluded_user_id is not None and candidate_user_id == excluded_user_id:
        return False
    return candidate_user_id not in blocked_candidate_ids


def _round_robin_collect(*, prepared_clusters, requested_count, scaled_matrix, cluster_centroids):
    selected_indices = []
    selected_cluster_ids = []
    centroid_distances = []
    seen_indices = set()
    cluster_offsets = [0] * len(prepared_clusters)

    while len(selected_indices) < requested_count:
        made_progress = False

        for position, cluster_payload in enumerate(prepared_clusters):
            candidate_indices = cluster_payload["candidate_indices"]
            cluster_id = int(cluster_payload["cluster_id"])
            cluster_centroid = cluster_centroids[cluster_id]
            offset = cluster_offsets[position]

            while offset < len(candidate_indices):
                candidate_index = int(candidate_indices[offset])
                offset += 1
                if candidate_index in seen_indices:
                    continue

                seen_indices.add(candidate_index)
                selected_indices.append(candidate_index)
                selected_cluster_ids.append(cluster_id)
                centroid_distances.append(
                    float(np.linalg.norm(scaled_matrix[candidate_index] - cluster_centroid))
                )
                made_progress = True
                break

            cluster_offsets[position] = offset
            if len(selected_indices) >= requested_count:
                break

        if not made_progress:
            break

    return selected_indices, selected_cluster_ids, centroid_distances
