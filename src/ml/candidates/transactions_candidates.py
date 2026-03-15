from dataclasses import dataclass
from threading import Lock

import faiss
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from core.data_config import Paths
from ml.infra.blocked_service import InteractionBlocklistService
from ml.services.user_id import normalize_user_id

np.random.seed(42)

DATA_PATH = Paths.out_data_path / "user_features_all_1000.csv"
ID_COLUMN = "party_rk"
DATE_COLUMNS = ["first_txn_dttm", "last_txn_dttm"]
TOP_CATEGORY_COLUMN = "top_cat_from_share"
TOP_CATEGORY_SHARE_COLUMN = "top_cat_share"
TOP_MCC_COLUMN = "top_mcc_from_share"
TOP_MCC_SHARE_COLUMN = "top_mcc_share"

_SEARCH_LOCK = Lock()
_SEARCH_BUNDLE = None
_INTERACTION_BLOCKLIST_SERVICE = InteractionBlocklistService()


@dataclass(frozen=True)
class CandidateConfig:
    n_rec_users: int = 5
    n_min_candidates: int = 100
    topk_trans_sim: int = 150
    topk_inter_sim: int = 150
    trans_inter_rate: float = 0.5
    an_impactful_features: int = 3


def infer_load_config(file_path, sample_rows=2000):
    sample_df = pd.read_csv(file_path, nrows=sample_rows)

    date_columns = [column for column in DATE_COLUMNS if column in sample_df.columns]
    categorical_columns = [
        column
        for column in sample_df.columns
        if column not in [ID_COLUMN, *date_columns]
        and not pd.api.types.is_numeric_dtype(sample_df[column])
    ]

    dtype_map = {ID_COLUMN: "string"}
    for column in sample_df.columns:
        if column in [ID_COLUMN, *date_columns, *categorical_columns]:
            continue

        if pd.api.types.is_integer_dtype(sample_df[column]):
            dtype_map[column] = "Int32"
        elif pd.api.types.is_float_dtype(sample_df[column]):
            dtype_map[column] = "float32"

    for column in categorical_columns:
        dtype_map[column] = "string"

    return dtype_map, date_columns, categorical_columns


def load_dataset(file_path):
    dtype_map, date_columns, categorical_columns = infer_load_config(file_path)

    df = pd.read_csv(
        file_path,
        dtype=dtype_map,
        parse_dates=date_columns,
        low_memory=False,
    )

    for column in categorical_columns:
        df[column] = df[column].astype("string")

    df[ID_COLUMN] = df[ID_COLUMN].map(lambda raw_id: normalize_user_id(raw_id, allow_hash=True)).astype("string")
    df = df[df[ID_COLUMN].notna()].copy()

    return df


def add_date_features(df):
    result = df.copy()
    available_date_columns = [column for column in DATE_COLUMNS if column in result.columns]

    if not available_date_columns:
        return result

    reference_date = max(result[column].max() for column in available_date_columns)

    if "first_txn_dttm" in result.columns:
        result["days_since_first_txn"] = (
            reference_date - result["first_txn_dttm"]
        ).dt.days.astype("float32")

    if "last_txn_dttm" in result.columns:
        result["days_since_last_txn"] = (
            reference_date - result["last_txn_dttm"]
        ).dt.days.astype("float32")

    if {"first_txn_dttm", "last_txn_dttm"}.issubset(result.columns):
        result["activity_length_days"] = (
            result["last_txn_dttm"] - result["first_txn_dttm"]
        ).dt.days.astype("float32")

    return result


def decode_top_share_feature(df, prefix, label_column, share_column):
    result = df.copy()
    share_columns = [column for column in result.columns if column.startswith(prefix)]

    if not share_columns:
        return result

    share_frame = result[share_columns].fillna(0.0).astype("float32")
    share_matrix = share_frame.to_numpy()
    max_positions = share_matrix.argmax(axis=1)
    max_values = share_matrix[np.arange(len(result)), max_positions]
    share_sums = share_matrix.sum(axis=1)

    labels = [share_columns[position].replace(prefix, "") for position in max_positions]
    labels = np.where(share_sums > 0, labels, "не_определено")
    max_values = np.where(share_sums > 0, max_values, np.nan)

    result[label_column] = labels
    result[share_column] = max_values
    return result


def add_interpretable_share_columns(df):
    result = df.copy()
    result = decode_top_share_feature(
        result,
        prefix="cat_share_",
        label_column=TOP_CATEGORY_COLUMN,
        share_column=TOP_CATEGORY_SHARE_COLUMN,
    )
    result = decode_top_share_feature(
        result,
        prefix="mcc_share_",
        label_column=TOP_MCC_COLUMN,
        share_column=TOP_MCC_SHARE_COLUMN,
    )
    return result


def prepare_search_artifacts(df):
    enriched_df = add_date_features(df)
    enriched_df = enriched_df[enriched_df[ID_COLUMN].notna()].copy().reset_index(drop=True)
    output_df = add_interpretable_share_columns(enriched_df).reset_index(drop=True)

    available_date_columns = [column for column in DATE_COLUMNS if column in enriched_df.columns]
    model_df = enriched_df.drop(columns=available_date_columns).copy()

    categorical_columns = []
    for column in model_df.columns:
        if column == ID_COLUMN:
            continue

        is_object = pd.api.types.is_object_dtype(model_df[column])
        is_string = pd.api.types.is_string_dtype(model_df[column])
        is_category = isinstance(model_df[column].dtype, pd.CategoricalDtype)

        if is_object or is_string or is_category:
            categorical_columns.append(column)

    encoded_df = pd.get_dummies(
        model_df,
        columns=categorical_columns,
        dummy_na=False,
        dtype=np.float32,
    )

    feature_df = encoded_df.drop(columns=[ID_COLUMN]).apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    imputed_matrix = imputer.fit_transform(feature_df).astype("float32")

    scaler = RobustScaler()
    scaled_matrix = scaler.fit_transform(imputed_matrix).astype("float32")

    return {
        "output_df": output_df,
        "feature_df": feature_df,
        "feature_names": feature_df.columns.to_numpy(dtype=object),
        "imputer": imputer,
        "scaler": scaler,
        "scaled_matrix": np.ascontiguousarray(scaled_matrix),
        "user_ids": output_df[ID_COLUMN].astype("string").to_numpy(),
        "id_index": pd.Index(output_df[ID_COLUMN].astype("string"), name=ID_COLUMN),
    }


def build_faiss_index(scaled_matrix, config):
    feature_count = scaled_matrix.shape[1]
    index = faiss.IndexHNSWFlat(feature_count, 32, faiss.METRIC_L2)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = max(config.topk_trans_sim * 2, config.n_min_candidates, 200)
    index.add(np.ascontiguousarray(scaled_matrix))
    return index


def _to_candidate_config(config_like=None):
    if isinstance(config_like, CandidateConfig):
        return config_like

    return CandidateConfig(
        n_rec_users=int(getattr(config_like, "top_n", CandidateConfig.n_rec_users)),
        n_min_candidates=int(getattr(config_like, "n_min_candidates", CandidateConfig.n_min_candidates)),
        topk_trans_sim=int(getattr(config_like, "topk_trans_sim", CandidateConfig.topk_trans_sim)),
        topk_inter_sim=int(getattr(config_like, "topk_inter_sim", CandidateConfig.topk_inter_sim)),
        trans_inter_rate=float(getattr(config_like, "trans_inter_rate", CandidateConfig.trans_inter_rate)),
        an_impactful_features=int(
            getattr(config_like, "an_impactful_features", CandidateConfig.an_impactful_features)
        ),
    )


def _resolve_user_id(user_or_conf):
    if isinstance(user_or_conf, str):
        normalized_user_id = normalize_user_id(user_or_conf, allow_hash=True)
        if normalized_user_id is not None:
            return normalized_user_id
        raw_user_id = str(user_or_conf).strip()
        if raw_user_id:
            return raw_user_id

    if hasattr(user_or_conf, "user_id"):
        raw_candidate_user_id = getattr(user_or_conf, "user_id", None)
        normalized_user_id = normalize_user_id(raw_candidate_user_id, allow_hash=True)
        if normalized_user_id is not None:
            return normalized_user_id
        raw_user_id = str(raw_candidate_user_id).strip() if raw_candidate_user_id is not None else ""
        if raw_user_id:
            return raw_user_id

    raise ValueError("Нужно передать непустой user_id.")


def _get_search_bundle(search_bundle=None, config=None):
    if search_bundle is not None:
        if "faiss_index" not in search_bundle:
            search_bundle["faiss_index"] = build_faiss_index(
                search_bundle["scaled_matrix"],
                config or CandidateConfig(),
            )
        return search_bundle

    global _SEARCH_BUNDLE
    if _SEARCH_BUNDLE is not None:
        return _SEARCH_BUNDLE

    with _SEARCH_LOCK:
        if _SEARCH_BUNDLE is None:
            df = load_dataset(DATA_PATH)
            bundle = prepare_search_artifacts(df)
            bundle["faiss_index"] = build_faiss_index(
                bundle["scaled_matrix"],
                config or CandidateConfig(),
            )
            _SEARCH_BUNDLE = bundle

    return _SEARCH_BUNDLE


def warmup_search_bundle(config=None):
    """Build and cache search artifacts + FAISS index once per process."""
    effective_config = _to_candidate_config(config)
    return _get_search_bundle(search_bundle=None, config=effective_config)


def _extract_impactful_features(query_vector, candidate_vector, feature_names, top_k):
    if top_k <= 0:
        return {}

    squared_diff = np.square(candidate_vector - query_vector)
    total = float(squared_diff.sum())
    if total <= 0:
        return {}

    contribution_shares = squared_diff / total
    top_indices = np.argsort(contribution_shares)[::-1][:top_k]
    top_feature_names = [
        str(feature_names[index])
        for index in top_indices
        if contribution_shares[index] > 0
    ]
    if not top_feature_names:
        return {}

    # Temporary placeholder for impact scoring until business formula is defined.
    random_impacts = np.random.uniform(0.4, 0.95, size=len(top_feature_names))
    return {
        feature_name: round(float(impact), 4)
        for feature_name, impact in zip(top_feature_names, random_impacts)
    }


def _extract_random_impactful_features(feature_names, top_k):
    if top_k <= 0:
        return {}
    if feature_names is None or len(feature_names) == 0:
        return {}

    sample_size = min(int(top_k), len(feature_names))
    sampled_indices = np.random.choice(len(feature_names), size=sample_size, replace=False)
    random_impacts = np.random.uniform(0.4, 0.95, size=sample_size)

    return {
        str(feature_names[index]): round(float(impact), 4)
        for index, impact in zip(sampled_indices, random_impacts)
    }


def get_transaction_candidates(user_id, config=CandidateConfig(), search_bundle=None):
    request_object = user_id
    user_id = _resolve_user_id(request_object)
    effective_config = _to_candidate_config(request_object if hasattr(request_object, "user_id") else config)

    search_bundle = _get_search_bundle(search_bundle=search_bundle, config=effective_config)
    id_index = search_bundle["id_index"]
    user_ids = search_bundle["user_ids"]
    scaled_matrix = search_bundle["scaled_matrix"]
    feature_names = search_bundle["feature_names"]

    fallback_random_mode = False

    if user_id not in id_index:
        if len(scaled_matrix) <= 1:
            return None
        fallback_random_mode = True
        query_position = int(np.random.randint(0, len(scaled_matrix)))
        query_user_id = str(user_ids[query_position])
    else:
        query_position = int(id_index.get_loc(user_id))
        query_user_id = user_id

    if fallback_random_mode:
        candidate_pool = np.delete(np.arange(len(scaled_matrix), dtype=np.int32), query_position)
        if candidate_pool.size == 0:
            return None

        sample_size = min(effective_config.n_rec_users, int(candidate_pool.size))
        top_indices = np.random.choice(candidate_pool, size=sample_size, replace=False).astype(np.int32)
        top_distances = np.random.uniform(0.1, 2.0, size=sample_size).astype(float)
        impactful_features = [
            _extract_random_impactful_features(
                feature_names=feature_names,
                top_k=effective_config.an_impactful_features,
            )
            for _ in top_indices
        ]
    else:
        index = search_bundle["faiss_index"]
        query_vector = scaled_matrix[query_position : query_position + 1]

        candidate_count = min(
            len(scaled_matrix),
            max(
                effective_config.topk_trans_sim,
                effective_config.n_min_candidates,
                effective_config.n_rec_users + 1,
            ),
        )

        _, raw_candidate_indices = index.search(query_vector, candidate_count + 1)
        raw_candidate_indices = raw_candidate_indices[0]

        unique_candidate_indices = []
        seen_indices = set()
        for candidate_index in raw_candidate_indices:
            candidate_index = int(candidate_index)
            if candidate_index < 0 or candidate_index == query_position:
                continue
            if candidate_index in seen_indices:
                continue
            seen_indices.add(candidate_index)
            unique_candidate_indices.append(candidate_index)

        if not unique_candidate_indices:
            return None

        candidate_indices = np.asarray(unique_candidate_indices, dtype=np.int32)
        exact_distances = np.linalg.norm(
            scaled_matrix[candidate_indices] - query_vector[0],
            axis=1,
        )

        filtered_indices, filtered_distances = _filter_blocked_candidates(
            actor_user_id=user_id,
            candidate_indices=candidate_indices,
            exact_distances=exact_distances,
            search_bundle=search_bundle,
        )
        if filtered_indices.size == 0:
            return None

        top_indices = filtered_indices[: effective_config.n_rec_users]
        top_distances = filtered_distances[: effective_config.n_rec_users]

        impactful_features = [
            _extract_impactful_features(
                query_vector=scaled_matrix[query_position],
                candidate_vector=scaled_matrix[candidate_index],
                feature_names=feature_names,
                top_k=effective_config.an_impactful_features,
            )
            for candidate_index in top_indices
        ]

    result_df = search_bundle["output_df"].iloc[top_indices].copy().reset_index(drop=True)
    result_df.insert(0, "rank", np.arange(1, len(result_df) + 1))
    result_df.insert(1, "distance_euclidean", top_distances.astype(float))
    result_df.insert(2, "query_party_rk", query_user_id)
    result_df.insert(3, "impactful_features", impactful_features)

    return result_df


def _filter_blocked_candidates(actor_user_id, candidate_indices, exact_distances, search_bundle):
    ranked_order = np.argsort(exact_distances)
    ranked_indices = candidate_indices[ranked_order]
    ranked_distances = exact_distances[ranked_order]

    candidate_user_ids = [
        str(search_bundle["user_ids"][candidate_index])
        for candidate_index in ranked_indices
    ]
    try:
        blocked_candidate_ids = _INTERACTION_BLOCKLIST_SERVICE.get_blocked_targets(
            actor_user_id=actor_user_id,
            candidate_user_ids=candidate_user_ids,
        )
    except Exception:
        # Recommendation serving should continue even if blocklist storage is unavailable.
        return ranked_indices, ranked_distances
    if not blocked_candidate_ids:
        return ranked_indices, ranked_distances

    allowed_positions = [
        position
        for position, candidate_user_id in enumerate(candidate_user_ids)
        if candidate_user_id not in blocked_candidate_ids
    ]
    if not allowed_positions:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=float)

    return ranked_indices[allowed_positions], ranked_distances[allowed_positions]
