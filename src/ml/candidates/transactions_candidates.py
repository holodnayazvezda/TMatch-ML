from pathlib import Path
from dataclasses import dataclass
from threading import Lock

import faiss
import numpy as np
import pandas as pd
import structlog
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from core.data_config import Paths
from ml.explanation import get_reason_mapper_values
from ml.infra.blocked_service import InteractionBlocklistService
from ml.services.user_id import normalize_user_id

DATA_PARQUET_PATH = Paths.transaction_user_features_parquet_path
DATA_CSV_PATH = Paths.transaction_user_features_csv_path
ID_COLUMN = "party_rk"
DATE_COLUMNS = ["first_txn_dttm", "last_txn_dttm"]
TOP_CATEGORY_COLUMN = "top_cat_from_share"
TOP_CATEGORY_SHARE_COLUMN = "top_cat_share"
TOP_MCC_COLUMN = "top_mcc_from_share"
TOP_MCC_SHARE_COLUMN = "top_mcc_share"
EXOTIC_MIN_CLUSTERS = 4
EXOTIC_MAX_CLUSTERS = 64
EXOTIC_CLUSTER_TRAIN_ITERS = 25
EXOTIC_CLUSTER_REDOS = 3
IMPACTFUL_FEATURE_SCORE_TAU = 0.85
IMPACTFUL_FEATURE_SCORE_DECIMALS = 4
IMPACTFUL_FALLBACK_SUPPORT_WEIGHT = 0.35

_SEARCH_LOCK = Lock()
_SEARCH_BUNDLE = None
_INTERACTION_BLOCKLIST_SERVICE = InteractionBlocklistService()
_MAPPER_REASON_VALUES = tuple(get_reason_mapper_values())
log: structlog.BoundLogger = structlog.get_logger(__name__)


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


def resolve_dataset_path(
    parquet_path: str | Path = DATA_PARQUET_PATH,
    csv_path: str | Path | None = DATA_CSV_PATH,
) -> Path:
    parquet_candidate = Path(parquet_path)
    if parquet_candidate.exists():
        return parquet_candidate

    if csv_path is not None:
        csv_candidate = Path(csv_path)
        if csv_candidate.exists():
            return csv_candidate

    raise FileNotFoundError(
        f"User feature dataset not found. Checked parquet={parquet_candidate} csv={csv_path}."
    )


def load_dataset(file_path: str | Path | None = None):
    resolved_path = Path(file_path) if file_path is not None else resolve_dataset_path()
    suffix = resolved_path.suffix.lower()

    if suffix == ".parquet":
        try:
            df = pd.read_parquet(resolved_path)
        except ImportError as exc:
            csv_fallback_path = Path(DATA_CSV_PATH)
            if not csv_fallback_path.exists():
                raise
            log.warning(
                "transaction_candidates_parquet_engine_missing_falling_back_to_csv",
                parquet_path=str(resolved_path),
                csv_fallback_path=str(csv_fallback_path),
                error=str(exc),
            )
            dtype_map, date_columns, categorical_columns = infer_load_config(csv_fallback_path)
            df = pd.read_csv(
                csv_fallback_path,
                dtype=dtype_map,
                parse_dates=date_columns,
                low_memory=False,
            )

            for column in categorical_columns:
                df[column] = df[column].astype("string")
    else:
        dtype_map, date_columns, categorical_columns = infer_load_config(resolved_path)

        df = pd.read_csv(
            resolved_path,
            dtype=dtype_map,
            parse_dates=date_columns,
            low_memory=False,
        )

        for column in categorical_columns:
            df[column] = df[column].astype("string")

    df = _coerce_date_columns(df)

    df[ID_COLUMN] = df[ID_COLUMN].map(lambda raw_id: normalize_user_id(raw_id, allow_hash=True)).astype("string")
    df = df[df[ID_COLUMN].notna()].copy()

    return df


def _coerce_date_columns(df, date_columns=None):
    result = df.copy()
    for column in _get_available_date_columns(result, date_columns=date_columns):
        result[column] = pd.to_datetime(result[column], errors="coerce", utc=False)
    return result


def _get_available_date_columns(df, date_columns=None):
    candidate_columns = DATE_COLUMNS if date_columns is None else list(date_columns)
    available_date_columns = []
    for column in candidate_columns:
        if column not in df.columns:
            continue
        series = df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            available_date_columns.append(column)
            continue
        coerced_series = pd.to_datetime(series, errors="coerce", utc=False)
        if coerced_series.notna().any():
            available_date_columns.append(column)
    return available_date_columns


def add_date_features(df, date_columns=None):
    result = df.copy()
    available_date_columns = _get_available_date_columns(result, date_columns=date_columns)

    if not available_date_columns:
        return result

    result = _coerce_date_columns(result, date_columns=available_date_columns)
    reference_date = max(result[column].max() for column in available_date_columns)

    if reference_date is pd.NaT or reference_date is None:
        return result

    for column in available_date_columns:
        result[f"days_since_{column}"] = (
            reference_date - result[column]
        ).dt.days.astype("float32")

    if "first_txn_dttm" in available_date_columns:
        result["days_since_first_txn"] = (
            reference_date - result["first_txn_dttm"]
        ).dt.days.astype("float32")

    if "last_txn_dttm" in available_date_columns:
        result["days_since_last_txn"] = (
            reference_date - result["last_txn_dttm"]
        ).dt.days.astype("float32")

    if {"first_txn_dttm", "last_txn_dttm"}.issubset(available_date_columns):
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


def prepare_search_artifacts(df, date_columns=None):
    enriched_df = add_date_features(df, date_columns=date_columns)
    enriched_df = enriched_df[enriched_df[ID_COLUMN].notna()].copy().reset_index(drop=True)
    output_df = add_interpretable_share_columns(enriched_df).reset_index(drop=True)

    available_date_columns = _get_available_date_columns(enriched_df, date_columns=date_columns)
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


def build_cluster_artifacts(scaled_matrix, config=None):
    user_count = int(scaled_matrix.shape[0])
    feature_count = int(scaled_matrix.shape[1]) if scaled_matrix.ndim == 2 else 0
    if user_count == 0 or feature_count == 0:
        empty_assignments = np.asarray([], dtype=np.int32)
        return {
            "cluster_count": 0,
            "cluster_assignments": empty_assignments,
            "cluster_member_indices": {},
            "cluster_order": empty_assignments,
            "cluster_centroids": np.empty((0, feature_count), dtype=np.float32),
        }

    cluster_count = _resolve_cluster_count(user_count=user_count, config=config)
    if cluster_count <= 1:
        member_indices = np.arange(user_count, dtype=np.int32)
        return {
            "cluster_count": 1,
            "cluster_assignments": np.zeros(user_count, dtype=np.int32),
            "cluster_member_indices": {0: member_indices},
            "cluster_order": np.asarray([0], dtype=np.int32),
            "cluster_centroids": np.mean(scaled_matrix, axis=0, keepdims=True).astype(np.float32),
        }

    train_matrix = np.ascontiguousarray(scaled_matrix)
    kmeans = faiss.Kmeans(
        feature_count,
        cluster_count,
        niter=EXOTIC_CLUSTER_TRAIN_ITERS,
        nredo=EXOTIC_CLUSTER_REDOS,
        seed=42,
        verbose=False,
    )
    kmeans.train(train_matrix)

    distances_to_centroid, assignments = kmeans.index.search(train_matrix, 1)
    distances_to_centroid = distances_to_centroid.reshape(-1)
    cluster_assignments = assignments.reshape(-1).astype(np.int32)

    cluster_member_indices = {}
    for cluster_id in range(cluster_count):
        member_indices = np.flatnonzero(cluster_assignments == cluster_id).astype(np.int32)
        if member_indices.size == 0:
            continue
        member_order = np.argsort(distances_to_centroid[member_indices], kind="stable")
        cluster_member_indices[int(cluster_id)] = member_indices[member_order]

    cluster_order = np.asarray(sorted(cluster_member_indices), dtype=np.int32)
    cluster_centroids = np.asarray(kmeans.centroids, dtype=np.float32).reshape(cluster_count, feature_count)

    return {
        "cluster_count": int(cluster_order.size),
        "cluster_assignments": cluster_assignments,
        "cluster_member_indices": cluster_member_indices,
        "cluster_order": cluster_order,
        "cluster_centroids": cluster_centroids,
    }


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


def _resolve_cluster_count(user_count, config=None):
    requested_cluster_count = int(getattr(config, "n_exotic_clusters", 0) or 0)
    if requested_cluster_count > 0:
        return max(1, min(user_count, requested_cluster_count))

    heuristic_cluster_count = int(np.sqrt(max(user_count, 1)))
    bounded_cluster_count = max(EXOTIC_MIN_CLUSTERS, heuristic_cluster_count)
    bounded_cluster_count = min(EXOTIC_MAX_CLUSTERS, bounded_cluster_count)
    bounded_cluster_count = min(bounded_cluster_count, max(1, user_count // 40))
    return max(1, min(user_count, bounded_cluster_count))


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
        _ensure_cluster_artifacts(search_bundle, config=config or CandidateConfig())
        return search_bundle

    global _SEARCH_BUNDLE
    if _SEARCH_BUNDLE is not None:
        return _SEARCH_BUNDLE

    with _SEARCH_LOCK:
        if _SEARCH_BUNDLE is None:
            df = load_dataset()
            bundle = prepare_search_artifacts(df)
            bundle["faiss_index"] = build_faiss_index(
                bundle["scaled_matrix"],
                config or CandidateConfig(),
            )
            bundle.update(build_cluster_artifacts(bundle["scaled_matrix"], config=config or CandidateConfig()))
            _SEARCH_BUNDLE = bundle

    return _SEARCH_BUNDLE


def warmup_search_bundle(config=None):
    """Build and cache search artifacts, HNSW index and cluster artifacts once per process."""
    effective_config = _to_candidate_config(config)
    return _get_search_bundle(search_bundle=None, config=effective_config)


def _ensure_cluster_artifacts(search_bundle, config=None):
    required_keys = {
        "cluster_count",
        "cluster_assignments",
        "cluster_member_indices",
        "cluster_order",
        "cluster_centroids",
    }
    if required_keys.issubset(search_bundle):
        return search_bundle

    search_bundle.update(
        build_cluster_artifacts(
            search_bundle["scaled_matrix"],
            config=config or CandidateConfig(),
        )
    )
    return search_bundle


def _load_prebuilt_search_bundle() -> dict[str, object] | None:
    bundle_path = Path(SEARCH_BUNDLE_PATH)
    index_path = Path(INDEX_PATH)
    if not bundle_path.exists():
        log_lazy_rebuild_blocked(
            source="transactions",
            reason="prebuilt_bundle_missing",
            artifact_path=bundle_path,
        )
        return None
    if not index_path.exists():
        log_lazy_rebuild_blocked(
            source="transactions",
            reason="prebuilt_index_missing",
            artifact_path=index_path,
        )
        return None

    try:
        with bundle_path.open("rb") as input_stream:
            bundle = pickle.load(input_stream)
    except Exception as exc:
        log.warning(
            "transaction_candidates_bundle_load_failed",
            bundle_path=str(bundle_path),
            error=str(exc),
        )
        return None

    required_fields = {"output_df", "feature_names", "scaled_matrix", "user_ids", "id_index"}
    if not required_fields.issubset(bundle):
        missing_fields = sorted(required_fields.difference(bundle))
        log.warning(
            "transaction_candidates_bundle_invalid",
            bundle_path=str(bundle_path),
            missing_fields=missing_fields,
        )
        return None

    try:
        index = faiss.read_index(str(index_path))
    except Exception as exc:
        log.warning(
            "transaction_candidates_index_load_failed",
            index_path=str(index_path),
            error=str(exc),
        )
        return None

    scaled_matrix = np.asarray(bundle["scaled_matrix"])
    feature_count = int(scaled_matrix.shape[1]) if scaled_matrix.ndim == 2 else 0
    if feature_count <= 0:
        return None
    if int(index.d) != feature_count or int(index.ntotal) != int(scaled_matrix.shape[0]):
        log.warning(
            "transaction_candidates_index_shape_mismatch",
            index_path=str(index_path),
            index_dim=int(index.d),
            feature_count=feature_count,
            index_size=int(index.ntotal),
            row_count=int(scaled_matrix.shape[0]),
        )
        return None

    bundle["faiss_index"] = index
    _ensure_cluster_artifacts(bundle, config=CandidateConfig())
    return bundle


def _build_empty_cluster_artifacts(*, feature_count: int) -> dict[str, object]:
    empty_assignments = np.asarray([], dtype=np.int32)
    return {
        "cluster_count": 0,
        "cluster_assignments": empty_assignments,
        "cluster_member_indices": {},
        "cluster_order": empty_assignments,
        "cluster_centroids": np.empty((0, max(0, int(feature_count))), dtype=np.float32),
    }


def _compute_empirical_percentiles(values) -> np.ndarray:
    prepared = np.asarray(values, dtype=np.float64).reshape(-1)
    if prepared.size == 0:
        return np.asarray([], dtype=np.float64)

    result = np.ones(prepared.shape, dtype=np.float64)
    finite_mask = np.isfinite(prepared)
    if not finite_mask.any():
        return result

    finite_values = prepared[finite_mask]
    sorted_order = np.argsort(finite_values, kind="mergesort")
    sorted_values = finite_values[sorted_order]
    right_positions = np.searchsorted(sorted_values, sorted_values, side="right")
    sorted_percentiles = right_positions.astype(np.float64) / float(sorted_values.size)

    finite_percentiles = np.empty(sorted_percentiles.shape, dtype=np.float64)
    finite_percentiles[sorted_order] = sorted_percentiles
    result[finite_mask] = finite_percentiles
    return np.clip(result, 0.0, 1.0)


def _compute_support_weights_from_scores(scores, *, lower_is_better: bool) -> np.ndarray:
    prepared = np.asarray(scores, dtype=np.float64).reshape(-1)
    if prepared.size == 0:
        return np.asarray([], dtype=np.float64)

    support_weights = np.zeros(prepared.shape, dtype=np.float64)
    finite_mask = np.isfinite(prepared)
    if not finite_mask.any():
        return support_weights

    finite_values = prepared[finite_mask]
    percentiles = _compute_empirical_percentiles(finite_values)
    if lower_is_better:
        support_values = 1.0 - percentiles
    else:
        support_values = percentiles

    support_weights[finite_mask] = np.clip(support_values, 0.0, 1.0)
    return support_weights


def _compute_impactful_feature_score(*, percentile_distance: float, support_weight: float, tau: float) -> float:
    bounded_tau = float(np.clip(float(tau), 0.0, 0.999999))
    bounded_percentile = float(np.clip(float(percentile_distance), 0.0, 1.0))
    bounded_support = float(np.clip(float(support_weight), 0.0, 1.0))

    denominator = max(1.0 - bounded_tau, 1e-6)
    normalized_signal = max(0.0, (1.0 - bounded_percentile) - bounded_tau) / denominator
    return float(np.clip(bounded_support * normalized_signal, 0.0, 1.0))


def _extract_impactful_features_batch(
    *,
    query_vector,
    candidate_matrix,
    feature_names,
    top_k,
    support_weights=None,
    tau: float = IMPACTFUL_FEATURE_SCORE_TAU,
):
    candidate_matrix = np.asarray(candidate_matrix, dtype=np.float32)
    if candidate_matrix.ndim != 2:
        return []

    candidate_count, feature_count = candidate_matrix.shape
    if candidate_count == 0:
        return []
    if top_k <= 0 or feature_count <= 0:
        return [{} for _ in range(candidate_count)]

    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if query.size != feature_count:
        return [{} for _ in range(candidate_count)]

    effective_top_k = min(int(top_k), feature_count)
    if effective_top_k <= 0:
        return [{} for _ in range(candidate_count)]

    abs_diff = np.abs(candidate_matrix - query)
    squared_diff = np.square(abs_diff)
    row_totals = squared_diff.sum(axis=1)
    contribution_shares = np.divide(
        squared_diff,
        row_totals[:, None],
        out=np.zeros_like(squared_diff),
        where=row_totals[:, None] > 0,
    )

    if effective_top_k >= feature_count:
        top_indices = np.tile(np.arange(feature_count, dtype=np.int32), (candidate_count, 1))
    else:
        partition_start = feature_count - effective_top_k
        top_indices = np.argpartition(contribution_shares, partition_start, axis=1)[:, partition_start:]
    row_indices = np.arange(candidate_count)[:, None]
    top_contributions = contribution_shares[row_indices, top_indices]
    order_within_top = np.argsort(top_contributions, axis=1)[:, ::-1]
    top_indices = top_indices[row_indices, order_within_top]

    if support_weights is None:
        pair_distances = np.linalg.norm(candidate_matrix - query, axis=1)
        prepared_support_weights = _compute_support_weights_from_scores(
            pair_distances,
            lower_is_better=True,
        )
    else:
        prepared_support_weights = np.asarray(support_weights, dtype=np.float64).reshape(-1)
        if prepared_support_weights.size != candidate_count:
            fallback_support_weights = _compute_support_weights_from_scores(
                np.linalg.norm(candidate_matrix - query, axis=1),
                lower_is_better=True,
            )
            merged_support_weights = np.array(fallback_support_weights, copy=True)
            limit = min(candidate_count, prepared_support_weights.size)
            if limit > 0:
                merged_support_weights[:limit] = prepared_support_weights[:limit]
            prepared_support_weights = merged_support_weights
        prepared_support_weights = np.nan_to_num(
            prepared_support_weights,
            nan=IMPACTFUL_FALLBACK_SUPPORT_WEIGHT,
            posinf=1.0,
            neginf=0.0,
        )
        prepared_support_weights = np.clip(prepared_support_weights, 0.0, 1.0)

    feature_percentiles: dict[int, np.ndarray] = {}
    for raw_feature_index in np.unique(top_indices):
        feature_index = int(raw_feature_index)
        feature_percentiles[feature_index] = _compute_empirical_percentiles(abs_diff[:, feature_index])

    feature_names_array = np.asarray(feature_names, dtype=object)
    impactful_features = []
    for row_position in range(candidate_count):
        row_support = float(prepared_support_weights[row_position]) if row_position < prepared_support_weights.size else 0.0
        if not np.isfinite(row_support):
            row_support = IMPACTFUL_FALLBACK_SUPPORT_WEIGHT
        row_support = float(np.clip(row_support, 0.0, 1.0))

        row_impacts = {}
        for raw_feature_index in top_indices[row_position]:
            feature_index = int(raw_feature_index)
            contribution_value = float(contribution_shares[row_position, feature_index])
            if contribution_value <= 0:
                continue

            percentiles_by_feature = feature_percentiles.get(feature_index)
            percentile_distance = (
                float(percentiles_by_feature[row_position])
                if percentiles_by_feature is not None and row_position < percentiles_by_feature.size
                else 1.0
            )
            feature_score = _compute_impactful_feature_score(
                percentile_distance=percentile_distance,
                support_weight=row_support,
                tau=tau,
            )
            feature_name = (
                str(feature_names_array[feature_index])
                if feature_index < feature_names_array.size
                else str(feature_index)
            )
            row_impacts[feature_name] = round(feature_score, IMPACTFUL_FEATURE_SCORE_DECIMALS)

        impactful_features.append(row_impacts)

    return impactful_features


def _score_seeded_reason_percentile(seed_value, reason_code) -> float:
    digest = hashlib.blake2b(
        f"{seed_value}:{reason_code}".encode("utf-8"),
        digest_size=8,
    ).digest()
    raw_value = int.from_bytes(digest, byteorder="big", signed=False)
    max_uint64 = float((1 << 64) - 1)
    if max_uint64 <= 0:
        return 1.0
    return float(raw_value / max_uint64)


def _extract_seeded_impactful_features(
    seed_value,
    top_k,
    *,
    support_weight: float = IMPACTFUL_FALLBACK_SUPPORT_WEIGHT,
    tau: float = IMPACTFUL_FEATURE_SCORE_TAU,
):
    if top_k <= 0:
        return {}
    if not _MAPPER_REASON_VALUES:
        return {}

    sample_size = min(int(top_k), len(_MAPPER_REASON_VALUES))
    if sample_size <= 0:
        return {}

    digest = hashlib.blake2b(str(seed_value).encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, byteorder="big", signed=False)
    random_generator = np.random.default_rng(seed)
    sampled_reasons = random_generator.choice(_MAPPER_REASON_VALUES, size=sample_size, replace=False)

    return {
        str(reason_code): round(
            _compute_impactful_feature_score(
                percentile_distance=_score_seeded_reason_percentile(seed_value, str(reason_code)),
                support_weight=support_weight,
                tau=tau,
            ),
            IMPACTFUL_FEATURE_SCORE_DECIMALS,
        )
        for reason_code in sampled_reasons
    }


def build_impactful_features_for_candidate_ids(
    *,
    query_user_id,
    candidate_user_ids,
    top_k,
    search_bundle=None,
    support_weights=None,
    seed_bases=None,
    tau: float = IMPACTFUL_FEATURE_SCORE_TAU,
):
    candidate_ids = [normalize_user_id(candidate_id, allow_hash=True) for candidate_id in candidate_user_ids]
    candidate_count = len(candidate_ids)
    if candidate_count == 0:
        return [], [], []
    if int(top_k) <= 0:
        return (
            [{} for _ in range(candidate_count)],
            ["seeded_fallback" for _ in range(candidate_count)],
            [None for _ in range(candidate_count)],
        )

    prepared_support_weights = np.asarray(
        support_weights if support_weights is not None else np.full(candidate_count, IMPACTFUL_FALLBACK_SUPPORT_WEIGHT),
        dtype=np.float64,
    ).reshape(-1)
    if prepared_support_weights.size != candidate_count:
        aligned_support_weights = np.full(candidate_count, IMPACTFUL_FALLBACK_SUPPORT_WEIGHT, dtype=np.float64)
        copy_count = min(candidate_count, prepared_support_weights.size)
        if copy_count > 0:
            aligned_support_weights[:copy_count] = prepared_support_weights[:copy_count]
        prepared_support_weights = aligned_support_weights
    prepared_support_weights = np.nan_to_num(
        prepared_support_weights,
        nan=IMPACTFUL_FALLBACK_SUPPORT_WEIGHT,
        posinf=1.0,
        neginf=0.0,
    )
    prepared_support_weights = np.clip(prepared_support_weights, 0.0, 1.0)

    impactful_features = [{} for _ in range(candidate_count)]
    impactful_origins = ["seeded_fallback" for _ in range(candidate_count)]
    impactful_seed_basis = [None for _ in range(candidate_count)]
    resolved_via_vectors = np.zeros(candidate_count, dtype=bool)

    bundle = _get_search_bundle(search_bundle=search_bundle, config=CandidateConfig())
    normalized_query_user_id = normalize_user_id(query_user_id, allow_hash=True)
    if (
        bundle
        and normalized_query_user_id is not None
        and "id_index" in bundle
        and "scaled_matrix" in bundle
        and "feature_names" in bundle
    ):
        id_index = bundle.get("id_index")
        scaled_matrix = np.asarray(bundle.get("scaled_matrix"))
        feature_names = bundle.get("feature_names")
        if id_index is not None and scaled_matrix.ndim == 2 and normalized_query_user_id in id_index:
            query_position = _resolve_id_index_position(id_index.get_loc(normalized_query_user_id))
            if query_position is not None and 0 <= query_position < int(scaled_matrix.shape[0]):
                vector_row_positions = []
                vector_candidate_positions = []
                vector_support_weights = []

                for row_position, candidate_id in enumerate(candidate_ids):
                    if candidate_id is None or candidate_id not in id_index:
                        continue
                    candidate_position = _resolve_id_index_position(id_index.get_loc(candidate_id))
                    if candidate_position is None:
                        continue
                    if candidate_position < 0 or candidate_position >= int(scaled_matrix.shape[0]):
                        continue
                    vector_row_positions.append(row_position)
                    vector_candidate_positions.append(candidate_position)
                    vector_support_weights.append(float(prepared_support_weights[row_position]))

                if vector_row_positions:
                    vector_features = _extract_impactful_features_batch(
                        query_vector=scaled_matrix[query_position],
                        candidate_matrix=scaled_matrix[np.asarray(vector_candidate_positions, dtype=np.int32)],
                        feature_names=feature_names,
                        top_k=int(top_k),
                        support_weights=np.asarray(vector_support_weights, dtype=np.float64),
                        tau=tau,
                    )
                    for row_position, candidate_features in zip(vector_row_positions, vector_features, strict=False):
                        impactful_features[row_position] = candidate_features
                        impactful_origins[row_position] = "vector_distance_contribution"
                        impactful_seed_basis[row_position] = None
                        resolved_via_vectors[row_position] = True

    for row_position, candidate_id in enumerate(candidate_ids):
        if resolved_via_vectors[row_position]:
            continue
        fallback_seed_basis = None
        if seed_bases is not None and row_position < len(seed_bases):
            raw_seed_basis = seed_bases[row_position]
            if raw_seed_basis is not None:
                prepared_seed_basis = str(raw_seed_basis).strip()
                if prepared_seed_basis and prepared_seed_basis.lower() != "<na>":
                    fallback_seed_basis = prepared_seed_basis
        if fallback_seed_basis is None:
            fallback_candidate_id = candidate_id if candidate_id is not None else f"unknown_{row_position}"
            fallback_seed_basis = f"{normalized_query_user_id or 'anonymous'}:{fallback_candidate_id}:impactful"
        impactful_features[row_position] = _extract_seeded_impactful_features(
            seed_value=fallback_seed_basis,
            top_k=int(top_k),
            support_weight=float(prepared_support_weights[row_position]),
            tau=tau,
        )
        impactful_origins[row_position] = "seeded_fallback"
        impactful_seed_basis[row_position] = fallback_seed_basis

    return impactful_features, impactful_origins, impactful_seed_basis


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
        top_distances = np.linalg.norm(
            scaled_matrix[top_indices] - scaled_matrix[query_position],
            axis=1,
        ).astype(float)
        scoring_indices = top_indices
        scoring_distances = top_distances
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

        scoring_indices = filtered_indices
        scoring_distances = filtered_distances
        top_indices = filtered_indices[: effective_config.n_rec_users]
        top_distances = filtered_distances[: effective_config.n_rec_users]

    support_weights = _compute_support_weights_from_scores(
        scoring_distances,
        lower_is_better=True,
    )
    scored_impactful_features = _extract_impactful_features_batch(
        query_vector=scaled_matrix[query_position],
        candidate_matrix=scaled_matrix[scoring_indices],
        feature_names=feature_names,
        top_k=effective_config.an_impactful_features,
        support_weights=support_weights,
        tau=IMPACTFUL_FEATURE_SCORE_TAU,
    )
    impactful_features = scored_impactful_features[: len(top_indices)]

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
