from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as joblib_load

from core.config import GlobalConfig
from core.data_config import Paths
from ml.ml_config import RerankConfig
from ml.services.recommendation_scoring import (
    DEFAULT_SCORE_WEIGHTS,
    PREDICTION_COLUMN_BY_TARGET,
    PROBABILITY_TARGETS,
    compute_linear_score_frame,
)
from ml.services.user_id import normalize_user_id


def rerank(conf: RerankConfig, g_conf: GlobalConfig) -> pd.DataFrame:
    rerank_mode = int(getattr(conf, "rerank_mode", 0) or 0)

    if rerank_mode == 0:
        return get_mode0_recommendations(conf=conf, g_conf=g_conf)
    if rerank_mode == 1:
        return load_or_fallback_1(conf=conf, g_conf=g_conf)
    if rerank_mode == 2:
        return load_or_fallback_2(conf=conf, g_conf=g_conf)
    if rerank_mode == 3:
        return load_or_fallback_3(conf=conf, g_conf=g_conf)

    raise ValueError(f"Unsupported rerank_mode={rerank_mode}. Expected one of: 0, 1, 2, 3.")


def load_or_fallback_1(conf: RerankConfig, g_conf: GlobalConfig) -> pd.DataFrame:
    return _rerank_with_probability_models(conf=conf, g_conf=g_conf, target_mode=1)


def load_or_fallback_2(conf: RerankConfig, g_conf: GlobalConfig) -> pd.DataFrame:
    return _rerank_with_probability_models(conf=conf, g_conf=g_conf, target_mode=2)


def load_or_fallback_3(conf: RerankConfig, g_conf: GlobalConfig) -> pd.DataFrame:
    return _rerank_with_probability_models(conf=conf, g_conf=g_conf, target_mode=3)


def _rerank_with_probability_models(
    conf: RerankConfig,
    g_conf: GlobalConfig,
    *,
    target_mode: int,
) -> pd.DataFrame:
    base_df = _get_mode0_candidate_pool(conf=conf, g_conf=g_conf)
    if base_df.empty:
        return _ensure_probability_columns(base_df.copy())

    n_rec_users = int(getattr(g_conf, "n_rec_users", 0) or 0)
    if n_rec_users <= 0:
        return pd.DataFrame()

    feature_matrix = _build_model_matrix(base_df)
    scored_df = base_df.copy()

    cache: dict[tuple[int, str], np.ndarray | None] = {}
    has_any_prediction = False

    for target in PROBABILITY_TARGETS:
        prediction = _predict_target_with_fallback(
            conf=conf,
            mode=target_mode,
            target=target,
            feature_matrix=feature_matrix,
            row_count=scored_df.shape[0],
            cache=cache,
        )
        target_column = PREDICTION_COLUMN_BY_TARGET[target]
        if prediction is None:
            scored_df[target_column] = np.nan
            continue

        scored_df[target_column] = prediction
        has_any_prediction = True

    if has_any_prediction:
        scored_df["rerank_score"] = compute_linear_score_frame(
            scored_df,
            weights=DEFAULT_SCORE_WEIGHTS,
        )
        sort_columns = ["rerank_score"]
        sort_order = [False]
        if "rank" in scored_df.columns:
            sort_columns.append("rank")
            sort_order.append(True)
        scored_df = scored_df.sort_values(sort_columns, ascending=sort_order, na_position="last").reset_index(
            drop=True
        )
    else:
        scored_df["rerank_score"] = np.nan

    normalized_df = _normalize_and_dedupe_candidates(scored_df, conf=conf)
    return normalized_df.head(n_rec_users).reset_index(drop=True)


def _predict_target_with_fallback(
    *,
    conf: RerankConfig,
    mode: int,
    target: str,
    feature_matrix: pd.DataFrame,
    row_count: int,
    cache: dict[tuple[int, str], np.ndarray | None],
) -> np.ndarray | None:
    cache_key = (mode, target)
    if cache_key in cache:
        return cache[cache_key]

    if mode <= 0:
        cache[cache_key] = None
        return None

    model = _load_target_model_for_mode(conf=conf, mode=mode, target=target)
    if model is None:
        fallback_prediction = _predict_target_with_fallback(
            conf=conf,
            mode=mode - 1,
            target=target,
            feature_matrix=feature_matrix,
            row_count=row_count,
            cache=cache,
        )
        cache[cache_key] = fallback_prediction
        return fallback_prediction

    try:
        aligned_matrix = _align_model_matrix(feature_matrix, model)
        prediction = _predict_scores(model=model, feature_matrix=aligned_matrix)
        if prediction.shape[0] != row_count:
            raise ValueError("Model produced unexpected score vector length.")
        cache[cache_key] = prediction
        return prediction
    except Exception:
        fallback_prediction = _predict_target_with_fallback(
            conf=conf,
            mode=mode - 1,
            target=target,
            feature_matrix=feature_matrix,
            row_count=row_count,
            cache=cache,
        )
        cache[cache_key] = fallback_prediction
        return fallback_prediction


def _load_target_model_for_mode(conf: RerankConfig, mode: int, target: str):
    model_path = _resolve_target_model_path(conf=conf, mode=mode, target=target)
    if model_path is None:
        return None
    try:
        return joblib_load(model_path)
    except Exception:
        return None


def _resolve_target_model_path(conf: RerankConfig, mode: int, target: str) -> Path | None:
    explicit_attr = f"rerank_mode_{mode}_{target}_model_path"
    explicit_value = getattr(conf, explicit_attr, None)
    if explicit_value:
        return Path(explicit_value)

    target_path_mapping = getattr(conf, "rerank_target_model_paths", None)
    if isinstance(target_path_mapping, dict):
        mode_mapping = target_path_mapping.get(mode)
        if isinstance(mode_mapping, dict):
            mode_target_path = mode_mapping.get(target)
            if mode_target_path:
                return Path(mode_target_path)

    generic_attr = f"rerank_mode_{mode}_model_path"
    generic_path = getattr(conf, generic_attr, None)
    if generic_path:
        return Path(generic_path)

    generic_path_mapping = getattr(conf, "rerank_model_paths", None)
    if isinstance(generic_path_mapping, dict):
        mapped_generic_path = generic_path_mapping.get(mode)
        if mapped_generic_path:
            return Path(mapped_generic_path)

    target_default = Paths.out_data_path / f"reranker_mode_{mode}_{target}.joblib"
    generic_default = Paths.out_data_path / f"reranker_mode_{mode}.joblib"

    if target_default.exists():
        return target_default
    if generic_default.exists():
        return generic_default

    return target_default


def _build_model_matrix(base_df: pd.DataFrame) -> pd.DataFrame:
    skip_columns = {
        "candidate_id",
        "party_rk",
        "user_id",
        "target_user_id",
        "id",
        "query_user_id",
        "query_party_rk",
        "request_user_id",
        "impactful_features",
        "rerank_score",
        *PREDICTION_COLUMN_BY_TARGET.values(),
    }

    feature_parts: dict[str, pd.Series] = {}
    for column in base_df.columns:
        if column in skip_columns:
            continue

        series = base_df[column]
        non_null = series.dropna()
        if not non_null.empty and isinstance(non_null.iloc[0], (dict, list, set, tuple)):
            continue

        if pd.api.types.is_numeric_dtype(series):
            feature_parts[column] = pd.to_numeric(series, errors="coerce").fillna(0.0)
        else:
            feature_parts[column] = series.astype("string").fillna("")

    if not feature_parts:
        return pd.DataFrame(index=base_df.index)

    feature_df = pd.DataFrame(feature_parts, index=base_df.index)
    categorical_columns = [
        column
        for column in feature_df.columns
        if not pd.api.types.is_numeric_dtype(feature_df[column])
    ]
    if categorical_columns:
        feature_df = pd.get_dummies(
            feature_df,
            columns=categorical_columns,
            dummy_na=False,
            dtype=float,
        )

    return feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _align_model_matrix(feature_matrix: pd.DataFrame, model) -> pd.DataFrame:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return feature_matrix
    return feature_matrix.reindex(columns=list(feature_names), fill_value=0.0)


def _predict_scores(model, feature_matrix: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(feature_matrix))
        if probabilities.ndim == 2:
            if probabilities.shape[1] > 1:
                return probabilities[:, 1].astype(float)
            return probabilities[:, 0].astype(float)
        return probabilities.reshape(-1).astype(float)

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(feature_matrix)).reshape(-1).astype(float)

    if hasattr(model, "predict"):
        return np.asarray(model.predict(feature_matrix)).reshape(-1).astype(float)

    raise ValueError("Model must implement predict_proba, decision_function, or predict.")


def get_mode0_recommendations(conf: RerankConfig, g_conf: GlobalConfig) -> pd.DataFrame:
    n_rec_users = int(getattr(g_conf, "n_rec_users", 0) or 0)
    if n_rec_users <= 0:
        return pd.DataFrame()

    source_candidates = _get_mode0_candidate_pool(conf=conf, g_conf=g_conf)
    result_df = _take_candidates_in_source_order(source_candidates, conf=conf, n_rec_users=n_rec_users)
    return _ensure_probability_columns(result_df)


def _get_mode0_candidate_pool(conf: RerankConfig, g_conf: GlobalConfig) -> pd.DataFrame:
    from ml.candidates import exotic_candidates, transactions_candidates
    from ml.retrieval import collect_retrieval_candidates

    n_rec_users = int(getattr(g_conf, "n_rec_users", 0) or 0)
    if n_rec_users <= 0:
        return pd.DataFrame()

    conf_top_n = int(getattr(conf, "top_n", 0) or 0)
    candidate_pool_size = max(conf_top_n, n_rec_users)
    effective_conf = replace(conf, top_n=candidate_pool_size)

    try:
        retrieval_df = collect_retrieval_candidates(conf=effective_conf, g_conf=g_conf)
    except Exception:
        retrieval_df = pd.DataFrame()
    if not retrieval_df.empty:
        return retrieval_df

    transactions_df = _safe_fetch_candidates(transactions_candidates.get_transaction_candidates, effective_conf)
    if not transactions_df.empty:
        return transactions_df

    return _safe_fetch_candidates(exotic_candidates.get_exotic_candidates, effective_conf)


def _safe_fetch_candidates(fetcher, conf: RerankConfig) -> pd.DataFrame:
    try:
        data_frame = fetcher(conf)
    except (FileNotFoundError, KeyError, ValueError):
        return pd.DataFrame()

    if data_frame is None:
        return pd.DataFrame()
    if not isinstance(data_frame, pd.DataFrame):
        raise TypeError("Candidates fetcher must return pandas.DataFrame or None.")
    return data_frame


def _take_candidates_in_source_order(source_df: pd.DataFrame, conf: RerankConfig, n_rec_users: int) -> pd.DataFrame:
    normalized_df = _normalize_and_dedupe_candidates(source_df, conf=conf)
    return normalized_df.head(n_rec_users).reset_index(drop=True)


def _normalize_and_dedupe_candidates(source_df: pd.DataFrame, conf: RerankConfig) -> pd.DataFrame:
    if source_df.empty:
        return source_df

    result = source_df.copy()

    candidate_id_col = _pick_first_existing_column(
        result,
        ["candidate_id", "party_rk", "user_id", "target_user_id", "id"],
    )
    if candidate_id_col is not None:
        result["candidate_id"] = (
            result[candidate_id_col]
            .map(lambda raw_id: normalize_user_id(raw_id, allow_hash=True))
            .astype("string")
        )
        result = result[result["candidate_id"].notna()].copy()

    query_col = _pick_first_existing_column(result, ["query_user_id", "query_party_rk", "request_user_id"])
    if query_col is not None:
        result["query_user_id"] = (
            result[query_col]
            .map(lambda raw_id: normalize_user_id(raw_id, allow_hash=True))
            .astype("string")
        )
    elif getattr(conf, "user_id", None) is not None:
        result["query_user_id"] = normalize_user_id(conf.user_id, allow_hash=True)

    if "candidate_id" in result.columns:
        dedupe_keys = ["candidate_id"]
        if "query_user_id" in result.columns:
            dedupe_keys = ["query_user_id", "candidate_id"]
        result = result.drop_duplicates(subset=dedupe_keys, keep="first")

    return result


def _ensure_probability_columns(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df.empty:
        return result_df

    for prediction_column in PREDICTION_COLUMN_BY_TARGET.values():
        if prediction_column not in result_df.columns:
            result_df[prediction_column] = np.nan

    if "rerank_score" not in result_df.columns:
        result_df["rerank_score"] = np.nan

    return result_df


def _pick_first_existing_column(df: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in df.columns:
            return column
    return None
