from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as joblib_load

from core.config import GlobalConfig
from core.data_config import Paths
from ml.ml_config import RerankConfig
from ml.services.user_id import normalize_user_id


def rerank(conf: RerankConfig, g_conf: GlobalConfig):
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
    fallback_df = get_mode0_recommendations(conf=conf, g_conf=g_conf)
    model = _load_model_for_mode(conf=conf, mode=1)
    if model is None:
        return fallback_df
    return _apply_model_or_fallback(
        model=model,
        base_df=fallback_df,
        fallback_df=fallback_df,
        conf=conf,
        g_conf=g_conf,
        mode=1,
    )


def load_or_fallback_2(conf: RerankConfig, g_conf: GlobalConfig) -> pd.DataFrame:
    fallback_df = load_or_fallback_1(conf=conf, g_conf=g_conf)
    model = _load_model_for_mode(conf=conf, mode=2)
    if model is None:
        return fallback_df
    return _apply_model_or_fallback(
        model=model,
        base_df=fallback_df,
        fallback_df=fallback_df,
        conf=conf,
        g_conf=g_conf,
        mode=2,
    )


def load_or_fallback_3(conf: RerankConfig, g_conf: GlobalConfig) -> pd.DataFrame:
    fallback_df = load_or_fallback_2(conf=conf, g_conf=g_conf)
    model = _load_model_for_mode(conf=conf, mode=3)
    if model is None:
        return fallback_df
    return _apply_model_or_fallback(
        model=model,
        base_df=fallback_df,
        fallback_df=fallback_df,
        conf=conf,
        g_conf=g_conf,
        mode=3,
    )


def _apply_model_or_fallback(
    model,
    base_df: pd.DataFrame,
    fallback_df: pd.DataFrame,
    conf: RerankConfig,
    g_conf: GlobalConfig,
    mode: int,
) -> pd.DataFrame:
    try:
        return _rerank_with_model(model=model, base_df=base_df, conf=conf, g_conf=g_conf, mode=mode)
    except Exception:
        return fallback_df


def _rerank_with_model(
    model,
    base_df: pd.DataFrame,
    conf: RerankConfig,
    g_conf: GlobalConfig,
    mode: int,
) -> pd.DataFrame:
    if base_df.empty:
        return base_df

    n_rec_users = int(getattr(g_conf, "n_rec_users", 0) or 0)
    if n_rec_users <= 0:
        return pd.DataFrame()

    feature_matrix = _build_model_matrix(base_df)
    feature_matrix = _align_model_matrix(feature_matrix, model)
    scores = _predict_scores(model=model, feature_matrix=feature_matrix)
    if scores.shape[0] != base_df.shape[0]:
        raise ValueError("Model produced unexpected score vector length.")

    score_col = f"rerank_mode_{mode}_score"
    scored_df = base_df.copy()
    scored_df[score_col] = scores

    sort_cols = [score_col]
    sort_asc = [False]
    if "rank" in scored_df.columns:
        sort_cols.append("rank")
        sort_asc.append(True)

    scored_df = scored_df.sort_values(sort_cols, ascending=sort_asc, na_position="last").reset_index(drop=True)
    return _take_candidates_in_source_order(scored_df, conf=conf, n_rec_users=n_rec_users)


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


def _load_model_for_mode(conf: RerankConfig, mode: int):
    model_path = _resolve_model_path(conf=conf, mode=mode)
    if model_path is None:
        return None
    try:
        return joblib_load(model_path)
    except Exception:
        return None


def _resolve_model_path(conf: RerankConfig, mode: int) -> Path | None:
    specific_attr = f"rerank_mode_{mode}_model_path"
    specific_path = getattr(conf, specific_attr, None)
    if specific_path:
        return Path(specific_path)

    model_paths = getattr(conf, "rerank_model_paths", None)
    if isinstance(model_paths, dict):
        dynamic_path = model_paths.get(mode)
        if dynamic_path:
            return Path(dynamic_path)

    return Paths.out_data_path / f"reranker_mode_{mode}.joblib"


def get_mode0_recommendations(conf: RerankConfig, g_conf: GlobalConfig) -> pd.DataFrame:
    from ml.candidates import exotic_candidates, transactions_candidates

    n_rec_users = int(getattr(g_conf, "n_rec_users", 0) or 0)
    if n_rec_users <= 0:
        return pd.DataFrame()

    # Ensure transaction source can return enough rows for final cut.
    conf_top_n = int(getattr(conf, "top_n", 0) or 0)
    effective_conf = replace(conf, top_n=max(conf_top_n, n_rec_users))

    transactions_df = _safe_fetch_candidates(transactions_candidates.get_transaction_candidates, effective_conf)
    if not transactions_df.empty:
        return _take_candidates_in_source_order(transactions_df, conf=effective_conf, n_rec_users=n_rec_users)

    exotic_df = _safe_fetch_candidates(exotic_candidates.get_exotic_candidates, effective_conf)
    return _take_candidates_in_source_order(exotic_df, conf=effective_conf, n_rec_users=n_rec_users)


def _safe_fetch_candidates(fetcher, conf: RerankConfig) -> pd.DataFrame:
    try:
        df = fetcher(conf)
    except (KeyError, ValueError):
        return pd.DataFrame()

    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Candidates fetcher must return pandas.DataFrame or None.")
    return df


def _take_candidates_in_source_order(source_df: pd.DataFrame, conf: RerankConfig, n_rec_users: int) -> pd.DataFrame:
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

    # Keep source-provided ordering; remove duplicates only preserving first appearance.
    if "candidate_id" in result.columns:
        dedupe_keys = ["candidate_id"]
        if "query_user_id" in result.columns:
            dedupe_keys = ["query_user_id", "candidate_id"]
        result = result.drop_duplicates(subset=dedupe_keys, keep="first")

    return result.head(n_rec_users).reset_index(drop=True)


def _pick_first_existing_column(df: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in df.columns:
            return column
    return None
