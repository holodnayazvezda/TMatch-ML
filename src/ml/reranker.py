from dataclasses import replace

import pandas as pd

from core.config import GlobalConfig
from ml.ml_config import RerankConfig


def rerank(conf: RerankConfig, g_conf: GlobalConfig):
    if conf.rerank_mode == 0:
        return get_mode0_recommendations(conf=conf, g_conf=g_conf)

    raise ValueError(f"Unsupported rerank_mode={conf.rerank_mode}")


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
        result["candidate_id"] = result[candidate_id_col].astype("string")

    query_col = _pick_first_existing_column(result, ["query_user_id", "query_party_rk", "request_user_id"])
    if query_col is not None:
        result["query_user_id"] = result[query_col].astype("string")
    elif getattr(conf, "user_id", None) is not None:
        result["query_user_id"] = str(conf.user_id)

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
