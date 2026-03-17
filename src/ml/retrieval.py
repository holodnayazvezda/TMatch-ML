from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import structlog

from core.config import GlobalConfig
from core.data_config import Paths
from ml.infra.blocked_service import InteractionBlocklistService
from ml.ml_config import RerankConfig
from ml.services.user_id import normalize_user_id


QUERY_ID_COLUMN = "query_user_id"
CANDIDATE_ID_COLUMN = "candidate_id"
log: structlog.BoundLogger = structlog.get_logger(__name__)
_BLOCKLIST_SERVICE = InteractionBlocklistService()


def save_candidates_dataset(candidates_df: pd.DataFrame, output_path: str | Path | None = None) -> Path:
    """Persist retrieval candidates for the reranker."""
    target_path = Path(output_path) if output_path is not None else Paths.out_data_path / "rerank_candidates.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(target_path, index=False)
    return target_path


def collect_retrieval_candidates(conf: RerankConfig, g_conf: GlobalConfig) -> pd.DataFrame:
    """Build one merged candidate dataset with per-source retrieval features."""
    source_frames = _fetch_source_candidates(conf)
    source_names = list(source_frames.keys())
    source_sizes = [frame.shape[0] for frame in source_frames.values()]
    source_limits = _compute_source_limits(
        source_sizes=source_sizes,
        ratios=getattr(conf, "ratios", None),
        n_max_candidates=g_conf.n_max_candidates,
    )

    prepared_frames = []
    for source_name, source_df, limit in zip(source_names, source_frames.values(), source_limits):
        prepared = _prepare_source_frame(
            source_name=source_name,
            source_df=source_df,
            conf=conf,
            limit=limit,
        )
        prepared_frames.append(prepared)

    merged = _merge_source_frames(prepared_frames, source_names)
    if merged.empty:
        return merged

    merged = _add_common_retrieval_features(merged, source_names)
    merged = _apply_block_filter(merged)
    merged = merged.sort_values(
        ["retrieval_sources_count", "retrieval_best_rank"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    return merged


def collect_and_save_retrieval_candidates(conf: RerankConfig, g_conf: GlobalConfig) -> pd.DataFrame:
    candidates_df = collect_retrieval_candidates(conf=conf, g_conf=g_conf)
    save_candidates_dataset(candidates_df, output_path=conf.retrieval_candidates_path)
    return candidates_df


def _fetch_source_candidates(conf: RerankConfig) -> dict[str, pd.DataFrame]:
    from ml.candidates import exotic_candidates, interactions_candidates, popularity_fallback, transactions_candidates

    fetchers: list[tuple[str, Callable[[RerankConfig], pd.DataFrame | None]]] = [
        ("transactions", transactions_candidates.get_transaction_candidates),
        ("exotic", exotic_candidates.get_exotic_candidates),
        ("popularity", popularity_fallback.get_popular_candidates),
        ("tx_interaction", interactions_candidates.get_interaction_candidates),
    ]

    source_frames: dict[str, pd.DataFrame] = {}
    for source_name, fetcher in fetchers:
        try:
            source_df = fetcher(conf)
        except Exception as exc:
            log.warning(
                "retrieval_source_fetch_failed",
                source_name=source_name,
                error=str(exc),
            )
            source_frames[source_name] = pd.DataFrame()
            continue
        if source_df is None:
            source_frames[source_name] = pd.DataFrame()
            continue
        if not isinstance(source_df, pd.DataFrame):
            raise TypeError(f"Source '{source_name}' must return pandas.DataFrame or None.")
        source_frames[source_name] = source_df.copy()
    return source_frames


def _compute_source_limits(
    source_sizes: list[int],
    ratios: list[float] | None,
    n_max_candidates: int | None,
) -> list[int]:
    if n_max_candidates is None:
        return source_sizes

    if not source_sizes:
        return []

    normalized_ratios = _normalize_ratios(ratios=ratios, source_count=len(source_sizes))
    raw_limits = [ratio * n_max_candidates for ratio in normalized_ratios]
    limits = [min(size, int(math.floor(raw_limit))) for size, raw_limit in zip(source_sizes, raw_limits)]

    remaining = max(0, n_max_candidates - sum(limits))
    if remaining == 0:
        return limits

    source_order = sorted(
        range(len(source_sizes)),
        key=lambda idx: (raw_limits[idx] - math.floor(raw_limits[idx]), source_sizes[idx]),
        reverse=True,
    )

    while remaining > 0:
        distributed = False
        for idx in source_order:
            if limits[idx] >= source_sizes[idx]:
                continue
            limits[idx] += 1
            remaining -= 1
            distributed = True
            if remaining == 0:
                break
        if not distributed:
            break

    return limits


def _normalize_ratios(ratios: list[float] | None, source_count: int) -> list[float]:
    if source_count <= 0:
        return []

    if ratios is None:
        return [1.0 / source_count] * source_count

    prepared = [max(0.0, float(ratio)) for ratio in list(ratios)[:source_count]]
    if len(prepared) < source_count:
        prepared.extend([0.0] * (source_count - len(prepared)))

    ratio_sum = sum(prepared)
    if ratio_sum <= 0:
        return [1.0 / source_count] * source_count

    return [ratio / ratio_sum for ratio in prepared]


def _prepare_source_frame(
    source_name: str,
    source_df: pd.DataFrame,
    conf: RerankConfig,
    limit: int,
) -> pd.DataFrame:
    is_candidate_col = f"retrieval_{source_name}_is_candidate"
    rank_col = f"retrieval_{source_name}_rank"
    score_col = f"retrieval_{source_name}_score"
    cluster_col = f"retrieval_{source_name}_cluster_id"

    if source_df.empty or limit <= 0:
        return pd.DataFrame(columns=[QUERY_ID_COLUMN, CANDIDATE_ID_COLUMN, is_candidate_col, rank_col, score_col])

    candidate_id_col = _pick_first_existing_column(
        source_df,
        ["candidate_id", "party_rk", "user_id", "target_user_id", "id"],
    )
    if candidate_id_col is None:
        raise ValueError(f"Cannot infer candidate id column for source '{source_name}'.")

    source_df = source_df.copy()
    source_df[CANDIDATE_ID_COLUMN] = (
        source_df[candidate_id_col]
        .map(lambda raw_id: normalize_user_id(raw_id, allow_hash=True))
        .astype("string")
    )

    source_query_column = _pick_first_existing_column(
        source_df,
        ["query_user_id", "query_party_rk", "request_user_id"],
    )
    query_from_conf = getattr(conf, "user_id", None)
    if source_query_column is not None:
        source_df[QUERY_ID_COLUMN] = (
            source_df[source_query_column]
            .map(lambda raw_id: normalize_user_id(raw_id, allow_hash=True))
            .astype("string")
        )
    elif query_from_conf is not None:
        normalized_query_user_id = normalize_user_id(query_from_conf, allow_hash=True)
        source_df[QUERY_ID_COLUMN] = normalized_query_user_id
    else:
        raise ValueError(f"Cannot infer query user id for source '{source_name}'. Pass conf.user_id.")

    source_df = source_df[
        source_df[CANDIDATE_ID_COLUMN].notna() & source_df[QUERY_ID_COLUMN].notna()
    ].copy()

    source_rank_column = _pick_first_existing_column(source_df, ["rank", "source_rank", "position"])
    if source_rank_column is not None:
        source_df["_source_rank"] = pd.to_numeric(source_df[source_rank_column], errors="coerce")
    else:
        source_df["_source_rank"] = np.arange(1, len(source_df) + 1, dtype=float)

    source_df = source_df.sort_values("_source_rank", ascending=True, na_position="last").reset_index(drop=True)
    source_df = source_df.iloc[:limit].copy()

    source_score_column = _pick_source_score_column(source_df)
    if source_score_column is not None:
        score_values = pd.to_numeric(source_df[source_score_column], errors="coerce")
    else:
        score_values = pd.Series(np.nan, index=source_df.index, dtype=float)

    base_frame = source_df[[QUERY_ID_COLUMN, CANDIDATE_ID_COLUMN]].copy()
    retrieval_flags_frame = pd.DataFrame(
        {
            is_candidate_col: pd.Series(1, index=source_df.index, dtype="int8"),
            rank_col: source_df["_source_rank"].astype(float),
            score_col: score_values.astype(float),
        },
        index=source_df.index,
    )
    source_cluster_column = _pick_first_existing_column(
        source_df,
        ["transaction_cluster_id", "cluster_id"],
    )
    if source_cluster_column is not None:
        retrieval_flags_frame[cluster_col] = source_df[source_cluster_column].astype("string")

    source_feature_columns = [
        column
        for column in ("impactful_features", "impactful_features_origin", "impactful_features_seed_basis")
        if column in source_df.columns
    ]
    if source_feature_columns:
        feature_frame = source_df[source_feature_columns].copy()
        feature_frame.columns = [f"retrieval_{source_name}_feature_{column}" for column in source_feature_columns]
        result = pd.concat(
            [base_frame, retrieval_flags_frame, feature_frame],
            axis=1,
            copy=False,
        )
    else:
        result = pd.concat(
            [base_frame, retrieval_flags_frame],
            axis=1,
            copy=False,
        )

    result = result.drop_duplicates(
        subset=[QUERY_ID_COLUMN, CANDIDATE_ID_COLUMN],
        keep="first",
    ).reset_index(drop=True)
    return result


def _merge_source_frames(prepared_frames: list[pd.DataFrame], source_names: list[str]) -> pd.DataFrame:
    if not prepared_frames:
        return pd.DataFrame()

    merged: pd.DataFrame | None = None
    for prepared in prepared_frames:
        if merged is None:
            merged = prepared
            continue
        merged = merged.merge(
            prepared,
            how="outer",
            on=[QUERY_ID_COLUMN, CANDIDATE_ID_COLUMN],
        )

    if merged is None or merged.empty:
        return pd.DataFrame()

    for source_name in source_names:
        is_candidate_col = f"retrieval_{source_name}_is_candidate"
        if is_candidate_col in merged.columns:
            merged[is_candidate_col] = merged[is_candidate_col].fillna(0).astype("int8")

    return merged


def _add_common_retrieval_features(merged: pd.DataFrame, source_names: list[str]) -> pd.DataFrame:
    is_candidate_cols = [f"retrieval_{name}_is_candidate" for name in source_names]
    rank_cols = [f"retrieval_{name}_rank" for name in source_names]
    score_cols = [f"retrieval_{name}_score" for name in source_names]
    source_presence = merged[is_candidate_cols].fillna(0).astype("int8")

    merged["retrieval_sources_count"] = source_presence.sum(axis=1).astype("int16")
    merged["retrieval_best_rank"] = merged[rank_cols].min(axis=1, skipna=True)
    merged["retrieval_mean_rank"] = merged[rank_cols].mean(axis=1, skipna=True)
    merged["retrieval_best_score"] = merged[score_cols].max(axis=1, skipna=True)
    merged["retrieval_mean_score"] = merged[score_cols].mean(axis=1, skipna=True)
    merged["retrieval_sources"] = [
        ",".join(source_name for source_name, is_present in zip(source_names, row, strict=False) if is_present == 1)
        for row in source_presence.to_numpy(dtype="int8")
    ]
    merged["retrieval_primary_source"] = merged[rank_cols].idxmin(axis=1, skipna=True).str.replace(
        "retrieval_", "",
    ).str.replace("_rank", "")
    cluster_cols = [
        f"retrieval_{source_name}_cluster_id"
        for source_name in source_names
        if f"retrieval_{source_name}_cluster_id" in merged.columns
    ]
    if cluster_cols:
        merged["transaction_cluster_id"] = (
            merged[cluster_cols]
            .astype("string")
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            .bfill(axis=1)
            .iloc[:, 0]
        )
    else:
        merged["transaction_cluster_id"] = pd.NA
    if "retrieval_exotic_cluster_id" in merged.columns:
        merged["cluster_id"] = merged["retrieval_exotic_cluster_id"]
    merged["query_party_rk"] = merged[QUERY_ID_COLUMN]
    merged["party_rk"] = merged[CANDIDATE_ID_COLUMN]
    merged["rank"] = merged["retrieval_best_rank"]
    merged["distance_euclidean"] = np.nan
    merged["impactful_features"] = None
    merged["impactful_features_origin"] = None
    merged["impactful_features_seed_basis"] = None

    for source_name in source_names:
        source_mask = merged["retrieval_primary_source"] == source_name
        source_score_column = f"retrieval_{source_name}_score"
        source_impactful_column = f"retrieval_{source_name}_feature_impactful_features"
        source_impactful_origin_column = f"retrieval_{source_name}_feature_impactful_features_origin"
        source_impactful_seed_column = f"retrieval_{source_name}_feature_impactful_features_seed_basis"
        if source_score_column in merged.columns:
            merged.loc[source_mask, "distance_euclidean"] = merged.loc[source_mask, source_score_column]
        if source_impactful_column in merged.columns:
            merged.loc[source_mask, "impactful_features"] = merged.loc[source_mask, source_impactful_column]
        if source_impactful_origin_column in merged.columns:
            merged.loc[source_mask, "impactful_features_origin"] = merged.loc[source_mask, source_impactful_origin_column]
        if source_impactful_seed_column in merged.columns:
            merged.loc[source_mask, "impactful_features_seed_basis"] = merged.loc[source_mask, source_impactful_seed_column]

    return merged


def _pick_first_existing_column(df: pd.DataFrame, columns: list[str]) -> str | None:
    for column in columns:
        if column in df.columns:
            return column
    return None


def _pick_source_score_column(df: pd.DataFrame) -> str | None:
    preferred = ["score", "distance_euclidean", "popularity_score", "similarity", "distance"]
    for column in preferred:
        if column in df.columns:
            return column

    exclude = {
        QUERY_ID_COLUMN,
        CANDIDATE_ID_COLUMN,
        "query_party_rk",
        "query_user_id",
        "party_rk",
        "user_id",
        "target_user_id",
        "id",
        "rank",
        "source_rank",
        "position",
        "_source_rank",
    }
    numeric_candidates = [
        column
        for column in df.columns
        if column not in exclude and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not numeric_candidates:
        return None
    return numeric_candidates[0]


def _apply_block_filter(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return merged

    query_user_id = normalize_user_id(merged[QUERY_ID_COLUMN].dropna().iloc[0], allow_hash=True) if QUERY_ID_COLUMN in merged.columns and not merged[QUERY_ID_COLUMN].dropna().empty else None
    if query_user_id is None:
        return merged

    result = merged.copy()
    result[CANDIDATE_ID_COLUMN] = (
        result[CANDIDATE_ID_COLUMN]
        .map(lambda raw_id: normalize_user_id(raw_id, allow_hash=True))
        .astype("string")
    )
    try:
        blocked_candidate_ids = _BLOCKLIST_SERVICE.get_blocked_targets(
            actor_user_id=query_user_id,
            candidate_user_ids=result[CANDIDATE_ID_COLUMN].dropna().astype("string").tolist(),
        )
    except Exception:
        blocked_candidate_ids = set()
    result = result[
        result[CANDIDATE_ID_COLUMN].notna()
        & (result[CANDIDATE_ID_COLUMN] != query_user_id)
        & (~result[CANDIDATE_ID_COLUMN].isin(blocked_candidate_ids))
    ].copy()
    return result.reset_index(drop=True)
