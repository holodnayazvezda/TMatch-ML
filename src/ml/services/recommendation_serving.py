from __future__ import annotations

from dataclasses import replace

import pandas as pd
import structlog

from core.config import GlobalConfig
from ml.explanation import build_exact_reasons
from ml.infra.blocked_service import InteractionBlocklistService
from ml.ml_config import RerankConfig
from ml.retrieval import collect_retrieval_candidates
from ml.services.recommendation_constraints import filter_candidate_frame_by_targeting
from ml.services.recommendation_feature_snapshot import attach_feature_snapshots
from ml.services.recommendation_ranking import resolve_model_family, score_candidate_frame
from ml.services.recommendation_runtime_config import RecommendationRuntimeConfig
from ml.services.recommendation_scoring import RecommendationStage
from ml.services.recommendation_store import get_shown_candidate_ids
from ml.services.user_id import normalize_user_id

log: structlog.BoundLogger = structlog.get_logger(__name__)
_BLOCKLIST_SERVICE = InteractionBlocklistService()
_REQUIRED_REASON_COUNT = 3


def get_recommendations_for_user(
    *,
    user_id: str,
    runtime_config: RecommendationRuntimeConfig,
    conf: RerankConfig,
    g_conf: GlobalConfig,
) -> pd.DataFrame:
    normalized_user_id = normalize_user_id(user_id, allow_hash=True)
    if normalized_user_id is None:
        return pd.DataFrame()

    stage = runtime_config.recommendation_stage
    has_transaction_profile = _has_transaction_profile(normalized_user_id)
    log.info(
        "recommendation_stage_selected",
        user_id=normalized_user_id,
        stage=stage.value,
        has_transaction_profile=has_transaction_profile,
    )

    primary_df = _assemble_primary_candidates(
        stage=stage,
        user_id=normalized_user_id,
        has_transaction_profile=has_transaction_profile,
        runtime_config=runtime_config,
        conf=conf,
        g_conf=g_conf,
    )
    filtered_primary_df = _apply_hard_filters(primary_df, actor_user_id=normalized_user_id)
    ranked_df = _apply_ranking_if_available(
        primary_df=filtered_primary_df,
        stage=stage,
        allow_ranking=has_transaction_profile
        and stage in {RecommendationStage.STAGE_2, RecommendationStage.STAGE_3, RecommendationStage.STAGE_4},
        runtime_config=runtime_config,
    )
    prepared_df = _prepare_candidate_reasons(ranked_df)
    prepared_df = _annotate_shown_history(prepared_df, actor_user_id=normalized_user_id)
    finalized_df = _fallback_if_needed(
        prepared_df,
        stage=stage,
        user_id=normalized_user_id,
        has_transaction_profile=has_transaction_profile,
        conf=conf,
        top_n=g_conf.n_rec_users,
    )
    result_df = _return_top_n(finalized_df, actor_user_id=normalized_user_id, top_n=g_conf.n_rec_users)
    log.info(
        "recommendation_candidates_finalized",
        user_id=normalized_user_id,
        stage=stage.value,
        final_count=int(result_df.shape[0]),
        final_candidate_ids=result_df.get("candidate_id", pd.Series(dtype="string")).astype("string").tolist(),
    )
    return result_df


def _assemble_primary_candidates(
    *,
    stage: RecommendationStage,
    user_id: str,
    has_transaction_profile: bool,
    runtime_config: RecommendationRuntimeConfig,
    conf: RerankConfig,
    g_conf: GlobalConfig,
) -> pd.DataFrame:
    from ml.candidates import exotic_candidates, popularity_fallback, transactions_candidates

    if stage == RecommendationStage.STAGE_1:
        source_df = _safe_fetch(
            transactions_candidates.get_transaction_candidates
            if has_transaction_profile
            else exotic_candidates.get_exotic_candidates,
            conf,
        )
        _log_source_counts(source_df, stage=stage, user_id=user_id)
        return attach_feature_snapshots(source_df, actor_user_id=user_id, stage=stage)

    if stage in {RecommendationStage.STAGE_2, RecommendationStage.STAGE_3}:
        if not has_transaction_profile:
            source_df = _safe_fetch(exotic_candidates.get_exotic_candidates, conf)
            _log_source_counts(source_df, stage=stage, user_id=user_id)
            return attach_feature_snapshots(source_df, actor_user_id=user_id, stage=stage)

        stage_conf = replace(conf, ratios=[0.5, 0.5, 0.0, 0.0])
        try:
            source_df = collect_retrieval_candidates(stage_conf, g_conf)
        except Exception:
            source_df = pd.DataFrame()
        _log_source_counts(source_df, stage=stage, user_id=user_id)
        return attach_feature_snapshots(source_df, actor_user_id=user_id, stage=stage)

    if has_transaction_profile:
        try:
            source_df = collect_retrieval_candidates(conf, g_conf)
        except Exception:
            source_df = pd.DataFrame()
        _log_source_counts(source_df, stage=stage, user_id=user_id)
        return attach_feature_snapshots(source_df, actor_user_id=user_id, stage=stage)

    source_df = _safe_fetch(popularity_fallback.get_popular_candidates, conf)
    if source_df.empty:
        source_df = _safe_fetch(exotic_candidates.get_exotic_candidates, conf)
    _log_source_counts(source_df, stage=stage, user_id=user_id)
    return attach_feature_snapshots(source_df, actor_user_id=user_id, stage=stage)


def _apply_ranking_if_available(
    *,
    primary_df: pd.DataFrame,
    stage: RecommendationStage,
    allow_ranking: bool,
    runtime_config: RecommendationRuntimeConfig,
) -> pd.DataFrame:
    if primary_df.empty or not allow_ranking:
        return primary_df

    family = resolve_model_family(stage.value)
    if family is None:
        return _sort_source_order(primary_df)

    model_dir = runtime_config.stage_23_model_dir if family == "stage_2_3" else runtime_config.stage_4_model_dir
    ranked_df = score_candidate_frame(
        primary_df,
        family=family,
        weights=runtime_config.ensemble_weights,
        model_dir=model_dir,
    )
    if "rerank_score" not in ranked_df.columns or ranked_df["rerank_score"].isna().all():
        log.info("recommendation_ranking_fallback_to_source_order", stage=stage.value, family=family)
        return _sort_source_order(primary_df)

    if "rank" in ranked_df.columns:
        return ranked_df.sort_values(
            ["rerank_score", "rank"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)

    return ranked_df.sort_values(
        ["rerank_score"],
        ascending=[False],
        na_position="last",
    ).reset_index(drop=True)


def _apply_hard_filters(candidate_df: pd.DataFrame, *, actor_user_id: str) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    result = _ensure_candidate_ids(candidate_df)
    result = result[
        result["candidate_id"].notna()
        & (result["candidate_id"] != "")
        & (result["candidate_id"] != actor_user_id)
    ].copy()
    if result.empty:
        return result.reset_index(drop=True)

    result = _apply_pair_state_filters(result, actor_user_id=actor_user_id)
    result = _apply_targeting_filters(result, actor_user_id=actor_user_id)
    return result.reset_index(drop=True)


def _apply_pair_state_filters(candidate_df: pd.DataFrame, *, actor_user_id: str) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    try:
        blocked_candidate_ids = _BLOCKLIST_SERVICE.get_blocked_targets(
            actor_user_id=actor_user_id,
            candidate_user_ids=candidate_df["candidate_id"].dropna().astype("string").tolist(),
        )
    except Exception as exc:
        log.warning(
            "recommendation_pair_state_filter_failed",
            actor_user_id=actor_user_id,
            error=str(exc),
        )
        return candidate_df.iloc[0:0].copy()

    before_count = int(candidate_df.shape[0])
    result = candidate_df[~candidate_df["candidate_id"].isin(blocked_candidate_ids)].copy()
    removed_count = before_count - int(result.shape[0])
    if removed_count > 0:
        log.info(
            "recommendation_pair_state_filter_applied",
            actor_user_id=actor_user_id,
            removed_count=removed_count,
        )
    return result.reset_index(drop=True)


def _apply_targeting_filters(candidate_df: pd.DataFrame, *, actor_user_id: str) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    try:
        result = filter_candidate_frame_by_targeting(
            candidate_df,
            actor_user_id=actor_user_id,
        )
    except Exception as exc:
        log.warning(
            "recommendation_targeting_filter_failed",
            actor_user_id=actor_user_id,
            error=str(exc),
        )
        return candidate_df.iloc[0:0].copy()

    removed_count = int(candidate_df.shape[0]) - int(result.shape[0])
    if removed_count > 0:
        log.info(
            "recommendation_targeting_filter_applied",
            actor_user_id=actor_user_id,
            removed_count=removed_count,
        )
    return result.reset_index(drop=True)


def _prepare_candidate_reasons(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    result = candidate_df.copy()
    reasons = []
    for row in result.to_dict(orient="records"):
        raw_reasons = row.get("impactful_features")
        if raw_reasons is None:
            raw_reasons = row.get("reasons")
        reasons.append(
            build_exact_reasons(
                raw_reasons,
                exact_count=_REQUIRED_REASON_COUNT,
            )
        )

    result["reasons"] = pd.Series(reasons, index=result.index, dtype=object)
    before_count = int(result.shape[0])
    result = result[result["reasons"].map(_has_exact_reasons)].copy()
    removed_count = before_count - int(result.shape[0])
    if removed_count > 0:
        log.info(
            "recommendation_reason_filter_applied",
            removed_count=removed_count,
        )
    return result.reset_index(drop=True)


def _annotate_shown_history(candidate_df: pd.DataFrame, *, actor_user_id: str) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    result = _ensure_candidate_ids(candidate_df)
    candidate_ids = result["candidate_id"].dropna().astype("string").tolist()
    try:
        shown_candidate_ids = get_shown_candidate_ids(
            actor_user_id,
            candidate_user_ids=candidate_ids,
        )
    except Exception as exc:
        log.warning(
            "recommendation_shown_history_lookup_failed",
            actor_user_id=actor_user_id,
            error=str(exc),
        )
        shown_candidate_ids = set()

    result["already_shown"] = result["candidate_id"].isin(shown_candidate_ids)
    shown_count = int(result["already_shown"].sum())
    if shown_count > 0:
        log.info(
            "recommendation_shown_history_applied",
            actor_user_id=actor_user_id,
            shown_count=shown_count,
        )
    return result.reset_index(drop=True)


def _fallback_if_needed(
    candidate_df: pd.DataFrame,
    *,
    stage: RecommendationStage,
    user_id: str,
    has_transaction_profile: bool,
    conf: RerankConfig,
    top_n: int,
) -> pd.DataFrame:
    from ml.candidates import exotic_candidates, popularity_fallback

    result = _dedupe_candidates(candidate_df)
    if _count_fresh_candidates(result) >= int(top_n):
        return result

    fallback_sequence = []
    if stage != RecommendationStage.STAGE_1 or has_transaction_profile:
        fallback_sequence.append(("popularity", popularity_fallback.get_popular_candidates))
    fallback_sequence.append(("exotic", exotic_candidates.get_exotic_candidates))

    for source_name, fetcher in fallback_sequence:
        if _count_fresh_candidates(result) >= int(top_n):
            break

        fallback_df = _safe_fetch(fetcher, conf)
        fallback_df = attach_feature_snapshots(fallback_df, actor_user_id=user_id, stage=stage)
        fallback_df = _apply_hard_filters(fallback_df, actor_user_id=user_id)
        fallback_df = _prepare_candidate_reasons(fallback_df)
        fallback_df = _annotate_shown_history(fallback_df, actor_user_id=user_id)
        result = _dedupe_candidates(pd.concat([result, fallback_df], ignore_index=True))
        log.info(
            "recommendation_fallback_applied",
            user_id=user_id,
            stage=stage.value,
            fallback_source=source_name,
            current_count=int(result.shape[0]),
            current_fresh_count=_count_fresh_candidates(result),
        )
    return result


def _return_top_n(candidate_df: pd.DataFrame, *, actor_user_id: str, top_n: int) -> pd.DataFrame:
    filtered_df = _apply_hard_filters(candidate_df, actor_user_id=actor_user_id)
    prepared_df = _prepare_candidate_reasons(filtered_df)
    annotated_df = _annotate_shown_history(prepared_df, actor_user_id=actor_user_id)
    deduped_df = _dedupe_candidates(annotated_df)

    if "already_shown" not in deduped_df.columns:
        deduped_df["already_shown"] = False

    fresh_df = deduped_df[~deduped_df["already_shown"]].copy()
    shown_df = deduped_df[deduped_df["already_shown"]].copy()
    result_df = pd.concat([fresh_df, shown_df], ignore_index=True).head(int(top_n)).reset_index(drop=True)

    recycled_count = int(result_df.get("already_shown", pd.Series(dtype=bool)).sum())
    if recycled_count > 0 and int(fresh_df.shape[0]) < int(top_n):
        log.info(
            "recommendation_recycled_shown_candidates",
            actor_user_id=actor_user_id,
            recycled_count=recycled_count,
        )
    return result_df


def _has_transaction_profile(user_id: str) -> bool:
    from ml.candidates import transactions_candidates

    try:
        bundle = transactions_candidates.warmup_search_bundle()
    except Exception:
        return False
    return user_id in bundle["id_index"]


def _ensure_candidate_ids(candidate_df: pd.DataFrame) -> pd.DataFrame:
    result = candidate_df.copy()
    result["candidate_id"] = pd.Series(
        _extract_candidate_ids(result),
        index=result.index,
        dtype="string",
    )
    return result


def _extract_candidate_ids(candidate_df: pd.DataFrame) -> list[str]:
    result: list[str] = []
    for row in candidate_df.to_dict(orient="records"):
        candidate_user_id = None
        for key in ("candidate_id", "party_rk", "user_id", "target_user_id", "id"):
            candidate_user_id = normalize_user_id(row.get(key), allow_hash=True)
            if candidate_user_id is not None:
                break
        result.append(candidate_user_id or "")
    return result


def _count_fresh_candidates(candidate_df: pd.DataFrame) -> int:
    if candidate_df.empty:
        return 0
    if "already_shown" not in candidate_df.columns:
        return int(candidate_df.shape[0])
    return int((~candidate_df["already_shown"].astype(bool)).sum())


def _has_exact_reasons(raw_reasons: object) -> bool:
    return isinstance(raw_reasons, dict) and len(raw_reasons) == _REQUIRED_REASON_COUNT


def _dedupe_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    result = _ensure_candidate_ids(candidate_df)
    result = result[result["candidate_id"].notna() & (result["candidate_id"] != "")].copy()
    if "rerank_score" in result.columns and "rank" in result.columns:
        result = result.sort_values(["rerank_score", "rank"], ascending=[False, True], na_position="last")
    elif "rerank_score" in result.columns:
        result = result.sort_values(["rerank_score"], ascending=[False], na_position="last")
    elif "rank" in result.columns:
        result = result.sort_values(["rank"], ascending=[True], na_position="last")
    result = result.drop_duplicates(subset=["candidate_id"], keep="first")
    return result.reset_index(drop=True)


def _sort_source_order(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty or "rank" not in candidate_df.columns:
        return candidate_df.reset_index(drop=True)
    return candidate_df.sort_values(["rank"], ascending=[True], na_position="last").reset_index(drop=True)


def _safe_fetch(fetcher, conf: RerankConfig) -> pd.DataFrame:
    try:
        source_df = fetcher(conf)
    except Exception:
        return pd.DataFrame()
    if source_df is None:
        return pd.DataFrame()
    return source_df.copy()


def _log_source_counts(candidate_df: pd.DataFrame, *, stage: RecommendationStage, user_id: str) -> None:
    if candidate_df.empty:
        log.info("recommendation_candidate_counts", user_id=user_id, stage=stage.value, total_count=0)
        return

    counts = {
        "total_count": int(candidate_df.shape[0]),
    }
    retrieval_flag_columns = {
        "transactions_count": "retrieval_transactions_is_candidate",
        "interaction_count": "retrieval_tx_interaction_is_candidate",
        "exotic_count": "retrieval_exotic_is_candidate",
        "popularity_count": "retrieval_popularity_is_candidate",
    }
    for output_name, column_name in retrieval_flag_columns.items():
        if column_name in candidate_df.columns:
            counts[output_name] = int(pd.to_numeric(candidate_df[column_name], errors="coerce").fillna(0).sum())

    if len(counts) == 1:
        counts["source_order_count"] = int(candidate_df.shape[0])

    log.info("recommendation_candidate_counts", user_id=user_id, stage=stage.value, **counts)
