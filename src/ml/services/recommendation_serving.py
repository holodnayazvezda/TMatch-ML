from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import pandas as pd
import structlog

from core.config import GlobalConfig
from ml.explanation import build_exact_reasons
from ml.infra.blocked_service import InteractionBlocklistService
from ml.ml_config import RerankConfig
from ml.retrieval import collect_retrieval_candidates
from ml.services.recommendation_constraints import UserTargetingProfile, filter_candidate_frame_by_targeting
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
    request_started = perf_counter()
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
    targeting_profile_cache: dict[str, UserTargetingProfile] = {}

    step_started = perf_counter()
    primary_df = _assemble_primary_candidates(
        stage=stage,
        user_id=normalized_user_id,
        has_transaction_profile=has_transaction_profile,
        runtime_config=runtime_config,
        conf=conf,
        g_conf=g_conf,
    )
    _log_step_timing(
        user_id=normalized_user_id,
        stage=stage,
        step="primary_retrieval",
        started_at=step_started,
        candidate_df=primary_df,
    )

    step_started = perf_counter()
    filtered_primary_df = _apply_hard_filters(
        primary_df,
        actor_user_id=normalized_user_id,
        targeting_profile_cache=targeting_profile_cache,
    )
    _log_step_timing(
        user_id=normalized_user_id,
        stage=stage,
        step="hard_filters_primary",
        started_at=step_started,
        candidate_df=filtered_primary_df,
    )

    step_started = perf_counter()
    ranked_df = _apply_ranking_if_available(
        primary_df=filtered_primary_df,
        stage=stage,
        allow_ranking=has_transaction_profile
        and stage in {RecommendationStage.STAGE_2, RecommendationStage.STAGE_3, RecommendationStage.STAGE_4},
        runtime_config=runtime_config,
    )
    _log_step_timing(
        user_id=normalized_user_id,
        stage=stage,
        step="ranking",
        started_at=step_started,
        candidate_df=ranked_df,
    )

    step_started = perf_counter()
    prepared_df = _prepare_candidate_reasons(ranked_df)
    _log_step_timing(
        user_id=normalized_user_id,
        stage=stage,
        step="reason_preparation",
        started_at=step_started,
        candidate_df=prepared_df,
    )

    step_started = perf_counter()
    prepared_df = _annotate_shown_history(prepared_df, actor_user_id=normalized_user_id)
    _log_step_timing(
        user_id=normalized_user_id,
        stage=stage,
        step="shown_history_annotation_primary",
        started_at=step_started,
        candidate_df=prepared_df,
    )

    step_started = perf_counter()
    finalized_df = _fallback_if_needed(
        prepared_df,
        stage=stage,
        user_id=normalized_user_id,
        has_transaction_profile=has_transaction_profile,
        conf=conf,
        top_n=g_conf.n_rec_users,
        targeting_profile_cache=targeting_profile_cache,
    )
    _log_step_timing(
        user_id=normalized_user_id,
        stage=stage,
        step="fallback_chain",
        started_at=step_started,
        candidate_df=finalized_df,
    )

    step_started = perf_counter()
    result_df = _return_top_n(
        finalized_df,
        actor_user_id=normalized_user_id,
        top_n=g_conf.n_rec_users,
        targeting_profile_cache=targeting_profile_cache,
    )
    _log_step_timing(
        user_id=normalized_user_id,
        stage=stage,
        step="final_assembly",
        started_at=step_started,
        candidate_df=result_df,
    )
    log.info(
        "recommendation_candidates_finalized",
        user_id=normalized_user_id,
        stage=stage.value,
        elapsed_ms=round((perf_counter() - request_started) * 1000, 2),
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


def _apply_hard_filters(
    candidate_df: pd.DataFrame,
    *,
    actor_user_id: str,
    targeting_profile_cache: dict[str, UserTargetingProfile] | None = None,
) -> pd.DataFrame:
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
    result = _apply_targeting_filters(
        result,
        actor_user_id=actor_user_id,
        targeting_profile_cache=targeting_profile_cache,
    )
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
            degraded_mode="fail_open",
        )
        return candidate_df.reset_index(drop=True)

    before_count = int(candidate_df.shape[0])
    result = candidate_df[~candidate_df["candidate_id"].isin(blocked_candidate_ids)].copy()
    removed_count = before_count - int(result.shape[0])
    if removed_count > 0:
        log.info(
            "recommendation_pair_state_filter_applied",
            actor_user_id=actor_user_id,
            before_count=before_count,
            after_count=int(result.shape[0]),
            removed_count=removed_count,
        )
    return result.reset_index(drop=True)


def _apply_targeting_filters(
    candidate_df: pd.DataFrame,
    *,
    actor_user_id: str,
    targeting_profile_cache: dict[str, UserTargetingProfile] | None = None,
) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    try:
        result = filter_candidate_frame_by_targeting(
            candidate_df,
            actor_user_id=actor_user_id,
            profile_cache=targeting_profile_cache,
        )
    except Exception as exc:
        log.warning(
            "recommendation_targeting_filter_failed",
            actor_user_id=actor_user_id,
            error=str(exc),
            degraded_mode="fail_open",
        )
        return candidate_df.reset_index(drop=True)

    before_count = int(candidate_df.shape[0])
    after_count = int(result.shape[0])
    removed_count = before_count - after_count
    if before_count > 0 and after_count == 0:
        log.warning(
            "recommendation_targeting_filter_degraded",
            actor_user_id=actor_user_id,
            before_count=before_count,
            removed_count=removed_count,
            degraded_mode="all_removed_fail_open",
        )
        return candidate_df.reset_index(drop=True)
    if removed_count > 0:
        log.info(
            "recommendation_targeting_filter_applied",
            actor_user_id=actor_user_id,
            before_count=before_count,
            after_count=after_count,
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
    targeting_profile_cache: dict[str, UserTargetingProfile] | None = None,
) -> pd.DataFrame:
    from ml.candidates import exotic_candidates, popularity_fallback, transactions_candidates

    result = _dedupe_candidates(candidate_df)
    if _count_fresh_candidates(result) >= int(top_n):
        return result

    fallback_conf = _build_fallback_conf(conf, top_n=top_n)
    fallback_sequence = []
    if has_transaction_profile:
        fallback_sequence.append(("transactions", transactions_candidates.get_transaction_candidates))
    if stage != RecommendationStage.STAGE_1 or has_transaction_profile:
        fallback_sequence.append(("popularity", popularity_fallback.get_popular_candidates))
    fallback_sequence.append(("exotic", exotic_candidates.get_exotic_candidates))

    for source_name, fetcher in fallback_sequence:
        if _count_fresh_candidates(result) >= int(top_n):
            break

        source_started = perf_counter()
        fallback_df = _safe_fetch(fetcher, fallback_conf)
        fetched_count = int(fallback_df.shape[0])
        fallback_df = attach_feature_snapshots(fallback_df, actor_user_id=user_id, stage=stage)
        with_features_count = int(fallback_df.shape[0])
        fallback_df = _apply_hard_filters(
            fallback_df,
            actor_user_id=user_id,
            targeting_profile_cache=targeting_profile_cache,
        )
        post_hard_filters_count = int(fallback_df.shape[0])
        fallback_df = _prepare_candidate_reasons(fallback_df)
        post_reason_count = int(fallback_df.shape[0])
        fallback_df = _annotate_shown_history(fallback_df, actor_user_id=user_id)
        post_shown_count = int(fallback_df.shape[0])
        result = _dedupe_candidates(pd.concat([result, fallback_df], ignore_index=True))
        log.info(
            "recommendation_fallback_applied",
            user_id=user_id,
            stage=stage.value,
            fallback_source=source_name,
            elapsed_ms=round((perf_counter() - source_started) * 1000, 2),
            fetched_count=fetched_count,
            with_features_count=with_features_count,
            post_hard_filters_count=post_hard_filters_count,
            post_reason_count=post_reason_count,
            post_shown_count=post_shown_count,
            current_count=int(result.shape[0]),
            current_fresh_count=_count_fresh_candidates(result),
        )
    return result


def _return_top_n(
    candidate_df: pd.DataFrame,
    *,
    actor_user_id: str,
    top_n: int,
    targeting_profile_cache: dict[str, UserTargetingProfile] | None = None,
) -> pd.DataFrame:
    filtered_df = _apply_hard_filters(
        candidate_df,
        actor_user_id=actor_user_id,
        targeting_profile_cache=targeting_profile_cache,
    )
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
    if not bundle:
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


def _build_fallback_conf(conf, *, top_n: int):
    if not isinstance(conf, RerankConfig):
        return conf
    fallback_top_n = max(int(top_n) * 12, 40)
    configured_top_n = int(getattr(conf, "top_n", 0) or 0)
    if configured_top_n > 0:
        fallback_top_n = min(fallback_top_n, configured_top_n)
    return replace(conf, top_n=max(fallback_top_n, int(top_n)))


def _log_step_timing(
    *,
    user_id: str,
    stage: RecommendationStage,
    step: str,
    started_at: float,
    candidate_df: pd.DataFrame | None = None,
) -> None:
    payload = {
        "user_id": user_id,
        "stage": stage.value,
        "step": step,
        "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
    }
    if candidate_df is not None:
        payload["candidate_count"] = int(candidate_df.shape[0])
        payload["fresh_candidate_count"] = _count_fresh_candidates(candidate_df)
    log.info("recommendation_step_timing", **payload)


def _log_source_counts(candidate_df: pd.DataFrame, *, stage: RecommendationStage, user_id: str) -> None:
    if candidate_df.empty:
        log.info(
            "recommendation_candidate_counts",
            user_id=user_id,
            stage=stage.value,
            phase="primary_retrieval",
            total_count=0,
        )
        return

    counts = {
        "phase": "primary_retrieval",
        "total_count": int(candidate_df.shape[0]),
    }
    has_source_specific_metrics = False
    retrieval_flag_columns = {
        "transactions_count": "retrieval_transactions_is_candidate",
        "interaction_count": "retrieval_tx_interaction_is_candidate",
        "exotic_count": "retrieval_exotic_is_candidate",
        "popularity_count": "retrieval_popularity_is_candidate",
    }
    for output_name, column_name in retrieval_flag_columns.items():
        if column_name in candidate_df.columns:
            counts[output_name] = int(pd.to_numeric(candidate_df[column_name], errors="coerce").fillna(0).sum())
            has_source_specific_metrics = True

    if not has_source_specific_metrics:
        counts["source_order_count"] = int(candidate_df.shape[0])

    log.info("recommendation_candidate_counts", user_id=user_id, stage=stage.value, **counts)
