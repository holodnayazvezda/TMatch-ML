from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np
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
_UNKNOWN_CLUSTER_ID = "unknown"
_UNCERTAINTY_TOP_SCORE_COUNT = 5
_UNCERTAINTY_TOP_POOL_SIZE = 15


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
    uncertainty_adjusted_df = _apply_uncertainty_strategy(
        candidate_df=ranked_df,
        stage=stage,
        actor_user_id=normalized_user_id,
        conf=conf,
        top_n=g_conf.n_rec_users,
        targeting_profile_cache=targeting_profile_cache,
    )
    _log_step_timing(
        user_id=normalized_user_id,
        stage=stage,
        step="uncertainty_strategy",
        started_at=step_started,
        candidate_df=uncertainty_adjusted_df,
    )

    step_started = perf_counter()
    prepared_df = _prepare_candidate_reasons(uncertainty_adjusted_df)
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


def _apply_uncertainty_strategy(
    *,
    candidate_df: pd.DataFrame,
    stage: RecommendationStage,
    actor_user_id: str,
    conf: RerankConfig,
    top_n: int,
    targeting_profile_cache: dict[str, UserTargetingProfile] | None = None,
) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    top_k = max(int(top_n), 0)
    if top_k <= 0:
        return candidate_df.iloc[0:0].copy()

    result = _ensure_candidate_ids(candidate_df)
    result = result[result["candidate_id"].notna() & (result["candidate_id"] != "")].copy()
    if result.empty:
        return result.reset_index(drop=True)

    result = _ensure_transaction_cluster_column(result)
    base_sorted_df = _sort_candidates_for_score(result, score_column="rerank_score")
    if stage not in {
        RecommendationStage.STAGE_2,
        RecommendationStage.STAGE_3,
        RecommendationStage.STAGE_4,
    }:
        return base_sorted_df.reset_index(drop=True)

    mean_all_score, gap_score, finite_score_count = _compute_uncertainty_metrics(base_sorted_df)
    if mean_all_score is None or gap_score is None:
        return base_sorted_df.reset_index(drop=True)

    threshold_mean = float(getattr(conf, "uncertainty_mean_score_threshold", 0.0) or 0.0)
    threshold_gap = float(getattr(conf, "uncertainty_gap_threshold", 0.05) or 0.05)
    check_1 = mean_all_score > threshold_mean
    check_2 = gap_score > threshold_gap

    unknown_cluster_count = int((base_sorted_df["transaction_cluster_id"] == _UNKNOWN_CLUSTER_ID).sum())
    log.info(
        "recommendation_uncertainty_metrics",
        actor_user_id=actor_user_id,
        stage=stage.value,
        candidate_count=int(base_sorted_df.shape[0]),
        finite_score_count=int(finite_score_count),
        mean_all_score=float(mean_all_score),
        gap_score=float(gap_score),
        uncertainty_mean_score_threshold=threshold_mean,
        uncertainty_gap_threshold=threshold_gap,
        check_1=bool(check_1),
        check_2=bool(check_2),
        unknown_cluster_count=unknown_cluster_count,
    )

    scenario = "s1"
    if check_1 and check_2:
        selected_df = base_sorted_df.head(top_k).reset_index(drop=True)
    elif check_1 and not check_2:
        scenario = "s2"
        top_pool_df = base_sorted_df.head(_UNCERTAINTY_TOP_POOL_SIZE).copy()
        diverse_df = _select_cluster_diverse_candidates(
            top_pool_df,
            top_k=top_k,
            score_column="rerank_score",
        )
        selected_df = _fill_with_ordered_candidates(
            selected_df=diverse_df,
            ordered_df=base_sorted_df,
            top_k=top_k,
        )
    elif (not check_1) and check_2:
        scenario = "s3"
        diverse_df = _select_cluster_diverse_candidates(
            base_sorted_df,
            top_k=top_k,
            score_column="rerank_score",
        )
        selected_df = _fill_with_ordered_candidates(
            selected_df=diverse_df,
            ordered_df=base_sorted_df,
            top_k=top_k,
        )
    else:
        scenario = "s4"
        selected_df = _build_popularity_diverse_candidates(
            actor_user_id=actor_user_id,
            stage=stage,
            conf=conf,
            top_k=top_k,
            base_sorted_df=base_sorted_df,
            targeting_profile_cache=targeting_profile_cache,
        )

    selected_df = _ensure_transaction_cluster_column(selected_df)
    selected_df = selected_df.head(top_k).reset_index(drop=True)
    log.info(
        "recommendation_uncertainty_scenario_selected",
        actor_user_id=actor_user_id,
        stage=stage.value,
        scenario=scenario,
        pre_count=int(base_sorted_df.shape[0]),
        post_count=int(selected_df.shape[0]),
    )
    return selected_df


def _compute_uncertainty_metrics(candidate_df: pd.DataFrame) -> tuple[float | None, float | None, int]:
    if candidate_df.empty or "rerank_score" not in candidate_df.columns:
        return None, None, 0

    score_series = pd.to_numeric(candidate_df["rerank_score"], errors="coerce")
    score_values = score_series.to_numpy(dtype=float, copy=False)
    finite_values = score_values[np.isfinite(score_values)]
    finite_score_count = int(finite_values.size)
    if finite_score_count <= 0:
        return None, None, 0

    mean_all_score = float(np.mean(finite_values))
    sorted_scores = np.sort(finite_values)[::-1]
    top_count = min(_UNCERTAINTY_TOP_SCORE_COUNT, finite_score_count)
    top_mean = float(np.mean(sorted_scores[:top_count]))
    rest_scores = sorted_scores[top_count:]
    if rest_scores.size == 0:
        gap_score = 0.0
    else:
        gap_score = float(top_mean - np.mean(rest_scores))

    return mean_all_score, gap_score, finite_score_count


def _select_cluster_diverse_candidates(
    candidate_df: pd.DataFrame,
    *,
    top_k: int,
    score_column: str = "rerank_score",
) -> pd.DataFrame:
    if candidate_df.empty or int(top_k) <= 0:
        return candidate_df.iloc[0:0].copy()

    normalized_df = _ensure_transaction_cluster_column(candidate_df)
    ordered_df = _sort_candidates_for_score(normalized_df, score_column=score_column)
    if ordered_df.empty:
        return ordered_df

    selected_positions: list[int] = []
    selected_clusters: set[str] = set()
    for row_position, row in ordered_df.iterrows():
        cluster_id = _normalize_cluster_id(row.get("transaction_cluster_id"))
        if cluster_id in selected_clusters:
            continue
        selected_positions.append(int(row_position))
        selected_clusters.add(cluster_id)
        if len(selected_positions) >= int(top_k):
            break

    selected_df = ordered_df.loc[selected_positions].copy() if selected_positions else ordered_df.iloc[0:0].copy()
    return _fill_with_ordered_candidates(
        selected_df=selected_df,
        ordered_df=ordered_df,
        top_k=int(top_k),
    )


def _build_popularity_diverse_candidates(
    *,
    actor_user_id: str,
    stage: RecommendationStage,
    conf: RerankConfig,
    top_k: int,
    base_sorted_df: pd.DataFrame,
    targeting_profile_cache: dict[str, UserTargetingProfile] | None = None,
) -> pd.DataFrame:
    from ml.candidates import popularity_fallback

    fallback_conf = _build_fallback_conf(conf, top_n=top_k)
    popularity_df = _safe_fetch(popularity_fallback.get_popular_candidates, fallback_conf)
    fetched_count = int(popularity_df.shape[0])
    if popularity_df.empty:
        return _fill_with_ordered_candidates(
            selected_df=popularity_df,
            ordered_df=base_sorted_df,
            top_k=top_k,
        )

    popularity_df = attach_feature_snapshots(popularity_df, actor_user_id=actor_user_id, stage=stage)
    popularity_df = _apply_hard_filters(
        popularity_df,
        actor_user_id=actor_user_id,
        targeting_profile_cache=targeting_profile_cache,
    )
    filtered_count = int(popularity_df.shape[0])
    popularity_df = _ensure_transaction_cluster_column(popularity_df)
    popularity_ordered_df = _sort_candidates_for_score(popularity_df, score_column="popularity_score")
    popularity_diverse_df = _select_cluster_diverse_candidates(
        popularity_ordered_df,
        top_k=top_k,
        score_column="popularity_score",
    )
    diversity_count = int(popularity_diverse_df.shape[0])
    selected_df = _fill_with_ordered_candidates(
        selected_df=popularity_diverse_df,
        ordered_df=popularity_ordered_df,
        top_k=top_k,
    )
    filled_from_popularity_count = int(selected_df.shape[0])

    if int(selected_df.shape[0]) < int(top_k):
        selected_df = _fill_with_ordered_candidates(
            selected_df=selected_df,
            ordered_df=base_sorted_df,
            top_k=top_k,
        )
    total_count = int(selected_df.shape[0])
    log.info(
        "recommendation_uncertainty_s4_popularity_applied",
        actor_user_id=actor_user_id,
        stage=stage.value,
        fetched_count=fetched_count,
        filtered_count=filtered_count,
        diversity_count=diversity_count,
        filled_from_popularity_count=filled_from_popularity_count,
        total_count=total_count,
    )

    return selected_df.reset_index(drop=True)


def _fill_with_ordered_candidates(
    *,
    selected_df: pd.DataFrame,
    ordered_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    if int(top_k) <= 0:
        return ordered_df.iloc[0:0].copy()

    selected = _ensure_candidate_ids(selected_df) if not selected_df.empty else selected_df.copy()
    ordered = _ensure_candidate_ids(ordered_df) if not ordered_df.empty else ordered_df.copy()
    if ordered.empty:
        return selected.head(int(top_k)).reset_index(drop=True)

    if selected.empty:
        return ordered.head(int(top_k)).reset_index(drop=True)

    selected_candidate_ids = set(selected["candidate_id"].dropna().astype("string").tolist())
    remaining_df = ordered[~ordered["candidate_id"].astype("string").isin(selected_candidate_ids)].copy()
    required_count = max(int(top_k) - int(selected.shape[0]), 0)
    if required_count > 0:
        selected = pd.concat([selected, remaining_df.head(required_count)], ignore_index=True)
    return selected.head(int(top_k)).reset_index(drop=True)


def _sort_candidates_for_score(candidate_df: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    result = _ensure_candidate_ids(candidate_df)
    result = result[result["candidate_id"].notna() & (result["candidate_id"] != "")].copy()
    if result.empty:
        return result.reset_index(drop=True)

    sort_columns: list[str] = []
    ascending: list[bool] = []

    if score_column in result.columns:
        result["_sort_score"] = pd.to_numeric(result[score_column], errors="coerce")
        sort_columns.append("_sort_score")
        ascending.append(False)
    if "rank" in result.columns:
        result["_sort_rank"] = pd.to_numeric(result["rank"], errors="coerce")
        sort_columns.append("_sort_rank")
        ascending.append(True)
    result["_sort_candidate_id"] = result["candidate_id"].astype("string")
    sort_columns.append("_sort_candidate_id")
    ascending.append(True)

    result = result.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    return result.drop(columns=["_sort_score", "_sort_rank", "_sort_candidate_id"], errors="ignore")


def _ensure_transaction_cluster_column(candidate_df: pd.DataFrame) -> pd.DataFrame:
    result = _ensure_candidate_ids(candidate_df)
    if result.empty:
        return result

    if "transaction_cluster_id" not in result.columns:
        result["transaction_cluster_id"] = pd.Series(pd.NA, index=result.index, dtype="string")
        cluster_lookup: dict[str, str] = {}
        candidate_ids = result["candidate_id"].dropna().astype("string").tolist()
        if candidate_ids:
            from ml.candidates import transactions_candidates

            try:
                cluster_lookup = transactions_candidates.get_transaction_cluster_ids(
                    candidate_ids,
                    unknown_cluster_id=_UNKNOWN_CLUSTER_ID,
                )
            except Exception:
                cluster_lookup = {}
        result["transaction_cluster_id"] = result["candidate_id"].map(
            lambda candidate_id: cluster_lookup.get(str(candidate_id), _UNKNOWN_CLUSTER_ID)
        )
    else:
        result["transaction_cluster_id"] = result["transaction_cluster_id"].astype("string")

    result["transaction_cluster_id"] = (
        result["transaction_cluster_id"]
        .map(_normalize_cluster_id)
        .astype("string")
    )
    return result


def _normalize_cluster_id(raw_cluster_id: object) -> str:
    if raw_cluster_id is None:
        return _UNKNOWN_CLUSTER_ID
    try:
        if pd.isna(raw_cluster_id):
            return _UNKNOWN_CLUSTER_ID
    except TypeError:
        pass
    if isinstance(raw_cluster_id, float) and pd.isna(raw_cluster_id):
        return _UNKNOWN_CLUSTER_ID
    cluster_text = str(raw_cluster_id).strip()
    if not cluster_text or cluster_text.lower() in {"nan", "none", "<na>"}:
        return _UNKNOWN_CLUSTER_ID
    return cluster_text


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

    result = _exclude_already_shown_candidates(_dedupe_candidates(candidate_df))
    if _count_fresh_candidates(result) >= int(top_n):
        return result

    candidate_pool_size = _get_offline_candidate_pool_size(actor_user_id=user_id)
    fallback_conf = _build_fallback_conf(
        conf,
        top_n=top_n,
        candidate_pool_size=candidate_pool_size,
    )
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
        fallback_df = _exclude_already_shown_candidates(fallback_df)
        post_fresh_only_count = int(fallback_df.shape[0])
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
            post_fresh_only_count=post_fresh_only_count,
            current_count=int(result.shape[0]),
            current_fresh_count=_count_fresh_candidates(result),
            candidate_pool_size=candidate_pool_size,
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
    result_df = fresh_df.head(int(top_n)).reset_index(drop=True)
    if int(result_df.shape[0]) < int(top_n):
        log.info(
            "recommendation_fresh_candidates_exhausted",
            actor_user_id=actor_user_id,
            requested_count=int(top_n),
            available_fresh_count=int(result_df.shape[0]),
            total_candidate_count=int(deduped_df.shape[0]),
        )
    return result_df


def get_recommendation_cycle_state(
    *,
    actor_user_id: str,
    stage: RecommendationStage,
    targeting_profile_cache: dict[str, UserTargetingProfile] | None = None,
) -> dict[str, int | bool]:
    pool_candidate_ids = _get_offline_candidate_ids(actor_user_id=actor_user_id)
    total_pool_count = len(pool_candidate_ids)
    if total_pool_count <= 0:
        payload = {
            "is_exhausted": False,
            "total_pool_count": 0,
            "eligible_pool_count": 0,
            "shown_eligible_count": 0,
        }
        log.info(
            "recommendation_cycle_exhaustion_evaluated",
            actor_user_id=actor_user_id,
            stage=stage.value,
            **payload,
        )
        return payload

    pool_df = pd.DataFrame({"candidate_id": sorted(pool_candidate_ids)})
    eligible_df = _apply_hard_filters(
        pool_df,
        actor_user_id=actor_user_id,
        targeting_profile_cache=targeting_profile_cache,
    )
    eligible_candidate_ids = (
        eligible_df.get("candidate_id", pd.Series(dtype="string"))
        .dropna()
        .astype("string")
        .tolist()
    )
    eligible_pool_count = len(eligible_candidate_ids)
    if eligible_pool_count <= 0:
        payload = {
            "is_exhausted": False,
            "total_pool_count": total_pool_count,
            "eligible_pool_count": 0,
            "shown_eligible_count": 0,
        }
        log.info(
            "recommendation_cycle_exhaustion_evaluated",
            actor_user_id=actor_user_id,
            stage=stage.value,
            **payload,
        )
        return payload

    shown_candidate_ids = get_shown_candidate_ids(
        actor_user_id,
        candidate_user_ids=eligible_candidate_ids,
    )
    shown_eligible_count = len(shown_candidate_ids)
    is_exhausted = shown_eligible_count >= eligible_pool_count
    payload = {
        "is_exhausted": is_exhausted,
        "total_pool_count": total_pool_count,
        "eligible_pool_count": eligible_pool_count,
        "shown_eligible_count": shown_eligible_count,
    }
    log.info(
        "recommendation_cycle_exhaustion_evaluated",
        actor_user_id=actor_user_id,
        stage=stage.value,
        **payload,
    )
    return payload


def _exclude_already_shown_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty or "already_shown" not in candidate_df.columns:
        return candidate_df.reset_index(drop=True)

    before_count = int(candidate_df.shape[0])
    result = candidate_df[~candidate_df["already_shown"].astype(bool)].copy()
    removed_count = before_count - int(result.shape[0])
    if removed_count > 0:
        log.info(
            "recommendation_shown_candidates_removed",
            before_count=before_count,
            after_count=int(result.shape[0]),
            removed_count=removed_count,
        )
    return result.reset_index(drop=True)


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


def _build_fallback_conf(conf, *, top_n: int, candidate_pool_size: int = 0):
    if not isinstance(conf, RerankConfig):
        return conf
    fallback_top_n = max(int(top_n) * 12, 40)
    configured_top_n = int(getattr(conf, "top_n", 0) or 0)
    if int(candidate_pool_size) > 0:
        fallback_top_n = max(fallback_top_n, candidate_pool_size)
    if configured_top_n > 0:
        fallback_top_n = max(fallback_top_n, configured_top_n)
    return replace(conf, top_n=max(fallback_top_n, int(top_n)))


def _get_offline_candidate_pool_size(*, actor_user_id: str | None) -> int:
    normalized_actor_user_id = normalize_user_id(actor_user_id, allow_hash=True)
    if normalized_actor_user_id is None:
        return 0
    return len(_get_offline_candidate_ids(actor_user_id=normalized_actor_user_id))


def _get_offline_candidate_ids(*, actor_user_id: str) -> set[str]:
    from ml.candidates import interactions_candidates, transactions_candidates

    candidate_ids: set[str] = set()
    warmup_by_source = {
        "transactions": transactions_candidates.warmup_search_bundle,
        "tx_interaction": interactions_candidates.warmup_search_bundle,
    }
    for source_name, warmup in warmup_by_source.items():
        try:
            bundle = warmup()
        except Exception as exc:
            log.warning(
                "recommendation_offline_pool_source_failed",
                actor_user_id=actor_user_id,
                source=source_name,
                error=str(exc),
            )
            continue
        if not bundle:
            continue

        source_user_ids = bundle.get("user_ids")
        if source_user_ids is None:
            continue
        for raw_candidate_user_id in source_user_ids:
            normalized_candidate_user_id = normalize_user_id(raw_candidate_user_id, allow_hash=True)
            if normalized_candidate_user_id is None:
                continue
            if normalized_candidate_user_id == actor_user_id:
                continue
            candidate_ids.add(normalized_candidate_user_id)
    return candidate_ids


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
