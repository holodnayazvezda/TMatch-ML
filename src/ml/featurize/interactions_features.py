from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import polars as pl

from src.core.data_config import Paths

PATHS = Paths()

INPUT_PATH = PATHS.in_data_path / "synth" / "synthetic_events.jsonl"
ARTIFACTS_DIR = PATHS.out_data_path
TMP_DIR = ARTIFACTS_DIR / "_tmp_interactions_features"

FILTERED_EVENTS_PATH = TMP_DIR / "filtered_events.parquet"
ACTOR_FEATURES_PATH = TMP_DIR / "actor_features.parquet"
RECEIVED_FEATURES_PATH = TMP_DIR / "received_features.parquet"
RECENT_FEATURES_PATH = TMP_DIR / "recent_features.parquet"
GAP_FEATURES_PATH = TMP_DIR / "gap_features.parquet"
PAIR_USER_FEATURES_PATH = TMP_DIR / "pair_user_features.parquet"

FINAL_PARQUET_PATH = ARTIFACTS_DIR / "user_interactions_all.parquet"
FINAL_CSV_PATH = ARTIFACTS_DIR / "user_interactions_all.csv"

STAGE_SOURCE_PATH = Path(__file__).resolve()

ACTIONS = [
    "like",
    "dislike",
    "skip",
    "match",
    "chat_started",
    "block",
    "unblock",
]

ACTION_SUFFIXES = {
    "like": "likes",
    "dislike": "dislikes",
    "skip": "skips",
    "match": "matches",
    "chat_started": "chat_started",
    "block": "blocks",
    "unblock": "unblocks",
}

POSITIVE_ACTIONS = ["like", "match", "chat_started"]
NEGATIVE_ACTIONS = ["dislike", "skip", "block"]

WINDOW_DELTAS = {
    "10m": timedelta(minutes=10),
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
}

RECENT_WINDOW_SIZES = [5, 10, 20, 50]
RECENT_ACTION_MAX_DEPTH = max(RECENT_WINDOW_SIZES)

PREFERENCE_WEIGHTS = {
    "like": 1.0,
    "dislike": -1.0,
    "skip": -0.5,
    "match": 3.0,
    "chat_started": 5.0,
    "block": -5.0,
    "unblock": 2.0,
}

DECAY_HALF_LIFE_DAYS = 7.0
DECAY_LAMBDA = math.log(2.0) / DECAY_HALF_LIFE_DAYS

FLOAT_DTYPES = {pl.Float32, pl.Float64}
INTEGER_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}

BASE_EVENT_SCHEMA = {
    "event_id": pl.Utf8,
    "event_type": pl.Utf8,
    "occurred_at": pl.Datetime,
    "actor_user_id": pl.Utf8,
    "target_user_id": pl.Utf8,
    "action": pl.Utf8,
}


def collect_stream(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def sink_parquet(path: Path, lf: pl.LazyFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lf.sink_parquet(path, compression="zstd", statistics=True)
    except TypeError:
        collect_stream(lf).write_parquet(path, compression="zstd")


def needs_rebuild(output_path: Path, dependencies: list[Path]) -> bool:
    if not output_path.exists():
        return True

    output_mtime = output_path.stat().st_mtime
    for dependency in dependencies:
        if dependency.exists() and dependency.stat().st_mtime > output_mtime:
            return True
    return False


def sink_parquet_stage(path: Path, lf: pl.LazyFrame, dependencies: list[Path]) -> None:
    if needs_rebuild(path, dependencies):
        sink_parquet(path, lf)


def bool_count_expr(condition: pl.Expr, alias: str) -> pl.Expr:
    return condition.cast(pl.UInt32).sum().alias(alias)


def safe_rate_expr(numerator: str, denominator: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(0.0)
        .cast(pl.Float64)
        .alias(alias)
    )


def safe_ratio_expr(numerator: str, denominator: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator) / pl.col(denominator))
        .otherwise(0.0)
        .cast(pl.Float64)
        .alias(alias)
    )


def seconds_since_expr(snapshot_ts: datetime, source_column: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(source_column).is_not_null())
        .then((pl.lit(snapshot_ts) - pl.col(source_column)).dt.total_seconds().cast(pl.Float64))
        .otherwise(-1.0)
        .alias(alias)
    )


def build_preference_weight_expr(action_expr: pl.Expr) -> pl.Expr:
    expr = pl.lit(0.0)
    for action, weight in PREFERENCE_WEIGHTS.items():
        expr = pl.when(action_expr == action).then(pl.lit(weight)).otherwise(expr)
    return expr


def utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def to_naive_utc(raw_value: Any) -> datetime | None:
    if raw_value is None:
        return None

    if isinstance(raw_value, datetime):
        dt = raw_value
    else:
        text = str(raw_value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)

    if dt.tzinfo is None:
        return dt.replace(microsecond=0)
    return dt.astimezone(timezone.utc).replace(tzinfo=None, microsecond=0)


def empty_base_events_df() -> pl.DataFrame:
    return pl.DataFrame(schema=BASE_EVENT_SCHEMA)


def base_jsonl_events_lf() -> pl.LazyFrame:
    return (
        pl.scan_ndjson(str(INPUT_PATH))
        .filter(pl.col("event_type") == "interaction.created")
        .with_columns(
            pl.col("payload").struct.field("action").cast(pl.Utf8).str.to_lowercase().alias("action"),
            pl.col("occurred_at").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%SZ", strict=False),
            pl.col("actor_user_id").cast(pl.Utf8),
            pl.col("target_user_id").cast(pl.Utf8),
        )
        .filter(pl.col("action").is_in(ACTIONS))
        .select(
            pl.col("event_id").cast(pl.Utf8).alias("event_id"),
            pl.col("event_type").cast(pl.Utf8).alias("event_type"),
            pl.col("occurred_at"),
            pl.col("actor_user_id"),
            pl.col("target_user_id"),
            pl.col("action"),
        )
    )


def build_base_events_df_from_records(records: Iterable[dict[str, Any]]) -> pl.DataFrame:
    normalized_rows: list[dict[str, Any]] = []
    for raw_row in records:
        event_id = str(raw_row.get("event_id", "")).strip()
        actor_user_id = str(raw_row.get("actor_user_id", "")).strip()
        target_user_id = str(raw_row.get("target_user_id", "")).strip()
        action = str(raw_row.get("action", "")).strip().lower()
        occurred_at = to_naive_utc(raw_row.get("occurred_at"))

        if not event_id or not actor_user_id or not target_user_id or not action or occurred_at is None:
            continue

        normalized_rows.append(
            {
                "event_id": event_id,
                "event_type": str(raw_row.get("event_type", "interaction.created")).strip() or "interaction.created",
                "occurred_at": occurred_at,
                "actor_user_id": actor_user_id,
                "target_user_id": target_user_id,
                "action": action,
            }
        )

    if not normalized_rows:
        return empty_base_events_df()

    return pl.DataFrame(normalized_rows, schema=BASE_EVENT_SCHEMA)


def enrich_interaction_events_lf(base_events_lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        base_events_lf.filter(pl.col("action").is_in(ACTIONS))
        .with_columns(
            pl.col("occurred_at").dt.date().alias("event_date"),
            pl.col("occurred_at").dt.hour().cast(pl.UInt8).alias("event_hour"),
            pl.col("occurred_at").dt.weekday().cast(pl.UInt8).alias("weekday_num"),
        )
        .with_columns(
            pl.col("weekday_num").is_in([6, 7]).alias("is_weekend"),
            build_preference_weight_expr(pl.col("action")).alias("preference_weight"),
            pl.col("action").is_in(POSITIVE_ACTIONS).alias("is_positive"),
            pl.col("action").is_in(NEGATIVE_ACTIONS).alias("is_negative"),
        )
        .with_columns(
            pl.when(pl.col("event_hour") < 6)
            .then(pl.lit("night"))
            .when(pl.col("event_hour") < 12)
            .then(pl.lit("morning"))
            .when(pl.col("event_hour") < 18)
            .then(pl.lit("afternoon"))
            .otherwise(pl.lit("evening"))
            .alias("daypart")
        )
        .select(
            "event_id",
            "event_type",
            "occurred_at",
            "event_date",
            "event_hour",
            "weekday_num",
            "is_weekend",
            "daypart",
            "actor_user_id",
            "target_user_id",
            "action",
            "is_positive",
            "is_negative",
            "preference_weight",
        )
    )


def normalized_events_lf() -> pl.LazyFrame:
    return enrich_interaction_events_lf(base_jsonl_events_lf())


def build_filtered_events_stage() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Interaction dataset not found: {INPUT_PATH}")

    sink_parquet_stage(
        FILTERED_EVENTS_PATH,
        normalized_events_lf(),
        dependencies=[INPUT_PATH, STAGE_SOURCE_PATH],
    )


def get_filtered_events_lf() -> pl.LazyFrame:
    return pl.scan_parquet(FILTERED_EVENTS_PATH)


def get_snapshot_ts(events_lf: pl.LazyFrame) -> datetime:
    snapshot_ts = collect_stream(
        events_lf.select(pl.max("occurred_at").alias("snapshot_ts"))
    ).item()
    if snapshot_ts is None:
        raise ValueError("No interaction events found after normalization")
    return snapshot_ts


def get_all_users_lf(events_lf: pl.LazyFrame) -> pl.LazyFrame:
    actor_users = events_lf.select(pl.col("actor_user_id").alias("user_id"))
    target_users = events_lf.select(pl.col("target_user_id").alias("user_id"))
    return pl.concat([actor_users, target_users]).unique().sort("user_id")


def build_actor_feature_expressions(snapshot_ts: datetime) -> list[pl.Expr]:
    decay_expr = (
        -pl.lit(DECAY_LAMBDA)
        * ((pl.lit(snapshot_ts) - pl.col("occurred_at")).dt.total_seconds() / 86_400.0)
    ).exp()

    agg_exprs: list[pl.Expr] = [
        pl.len().alias("user_total_interactions"),
        pl.len().alias("user_history_size"),
        pl.col("target_user_id").n_unique().alias("user_unique_targets_interacted"),
        pl.col("event_date").n_unique().alias("user_active_days_count"),
        pl.col("occurred_at").min().alias("user_first_interaction_at"),
        pl.col("occurred_at").max().alias("user_last_interaction_at"),
        bool_count_expr(pl.col("is_positive"), "user_total_positive_actions"),
        bool_count_expr(pl.col("is_negative"), "user_total_negative_actions"),
        pl.col("preference_weight").sum().cast(pl.Float64).alias("user_preference_score_lifetime"),
        (pl.col("preference_weight") * decay_expr).sum().cast(pl.Float64).alias("user_preference_score_decay_7d"),
        pl.col("target_user_id").filter(pl.col("action") == "like").n_unique().alias("user_unique_targets_liked"),
        pl.col("target_user_id").filter(pl.col("action") == "dislike").n_unique().alias("user_unique_targets_disliked"),
        pl.col("target_user_id").filter(pl.col("action") == "skip").n_unique().alias("user_unique_targets_skipped"),
        pl.col("target_user_id").filter(pl.col("action") == "block").n_unique().alias("user_unique_targets_blocked"),
        bool_count_expr(pl.col("daypart") == "night", "user_night_interactions"),
        bool_count_expr(pl.col("daypart") == "morning", "user_morning_interactions"),
        bool_count_expr(pl.col("daypart") == "afternoon", "user_afternoon_interactions"),
        bool_count_expr(pl.col("daypart") == "evening", "user_evening_interactions"),
        bool_count_expr(pl.col("is_weekend"), "user_weekend_interactions"),
        bool_count_expr(~pl.col("is_weekend"), "user_weekday_interactions"),
    ]

    for action, suffix in ACTION_SUFFIXES.items():
        agg_exprs.extend(
            [
                bool_count_expr(pl.col("action") == action, f"user_total_{suffix}"),
                pl.col("occurred_at").filter(pl.col("action") == action).max().alias(f"user_last_{action}_at"),
                decay_expr.filter(pl.col("action") == action).sum().cast(pl.Float64).alias(
                    f"user_decay_7d_{suffix}_score"
                ),
            ]
        )

    for daypart in ("night", "morning", "afternoon", "evening"):
        agg_exprs.extend(
            [
                bool_count_expr(
                    (pl.col("daypart") == daypart) & (pl.col("action") == "like"),
                    f"user_{daypart}_likes",
                ),
                bool_count_expr(
                    (pl.col("daypart") == daypart) & pl.col("is_negative"),
                    f"user_{daypart}_negative_actions",
                ),
            ]
        )

    for period_name, is_weekend in (("weekday", False), ("weekend", True)):
        agg_exprs.extend(
            [
                bool_count_expr(
                    (pl.col("is_weekend") == is_weekend) & (pl.col("action") == "like"),
                    f"user_{period_name}_likes",
                ),
                bool_count_expr(
                    (pl.col("is_weekend") == is_weekend) & (pl.col("action") == "dislike"),
                    f"user_{period_name}_dislikes",
                ),
                bool_count_expr(
                    (pl.col("is_weekend") == is_weekend) & (pl.col("action") == "skip"),
                    f"user_{period_name}_skips",
                ),
            ]
        )

    for window_name, delta in WINDOW_DELTAS.items():
        window_start = snapshot_ts - delta
        window_condition = pl.col("occurred_at") >= pl.lit(window_start)
        agg_exprs.extend(
            [
                bool_count_expr(window_condition, f"user_interactions_{window_name}"),
                bool_count_expr(window_condition & pl.col("is_positive"), f"user_positive_actions_{window_name}"),
                bool_count_expr(window_condition & pl.col("is_negative"), f"user_negative_actions_{window_name}"),
            ]
        )

        for action, suffix in ACTION_SUFFIXES.items():
            agg_exprs.append(
                bool_count_expr(
                    window_condition & (pl.col("action") == action),
                    f"user_total_{suffix}_{window_name}",
                )
            )

    return agg_exprs


def build_actor_features_lf(events_lf: pl.LazyFrame, snapshot_ts: datetime) -> pl.LazyFrame:
    rate_exprs: list[pl.Expr] = [
        safe_rate_expr("user_total_positive_actions", "user_total_interactions", "user_positive_rate"),
        safe_rate_expr("user_total_negative_actions", "user_total_interactions", "user_negative_rate"),
        safe_rate_expr("user_night_interactions", "user_total_interactions", "user_night_interaction_share"),
        safe_rate_expr("user_morning_interactions", "user_total_interactions", "user_morning_interaction_share"),
        safe_rate_expr("user_afternoon_interactions", "user_total_interactions", "user_afternoon_interaction_share"),
        safe_rate_expr("user_evening_interactions", "user_total_interactions", "user_evening_interaction_share"),
        safe_rate_expr("user_weekend_interactions", "user_total_interactions", "user_weekend_interaction_share"),
        safe_rate_expr("user_weekday_interactions", "user_total_interactions", "user_weekday_interaction_share"),
        safe_ratio_expr("user_total_interactions", "user_unique_targets_interacted", "user_repeat_target_ratio"),
        safe_rate_expr("user_total_negative_actions", "user_total_interactions", "user_harshness_score"),
    ]

    for action, suffix in ACTION_SUFFIXES.items():
        rate_exprs.append(
            safe_rate_expr(f"user_total_{suffix}", "user_total_interactions", f"user_{action}_rate")
        )

    for window_name in ("24h", "7d"):
        rate_exprs.extend(
            [
                safe_rate_expr(
                    f"user_positive_actions_{window_name}",
                    f"user_interactions_{window_name}",
                    f"user_positive_rate_{window_name}",
                ),
                safe_rate_expr(
                    f"user_negative_actions_{window_name}",
                    f"user_interactions_{window_name}",
                    f"user_negative_rate_{window_name}",
                ),
            ]
        )
        for action, suffix in ACTION_SUFFIXES.items():
            rate_exprs.append(
                safe_rate_expr(
                    f"user_total_{suffix}_{window_name}",
                    f"user_interactions_{window_name}",
                    f"user_{action}_rate_{window_name}",
                )
            )

    for daypart in ("night", "morning", "afternoon", "evening"):
        rate_exprs.extend(
            [
                safe_rate_expr(
                    f"user_{daypart}_likes",
                    f"user_{daypart}_interactions",
                    f"user_{daypart}_like_rate",
                ),
                safe_rate_expr(
                    f"user_{daypart}_negative_actions",
                    f"user_{daypart}_interactions",
                    f"user_{daypart}_negative_rate",
                ),
            ]
        )

    for period_name in ("weekday", "weekend"):
        rate_exprs.extend(
            [
                safe_rate_expr(
                    f"user_{period_name}_likes",
                    f"user_{period_name}_interactions",
                    f"user_{period_name}_like_rate",
                ),
                safe_rate_expr(
                    f"user_{period_name}_dislikes",
                    f"user_{period_name}_interactions",
                    f"user_{period_name}_dislike_rate",
                ),
                safe_rate_expr(
                    f"user_{period_name}_skips",
                    f"user_{period_name}_interactions",
                    f"user_{period_name}_skip_rate",
                ),
            ]
        )

    recency_exprs = [
        seconds_since_expr(snapshot_ts, "user_last_interaction_at", "user_seconds_since_last_interaction"),
    ]
    for action in ACTIONS:
        recency_exprs.append(
            seconds_since_expr(snapshot_ts, f"user_last_{action}_at", f"user_seconds_since_last_{action}")
        )

    derived_exprs: list[pl.Expr] = [
        (pl.col("user_last_interaction_at") - pl.col("user_first_interaction_at"))
        .dt.total_days()
        .fill_null(0)
        .cast(pl.Int32)
        .alias("user_activity_span_days"),
        safe_rate_expr("user_total_interactions", "user_active_days_count", "user_avg_interactions_per_active_day"),
        safe_ratio_expr("user_total_likes", "user_total_dislikes", "user_like_to_dislike_ratio"),
        safe_ratio_expr("user_total_likes", "user_total_skips", "user_like_to_skip_ratio"),
        safe_ratio_expr("user_total_matches", "user_total_likes", "user_match_to_like_ratio"),
        safe_ratio_expr("user_total_chat_started", "user_total_matches", "user_chat_to_match_ratio"),
        safe_ratio_expr("user_total_blocks", "user_total_dislikes", "user_block_to_dislike_ratio"),
        safe_ratio_expr("user_total_unblocks", "user_total_blocks", "user_block_reversal_rate"),
        safe_ratio_expr("user_total_matches", "user_total_likes", "user_like_to_match_conversion"),
        safe_ratio_expr("user_total_chat_started", "user_total_matches", "user_match_to_chat_conversion"),
        safe_ratio_expr("user_total_chat_started", "user_total_likes", "user_like_to_chat_conversion"),
        (pl.col("user_interactions_24h") / 24.0).cast(pl.Float64).alias("user_events_per_hour_24h"),
        (pl.col("user_interactions_7d") / 7.0).cast(pl.Float64).alias("user_events_per_day_7d"),
        (pl.col("user_total_likes_24h") / 24.0).cast(pl.Float64).alias("user_likes_per_hour_24h"),
        (pl.col("user_total_skips_10m") / 10.0).cast(pl.Float64).alias("user_skips_per_minute_10m"),
        (pl.col("user_total_blocks_7d") / 7.0).cast(pl.Float64).alias("user_blocks_per_day_7d"),
        (
            (pl.col("user_total_dislikes") + 0.5 * pl.col("user_total_skips") + 2.0 * pl.col("user_total_blocks"))
            / pl.when(pl.col("user_total_interactions") > 0)
            .then(pl.col("user_total_interactions"))
            .otherwise(1.0)
        )
        .cast(pl.Float64)
        .alias("user_interaction_harshness_score"),
        (
            pl.col("user_total_blocks_24h")
            / ((pl.col("user_total_blocks_7d") / 7.0) + 1e-6)
        )
        .cast(pl.Float64)
        .alias("user_block_burst_score"),
        (pl.col("user_history_size") >= 10).alias("user_has_enough_history_10"),
        (pl.col("user_history_size") >= 50).alias("user_has_enough_history_50"),
    ]

    return (
        events_lf.group_by("actor_user_id")
        .agg(build_actor_feature_expressions(snapshot_ts))
        .rename({"actor_user_id": "user_id"})
        .with_columns(rate_exprs + recency_exprs + derived_exprs)
    )


def build_received_feature_expressions(snapshot_ts: datetime) -> list[pl.Expr]:
    agg_exprs: list[pl.Expr] = [
        pl.len().alias("user_received_total_interactions"),
        pl.len().alias("user_received_history_size"),
        pl.col("actor_user_id").n_unique().alias("user_unique_actors_interacted_with"),
        pl.col("occurred_at").min().alias("user_first_received_interaction_at"),
        pl.col("occurred_at").max().alias("user_last_received_interaction_at"),
        bool_count_expr(pl.col("is_positive"), "user_received_total_positive_actions"),
        bool_count_expr(pl.col("is_negative"), "user_received_total_negative_actions"),
    ]

    for action, suffix in ACTION_SUFFIXES.items():
        agg_exprs.extend(
            [
                bool_count_expr(pl.col("action") == action, f"user_received_total_{suffix}"),
                pl.col("occurred_at").filter(pl.col("action") == action).max().alias(
                    f"user_last_received_{action}_at"
                ),
            ]
        )

    for window_name, delta in WINDOW_DELTAS.items():
        if window_name not in {"24h", "7d"}:
            continue
        window_start = snapshot_ts - delta
        window_condition = pl.col("occurred_at") >= pl.lit(window_start)
        agg_exprs.extend(
            [
                bool_count_expr(window_condition, f"user_received_interactions_{window_name}"),
                bool_count_expr(
                    window_condition & pl.col("is_positive"),
                    f"user_received_positive_actions_{window_name}",
                ),
                bool_count_expr(
                    window_condition & pl.col("is_negative"),
                    f"user_received_negative_actions_{window_name}",
                ),
            ]
        )

        for action, suffix in ACTION_SUFFIXES.items():
            agg_exprs.append(
                bool_count_expr(
                    window_condition & (pl.col("action") == action),
                    f"user_received_{suffix}_{window_name}",
                )
            )

    return agg_exprs


def build_received_features_lf(events_lf: pl.LazyFrame, snapshot_ts: datetime) -> pl.LazyFrame:
    rate_exprs: list[pl.Expr] = [
        safe_rate_expr(
            "user_received_total_positive_actions",
            "user_received_total_interactions",
            "user_received_positive_rate",
        ),
        safe_rate_expr(
            "user_received_total_negative_actions",
            "user_received_total_interactions",
            "user_received_negative_rate",
        ),
        safe_ratio_expr(
            "user_received_total_interactions",
            "user_unique_actors_interacted_with",
            "user_received_repeat_actor_ratio",
        ),
        safe_ratio_expr("user_received_total_matches", "user_received_total_likes", "user_received_match_per_like"),
        safe_ratio_expr(
            "user_received_total_chat_started",
            "user_received_total_matches",
            "user_received_chat_per_match",
        ),
        safe_ratio_expr(
            "user_received_total_chat_started",
            "user_received_total_likes",
            "user_received_chat_per_like",
        ),
    ]

    for action, suffix in ACTION_SUFFIXES.items():
        rate_exprs.append(
            safe_rate_expr(
                f"user_received_total_{suffix}",
                "user_received_total_interactions",
                f"user_received_{action}_rate",
            )
        )

    for window_name in ("24h", "7d"):
        rate_exprs.extend(
            [
                safe_rate_expr(
                    f"user_received_positive_actions_{window_name}",
                    f"user_received_interactions_{window_name}",
                    f"user_received_positive_rate_{window_name}",
                ),
                safe_rate_expr(
                    f"user_received_negative_actions_{window_name}",
                    f"user_received_interactions_{window_name}",
                    f"user_received_negative_rate_{window_name}",
                ),
            ]
        )

        for action, suffix in ACTION_SUFFIXES.items():
            rate_exprs.append(
                safe_rate_expr(
                    f"user_received_{suffix}_{window_name}",
                    f"user_received_interactions_{window_name}",
                    f"user_received_{action}_rate_{window_name}",
                )
            )

    recency_exprs = [
        seconds_since_expr(
            snapshot_ts,
            "user_last_received_interaction_at",
            "user_seconds_since_last_received_interaction",
        )
    ]
    for action in ACTIONS:
        recency_exprs.append(
            seconds_since_expr(
                snapshot_ts,
                f"user_last_received_{action}_at",
                f"user_seconds_since_last_received_{action}",
            )
        )

    return (
        events_lf.group_by("target_user_id")
        .agg(build_received_feature_expressions(snapshot_ts))
        .rename({"target_user_id": "user_id"})
        .with_columns(rate_exprs + recency_exprs)
    )


def action_list_count_expr(list_column: str, values: list[str], alias: str) -> pl.Expr:
    return (
        pl.col(list_column)
        .list.eval(pl.element().is_in(values).cast(pl.UInt16))
        .list.sum()
        .fill_null(0)
        .cast(pl.UInt32)
        .alias(alias)
    )


def streak_length_expr(list_column: str, values: list[str], alias: str) -> pl.Expr:
    return (
        pl.col(list_column)
        .list.eval(pl.element().is_in(values).cast(pl.UInt16).cum_min())
        .list.sum()
        .fill_null(0)
        .cast(pl.UInt32)
        .alias(alias)
    )


def build_recent_features_lf(events_lf: pl.LazyFrame) -> pl.LazyFrame:
    recent_base_lf = (
        events_lf.group_by("actor_user_id")
        .agg(
            pl.col("action")
            .sort_by("occurred_at", descending=True)
            .head(RECENT_ACTION_MAX_DEPTH)
            .alias("recent_actions")
        )
        .rename({"actor_user_id": "user_id"})
        .with_columns(
            pl.col("recent_actions").list.slice(0, 3).alias("__recent_3"),
            pl.col("recent_actions").list.slice(0, 5).alias("__recent_5"),
            pl.col("recent_actions").list.slice(0, 10).alias("__recent_10"),
            pl.col("recent_actions").list.slice(0, 20).alias("__recent_20"),
            pl.col("recent_actions").list.slice(0, 50).alias("__recent_50"),
        )
    )

    exprs: list[pl.Expr] = []
    for n in RECENT_WINDOW_SIZES:
        list_column = f"__recent_{n}"
        total_column = f"user_recent_{n}_total_actions"
        exprs.append(pl.col(list_column).list.len().cast(pl.UInt32).alias(total_column))

        for action in ACTIONS:
            exprs.append(
                action_list_count_expr(
                    list_column,
                    [action],
                    f"user_last_{n}_{action}_count",
                )
            )

        exprs.extend(
            [
                action_list_count_expr(list_column, POSITIVE_ACTIONS, f"user_last_{n}_positive_count"),
                action_list_count_expr(list_column, NEGATIVE_ACTIONS, f"user_last_{n}_negative_count"),
            ]
        )

    exprs.extend(
        [
            streak_length_expr("recent_actions", NEGATIVE_ACTIONS, "user_negative_streak_len"),
            streak_length_expr("recent_actions", POSITIVE_ACTIONS, "user_positive_streak_len"),
            streak_length_expr("recent_actions", ["like"], "user_like_streak_len"),
            streak_length_expr("recent_actions", ["match"], "user_match_streak_len"),
            (
                (pl.col("__recent_3").list.len() == 3)
                & (
                    pl.col("__recent_3")
                    .list.eval(pl.element().is_in(NEGATIVE_ACTIONS).cast(pl.UInt8))
                    .list.min()
                    == 1
                )
            ).alias("user_last_3_all_negative"),
            (
                pl.col("__recent_5")
                .list.eval((pl.element() == "block").cast(pl.UInt8))
                .list.sum()
                .fill_null(0)
                > 0
            ).alias("user_last_5_contains_block"),
            (
                pl.col("__recent_10")
                .list.eval((pl.element() == "match").cast(pl.UInt8))
                .list.sum()
                .fill_null(0)
                > 0
            ).alias("user_last_10_has_match"),
            (
                pl.col("__recent_10")
                .list.eval((pl.element() == "chat_started").cast(pl.UInt8))
                .list.sum()
                .fill_null(0)
                > 0
            ).alias("user_last_10_has_chat_started"),
        ]
    )

    recent_lf = recent_base_lf.with_columns(exprs)

    rate_exprs: list[pl.Expr] = []
    for n in (10, 20, 50):
        total_column = f"user_recent_{n}_total_actions"
        for action in ("like", "dislike", "skip", "match", "chat_started", "block"):
            rate_exprs.append(
                safe_rate_expr(
                    f"user_last_{n}_{action}_count",
                    total_column,
                    f"user_last_{n}_{action}_rate",
                )
            )
        rate_exprs.extend(
            [
                safe_rate_expr(
                    f"user_last_{n}_positive_count",
                    total_column,
                    f"user_recent_positive_share_{n}",
                ),
                safe_rate_expr(
                    f"user_last_{n}_negative_count",
                    total_column,
                    f"user_recent_negative_share_{n}",
                ),
            ]
        )

    return recent_lf.with_columns(rate_exprs).drop(
        "recent_actions",
        "__recent_3",
        "__recent_5",
        "__recent_10",
        "__recent_20",
        "__recent_50",
    )


def build_gap_features_lf(events_lf: pl.LazyFrame) -> pl.LazyFrame:
    base_gap_lf = (
        events_lf.select("actor_user_id", "occurred_at")
        .sort(["actor_user_id", "occurred_at"])
        .with_columns(
            pl.col("occurred_at")
            .diff()
            .over("actor_user_id")
            .dt.total_seconds()
            .cast(pl.Float64)
            .alias("gap_seconds")
        )
        .group_by("actor_user_id")
        .agg(
            pl.col("gap_seconds").mean().alias("user_avg_time_between_interactions"),
            pl.col("gap_seconds").median().alias("user_median_time_between_interactions"),
        )
        .rename({"actor_user_id": "user_id"})
    )

    result_lf = base_gap_lf
    for action in ("like", "dislike", "skip"):
        alias_suffix = {
            "like": "likes",
            "dislike": "dislikes",
            "skip": "skips",
        }[action]
        action_gap_lf = (
            events_lf.filter(pl.col("action") == action)
            .select("actor_user_id", "occurred_at")
            .sort(["actor_user_id", "occurred_at"])
            .with_columns(
                pl.col("occurred_at")
                .diff()
                .over("actor_user_id")
                .dt.total_seconds()
                .cast(pl.Float64)
                .alias("gap_seconds")
            )
            .group_by("actor_user_id")
            .agg(
                pl.col("gap_seconds").mean().alias(f"user_avg_time_between_{alias_suffix}"),
                pl.col("gap_seconds").median().alias(f"user_median_time_between_{alias_suffix}"),
            )
            .rename({"actor_user_id": "user_id"})
        )
        result_lf = result_lf.join(action_gap_lf, on="user_id", how="left")

    return result_lf


def build_pair_user_features_lf(events_lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        events_lf.group_by(["actor_user_id", "target_user_id"])
        .agg(
            pl.len().alias("pair_total_interactions"),
            pl.col("action").eq("dislike").any().alias("pair_has_dislike"),
            pl.col("action").eq("block").any().alias("pair_has_block"),
            pl.col("action").eq("unblock").any().alias("pair_has_unblock"),
            pl.col("occurred_at").filter(pl.col("action") == "dislike").min().alias("pair_first_dislike_at"),
            pl.col("occurred_at").filter(pl.col("action") == "block").min().alias("pair_first_block_at"),
            pl.col("occurred_at").filter(pl.col("action") == "block").max().alias("pair_last_block_at"),
            pl.col("occurred_at").filter(pl.col("action") == "unblock").max().alias("pair_last_unblock_at"),
        )
        .with_columns(
            (
                pl.col("pair_last_block_at").is_not_null()
                & (
                    pl.col("pair_last_unblock_at").is_null()
                    | (pl.col("pair_last_block_at") > pl.col("pair_last_unblock_at"))
                )
            ).alias("pair_is_currently_blocked"),
            (
                pl.col("pair_first_dislike_at").is_not_null()
                & pl.col("pair_first_block_at").is_not_null()
                & (pl.col("pair_first_block_at") > pl.col("pair_first_dislike_at"))
            ).alias("pair_block_after_dislike"),
        )
        .group_by("actor_user_id")
        .agg(
            pl.len().alias("user_pair_history_size"),
            pl.col("pair_has_dislike").cast(pl.UInt32).sum().alias("user_pairs_with_dislike"),
            (pl.col("pair_total_interactions") > 1).cast(pl.UInt32).sum().alias("user_pairs_with_repeated_interactions"),
            pl.col("pair_is_currently_blocked").cast(pl.UInt32).sum().alias("user_active_block_count"),
            pl.col("pair_block_after_dislike").cast(pl.UInt32).sum().alias("user_block_after_dislike_pairs"),
        )
        .rename({"actor_user_id": "user_id"})
        .with_columns(
            safe_rate_expr("user_pairs_with_repeated_interactions", "user_pair_history_size", "user_repeated_pair_rate"),
            safe_rate_expr("user_active_block_count", "user_pair_history_size", "user_active_block_ratio"),
            safe_rate_expr("user_block_after_dislike_pairs", "user_pairs_with_dislike", "user_block_after_dislike_rate"),
        )
    )


def build_feature_stages(events_lf: pl.LazyFrame, snapshot_ts: datetime) -> None:
    sink_parquet_stage(
        ACTOR_FEATURES_PATH,
        build_actor_features_lf(events_lf, snapshot_ts),
        dependencies=[FILTERED_EVENTS_PATH, STAGE_SOURCE_PATH],
    )
    sink_parquet_stage(
        RECEIVED_FEATURES_PATH,
        build_received_features_lf(events_lf, snapshot_ts),
        dependencies=[FILTERED_EVENTS_PATH, STAGE_SOURCE_PATH],
    )
    sink_parquet_stage(
        RECENT_FEATURES_PATH,
        build_recent_features_lf(events_lf),
        dependencies=[FILTERED_EVENTS_PATH, STAGE_SOURCE_PATH],
    )
    sink_parquet_stage(
        GAP_FEATURES_PATH,
        build_gap_features_lf(events_lf),
        dependencies=[FILTERED_EVENTS_PATH, STAGE_SOURCE_PATH],
    )
    sink_parquet_stage(
        PAIR_USER_FEATURES_PATH,
        build_pair_user_features_lf(events_lf),
        dependencies=[FILTERED_EVENTS_PATH, STAGE_SOURCE_PATH],
    )


def build_final_feature_lf(
    events_lf: pl.LazyFrame,
    snapshot_ts: datetime,
    all_users_lf: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    if all_users_lf is None:
        all_users_lf = get_all_users_lf(events_lf)

    return (
        all_users_lf.with_columns(pl.lit(snapshot_ts).alias("snapshot_at"))
        .join(build_actor_features_lf(events_lf, snapshot_ts), on="user_id", how="left")
        .join(build_received_features_lf(events_lf, snapshot_ts), on="user_id", how="left")
        .join(build_recent_features_lf(events_lf), on="user_id", how="left")
        .join(build_gap_features_lf(events_lf), on="user_id", how="left")
        .join(build_pair_user_features_lf(events_lf), on="user_id", how="left")
        .with_columns(
            safe_rate_expr("user_active_block_count", "user_unique_targets_interacted", "user_active_block_target_rate")
        )
        .sort("user_id")
    )


def finalize_missing_values(df: pl.DataFrame) -> pl.DataFrame:
    recency_markers = (
        "seconds_since",
        "avg_time_between",
        "median_time_between",
    )

    exprs: list[pl.Expr] = []
    for column_name, dtype in df.schema.items():
        column_expr = pl.col(column_name)
        if dtype == pl.Boolean:
            exprs.append(column_expr.fill_null(False).alias(column_name))
            continue

        if dtype in FLOAT_DTYPES:
            fill_value = -1.0 if any(marker in column_name for marker in recency_markers) else 0.0
            exprs.append(column_expr.fill_null(fill_value).alias(column_name))
            continue

        if dtype in INTEGER_DTYPES:
            exprs.append(column_expr.fill_null(0).alias(column_name))
            continue

        exprs.append(column_expr)

    return df.with_columns(exprs)


def validate_final_dataset(df: pl.DataFrame) -> None:
    if df["user_id"].n_unique() != df.height:
        raise ValueError("Final dataset validation failed: user_id is not unique")

    ratio_like_columns = [
        column
        for column in df.columns
        if column.endswith("_rate")
        or column.endswith("_share")
        or "_share_" in column
    ]

    for column_name in ratio_like_columns:
        series = df[column_name]
        if series.null_count() > 0:
            raise ValueError(f"Final dataset validation failed: nulls left in {column_name}")
        if not bool(((series >= 0.0) & (series <= 1.0)).all()):
            raise ValueError(f"Final dataset validation failed: {column_name} is outside [0, 1]")

    if not bool((df["user_total_interactions"] >= df["user_total_positive_actions"]).all()):
        raise ValueError("Final dataset validation failed: positive actions exceed total interactions")

    if not bool((df["user_total_interactions"] >= df["user_total_negative_actions"]).all()):
        raise ValueError("Final dataset validation failed: negative actions exceed total interactions")


def infer_snapshot_ts_from_base_events(base_events_df: pl.DataFrame) -> datetime:
    if base_events_df.height == 0:
        return utc_now_naive()

    occurred_at_series = base_events_df["occurred_at"]
    max_event_ts = occurred_at_series.max()
    if max_event_ts is None:
        return utc_now_naive()

    max_event_ts = to_naive_utc(max_event_ts)
    if max_event_ts is None:
        return utc_now_naive()

    return max(max_event_ts, utc_now_naive())


def build_feature_frame_from_base_events(
    base_events_df: pl.DataFrame,
    *,
    all_user_ids: Iterable[str] | None = None,
    snapshot_ts: datetime | None = None,
) -> pl.DataFrame:
    base_events_lf = base_events_df.lazy()
    events_lf = enrich_interaction_events_lf(base_events_lf)

    if snapshot_ts is None:
        snapshot_ts = infer_snapshot_ts_from_base_events(base_events_df)

    if all_user_ids is None:
        all_users_lf = get_all_users_lf(events_lf)
    else:
        normalized_user_ids = [
            str(user_id).strip()
            for user_id in all_user_ids
            if str(user_id).strip()
        ]
        all_users_lf = (
            pl.DataFrame({"user_id": normalized_user_ids}, schema={"user_id": pl.Utf8})
            .lazy()
            .unique()
            .sort("user_id")
        )

    final_df = finalize_missing_values(
        collect_stream(build_final_feature_lf(events_lf, snapshot_ts, all_users_lf))
    )
    validate_final_dataset(final_df)
    return final_df


@lru_cache(maxsize=1)
def get_user_interaction_feature_schema() -> dict[str, pl.DataType]:
    empty_events_lf = enrich_interaction_events_lf(empty_base_events_df().lazy())
    empty_users_lf = pl.DataFrame(schema={"user_id": pl.Utf8}).lazy()
    return dict(
        build_final_feature_lf(
            empty_events_lf,
            datetime(1970, 1, 1),
            empty_users_lf,
        ).collect_schema()
    )


def get_user_interaction_feature_columns() -> list[str]:
    return list(get_user_interaction_feature_schema().keys())


def assemble_final_dataset(snapshot_ts: datetime, all_users_lf: pl.LazyFrame) -> pl.DataFrame:
    final_lf = build_final_feature_lf(
        events_lf=pl.scan_parquet(FILTERED_EVENTS_PATH),
        snapshot_ts=snapshot_ts,
        all_users_lf=all_users_lf,
    )
    final_df = finalize_missing_values(collect_stream(final_lf))
    validate_final_dataset(final_df)
    return final_df


def save_final_outputs(df: pl.DataFrame) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(FINAL_PARQUET_PATH, compression="zstd")
    df.write_csv(FINAL_CSV_PATH)


def run() -> None:
    from ml.services.interaction_feature_store import backfill_jsonl_to_database

    summary = backfill_jsonl_to_database(INPUT_PATH)
    print(f"JSONL lines processed: {summary['jsonl_line_count']}")
    print(f"Stored interactions: {summary['stored_interaction_count']}")
    print(f"Affected users: {summary['affected_user_count']}")
    print(f"Upserted feature rows: {summary['upserted_feature_row_count']}")
    print(f"Deleted feature rows: {summary['deleted_feature_row_count']}")


if __name__ == "__main__":
    run()
