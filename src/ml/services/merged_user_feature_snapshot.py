from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import func, select

from core.data_config import Paths
from ml.infra.db.interaction_feature_tables import user_interaction_features_table
from ml.infra.db.session import engine
from ml.services.user_id import normalize_user_id

PATHS = Paths()
TRANSACTION_SNAPSHOT_PATH = PATHS.transaction_user_features_parquet_path
MERGED_SNAPSHOT_PATH = PATHS.merged_user_features_parquet_path
MATCH_RATE_WARNING_THRESHOLD = 0.80

log: structlog.BoundLogger = structlog.get_logger(__name__)


def ensure_merged_user_feature_snapshot() -> Path:
    return build_merged_user_feature_snapshot(force=False)


def build_merged_user_feature_snapshot(force: bool = False) -> Path:
    if not TRANSACTION_SNAPSHOT_PATH.exists():
        log.warning(
            "merged_user_feature_snapshot_missing_transaction_snapshot",
            transaction_snapshot_path=str(TRANSACTION_SNAPSHOT_PATH),
        )
        return MERGED_SNAPSHOT_PATH

    interaction_snapshot_at: datetime | None = None
    try:
        interaction_snapshot_at = _load_max_interaction_snapshot_at()
    except Exception as exc:
        if MERGED_SNAPSHOT_PATH.exists():
            log.warning(
                "merged_user_feature_snapshot_db_freshness_check_failed",
                error=str(exc),
                merged_snapshot_path=str(MERGED_SNAPSHOT_PATH),
            )
            if not force and not _is_file_newer(TRANSACTION_SNAPSHOT_PATH, MERGED_SNAPSHOT_PATH):
                return MERGED_SNAPSHOT_PATH
        else:
            log.warning(
                "merged_user_feature_snapshot_db_unavailable",
                error=str(exc),
                merged_snapshot_path=str(MERGED_SNAPSHOT_PATH),
            )
            return MERGED_SNAPSHOT_PATH

    if not force and not _needs_refresh(
        transaction_snapshot_path=TRANSACTION_SNAPSHOT_PATH,
        merged_snapshot_path=MERGED_SNAPSHOT_PATH,
        interaction_snapshot_at=interaction_snapshot_at,
    ):
        return MERGED_SNAPSHOT_PATH

    try:
        transaction_df = pd.read_parquet(TRANSACTION_SNAPSHOT_PATH)
    except Exception as exc:
        log.warning(
            "merged_user_feature_snapshot_transaction_read_failed",
            error=str(exc),
            transaction_snapshot_path=str(TRANSACTION_SNAPSHOT_PATH),
        )
        return MERGED_SNAPSHOT_PATH

    try:
        interaction_df = _load_interaction_features_frame()
    except Exception as exc:
        if MERGED_SNAPSHOT_PATH.exists():
            log.warning(
                "merged_user_feature_snapshot_interaction_read_failed_using_existing",
                error=str(exc),
                merged_snapshot_path=str(MERGED_SNAPSHOT_PATH),
            )
            return MERGED_SNAPSHOT_PATH

        log.warning(
            "merged_user_feature_snapshot_interaction_read_failed_no_snapshot",
            error=str(exc),
            merged_snapshot_path=str(MERGED_SNAPSHOT_PATH),
        )
        return MERGED_SNAPSHOT_PATH

    normalized_transaction_df = _normalize_transaction_frame(transaction_df)
    normalized_interaction_df = _normalize_interaction_frame(interaction_df)
    _log_join_metrics(
        transaction_df=normalized_transaction_df,
        interaction_df=normalized_interaction_df,
    )

    merged_df = normalized_transaction_df.merge(
        normalized_interaction_df,
        on="party_rk",
        how="left",
    )

    MERGED_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(MERGED_SNAPSHOT_PATH, index=False)
    log.info(
        "merged_user_feature_snapshot_built",
        merged_snapshot_path=str(MERGED_SNAPSHOT_PATH),
        row_count=int(len(merged_df)),
        column_count=int(len(merged_df.columns)),
    )
    return MERGED_SNAPSHOT_PATH


def _needs_refresh(
    *,
    transaction_snapshot_path: Path,
    merged_snapshot_path: Path,
    interaction_snapshot_at: datetime | None,
) -> bool:
    if not merged_snapshot_path.exists():
        return True
    if _is_file_newer(transaction_snapshot_path, merged_snapshot_path):
        return True
    if interaction_snapshot_at is None:
        return False

    return _to_utc_timestamp(interaction_snapshot_at) > merged_snapshot_path.stat().st_mtime


def _is_file_newer(source_path: Path, target_path: Path) -> bool:
    if not source_path.exists() or not target_path.exists():
        return False
    return source_path.stat().st_mtime > target_path.stat().st_mtime


def _load_max_interaction_snapshot_at() -> datetime | None:
    stmt = select(func.max(user_interaction_features_table.c.snapshot_at))
    with engine.connect() as connection:
        return connection.execute(stmt).scalar_one_or_none()


def _load_interaction_features_frame() -> pd.DataFrame:
    stmt = select(user_interaction_features_table)
    with engine.connect() as connection:
        return pd.read_sql_query(stmt, connection)


def _normalize_transaction_frame(transaction_df: pd.DataFrame) -> pd.DataFrame:
    result = transaction_df.copy()
    result["party_rk"] = _normalize_id_series(result.get("party_rk"))
    result = result[result["party_rk"].notna()].copy()
    result = result.drop_duplicates(subset=["party_rk"], keep="last").reset_index(drop=True)
    return result


def _normalize_interaction_frame(interaction_df: pd.DataFrame) -> pd.DataFrame:
    if interaction_df.empty:
        return pd.DataFrame(columns=["party_rk"])

    result = interaction_df.copy()
    result["party_rk"] = _normalize_id_series(result.get("user_id"))
    result = result[result["party_rk"].notna()].copy()
    result = result.drop(columns=["user_id"], errors="ignore")
    result = result.drop_duplicates(subset=["party_rk"], keep="last").reset_index(drop=True)
    return result


def _normalize_id_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="string")
    normalized = series.map(lambda raw_id: normalize_user_id(raw_id, allow_hash=True))
    return normalized.astype("string")


def _log_join_metrics(*, transaction_df: pd.DataFrame, interaction_df: pd.DataFrame) -> None:
    transaction_user_ids = {
        str(user_id)
        for user_id in transaction_df.get("party_rk", pd.Series(dtype="string")).dropna().tolist()
    }
    interaction_user_ids = {
        str(user_id)
        for user_id in interaction_df.get("party_rk", pd.Series(dtype="string")).dropna().tolist()
    }
    matched_user_count = len(transaction_user_ids & interaction_user_ids)
    transaction_user_count = len(transaction_user_ids)
    interaction_user_count = len(interaction_user_ids)

    transaction_match_rate = (
        float(matched_user_count) / float(transaction_user_count)
        if transaction_user_count > 0
        else 0.0
    )
    interaction_match_rate = (
        float(matched_user_count) / float(interaction_user_count)
        if interaction_user_count > 0
        else 0.0
    )

    log.info(
        "merged_user_feature_snapshot_join_metrics",
        transaction_user_count=transaction_user_count,
        interaction_user_count=interaction_user_count,
        matched_user_count=matched_user_count,
        transaction_match_rate=round(transaction_match_rate, 4),
        interaction_match_rate=round(interaction_match_rate, 4),
    )

    if matched_user_count == 0 or interaction_match_rate < MATCH_RATE_WARNING_THRESHOLD:
        log.warning(
            "merged_user_feature_snapshot_low_match_rate",
            transaction_user_count=transaction_user_count,
            interaction_user_count=interaction_user_count,
            matched_user_count=matched_user_count,
            transaction_match_rate=round(transaction_match_rate, 4),
            interaction_match_rate=round(interaction_match_rate, 4),
            warning_threshold=MATCH_RATE_WARNING_THRESHOLD,
        )


def _to_utc_timestamp(raw_value: Any) -> float:
    if raw_value is None:
        return 0.0
    if isinstance(raw_value, datetime):
        date_time = raw_value
    else:
        date_time = datetime.fromisoformat(str(raw_value).strip())

    if date_time.tzinfo is None:
        date_time = date_time.replace(tzinfo=timezone.utc)
    else:
        date_time = date_time.astimezone(timezone.utc)
    return date_time.timestamp()
