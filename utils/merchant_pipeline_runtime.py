from __future__ import annotations

import argparse
import json
import re
from contextlib import contextmanager
from datetime import datetime
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Iterable

import polars as pl


ROOT = Path(".")
DATA_DIR = ROOT / "data" / "Data_1"
ARTIFACTS_DIR = ROOT / "artifacts"
INTERMEDIATE_DIR = ARTIFACTS_DIR / "intermediate"
FEATURES_DIR = INTERMEDIATE_DIR / "merchant_feature_stages"

PIPELINE_VERSION = "merchant_v6_crash_safe"
TOP_CATEGORY_LIMIT = 30
TOP_MCC_LIMIT = 30
DT_FORMAT = "%Y-%m-%d %H:%M:%S"

PREPARED_SHARDS_DIR = INTERMEDIATE_DIR / f"merchant_prepared_transactions_{PIPELINE_VERSION}_shards"
DAILY_COUNT_SHARDS_DIR = INTERMEDIATE_DIR / f"merchant_daily_counts_{PIPELINE_VERSION}_shards"
CATEGORY_COUNT_SHARDS_DIR = INTERMEDIATE_DIR / f"merchant_category_counts_{PIPELINE_VERSION}_shards"
MCC_COUNT_SHARDS_DIR = INTERMEDIATE_DIR / f"merchant_mcc_counts_{PIPELINE_VERSION}_shards"
MERCHANT_USER_SHARDS_DIR = INTERMEDIATE_DIR / f"merchant_user_counts_{PIPELINE_VERSION}_shards"

DAILY_COUNTS_PATH = INTERMEDIATE_DIR / f"merchant_daily_counts_{PIPELINE_VERSION}.parquet"
CATEGORY_COUNTS_PATH = INTERMEDIATE_DIR / f"merchant_category_counts_{PIPELINE_VERSION}.parquet"
MCC_COUNTS_PATH = INTERMEDIATE_DIR / f"merchant_mcc_counts_{PIPELINE_VERSION}.parquet"
MERCHANT_USER_COUNTS_PATH = INTERMEDIATE_DIR / f"merchant_user_counts_{PIPELINE_VERSION}.parquet"
BUCKET_CONFIG_PATH = INTERMEDIATE_DIR / f"merchant_bucket_config_{PIPELINE_VERSION}.json"

BASE_FEATURES_PATH = FEATURES_DIR / f"merchant_base_features_{PIPELINE_VERSION}.parquet"
DAILY_FEATURES_PATH = FEATURES_DIR / f"merchant_daily_features_{PIPELINE_VERSION}.parquet"
CATEGORY_FEATURES_PATH = FEATURES_DIR / f"merchant_category_features_{PIPELINE_VERSION}.parquet"
MCC_FEATURES_PATH = FEATURES_DIR / f"merchant_mcc_features_{PIPELINE_VERSION}.parquet"
USER_INTERACTION_FEATURES_PATH = FEATURES_DIR / f"merchant_user_interaction_features_{PIPELINE_VERSION}.parquet"
FINAL_PARQUET_PATH = ARTIFACTS_DIR / f"merchant_features_data1_{PIPELINE_VERSION}.parquet"
FINAL_CSV_PATH = ARTIFACTS_DIR / f"merchant_features_data1_{PIPELINE_VERSION}.csv"

EXPECTED_COLUMNS = [
    "real_transaction_dttm",
    "party_rk",
    "transaction_rk",
    "merchant_type_code",
    "merchant_nm",
    "category_nm",
]
SCHEMA_OVERRIDES = {col: pl.Utf8 for col in EXPECTED_COLUMNS}
CATEGORY_UNKNOWN_VALUE = "unknown_category"
MERCHANT_UNKNOWN_VALUE = "unknown_merchant"
MCC_UNKNOWN_VALUE = -1
MCC_UNKNOWN_LABEL = "unknown"

for path in [
    ARTIFACTS_DIR,
    INTERMEDIATE_DIR,
    FEATURES_DIR,
    PREPARED_SHARDS_DIR,
    DAILY_COUNT_SHARDS_DIR,
    CATEGORY_COUNT_SHARDS_DIR,
    MCC_COUNT_SHARDS_DIR,
    MERCHANT_USER_SHARDS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


@contextmanager
def timed(label: str):
    started = perf_counter()
    log(f"{label} started")
    try:
        yield
    finally:
        log(f"{label} finished in {perf_counter() - started:,.2f} sec")


def collect_stream(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def sink_parquet_stage(label: str, path: Path, lf: pl.LazyFrame, rebuild: bool = False) -> None:
    if path.exists() and not rebuild:
        log(f"Skip existing stage: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with timed(label):
        try:
            lf.sink_parquet(path, compression="zstd", statistics=True)
        except TypeError:
            collect_stream(lf).write_parquet(path, compression="zstd")
    log(f"Stage parquet saved: {path}")


def get_schema_names(lf: pl.LazyFrame) -> list[str]:
    collect_schema = getattr(lf, "collect_schema", None)
    if callable(collect_schema):
        return list(collect_schema().names())
    return list(lf.schema.keys())


def sanitize_token(value: object) -> str:
    token = str(value).strip().lower()
    token = re.sub(r"\s+", "_", token)
    token = re.sub(r"[^\w]+", "_", token, flags=re.UNICODE)
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def build_unique_feature_name_map(values: Iterable[object], prefix: str) -> dict[object, str]:
    mapping: dict[object, str] = {}
    seen: dict[str, int] = {}
    for raw_value in values:
        base_name = f"{prefix}{sanitize_token(raw_value)}"
        suffix = seen.get(base_name, 0)
        feature_name = base_name if suffix == 0 else f"{base_name}_{suffix}"
        seen[base_name] = suffix + 1
        mapping[raw_value] = feature_name
    return mapping


def extend_with_required(values: list, required_values: list) -> list:
    result: list = []
    seen = set()
    for value in values + required_values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def safe_divide_expr(numerator: pl.Expr, denominator: pl.Expr, alias: str) -> pl.Expr:
    return (
        pl.when(denominator.fill_null(0) > 0)
        .then(numerator / denominator)
        .otherwise(0.0)
        .cast(pl.Float64)
        .alias(alias)
    )


def weekday_name_expr(column_name: str, alias: str = "dominant_weekday") -> pl.Expr:
    return (
        pl.when(pl.col(column_name) == 1).then(pl.lit("Monday"))
        .when(pl.col(column_name) == 2).then(pl.lit("Tuesday"))
        .when(pl.col(column_name) == 3).then(pl.lit("Wednesday"))
        .when(pl.col(column_name) == 4).then(pl.lit("Thursday"))
        .when(pl.col(column_name) == 5).then(pl.lit("Friday"))
        .when(pl.col(column_name) == 6).then(pl.lit("Saturday"))
        .when(pl.col(column_name) == 7).then(pl.lit("Sunday"))
        .otherwise(pl.lit("unknown"))
        .alias(alias)
    )


def normalize_text_bucket(column_name: str, unknown_value: str) -> pl.Expr:
    cleaned = (
        pl.col(column_name)
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.replace_all(r"^\s+|\s+$", "")
        .str.replace_all(r"\s+", " ")
    )
    return pl.when(cleaned == "").then(pl.lit(unknown_value)).otherwise(cleaned).alias(column_name)


def normalize_merchant_name(column: str | pl.Expr = "merchant_nm") -> pl.Expr:
    expr = pl.col(column) if isinstance(column, str) else column
    cleaned = (
        expr.cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.replace_all(r"^\s+|\s+$", "")
        .str.replace_all(r"^[\"'`]+|[\"'`]+$", "")
        .str.to_lowercase()
        .str.replace_all("\u0451", "\u0435")
        .str.replace_all(r"^\s+|\s+$", "")
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"^[\"'`]+|[\"'`]+$", "")
        .str.replace_all(r"^\s+|\s+$", "")
    )
    return pl.when(cleaned == "").then(pl.lit(MERCHANT_UNKNOWN_VALUE)).otherwise(cleaned).alias("normalized_merchant_nm")


def discover_csv_files(data_dir: Path) -> list[Path]:
    csv_files = sorted(
        path for path in data_dir.rglob("*.csv") if "__MACOSX" not in path.parts and not path.name.startswith("._")
    )
    if not csv_files:
        raise FileNotFoundError(f"No valid CSV files found in {data_dir}")
    return csv_files


def get_polars_thread_pool_size() -> int:
    thread_pool_fn = getattr(pl, "thread_pool_size", None)
    if callable(thread_pool_fn):
        return int(thread_pool_fn())
    thread_pool_fn = getattr(pl, "threadpool_size", None)
    if callable(thread_pool_fn):
        return int(thread_pool_fn())
    return 1


def split_into_batches(items: list[Path], batch_count: int) -> list[list[Path]]:
    batch_count = max(1, min(batch_count, len(items)))
    batch_size = ceil(len(items) / batch_count)
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


CSV_FILES = discover_csv_files(DATA_DIR)
THREAD_POOL_SIZE = get_polars_thread_pool_size()
STAGE1_BATCH_COUNT = min(len(CSV_FILES), max(1, THREAD_POOL_SIZE))
STAGE1_BATCHES = split_into_batches(CSV_FILES, STAGE1_BATCH_COUNT)
PREPARED_SHARD_PATHS = [PREPARED_SHARDS_DIR / f"part_{idx:03d}.parquet" for idx in range(len(STAGE1_BATCHES))]


def prepared_shard_files() -> list[Path]:
    files = sorted(PREPARED_SHARDS_DIR.glob("part_*.parquet"))
    if not files:
        raise FileNotFoundError(f"Prepared shard parquet files not found in {PREPARED_SHARDS_DIR}")
    return files


def daily_count_shard_files() -> list[Path]:
    files = sorted(DAILY_COUNT_SHARDS_DIR.glob("part_*.parquet"))
    if not files:
        raise FileNotFoundError(f"Daily count shard parquet files not found in {DAILY_COUNT_SHARDS_DIR}")
    return files


def category_count_shard_files() -> list[Path]:
    files = sorted(CATEGORY_COUNT_SHARDS_DIR.glob("part_*.parquet"))
    if not files:
        raise FileNotFoundError(f"Category count shard parquet files not found in {CATEGORY_COUNT_SHARDS_DIR}")
    return files


def mcc_count_shard_files() -> list[Path]:
    files = sorted(MCC_COUNT_SHARDS_DIR.glob("part_*.parquet"))
    if not files:
        raise FileNotFoundError(f"MCC count shard parquet files not found in {MCC_COUNT_SHARDS_DIR}")
    return files


def merchant_user_shard_files() -> list[Path]:
    files = sorted(MERCHANT_USER_SHARDS_DIR.glob("part_*.parquet"))
    if not files:
        raise FileNotFoundError(f"Merchant-user shard parquet files not found in {MERCHANT_USER_SHARDS_DIR}")
    return files


def tx_lf() -> pl.LazyFrame:
    return pl.scan_parquet([str(path) for path in prepared_shard_files()])


def build_prepared_transactions_lf(csv_batch: list[Path]) -> pl.LazyFrame:
    raw_lf = pl.scan_csv(
        [str(path) for path in csv_batch],
        has_header=True,
        schema_overrides=SCHEMA_OVERRIDES,
        null_values=["", "null", "None"],
        infer_schema=False,
        low_memory=True,
        rechunk=False,
    )
    input_columns = get_schema_names(raw_lf)
    if input_columns != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected transaction schema: {input_columns}")
    return (
        raw_lf.select(EXPECTED_COLUMNS)
        .with_columns(
            pl.col("real_transaction_dttm").str.strptime(pl.Datetime, format=DT_FORMAT, strict=False),
            pl.col("party_rk").cast(pl.Utf8, strict=False),
            pl.col("transaction_rk").cast(pl.Utf8, strict=False),
            pl.col("merchant_nm").cast(pl.Utf8, strict=False).alias("merchant_nm_raw"),
            normalize_text_bucket("category_nm", CATEGORY_UNKNOWN_VALUE),
            pl.col("merchant_type_code").cast(pl.Int32, strict=False).fill_null(MCC_UNKNOWN_VALUE),
            normalize_merchant_name("merchant_nm"),
        )
        .with_columns(
            pl.when(pl.col("merchant_type_code") == MCC_UNKNOWN_VALUE)
            .then(pl.lit(MCC_UNKNOWN_LABEL))
            .otherwise(pl.col("merchant_type_code").cast(pl.Utf8))
            .alias("merchant_type_code_label"),
            pl.col("real_transaction_dttm").dt.truncate("1d").alias("txn_date"),
            pl.col("real_transaction_dttm").dt.hour().cast(pl.Int8).alias("hour"),
            pl.col("real_transaction_dttm").dt.weekday().cast(pl.Int8).alias("weekday_num"),
        )
        .select(
            "party_rk",
            "transaction_rk",
            "merchant_nm_raw",
            "normalized_merchant_nm",
            "category_nm",
            "merchant_type_code",
            "merchant_type_code_label",
            "real_transaction_dttm",
            "txn_date",
            "hour",
            "weekday_num",
        )
    )


def stage_prepare_shards(rebuild: bool) -> None:
    log(f"Polars thread pool size: {THREAD_POOL_SIZE}")
    log(f"CSV file count in Data_1: {len(CSV_FILES)}")
    with timed("Stage 1: prepared transaction shards"):
        for idx, (batch_files, shard_path) in enumerate(zip(STAGE1_BATCHES, PREPARED_SHARD_PATHS), start=1):
            label = f"Stage 1 shard {idx}/{len(STAGE1_BATCHES)}"
            if shard_path.exists() and not rebuild:
                log(f"Skip existing shard: {shard_path.name}")
                continue
            log(f"{label}: {len(batch_files)} files -> {shard_path.name}")
            sink_parquet_stage(label, shard_path, build_prepared_transactions_lf(batch_files), rebuild=True)
    log(f"Prepared shard count: {len(prepared_shard_files())}")


def stage_top_buckets(rebuild: bool) -> None:
    if BUCKET_CONFIG_PATH.exists() and not rebuild:
        log(f"Skip existing stage: {BUCKET_CONFIG_PATH}")
        return
    source_lf = tx_lf()
    with timed("Stage: top bucket config"):
        raw_top_categories = collect_stream(
            source_lf.group_by("category_nm").len().sort("len", descending=True).head(TOP_CATEGORY_LIMIT).select("category_nm")
        )["category_nm"].to_list()
        top_category_values = [value for value in raw_top_categories if value != CATEGORY_UNKNOWN_VALUE]
        top_category_values = extend_with_required(top_category_values, [CATEGORY_UNKNOWN_VALUE])
        category_share_values = extend_with_required(top_category_values, ["other"])
        category_share_name_map = build_unique_feature_name_map(category_share_values, "merchant_cat_share_")

        raw_top_mcc = collect_stream(
            source_lf.group_by("merchant_type_code_label")
            .len()
            .sort("len", descending=True)
            .head(TOP_MCC_LIMIT)
            .select("merchant_type_code_label")
        )["merchant_type_code_label"].to_list()
        top_mcc_values = [value for value in raw_top_mcc if value != MCC_UNKNOWN_LABEL]
        top_mcc_values = extend_with_required(top_mcc_values, [MCC_UNKNOWN_LABEL])
        mcc_share_values = extend_with_required(top_mcc_values, ["other"])
        mcc_share_name_map = build_unique_feature_name_map(mcc_share_values, "merchant_mcc_share_")

        config = {
            "top_category_values": top_category_values,
            "category_share_values": category_share_values,
            "category_share_name_map": category_share_name_map,
            "top_mcc_values": top_mcc_values,
            "mcc_share_values": mcc_share_values,
            "mcc_share_name_map": mcc_share_name_map,
        }
        BUCKET_CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Bucket config saved: {BUCKET_CONFIG_PATH}")


def load_bucket_config() -> dict:
    if not BUCKET_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Bucket config not found: {BUCKET_CONFIG_PATH}")
    return json.loads(BUCKET_CONFIG_PATH.read_text(encoding="utf-8"))


def build_base_features_lf(source_lf: pl.LazyFrame) -> pl.LazyFrame:
    hour_cols = [f"__hour_{hour:02d}_cnt" for hour in range(24)]
    weekday_cols = [f"__weekday_{weekday}_cnt" for weekday in range(1, 8)]
    agg_exprs: list[pl.Expr] = [
        pl.len().alias("merchant_txn_cnt"),
        pl.col("transaction_rk").drop_nulls().n_unique().alias("unique_transaction_cnt"),
        pl.col("real_transaction_dttm").min().alias("first_txn_dttm"),
        pl.col("real_transaction_dttm").max().alias("last_txn_dttm"),
    ]
    agg_exprs += [pl.col("hour").eq(hour).fill_null(False).cast(pl.UInt32).sum().alias(f"__hour_{hour:02d}_cnt") for hour in range(24)]
    agg_exprs += [pl.col("weekday_num").eq(weekday).fill_null(False).cast(pl.UInt32).sum().alias(f"__weekday_{weekday}_cnt") for weekday in range(1, 8)]
    return (
        source_lf.group_by("normalized_merchant_nm")
        .agg(agg_exprs)
        .with_columns(
            (pl.col("last_txn_dttm") - pl.col("first_txn_dttm")).dt.total_days().fill_null(0).cast(pl.Int32).alias("activity_span_days"),
            pl.concat_list([pl.col(col) for col in hour_cols]).list.arg_max().cast(pl.Int8).alias("dominant_hour"),
            (pl.concat_list([pl.col(col) for col in weekday_cols]).list.arg_max() + 1).cast(pl.Int8).alias("dominant_weekday_num"),
            pl.sum_horizontal([pl.col(f"__hour_{hour:02d}_cnt") for hour in range(0, 6)]).alias("__night_cnt"),
            pl.sum_horizontal([pl.col(f"__hour_{hour:02d}_cnt") for hour in range(6, 12)]).alias("__morning_cnt"),
            pl.sum_horizontal([pl.col(f"__hour_{hour:02d}_cnt") for hour in range(12, 18)]).alias("__afternoon_cnt"),
            pl.sum_horizontal([pl.col(f"__hour_{hour:02d}_cnt") for hour in range(18, 24)]).alias("__evening_cnt"),
            pl.sum_horizontal(
                [pl.col("__weekday_1_cnt"), pl.col("__weekday_2_cnt"), pl.col("__weekday_3_cnt"), pl.col("__weekday_4_cnt"), pl.col("__weekday_5_cnt")]
            ).alias("__weekday_cnt"),
            pl.sum_horizontal([pl.col("__weekday_6_cnt"), pl.col("__weekday_7_cnt")]).alias("__weekend_cnt"),
        )
        .with_columns(
            safe_divide_expr(pl.col("__weekend_cnt"), pl.col("merchant_txn_cnt"), "share_weekend_txn"),
            safe_divide_expr(pl.col("__weekday_cnt"), pl.col("merchant_txn_cnt"), "share_weekday_txn"),
            safe_divide_expr(pl.col("__night_cnt"), pl.col("merchant_txn_cnt"), "share_night_txn"),
            safe_divide_expr(pl.col("__morning_cnt"), pl.col("merchant_txn_cnt"), "share_morning_txn"),
            safe_divide_expr(pl.col("__afternoon_cnt"), pl.col("merchant_txn_cnt"), "share_afternoon_txn"),
            safe_divide_expr(pl.col("__evening_cnt"), pl.col("merchant_txn_cnt"), "share_evening_txn"),
            pl.concat_list([pl.col("__night_cnt"), pl.col("__morning_cnt"), pl.col("__afternoon_cnt"), pl.col("__evening_cnt")]).list.arg_max().cast(pl.Int8).alias("__dominant_daypart_idx"),
        )
        .with_columns(
            pl.when(pl.col("__dominant_daypart_idx") == 0).then(pl.lit("night"))
            .when(pl.col("__dominant_daypart_idx") == 1).then(pl.lit("morning"))
            .when(pl.col("__dominant_daypart_idx") == 2).then(pl.lit("afternoon"))
            .otherwise(pl.lit("evening"))
            .alias("dominant_daypart")
        )
        .select(
            "normalized_merchant_nm",
            "merchant_txn_cnt",
            "unique_transaction_cnt",
            "first_txn_dttm",
            "last_txn_dttm",
            "activity_span_days",
            "share_weekend_txn",
            "share_weekday_txn",
            "share_night_txn",
            "share_morning_txn",
            "share_afternoon_txn",
            "share_evening_txn",
            "dominant_hour",
            "dominant_weekday_num",
            "dominant_daypart",
        )
    )


def build_daily_count_shard_lf(source_lf: pl.LazyFrame) -> pl.LazyFrame:
    return source_lf.group_by(["normalized_merchant_nm", "txn_date"]).agg(pl.len().alias("daily_txn_cnt"))


def build_daily_features_lf(daily_counts_lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        daily_counts_lf.with_columns(
            pl.col("txn_date").dt.truncate("1w").alias("week_start"),
            pl.col("txn_date").dt.strftime("%Y-%m").alias("month_key"),
        )
        .group_by("normalized_merchant_nm")
        .agg(
            pl.len().alias("active_days_cnt"),
            pl.col("week_start").n_unique().alias("active_weeks_cnt"),
            pl.col("month_key").n_unique().alias("active_months_cnt"),
            pl.col("daily_txn_cnt").max().alias("max_txn_in_one_day"),
            pl.col("daily_txn_cnt").median().cast(pl.Float64).alias("median_txn_per_active_day"),
        )
    )


def build_category_count_shard_lf(source_lf: pl.LazyFrame) -> pl.LazyFrame:
    return source_lf.group_by(["normalized_merchant_nm", "category_nm"]).agg(pl.len().alias("category_txn_cnt"))


def build_category_features_lf(category_counts_lf: pl.LazyFrame, cfg: dict) -> pl.LazyFrame:
    top_values = cfg["top_category_values"]
    share_values = cfg["category_share_values"]
    share_name_map = cfg["category_share_name_map"]
    merchant_totals_lf = pl.scan_parquet(BASE_FEATURES_PATH).select("normalized_merchant_nm", "merchant_txn_cnt")
    count_columns = {value: f"__category_bucket_cnt_{idx}" for idx, value in enumerate(share_values)}
    category_share_lf = (
        category_counts_lf.with_columns(
            pl.when(pl.col("category_nm").is_in(top_values)).then(pl.col("category_nm")).otherwise(pl.lit("other")).alias("__category_bucket")
        )
        .group_by("normalized_merchant_nm")
        .agg([pl.col("__category_bucket").eq(value).cast(pl.UInt32).sum().alias(count_columns[value]) for value in share_values])
        .join(merchant_totals_lf, on="normalized_merchant_nm", how="left")
        .with_columns([safe_divide_expr(pl.col(count_columns[value]), pl.col("merchant_txn_cnt"), share_name_map[value]) for value in share_values])
        .select("normalized_merchant_nm", *[share_name_map[value] for value in share_values])
    )
    category_distribution_lf = (
        category_counts_lf.join(merchant_totals_lf, on="normalized_merchant_nm", how="left")
        .with_columns(safe_divide_expr(pl.col("category_txn_cnt"), pl.col("merchant_txn_cnt"), "__category_p"))
        .group_by("normalized_merchant_nm")
        .agg(
            pl.len().alias("unique_categories_cnt"),
            pl.col("__category_p").max().alias("top_category_txn_share"),
            (-pl.col("__category_p") * pl.col("__category_p").log()).sum().alias("category_entropy"),
        )
    )
    return category_distribution_lf.join(category_share_lf, on="normalized_merchant_nm", how="left")


def build_mcc_count_shard_lf(source_lf: pl.LazyFrame) -> pl.LazyFrame:
    return source_lf.group_by(["normalized_merchant_nm", "merchant_type_code_label"]).agg(pl.len().alias("mcc_txn_cnt"))


def build_mcc_features_lf(mcc_counts_lf: pl.LazyFrame, cfg: dict) -> pl.LazyFrame:
    top_values = cfg["top_mcc_values"]
    share_values = cfg["mcc_share_values"]
    share_name_map = cfg["mcc_share_name_map"]
    merchant_totals_lf = pl.scan_parquet(BASE_FEATURES_PATH).select("normalized_merchant_nm", "merchant_txn_cnt")
    count_columns = {value: f"__mcc_bucket_cnt_{idx}" for idx, value in enumerate(share_values)}
    mcc_share_lf = (
        mcc_counts_lf.with_columns(
            pl.when(pl.col("merchant_type_code_label").is_in(top_values)).then(pl.col("merchant_type_code_label")).otherwise(pl.lit("other")).alias("__mcc_bucket")
        )
        .group_by("normalized_merchant_nm")
        .agg([pl.col("__mcc_bucket").eq(value).cast(pl.UInt32).sum().alias(count_columns[value]) for value in share_values])
        .join(merchant_totals_lf, on="normalized_merchant_nm", how="left")
        .with_columns([safe_divide_expr(pl.col(count_columns[value]), pl.col("merchant_txn_cnt"), share_name_map[value]) for value in share_values])
        .select("normalized_merchant_nm", *[share_name_map[value] for value in share_values])
    )
    mcc_distribution_lf = (
        mcc_counts_lf.join(merchant_totals_lf, on="normalized_merchant_nm", how="left")
        .with_columns(safe_divide_expr(pl.col("mcc_txn_cnt"), pl.col("merchant_txn_cnt"), "__mcc_p"))
        .group_by("normalized_merchant_nm")
        .agg(
            pl.len().alias("unique_mcc_cnt"),
            pl.col("__mcc_p").max().alias("top_mcc_txn_share"),
            (-pl.col("__mcc_p") * pl.col("__mcc_p").log()).sum().alias("mcc_entropy"),
        )
    )
    return mcc_distribution_lf.join(mcc_share_lf, on="normalized_merchant_nm", how="left")


def build_merchant_user_count_shard_lf(source_lf: pl.LazyFrame) -> pl.LazyFrame:
    return source_lf.group_by(["normalized_merchant_nm", "party_rk"]).agg(pl.len().alias("merchant_user_txn_cnt"))


def build_user_interaction_features_lf(counts_lf: pl.LazyFrame) -> pl.LazyFrame:
    merchant_totals_lf = pl.scan_parquet(BASE_FEATURES_PATH).select("normalized_merchant_nm", "merchant_txn_cnt")
    return (
        counts_lf.group_by("normalized_merchant_nm")
        .agg(
            pl.len().alias("unique_users_cnt"),
            pl.col("merchant_user_txn_cnt").median().cast(pl.Float64).alias("median_txn_per_user"),
            pl.col("merchant_user_txn_cnt").max().alias("max_txn_per_user"),
            pl.col("merchant_user_txn_cnt").eq(1).cast(pl.UInt32).sum().alias("__single_visit_users_cnt"),
            pl.col("merchant_user_txn_cnt").max().alias("__top_user_txn_cnt"),
        )
        .join(merchant_totals_lf, on="normalized_merchant_nm", how="left")
        .with_columns(
            safe_divide_expr(pl.col("__top_user_txn_cnt"), pl.col("merchant_txn_cnt"), "top_user_txn_share"),
            safe_divide_expr(pl.col("__single_visit_users_cnt"), pl.col("unique_users_cnt"), "share_of_single_visit_users"),
            safe_divide_expr(pl.col("merchant_txn_cnt"), pl.col("unique_users_cnt"), "avg_txn_per_user"),
            safe_divide_expr(pl.col("unique_users_cnt"), pl.col("merchant_txn_cnt"), "user_concentration_ratio"),
        )
        .select(
            "normalized_merchant_nm",
            "unique_users_cnt",
            "median_txn_per_user",
            "max_txn_per_user",
            "top_user_txn_share",
            "share_of_single_visit_users",
            "avg_txn_per_user",
            "user_concentration_ratio",
        )
    )


def build_final_features_lf() -> pl.LazyFrame:
    base_lf = pl.scan_parquet(BASE_FEATURES_PATH)
    daily_lf = pl.scan_parquet(DAILY_FEATURES_PATH)
    category_lf = pl.scan_parquet(CATEGORY_FEATURES_PATH)
    mcc_lf = pl.scan_parquet(MCC_FEATURES_PATH)
    user_lf = pl.scan_parquet(USER_INTERACTION_FEATURES_PATH)
    return (
        base_lf.join(daily_lf, on="normalized_merchant_nm", how="left")
        .join(category_lf, on="normalized_merchant_nm", how="left")
        .join(mcc_lf, on="normalized_merchant_nm", how="left")
        .join(user_lf, on="normalized_merchant_nm", how="left")
        .with_columns(
            pl.col("active_days_cnt").fill_null(0).cast(pl.UInt32),
            pl.col("active_weeks_cnt").fill_null(0).cast(pl.UInt32),
            pl.col("active_months_cnt").fill_null(0).cast(pl.UInt32),
            pl.col("max_txn_in_one_day").fill_null(0).cast(pl.UInt32),
            pl.col("median_txn_per_active_day").fill_null(0.0).cast(pl.Float64),
            pl.col("unique_categories_cnt").fill_null(0).cast(pl.UInt32),
            pl.col("unique_mcc_cnt").fill_null(0).cast(pl.UInt32),
            pl.col("unique_users_cnt").fill_null(0).cast(pl.UInt32),
            pl.col("median_txn_per_user").fill_null(0.0).cast(pl.Float64),
            pl.col("max_txn_per_user").fill_null(0).cast(pl.UInt32),
            pl.col("top_user_txn_share").fill_null(0.0).cast(pl.Float64),
            pl.col("share_of_single_visit_users").fill_null(0.0).cast(pl.Float64),
            pl.col("avg_txn_per_user").fill_null(0.0).cast(pl.Float64),
            pl.col("user_concentration_ratio").fill_null(0.0).cast(pl.Float64),
            pl.col("top_category_txn_share").fill_null(0.0).cast(pl.Float64),
            pl.col("top_mcc_txn_share").fill_null(0.0).cast(pl.Float64),
            pl.col("category_entropy").fill_null(0.0).cast(pl.Float64),
            pl.col("mcc_entropy").fill_null(0.0).cast(pl.Float64),
        )
        .with_columns(
            safe_divide_expr(pl.col("merchant_txn_cnt"), pl.col("active_days_cnt"), "avg_txn_per_active_day"),
            safe_divide_expr(pl.col("merchant_txn_cnt"), pl.col("active_weeks_cnt"), "avg_txn_per_active_week"),
            safe_divide_expr(pl.col("merchant_txn_cnt"), pl.col("active_months_cnt"), "avg_txn_per_active_month"),
            weekday_name_expr("dominant_weekday_num"),
            pl.lit(None, dtype=pl.Float64).alias("avg_unique_users_per_active_day"),
            pl.col("merchant_txn_cnt").rank(method="dense", descending=True).cast(pl.UInt32).alias("merchant_popularity_rank_proxy"),
        )
    )


def stage_pair_count_shards(label_prefix: str, shard_dir: Path, builder, rebuild: bool) -> None:
    source_shards = prepared_shard_files()
    target_paths = [shard_dir / f"part_{idx:03d}.parquet" for idx in range(len(source_shards))]
    with timed(label_prefix):
        for idx, (source_path, target_path) in enumerate(zip(source_shards, target_paths), start=1):
            if target_path.exists() and not rebuild:
                log(f"Skip existing shard: {target_path.name}")
                continue
            sink_parquet_stage(
                f"{label_prefix} shard {idx}/{len(source_shards)}",
                target_path,
                builder(pl.scan_parquet(str(source_path))),
                rebuild=True,
            )


def merge_count_shards(
    label: str,
    shard_files: list[Path],
    merged_path: Path,
    keys: list[str],
    count_col: str,
    rebuild: bool,
) -> None:
    if merged_path.exists() and not rebuild:
        log(f"Skip existing stage: {merged_path}")
        return
    if not shard_files:
        raise FileNotFoundError(f"No shard files found for merge into {merged_path}")
    merged_lf = pl.scan_parquet([str(path) for path in shard_files]).group_by(keys).agg(pl.col(count_col).sum().alias(count_col))
    sink_parquet_stage(label, merged_path, merged_lf, rebuild=True)


def stage_base_features(rebuild: bool) -> None:
    sink_parquet_stage("Stage 2: merchant base features", BASE_FEATURES_PATH, build_base_features_lf(tx_lf()), rebuild=rebuild)


def stage_daily_count_shards(rebuild: bool) -> None:
    stage_pair_count_shards("Stage 3a: merchant daily count shards", DAILY_COUNT_SHARDS_DIR, build_daily_count_shard_lf, rebuild)


def stage_daily_features(rebuild: bool) -> None:
    merge_count_shards(
        "Stage 3b: merge merchant daily counts",
        daily_count_shard_files(),
        DAILY_COUNTS_PATH,
        ["normalized_merchant_nm", "txn_date"],
        "daily_txn_cnt",
        rebuild,
    )
    sink_parquet_stage(
        "Stage 3c: merchant daily features",
        DAILY_FEATURES_PATH,
        build_daily_features_lf(pl.scan_parquet(DAILY_COUNTS_PATH)),
        rebuild=rebuild,
    )


def stage_category_count_shards(rebuild: bool) -> None:
    stage_pair_count_shards("Stage 4a: merchant category count shards", CATEGORY_COUNT_SHARDS_DIR, build_category_count_shard_lf, rebuild)


def stage_category_features(rebuild: bool) -> None:
    merge_count_shards(
        "Stage 4b: merge merchant category counts",
        category_count_shard_files(),
        CATEGORY_COUNTS_PATH,
        ["normalized_merchant_nm", "category_nm"],
        "category_txn_cnt",
        rebuild,
    )
    sink_parquet_stage(
        "Stage 4c: merchant category features",
        CATEGORY_FEATURES_PATH,
        build_category_features_lf(pl.scan_parquet(CATEGORY_COUNTS_PATH), load_bucket_config()),
        rebuild=rebuild,
    )


def stage_mcc_count_shards(rebuild: bool) -> None:
    stage_pair_count_shards("Stage 5a: merchant MCC count shards", MCC_COUNT_SHARDS_DIR, build_mcc_count_shard_lf, rebuild)


def stage_mcc_features(rebuild: bool) -> None:
    merge_count_shards(
        "Stage 5b: merge merchant MCC counts",
        mcc_count_shard_files(),
        MCC_COUNTS_PATH,
        ["normalized_merchant_nm", "merchant_type_code_label"],
        "mcc_txn_cnt",
        rebuild,
    )
    sink_parquet_stage(
        "Stage 5c: merchant MCC features",
        MCC_FEATURES_PATH,
        build_mcc_features_lf(pl.scan_parquet(MCC_COUNTS_PATH), load_bucket_config()),
        rebuild=rebuild,
    )


def stage_merchant_user_count_shards(rebuild: bool) -> None:
    stage_pair_count_shards("Stage 6a: merchant-user count shards", MERCHANT_USER_SHARDS_DIR, build_merchant_user_count_shard_lf, rebuild)


def stage_user_features(rebuild: bool) -> None:
    merge_count_shards(
        "Stage 6b: merge merchant-user counts",
        merchant_user_shard_files(),
        MERCHANT_USER_COUNTS_PATH,
        ["normalized_merchant_nm", "party_rk"],
        "merchant_user_txn_cnt",
        rebuild,
    )
    sink_parquet_stage(
        "Stage 6c: merchant user interaction features",
        USER_INTERACTION_FEATURES_PATH,
        build_user_interaction_features_lf(pl.scan_parquet(MERCHANT_USER_COUNTS_PATH)),
        rebuild=rebuild,
    )


def run_final_checks(final_df: pl.DataFrame) -> None:
    merchant_count = final_df.height
    unique_merchant_count = final_df.select(pl.col("normalized_merchant_nm").n_unique()).item()
    if merchant_count != unique_merchant_count:
        raise ValueError(f"Duplicate normalized_merchant_nm detected: rows={merchant_count}, unique={unique_merchant_count}")

    invalid_unique_txn = final_df.filter(pl.col("merchant_txn_cnt") < pl.col("unique_transaction_cnt")).height
    if invalid_unique_txn:
        raise ValueError(f"merchant_txn_cnt < unique_transaction_cnt detected for {invalid_unique_txn} merchants")

    share_columns = [
        "share_weekend_txn",
        "share_weekday_txn",
        "share_night_txn",
        "share_morning_txn",
        "share_afternoon_txn",
        "share_evening_txn",
    ]
    for col_name in share_columns:
        invalid_cnt = final_df.filter((pl.col(col_name) < 0) | (pl.col(col_name) > 1)).height
        if invalid_cnt:
            raise ValueError(f"{col_name} is outside [0, 1] for {invalid_cnt} merchants")

    weekday_sum_bad = final_df.filter(
        (pl.col("merchant_txn_cnt") > 0) & ((pl.col("share_weekend_txn") + pl.col("share_weekday_txn") - 1.0).abs() > 1e-6)
    ).height
    if weekday_sum_bad:
        raise ValueError(f"share_weekend_txn + share_weekday_txn is inconsistent for {weekday_sum_bad} merchants")

    weekday_name_mismatch = final_df.filter(
        weekday_name_expr("dominant_weekday_num", "__expected_weekday") != pl.col("dominant_weekday")
    ).height
    if weekday_name_mismatch:
        raise ValueError(f"dominant_weekday_num is inconsistent with dominant_weekday for {weekday_name_mismatch} merchants")


def stage_final(rebuild: bool) -> None:
    if FINAL_PARQUET_PATH.exists() and FINAL_CSV_PATH.exists() and not rebuild:
        log(f"Skip existing final outputs: {FINAL_PARQUET_PATH} and {FINAL_CSV_PATH}")
        return
    with timed("Final stage: assemble merchant features"):
        final_df = collect_stream(build_final_features_lf().sort("merchant_txn_cnt", descending=True))
        run_final_checks(final_df)
        final_df.write_parquet(FINAL_PARQUET_PATH, compression="zstd")
        final_df.write_csv(FINAL_CSV_PATH)
    log(f"Final parquet saved: {FINAL_PARQUET_PATH}")
    log(f"Final csv saved: {FINAL_CSV_PATH}")
    log(f"Final shape: {final_df.shape}")
    log(f"Merchant count: {final_df.height}")
    log(f"First columns: {final_df.columns[:12]}")
    log(str(final_df.head(5)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crash-safe merchant feature pipeline runtime")
    parser.add_argument(
        "stage",
        choices=[
            "prepare_shards",
            "top_buckets",
            "base_features",
            "daily_count_shards",
            "daily_features",
            "category_count_shards",
            "category_features",
            "mcc_count_shards",
            "mcc_features",
            "merchant_user_count_shards",
            "user_features",
            "final",
            "all",
        ],
    )
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log(f"Pipeline version: {PIPELINE_VERSION}")
    log(f"Data source: {DATA_DIR}")
    log(f"Polars thread pool size: {THREAD_POOL_SIZE}")

    if args.stage == "prepare_shards":
        stage_prepare_shards(rebuild=args.rebuild)
    elif args.stage == "top_buckets":
        stage_top_buckets(rebuild=args.rebuild)
    elif args.stage == "base_features":
        stage_base_features(rebuild=args.rebuild)
    elif args.stage == "daily_count_shards":
        stage_daily_count_shards(rebuild=args.rebuild)
    elif args.stage == "daily_features":
        stage_daily_features(rebuild=args.rebuild)
    elif args.stage == "category_count_shards":
        stage_category_count_shards(rebuild=args.rebuild)
    elif args.stage == "category_features":
        stage_category_features(rebuild=args.rebuild)
    elif args.stage == "mcc_count_shards":
        stage_mcc_count_shards(rebuild=args.rebuild)
    elif args.stage == "mcc_features":
        stage_mcc_features(rebuild=args.rebuild)
    elif args.stage == "merchant_user_count_shards":
        stage_merchant_user_count_shards(rebuild=args.rebuild)
    elif args.stage == "user_features":
        stage_user_features(rebuild=args.rebuild)
    elif args.stage == "final":
        stage_final(rebuild=args.rebuild)
    elif args.stage == "all":
        stage_prepare_shards(rebuild=args.rebuild)
        stage_top_buckets(rebuild=args.rebuild)
        stage_base_features(rebuild=args.rebuild)
        stage_daily_count_shards(rebuild=args.rebuild)
        stage_daily_features(rebuild=args.rebuild)
        stage_category_count_shards(rebuild=args.rebuild)
        stage_category_features(rebuild=args.rebuild)
        stage_mcc_count_shards(rebuild=args.rebuild)
        stage_mcc_features(rebuild=args.rebuild)
        stage_merchant_user_count_shards(rebuild=args.rebuild)
        stage_user_features(rebuild=args.rebuild)
        stage_final(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
