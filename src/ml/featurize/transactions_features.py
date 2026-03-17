import json
import re
from pathlib import Path

import polars as pl

from core.data_config import Paths
from ml.services.user_id import normalize_user_id

PATHS = Paths()

DATA_DIR = PATHS.in_data_path
ARTIFACTS_DIR = PATHS.out_data_path
INTERMEDIATE_DIR = ARTIFACTS_DIR / "intermediate"
FEATURES_DIR = INTERMEDIATE_DIR / "user_feature_stages"

PIPELINE_VERSION = "v2"

TX_LAYER_PATH = INTERMEDIATE_DIR / f"tx_enriched_all_{PIPELINE_VERSION}.parquet"
BASE_FEATURES_PATH = FEATURES_DIR / f"base_activity_features_{PIPELINE_VERSION}.parquet"
DAILY_FEATURES_PATH = FEATURES_DIR / f"daily_features_{PIPELINE_VERSION}.parquet"
CATEGORY_FEATURES_PATH = FEATURES_DIR / f"category_features_{PIPELINE_VERSION}.parquet"
MCC_FEATURES_PATH = FEATURES_DIR / f"mcc_features_{PIPELINE_VERSION}.parquet"
MERCHANT_FEATURES_PATH = FEATURES_DIR / f"merchant_features_{PIPELINE_VERSION}.parquet"
FINAL_PARQUET_PATH = PATHS.transaction_user_features_parquet_path
FINAL_CSV_PATH = PATHS.transaction_user_features_csv_path

EXPECTED_COLUMNS = [
    "real_transaction_dttm",
    "party_rk",
    "transaction_rk",
    "merchant_type_code",
    "merchant_nm",
    "category_nm",
]

DT_FORMAT = "%Y-%m-%d %H:%M:%S"
TOP_CATEGORY_LIMIT = 30
TOP_MCC_LIMIT = 30

REBUILD_TX_LAYER = True
REBUILD_FEATURE_STAGES = True
REBUILD_FINAL_DATASET = True
DEDUPLICATE_BY_TRANSACTION_ID = False

SCHEMA_OVERRIDES = {
    "real_transaction_dttm": pl.Utf8,
    "party_rk": pl.Utf8,
    "transaction_rk": pl.Utf8,
    "merchant_type_code": pl.Utf8,
    "merchant_nm": pl.Utf8,
    "category_nm": pl.Utf8,
}

hour_count_cols = [f"__hour_{hour:02d}_cnt" for hour in range(24)]
weekday_count_cols = [f"__weekday_{weekday}_cnt" for weekday in range(1, 8)]

TRAIT_INTERNAL_CODES = (
    "owl",
    "early_bird",
    "weekend_lover",
    "explorer",
    "routine",
    "out_of_home_leisure",
    "traveler",
)

TRAIT_SCORE_COLUMN_BY_CODE = {
    "owl": "trait_score_owl",
    "early_bird": "trait_score_early_bird",
    "weekend_lover": "trait_score_weekend_lover",
    "explorer": "trait_score_explorer",
    "routine": "trait_score_routine",
    "out_of_home_leisure": "trait_score_out_of_home_leisure",
    "traveler": "trait_score_traveler",
}

TRAIT_DISPLAY_TAG_BY_CODE = {
    "owl": "сова",
    "early_bird": "ранняя пташка",
    "weekend_lover": "любит активные выходные",
    "explorer": "любит новые места",
    "routine": "предпочитает стабильный ритм",
    "out_of_home_leisure": "любит досуг вне дома",
    "traveler": "любит поездки",
}

TRAIT_PERCENTILE_SOURCE_FEATURES = [
    "share_night_txn",
    "share_weekend_txn",
    "category_entropy",
    "mcc_entropy",
    "unique_merchant_cnt",
    "merchant_diversity_ratio",
    "category_diversity_ratio",
    "mcc_diversity_ratio",
    "cat_share_рестораны",
    "cat_share_развлечения",
    "cat_share_кино",
    "cat_share_авиабилеты",
    "cat_share_ж_д_билеты",
    "cat_share_турагентства",
    "cat_share_аренда_авто",
]

TRAIT_EXPLORER_COMPONENTS = [
    "category_entropy",
    "mcc_entropy",
    "unique_merchant_cnt",
    "merchant_diversity_ratio",
    "category_diversity_ratio",
    "mcc_diversity_ratio",
]
TRAIT_OUT_OF_HOME_COMPONENTS = [
    "cat_share_рестораны",
    "cat_share_развлечения",
    "cat_share_кино",
]
TRAIT_TRAVEL_COMPONENTS = [
    "cat_share_авиабилеты",
    "cat_share_ж_д_билеты",
    "cat_share_турагентства",
    "cat_share_аренда_авто",
]

TRAIT_MIN_VALID_COMPONENTS_DEFAULT = 1
TRAIT_MIN_VALID_COMPONENTS_OUT_OF_HOME = 2
TRAIT_MIN_VALID_COMPONENTS_TRAVELER = 2

TRAIT_ELIGIBILITY_MIN_TXN_CNT = 20
TRAIT_ELIGIBILITY_MIN_ACTIVE_DAYS_CNT = 10
TRAIT_ELIGIBILITY_MIN_ACTIVITY_SPAN_DAYS = 30

TRAIT_SUPPORT_TXN_REFERENCE = 50.0
TRAIT_SUPPORT_ACTIVE_DAYS_REFERENCE = 20.0
TRAIT_SUPPORT_ACTIVITY_SPAN_REFERENCE = 90.0

TRAIT_UPPER_TAIL_START = 0.85
TRAIT_LOWER_TAIL_END = 0.15
TRAIT_TAIL_WIDTH = 0.15

TRAIT_TAG_SCORE_THRESHOLD = 0.35
TRAIT_TAG_MAX_COUNT = 3
TRAIT_INTERNAL_ELIGIBLE_COLUMN = "__traits_eligible"
TRAIT_INTERNAL_SUPPORT_WEIGHT_COLUMN = "__trait_support_weight"


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


def sink_parquet_stage(path: Path, lf: pl.LazyFrame, rebuild: bool = False) -> None:
    if path.exists() and not rebuild:
        return
    sink_parquet(path, lf)


def sanitize_token(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def normalize_party_rk(value: str | None) -> str | None:
    return normalize_user_id(value, allow_hash=True)


def build_unique_feature_name_map(values: list, prefix: str) -> dict:
    mapping = {}
    seen = {}
    for raw_value in values:
        base_name = f"{prefix}{sanitize_token(raw_value)}"
        suffix = seen.get(base_name, 0)
        feature_name = base_name if suffix == 0 else f"{base_name}_{suffix}"
        seen[base_name] = suffix + 1
        mapping[raw_value] = feature_name
    return mapping


def extend_with_required(values: list, required_values: list) -> list:
    result = []
    seen = set()
    for value in values + required_values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def weekday_name_expr(column_name: str, alias: str = "dominant_weekday") -> pl.Expr:
    return (
        pl.when(pl.col(column_name) == 1).then(pl.lit("Monday"))
        .when(pl.col(column_name) == 2).then(pl.lit("Tuesday"))
        .when(pl.col(column_name) == 3).then(pl.lit("Wednesday"))
        .when(pl.col(column_name) == 4).then(pl.lit("Thursday"))
        .when(pl.col(column_name) == 5).then(pl.lit("Friday"))
        .when(pl.col(column_name) == 6).then(pl.lit("Saturday"))
        .when(pl.col(column_name) == 7).then(pl.lit("Sunday"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias(alias)
    )


def validate_inputs() -> list[Path]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Input data directory not found: {DATA_DIR}")

    csv_files = []
    for i in range(1, 2):
        folder = DATA_DIR / f"Data_{i}"
        print(folder)
        if folder.exists():
            csv_files.extend(sorted(folder.glob("*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {DATA_DIR}")

    sample_lf = pl.scan_csv(
        [str(csv_files[0])],
        has_header=True,
        schema_overrides=SCHEMA_OVERRIDES,
        null_values=["", "null", "None"],
    )

    input_columns = sample_lf.collect_schema().names()
    if input_columns != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected input columns: {input_columns}")

    return csv_files


def build_tx_layer(
    csv_files: list[Path],
    *,
    tx_layer_path: Path = TX_LAYER_PATH,
    rebuild_tx_layer: bool = REBUILD_TX_LAYER,
) -> None:
    if tx_layer_path.exists() and not rebuild_tx_layer:
        return

    raw_lf = pl.scan_csv(
        [str(path) for path in csv_files],
        has_header=True,
        schema_overrides=SCHEMA_OVERRIDES,
        null_values=["", "null", "None"],
    )

    input_columns = raw_lf.collect_schema().names()
    if input_columns != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected input columns in source CSV files: {input_columns}")

    prepared_lf = raw_lf.select(EXPECTED_COLUMNS).with_columns(
        pl.col("real_transaction_dttm").str.strptime(pl.Datetime, format=DT_FORMAT, strict=False),
        pl.col("party_rk")
        .cast(pl.Utf8)
        .map_elements(normalize_party_rk, return_dtype=pl.Utf8)
        .alias("party_rk"),
        pl.col("transaction_rk").cast(pl.Utf8),
        pl.coalesce([pl.col("merchant_nm").cast(pl.Utf8), pl.lit("unknown_merchant")]).alias("merchant_nm"),
        pl.coalesce([pl.col("category_nm").cast(pl.Utf8), pl.lit("unknown")]).alias("category_nm"),
        pl.col("merchant_type_code").cast(pl.Int32, strict=False).fill_null(-1).alias("merchant_type_code"),
    ).filter(pl.col("party_rk").is_not_null())

    if DEDUPLICATE_BY_TRANSACTION_ID:
        prepared_lf = prepared_lf.unique(subset=["transaction_rk"], keep="any")

    tx_layer_lf = prepared_lf.select(
        "party_rk",
        "transaction_rk",
        "merchant_nm",
        "category_nm",
        "merchant_type_code",
        "real_transaction_dttm",
    ).with_columns(
        pl.col("real_transaction_dttm").dt.truncate("1d").alias("txn_date"),
        pl.col("real_transaction_dttm").dt.strftime("%Y-%m").alias("month_key"),
        pl.col("real_transaction_dttm").dt.truncate("1w").alias("week_start"),
        pl.col("real_transaction_dttm").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("real_transaction_dttm").dt.weekday().cast(pl.Int8).alias("weekday_num"),
    )

    sink_parquet(tx_layer_path, tx_layer_lf)


def get_top_buckets(tx_lf: pl.LazyFrame):
    raw_top_categories = collect_stream(
        tx_lf.group_by("category_nm")
        .len()
        .sort("len", descending=True)
        .head(TOP_CATEGORY_LIMIT)
        .select("category_nm")
    )["category_nm"].to_list()

    top_category_values = [value for value in raw_top_categories if value != "unknown"]

    category_unknown_present = (
        collect_stream(
            tx_lf.filter(pl.col("category_nm") == "unknown").select(pl.len().alias("cnt"))
        )["cnt"][0] > 0
    )

    if category_unknown_present:
        top_category_values = extend_with_required(top_category_values, ["unknown"])

    raw_top_mcc = collect_stream(
        tx_lf.group_by("merchant_type_code")
        .len()
        .sort("len", descending=True)
        .head(TOP_MCC_LIMIT)
        .select("merchant_type_code")
    )["merchant_type_code"].to_list()

    top_mcc_values = [int(value) for value in raw_top_mcc if int(value) != -1]

    mcc_unknown_present = (
        collect_stream(
            tx_lf.filter(pl.col("merchant_type_code") == -1).select(pl.len().alias("cnt"))
        )["cnt"][0] > 0
    )

    mcc_bucket_values = [str(value) for value in top_mcc_values]
    if mcc_unknown_present:
        mcc_bucket_values = extend_with_required(mcc_bucket_values, ["unknown"])

    category_share_name_map = build_unique_feature_name_map(top_category_values + ["other"], "cat_share_")
    mcc_share_name_map = build_unique_feature_name_map(mcc_bucket_values + ["other"], "mcc_share_")

    return top_category_values, top_mcc_values, mcc_bucket_values, category_share_name_map, mcc_share_name_map


def build_base_activity_lf(tx_lf: pl.LazyFrame) -> pl.LazyFrame:
    agg_exprs = [
        pl.len().alias("txn_cnt"),
        pl.col("transaction_rk").n_unique().alias("unique_transaction_cnt"),
        pl.col("txn_date").drop_nulls().n_unique().alias("active_days_cnt"),
        pl.col("month_key").drop_nulls().n_unique().alias("active_months_cnt"),
        pl.col("week_start").drop_nulls().n_unique().alias("active_weeks_cnt"),
        pl.col("real_transaction_dttm").min().alias("first_txn_dttm"),
        pl.col("real_transaction_dttm").max().alias("last_txn_dttm"),
    ]

    agg_exprs.extend([
        pl.col("hour").eq(hour).fill_null(False).cast(pl.UInt32).sum().alias(f"__hour_{hour:02d}_cnt")
        for hour in range(24)
    ])

    agg_exprs.extend([
        pl.col("weekday_num").eq(weekday).fill_null(False).cast(pl.UInt32).sum().alias(f"__weekday_{weekday}_cnt")
        for weekday in range(1, 8)
    ])

    return (
        tx_lf.group_by("party_rk")
        .agg(agg_exprs)
        .with_columns(
            (pl.col("last_txn_dttm") - pl.col("first_txn_dttm")).dt.total_days().fill_null(0).cast(pl.Int32).alias("activity_span_days"),
            pl.when(pl.col("active_days_cnt") > 0).then(pl.col("txn_cnt") / pl.col("active_days_cnt")).otherwise(0.0).alias("avg_txn_per_active_day"),
            pl.when(pl.col("active_months_cnt") > 0).then(pl.col("txn_cnt") / pl.col("active_months_cnt")).otherwise(0.0).alias("avg_txn_per_month_active"),
            pl.when(pl.col("active_weeks_cnt") > 0).then(pl.col("txn_cnt") / pl.col("active_weeks_cnt")).otherwise(0.0).alias("avg_txn_per_week_active"),
            pl.concat_list([pl.col(col) for col in hour_count_cols]).list.arg_max().cast(pl.Int8).alias("dominant_hour"),
            (pl.concat_list([pl.col(col) for col in weekday_count_cols]).list.arg_max() + 1).cast(pl.Int8).alias("dominant_weekday_num"),
            pl.sum_horizontal([pl.col(f"__hour_{hour:02d}_cnt") for hour in range(0, 6)]).alias("__night_cnt"),
            pl.sum_horizontal([pl.col(f"__hour_{hour:02d}_cnt") for hour in range(6, 12)]).alias("__morning_cnt"),
            pl.sum_horizontal([pl.col(f"__hour_{hour:02d}_cnt") for hour in range(12, 18)]).alias("__afternoon_cnt"),
            pl.sum_horizontal([pl.col(f"__hour_{hour:02d}_cnt") for hour in range(18, 24)]).alias("__evening_cnt"),
            pl.sum_horizontal([
                pl.col("__weekday_1_cnt"),
                pl.col("__weekday_2_cnt"),
                pl.col("__weekday_3_cnt"),
                pl.col("__weekday_4_cnt"),
                pl.col("__weekday_5_cnt"),
            ]).alias("__weekday_cnt"),
            pl.sum_horizontal([pl.col("__weekday_6_cnt"), pl.col("__weekday_7_cnt")]).alias("__weekend_cnt"),
        )
        .with_columns(
            pl.when(pl.col("txn_cnt") > 0).then(pl.col("__weekend_cnt") / pl.col("txn_cnt")).otherwise(0.0).alias("share_weekend_txn"),
            pl.when(pl.col("txn_cnt") > 0).then(pl.col("__weekday_cnt") / pl.col("txn_cnt")).otherwise(0.0).alias("share_weekday_txn"),
            pl.when(pl.col("txn_cnt") > 0).then(pl.col("__night_cnt") / pl.col("txn_cnt")).otherwise(0.0).alias("share_night_txn"),
            pl.when(pl.col("txn_cnt") > 0).then(pl.col("__morning_cnt") / pl.col("txn_cnt")).otherwise(0.0).alias("share_morning_txn"),
            pl.when(pl.col("txn_cnt") > 0).then(pl.col("__afternoon_cnt") / pl.col("txn_cnt")).otherwise(0.0).alias("share_afternoon_txn"),
            pl.when(pl.col("txn_cnt") > 0).then(pl.col("__evening_cnt") / pl.col("txn_cnt")).otherwise(0.0).alias("share_evening_txn"),
            pl.concat_list([
                pl.col("__night_cnt"),
                pl.col("__morning_cnt"),
                pl.col("__afternoon_cnt"),
                pl.col("__evening_cnt"),
            ]).list.arg_max().cast(pl.Int8).alias("__dominant_daypart_idx"),
        )
        .with_columns(
            pl.when(pl.col("__dominant_daypart_idx") == 0).then(pl.lit("night"))
            .when(pl.col("__dominant_daypart_idx") == 1).then(pl.lit("morning"))
            .when(pl.col("__dominant_daypart_idx") == 2).then(pl.lit("afternoon"))
            .otherwise(pl.lit("evening"))
            .alias("dominant_daypart")
        )
        .drop(
            hour_count_cols
            + weekday_count_cols
            + [
                "__night_cnt",
                "__morning_cnt",
                "__afternoon_cnt",
                "__evening_cnt",
                "__weekday_cnt",
                "__weekend_cnt",
                "__dominant_daypart_idx",
            ]
        )
    )


def build_daily_features_lf(tx_lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        tx_lf.select("party_rk", "txn_date")
        .drop_nulls()
        .group_by(["party_rk", "txn_date"])
        .len()
        .rename({"len": "txn_day_cnt"})
        .group_by("party_rk")
        .agg(
            pl.col("txn_day_cnt").max().alias("max_txn_in_one_day"),
            pl.col("txn_day_cnt").median().alias("median_txn_per_active_day"),
        )
    )


def build_category_features_lf(
    tx_lf: pl.LazyFrame,
    top_category_values: list,
    category_share_name_map: dict,
) -> pl.LazyFrame:
    category_counts_lf = (
        tx_lf.select("party_rk", "category_nm")
        .group_by(["party_rk", "category_nm"])
        .len()
        .rename({"len": "category_txn_cnt"})
        .with_columns(
            (pl.col("category_txn_cnt") / pl.col("category_txn_cnt").sum().over("party_rk")).alias("share"),
            pl.when(pl.col("category_nm") == "unknown").then(pl.lit("unknown"))
            .when(pl.col("category_nm").is_in(top_category_values)).then(pl.col("category_nm"))
            .otherwise(pl.lit("other"))
            .alias("category_bucket"),
        )
    )

    agg_exprs = [
        pl.len().alias("unique_category_cnt"),
        pl.col("share").max().alias("top_category_txn_share"),
        (-(pl.col("share") * pl.col("share").log()).sum()).fill_null(0.0).alias("category_entropy"),
    ]

    agg_exprs.extend([
        pl.col("share")
        .filter(pl.col("category_bucket") == bucket_value)
        .sum()
        .fill_null(0.0)
        .alias(category_share_name_map[bucket_value])
        for bucket_value in top_category_values
    ])

    agg_exprs.append(
        pl.col("share")
        .filter(pl.col("category_bucket") == "other")
        .sum()
        .fill_null(0.0)
        .alias(category_share_name_map["other"])
    )

    return category_counts_lf.group_by("party_rk").agg(agg_exprs)


def build_mcc_features_lf(
    tx_lf: pl.LazyFrame,
    top_mcc_values: list,
    mcc_bucket_values: list,
    mcc_share_name_map: dict,
) -> pl.LazyFrame:
    mcc_counts_lf = (
        tx_lf.select("party_rk", "merchant_type_code")
        .group_by(["party_rk", "merchant_type_code"])
        .len()
        .rename({"len": "mcc_txn_cnt"})
        .with_columns(
            (pl.col("mcc_txn_cnt") / pl.col("mcc_txn_cnt").sum().over("party_rk")).alias("share"),
            pl.when(pl.col("merchant_type_code") == -1).then(pl.lit("unknown"))
            .when(pl.col("merchant_type_code").is_in(top_mcc_values)).then(pl.col("merchant_type_code").cast(pl.Utf8))
            .otherwise(pl.lit("other"))
            .alias("mcc_bucket"),
        )
    )

    agg_exprs = [
        pl.len().alias("unique_mcc_cnt"),
        pl.col("share").max().alias("top_mcc_txn_share"),
        (-(pl.col("share") * pl.col("share").log()).sum()).fill_null(0.0).alias("mcc_entropy"),
    ]

    agg_exprs.extend([
        pl.col("share")
        .filter(pl.col("mcc_bucket") == bucket_value)
        .sum()
        .fill_null(0.0)
        .alias(mcc_share_name_map[bucket_value])
        for bucket_value in mcc_bucket_values
    ])

    agg_exprs.append(
        pl.col("share")
        .filter(pl.col("mcc_bucket") == "other")
        .sum()
        .fill_null(0.0)
        .alias(mcc_share_name_map["other"])
    )

    return mcc_counts_lf.group_by("party_rk").agg(agg_exprs)


def build_merchant_features_lf(tx_lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        tx_lf.select("party_rk", "merchant_nm")
        .group_by(["party_rk", "merchant_nm"])
        .len()
        .rename({"len": "merchant_txn_cnt"})
        .group_by("party_rk")
        .agg(
            pl.len().alias("unique_merchant_cnt"),
            (pl.col("merchant_txn_cnt").max() / pl.col("merchant_txn_cnt").sum()).alias("top_merchant_txn_share"),
        )
    )


def build_feature_stages(
    tx_lf: pl.LazyFrame,
    *,
    base_features_path: Path = BASE_FEATURES_PATH,
    daily_features_path: Path = DAILY_FEATURES_PATH,
    category_features_path: Path = CATEGORY_FEATURES_PATH,
    mcc_features_path: Path = MCC_FEATURES_PATH,
    merchant_features_path: Path = MERCHANT_FEATURES_PATH,
    rebuild_feature_stages: bool = REBUILD_FEATURE_STAGES,
) -> None:
    top_category_values, top_mcc_values, mcc_bucket_values, category_share_name_map, mcc_share_name_map = get_top_buckets(tx_lf)

    sink_parquet_stage(
        base_features_path,
        build_base_activity_lf(tx_lf),
        rebuild=rebuild_feature_stages,
    )

    sink_parquet_stage(
        daily_features_path,
        build_daily_features_lf(tx_lf),
        rebuild=rebuild_feature_stages,
    )

    sink_parquet_stage(
        category_features_path,
        build_category_features_lf(
            tx_lf=tx_lf,
            top_category_values=top_category_values,
            category_share_name_map=category_share_name_map,
        ),
        rebuild=rebuild_feature_stages,
    )

    sink_parquet_stage(
        mcc_features_path,
        build_mcc_features_lf(
            tx_lf=tx_lf,
            top_mcc_values=top_mcc_values,
            mcc_bucket_values=mcc_bucket_values,
            mcc_share_name_map=mcc_share_name_map,
        ),
        rebuild=rebuild_feature_stages,
    )

    sink_parquet_stage(
        merchant_features_path,
        build_merchant_features_lf(tx_lf),
        rebuild=rebuild_feature_stages,
    )


def add_deterministic_user_traits(user_features_pl: pl.DataFrame) -> pl.DataFrame:
    result = _prepare_trait_percentiles(user_features_pl)
    result = _compute_trait_scores(result)
    result = _resolve_trait_conflicts(result)
    result = _build_trait_tag_columns(result)

    trait_internal_columns = [
        TRAIT_INTERNAL_ELIGIBLE_COLUMN,
        TRAIT_INTERNAL_SUPPORT_WEIGHT_COLUMN,
        *[_trait_percentile_column_name(feature_name) for feature_name in TRAIT_PERCENTILE_SOURCE_FEATURES],
    ]
    columns_to_drop = [column_name for column_name in trait_internal_columns if column_name in result.columns]
    if columns_to_drop:
        result = result.drop(columns_to_drop)
    return result


def _prepare_trait_percentiles(user_features_pl: pl.DataFrame) -> pl.DataFrame:
    result = user_features_pl.clone()

    for feature_name in TRAIT_PERCENTILE_SOURCE_FEATURES:
        if feature_name not in result.columns:
            result = result.with_columns(pl.lit(None, dtype=pl.Float64).alias(feature_name))

    result = result.with_columns([
        pl.col(feature_name).cast(pl.Float64, strict=False).alias(feature_name)
        for feature_name in TRAIT_PERCENTILE_SOURCE_FEATURES
    ])

    result = result.with_columns(
        (
            (pl.col("txn_cnt").fill_null(0) >= TRAIT_ELIGIBILITY_MIN_TXN_CNT)
            & (pl.col("active_days_cnt").fill_null(0) >= TRAIT_ELIGIBILITY_MIN_ACTIVE_DAYS_CNT)
            & (pl.col("activity_span_days").fill_null(0) >= TRAIT_ELIGIBILITY_MIN_ACTIVITY_SPAN_DAYS)
        ).alias(TRAIT_INTERNAL_ELIGIBLE_COLUMN),
        pl.min_horizontal(
            pl.lit(1.0),
            _sqrt_non_negative_ratio_expr("txn_cnt", TRAIT_SUPPORT_TXN_REFERENCE),
            _sqrt_non_negative_ratio_expr("active_days_cnt", TRAIT_SUPPORT_ACTIVE_DAYS_REFERENCE),
            _sqrt_non_negative_ratio_expr("activity_span_days", TRAIT_SUPPORT_ACTIVITY_SPAN_REFERENCE),
        ).alias(TRAIT_INTERNAL_SUPPORT_WEIGHT_COLUMN),
    )

    percentile_exprs = [
        _build_percentile_expr(feature_name=feature_name, eligible_column=TRAIT_INTERNAL_ELIGIBLE_COLUMN)
        for feature_name in TRAIT_PERCENTILE_SOURCE_FEATURES
    ]
    return result.with_columns(percentile_exprs)


def _compute_trait_scores(user_features_pl: pl.DataFrame) -> pl.DataFrame:
    percentile_expr_by_feature = {
        feature_name: pl.col(_trait_percentile_column_name(feature_name))
        for feature_name in TRAIT_PERCENTILE_SOURCE_FEATURES
    }

    explorer_raw_expr = _composite_mean_expr(
        [percentile_expr_by_feature[feature_name] for feature_name in TRAIT_EXPLORER_COMPONENTS],
        min_valid_components=TRAIT_MIN_VALID_COMPONENTS_DEFAULT,
    )
    routine_raw_expr = _composite_mean_expr(
        [
            pl.when(percentile_expr_by_feature[feature_name].is_not_null())
            .then(1.0 - percentile_expr_by_feature[feature_name])
            .otherwise(None)
            for feature_name in TRAIT_EXPLORER_COMPONENTS
        ],
        min_valid_components=TRAIT_MIN_VALID_COMPONENTS_DEFAULT,
    )
    out_of_home_raw_expr = _composite_mean_expr(
        [percentile_expr_by_feature[feature_name] for feature_name in TRAIT_OUT_OF_HOME_COMPONENTS],
        min_valid_components=TRAIT_MIN_VALID_COMPONENTS_OUT_OF_HOME,
    )
    traveler_raw_expr = _composite_mean_expr(
        [percentile_expr_by_feature[feature_name] for feature_name in TRAIT_TRAVEL_COMPONENTS],
        min_valid_components=TRAIT_MIN_VALID_COMPONENTS_TRAVELER,
    )
    traveler_signal_expr = (
        pl.sum_horizontal([
            pl.when(pl.col(feature_name).fill_null(0.0) > 0.0).then(1).otherwise(0)
            for feature_name in TRAIT_TRAVEL_COMPONENTS
        ]) > 0
    )

    return user_features_pl.with_columns(
        _trait_score_expr(_upper_tail_score_expr(percentile_expr_by_feature["share_night_txn"]))
        .alias(TRAIT_SCORE_COLUMN_BY_CODE["owl"]),
        _trait_score_expr(_lower_tail_score_expr(percentile_expr_by_feature["share_night_txn"]))
        .alias(TRAIT_SCORE_COLUMN_BY_CODE["early_bird"]),
        _trait_score_expr(_upper_tail_score_expr(percentile_expr_by_feature["share_weekend_txn"]))
        .alias(TRAIT_SCORE_COLUMN_BY_CODE["weekend_lover"]),
        _trait_score_expr(_upper_tail_score_expr(explorer_raw_expr)).alias(TRAIT_SCORE_COLUMN_BY_CODE["explorer"]),
        _trait_score_expr(_upper_tail_score_expr(routine_raw_expr)).alias(TRAIT_SCORE_COLUMN_BY_CODE["routine"]),
        _trait_score_expr(_upper_tail_score_expr(out_of_home_raw_expr))
        .alias(TRAIT_SCORE_COLUMN_BY_CODE["out_of_home_leisure"]),
        _trait_score_expr(
            pl.when(traveler_signal_expr)
            .then(_upper_tail_score_expr(traveler_raw_expr))
            .otherwise(0.0)
        ).alias(TRAIT_SCORE_COLUMN_BY_CODE["traveler"]),
    )


def _resolve_trait_conflicts(user_features_pl: pl.DataFrame) -> pl.DataFrame:
    owl_score_expr = pl.col(TRAIT_SCORE_COLUMN_BY_CODE["owl"]).fill_null(0.0)
    early_bird_score_expr = pl.col(TRAIT_SCORE_COLUMN_BY_CODE["early_bird"]).fill_null(0.0)
    explorer_score_expr = pl.col(TRAIT_SCORE_COLUMN_BY_CODE["explorer"]).fill_null(0.0)
    routine_score_expr = pl.col(TRAIT_SCORE_COLUMN_BY_CODE["routine"]).fill_null(0.0)

    return user_features_pl.with_columns(
        pl.when(owl_score_expr >= early_bird_score_expr)
        .then(owl_score_expr)
        .otherwise(0.0)
        .alias(TRAIT_SCORE_COLUMN_BY_CODE["owl"]),
        pl.when(early_bird_score_expr > owl_score_expr)
        .then(early_bird_score_expr)
        .otherwise(0.0)
        .alias(TRAIT_SCORE_COLUMN_BY_CODE["early_bird"]),
        pl.when(explorer_score_expr >= routine_score_expr)
        .then(explorer_score_expr)
        .otherwise(0.0)
        .alias(TRAIT_SCORE_COLUMN_BY_CODE["explorer"]),
        pl.when(routine_score_expr > explorer_score_expr)
        .then(routine_score_expr)
        .otherwise(0.0)
        .alias(TRAIT_SCORE_COLUMN_BY_CODE["routine"]),
    )


def _build_trait_tag_columns(user_features_pl: pl.DataFrame) -> pl.DataFrame:
    score_struct_expr = pl.struct([
        pl.col(TRAIT_SCORE_COLUMN_BY_CODE[trait_code])
        for trait_code in TRAIT_INTERNAL_CODES
    ])
    return user_features_pl.with_columns(
        score_struct_expr.map_elements(
            _build_trait_internal_tags,
            return_dtype=pl.Utf8,
        ).alias("trait_tags_internal"),
        score_struct_expr.map_elements(
            _build_trait_display_tags,
            return_dtype=pl.Utf8,
        ).alias("trait_tags_display"),
    )


def _trait_percentile_column_name(feature_name: str) -> str:
    return f"__trait_pct_{sanitize_token(feature_name)}"


def _sqrt_non_negative_ratio_expr(column_name: str, reference_value: float) -> pl.Expr:
    numeric_value_expr = pl.col(column_name).fill_null(0.0).cast(pl.Float64, strict=False)
    return (
        pl.when(numeric_value_expr > 0.0)
        .then((numeric_value_expr / float(reference_value)).sqrt())
        .otherwise(0.0)
    )


def _build_percentile_expr(*, feature_name: str, eligible_column: str) -> pl.Expr:
    source_expr = (
        pl.when(pl.col(eligible_column) & pl.col(feature_name).is_not_null())
        .then(pl.col(feature_name).cast(pl.Float64, strict=False))
        .otherwise(None)
    )
    rank_expr = source_expr.rank(method="average")
    valid_count_expr = source_expr.is_not_null().sum()
    percentile_expr = (
        pl.when(rank_expr.is_not_null())
        .then(
            pl.when(valid_count_expr > 1)
            .then((rank_expr - 1.0) / (valid_count_expr - 1.0))
            .otherwise(0.0)
        )
        .otherwise(None)
    )
    return (
        pl.when(percentile_expr.is_not_null())
        .then(_clamp_0_1_expr(percentile_expr))
        .otherwise(None)
        .alias(_trait_percentile_column_name(feature_name))
    )


def _composite_mean_expr(component_exprs: list[pl.Expr], *, min_valid_components: int) -> pl.Expr:
    valid_count_expr = pl.sum_horizontal([
        pl.when(component_expr.is_not_null()).then(1).otherwise(0)
        for component_expr in component_exprs
    ]).cast(pl.Float64)
    value_sum_expr = pl.sum_horizontal([
        pl.when(component_expr.is_not_null()).then(component_expr).otherwise(0.0)
        for component_expr in component_exprs
    ])
    return (
        pl.when(valid_count_expr >= float(min_valid_components))
        .then(value_sum_expr / valid_count_expr)
        .otherwise(None)
    )


def _upper_tail_score_expr(percentile_expr: pl.Expr) -> pl.Expr:
    return _clamp_0_1_expr((percentile_expr - TRAIT_UPPER_TAIL_START) / TRAIT_TAIL_WIDTH)


def _lower_tail_score_expr(percentile_expr: pl.Expr) -> pl.Expr:
    return _clamp_0_1_expr((TRAIT_LOWER_TAIL_END - percentile_expr) / TRAIT_TAIL_WIDTH)


def _trait_score_expr(base_score_expr: pl.Expr) -> pl.Expr:
    support_weight_expr = pl.col(TRAIT_INTERNAL_SUPPORT_WEIGHT_COLUMN).fill_null(0.0)
    return (
        pl.when(pl.col(TRAIT_INTERNAL_ELIGIBLE_COLUMN))
        .then(_clamp_0_1_expr(base_score_expr * support_weight_expr))
        .otherwise(0.0)
    )


def _clamp_0_1_expr(value_expr: pl.Expr) -> pl.Expr:
    return pl.min_horizontal(
        pl.lit(1.0),
        pl.max_horizontal(pl.lit(0.0), value_expr),
    )


def _build_trait_internal_tags(score_row: dict[str, object]) -> str:
    return _serialize_trait_tags([trait_code for trait_code, _, _ in _select_top_trait_payload(score_row)])


def _build_trait_display_tags(score_row: dict[str, object]) -> str:
    return _serialize_trait_tags([display_tag for _, display_tag, _ in _select_top_trait_payload(score_row)])


def _select_top_trait_payload(score_row: dict[str, object] | None) -> list[tuple[str, str, float]]:
    if not isinstance(score_row, dict):
        return []

    selected: list[tuple[str, str, float, int]] = []
    for order_index, trait_code in enumerate(TRAIT_INTERNAL_CODES):
        score_column = TRAIT_SCORE_COLUMN_BY_CODE[trait_code]
        try:
            score_value = float(score_row.get(score_column, 0.0) or 0.0)
        except (TypeError, ValueError):
            score_value = 0.0
        if score_value < TRAIT_TAG_SCORE_THRESHOLD:
            continue
        selected.append((trait_code, TRAIT_DISPLAY_TAG_BY_CODE[trait_code], score_value, order_index))

    selected.sort(key=lambda trait_payload: (-trait_payload[2], trait_payload[3]))
    top_traits = selected[:TRAIT_TAG_MAX_COUNT]
    return [
        (trait_code, display_tag, score_value)
        for trait_code, display_tag, score_value, _ in top_traits
    ]


def _serialize_trait_tags(tag_values: list[str]) -> str:
    return json.dumps(tag_values, ensure_ascii=False, separators=(",", ":"))


def assemble_final_dataset(
    *,
    final_parquet_path: Path = FINAL_PARQUET_PATH,
    base_features_path: Path = BASE_FEATURES_PATH,
    daily_features_path: Path = DAILY_FEATURES_PATH,
    category_features_path: Path = CATEGORY_FEATURES_PATH,
    mcc_features_path: Path = MCC_FEATURES_PATH,
    merchant_features_path: Path = MERCHANT_FEATURES_PATH,
    rebuild_final_dataset: bool = REBUILD_FINAL_DATASET,
) -> None:
    if final_parquet_path.exists() and not rebuild_final_dataset:
        return

    final_lf = (
        pl.scan_parquet(base_features_path)
        .join(pl.scan_parquet(daily_features_path), on="party_rk", how="left")
        .join(pl.scan_parquet(category_features_path), on="party_rk", how="left")
        .join(pl.scan_parquet(mcc_features_path), on="party_rk", how="left")
        .join(pl.scan_parquet(merchant_features_path), on="party_rk", how="left")
        .with_columns(
            weekday_name_expr("dominant_weekday_num", "dominant_weekday"),
            pl.when(pl.col("txn_cnt") > 0).then(pl.col("unique_merchant_cnt") / pl.col("txn_cnt")).otherwise(0.0).alias("merchant_diversity_ratio"),
            pl.when(pl.col("txn_cnt") > 0).then(pl.col("unique_category_cnt") / pl.col("txn_cnt")).otherwise(0.0).alias("category_diversity_ratio"),
            pl.when(pl.col("txn_cnt") > 0).then(pl.col("unique_mcc_cnt") / pl.col("txn_cnt")).otherwise(0.0).alias("mcc_diversity_ratio"),
        )
    )

    final_df = collect_stream(final_lf)
    final_df = add_deterministic_user_traits(final_df)
    final_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(final_parquet_path)


def validate_final_dataset(user_features_pl: pl.DataFrame) -> None:
    if user_features_pl["party_rk"].n_unique() != user_features_pl.height:
        raise ValueError("Final dataset validation failed: party_rk is not unique")

    if not bool((user_features_pl["unique_transaction_cnt"] <= user_features_pl["txn_cnt"]).all()):
        raise ValueError("Final dataset validation failed: unique_transaction_cnt > txn_cnt")

    user_features_check_pl = user_features_pl.with_columns(
        weekday_name_expr("dominant_weekday_num", "_expected_dominant_weekday")
    )

    consistency_df = user_features_check_pl.select(
        (pl.col("party_rk").n_unique() == pl.len()).alias("one_row_per_user"),
        ((pl.col("dominant_weekday_num") >= 1) & (pl.col("dominant_weekday_num") <= 7)).all().alias("dominant_weekday_num_in_range"),
        (pl.col("_expected_dominant_weekday") == pl.col("dominant_weekday")).all().alias("dominant_weekday_consistent"),
        ((pl.col("share_weekend_txn") >= 0.0) & (pl.col("share_weekend_txn") <= 1.0)).all().alias("share_weekend_in_range"),
        ((pl.col("share_weekday_txn") >= 0.0) & (pl.col("share_weekday_txn") <= 1.0)).all().alias("share_weekday_in_range"),
        pl.when(pl.col("txn_cnt") > 0)
        .then((pl.col("share_weekend_txn") + pl.col("share_weekday_txn") - 1.0).abs() <= 1e-6)
        .otherwise(True)
        .all()
        .alias("weekday_share_sum_close_to_one"),
    )

    consistency_row = consistency_df.row(0, named=True)
    for check_name, check_value in consistency_row.items():
        if not bool(check_value):
            raise ValueError(f"Final dataset validation failed: {check_name}")


def save_final_outputs(
    user_features_pl: pl.DataFrame,
    *,
    final_parquet_path: Path = FINAL_PARQUET_PATH,
    final_csv_path: Path = FINAL_CSV_PATH,
) -> None:
    final_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    final_csv_path.parent.mkdir(parents=True, exist_ok=True)
    user_features_pl.write_parquet(final_parquet_path)
    user_features_pl.write_csv(final_csv_path)


def build_unique_user_ids_frame(user_features_pl: pl.DataFrame) -> pl.DataFrame:
    return (
        user_features_pl
        .select(pl.col("party_rk").alias("user_id"))
        .drop_nulls()
        .unique()
        .sort("user_id")
    )


def run_pipeline(
    *,
    intermediate_dir: Path = INTERMEDIATE_DIR,
    features_dir: Path = FEATURES_DIR,
    tx_layer_path: Path = TX_LAYER_PATH,
    base_features_path: Path = BASE_FEATURES_PATH,
    daily_features_path: Path = DAILY_FEATURES_PATH,
    category_features_path: Path = CATEGORY_FEATURES_PATH,
    mcc_features_path: Path = MCC_FEATURES_PATH,
    merchant_features_path: Path = MERCHANT_FEATURES_PATH,
    final_parquet_path: Path = FINAL_PARQUET_PATH,
    final_csv_path: Path = FINAL_CSV_PATH,
    rebuild_tx_layer: bool = REBUILD_TX_LAYER,
    rebuild_feature_stages: bool = REBUILD_FEATURE_STAGES,
    rebuild_final_dataset: bool = REBUILD_FINAL_DATASET,
) -> pl.DataFrame:
    final_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    csv_files = validate_inputs()
    build_tx_layer(
        csv_files,
        tx_layer_path=tx_layer_path,
        rebuild_tx_layer=rebuild_tx_layer,
    )

    tx_lf = pl.scan_parquet(tx_layer_path)
    build_feature_stages(
        tx_lf,
        base_features_path=base_features_path,
        daily_features_path=daily_features_path,
        category_features_path=category_features_path,
        mcc_features_path=mcc_features_path,
        merchant_features_path=merchant_features_path,
        rebuild_feature_stages=rebuild_feature_stages,
    )
    assemble_final_dataset(
        final_parquet_path=final_parquet_path,
        base_features_path=base_features_path,
        daily_features_path=daily_features_path,
        category_features_path=category_features_path,
        mcc_features_path=mcc_features_path,
        merchant_features_path=merchant_features_path,
        rebuild_final_dataset=rebuild_final_dataset,
    )

    user_features_pl = pl.read_parquet(final_parquet_path)
    validate_final_dataset(user_features_pl)
    save_final_outputs(
        user_features_pl,
        final_parquet_path=final_parquet_path,
        final_csv_path=final_csv_path,
    )
    return user_features_pl


def run() -> None:
    user_features_pl = run_pipeline()

    print(f"Parquet saved to: {FINAL_PARQUET_PATH.resolve()}")
    print(f"CSV saved to: {FINAL_CSV_PATH.resolve()}")
    user_ids_pl = build_unique_user_ids_frame(user_features_pl)
    print(f"User ids prepared: {user_ids_pl.height}")


if __name__ == "__main__":
    run()
