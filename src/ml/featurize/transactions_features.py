import re
from pathlib import Path

import polars as pl

from src.core.data_config import Paths
from src.ml.services.user_id import normalize_user_id

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


def build_tx_layer(csv_files: list[Path]) -> None:
    if TX_LAYER_PATH.exists() and not REBUILD_TX_LAYER:
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

    sink_parquet(TX_LAYER_PATH, tx_layer_lf)


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


def build_feature_stages(tx_lf: pl.LazyFrame) -> None:
    top_category_values, top_mcc_values, mcc_bucket_values, category_share_name_map, mcc_share_name_map = get_top_buckets(tx_lf)

    sink_parquet_stage(
        BASE_FEATURES_PATH,
        build_base_activity_lf(tx_lf),
        rebuild=REBUILD_FEATURE_STAGES,
    )

    sink_parquet_stage(
        DAILY_FEATURES_PATH,
        build_daily_features_lf(tx_lf),
        rebuild=REBUILD_FEATURE_STAGES,
    )

    sink_parquet_stage(
        CATEGORY_FEATURES_PATH,
        build_category_features_lf(
            tx_lf=tx_lf,
            top_category_values=top_category_values,
            category_share_name_map=category_share_name_map,
        ),
        rebuild=REBUILD_FEATURE_STAGES,
    )

    sink_parquet_stage(
        MCC_FEATURES_PATH,
        build_mcc_features_lf(
            tx_lf=tx_lf,
            top_mcc_values=top_mcc_values,
            mcc_bucket_values=mcc_bucket_values,
            mcc_share_name_map=mcc_share_name_map,
        ),
        rebuild=REBUILD_FEATURE_STAGES,
    )

    sink_parquet_stage(
        MERCHANT_FEATURES_PATH,
        build_merchant_features_lf(tx_lf),
        rebuild=REBUILD_FEATURE_STAGES,
    )


def assemble_final_dataset() -> None:
    if FINAL_PARQUET_PATH.exists() and not REBUILD_FINAL_DATASET:
        return

    final_lf = (
        pl.scan_parquet(BASE_FEATURES_PATH)
        .join(pl.scan_parquet(DAILY_FEATURES_PATH), on="party_rk", how="left")
        .join(pl.scan_parquet(CATEGORY_FEATURES_PATH), on="party_rk", how="left")
        .join(pl.scan_parquet(MCC_FEATURES_PATH), on="party_rk", how="left")
        .join(pl.scan_parquet(MERCHANT_FEATURES_PATH), on="party_rk", how="left")
        .with_columns(
            weekday_name_expr("dominant_weekday_num", "dominant_weekday"),
            pl.when(pl.col("txn_cnt") > 0).then(pl.col("unique_merchant_cnt") / pl.col("txn_cnt")).otherwise(0.0).alias("merchant_diversity_ratio"),
            pl.when(pl.col("txn_cnt") > 0).then(pl.col("unique_category_cnt") / pl.col("txn_cnt")).otherwise(0.0).alias("category_diversity_ratio"),
            pl.when(pl.col("txn_cnt") > 0).then(pl.col("unique_mcc_cnt") / pl.col("txn_cnt")).otherwise(0.0).alias("mcc_diversity_ratio"),
        )
    )

    sink_parquet(FINAL_PARQUET_PATH, final_lf)


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


def save_final_outputs(user_features_pl: pl.DataFrame) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    user_features_pl.write_parquet(FINAL_PARQUET_PATH)
    user_features_pl.write_csv(FINAL_CSV_PATH)


def run() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = validate_inputs()
    build_tx_layer(csv_files)

    tx_lf = pl.scan_parquet(TX_LAYER_PATH)
    build_feature_stages(tx_lf)
    assemble_final_dataset()

    user_features_pl = pl.read_parquet(FINAL_PARQUET_PATH)
    validate_final_dataset(user_features_pl)
    save_final_outputs(user_features_pl)

    print(f"Parquet saved to: {FINAL_PARQUET_PATH.resolve()}")
    print(f"CSV saved to: {FINAL_CSV_PATH.resolve()}")


if __name__ == "__main__":
    run()
