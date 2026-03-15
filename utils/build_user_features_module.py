from pathlib import Path
import re
import polars as pl

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REFERENCE_COLUMNS = [
    "real_transaction_dttm",
    "party_rk",
    "transaction_rk",
    "merchant_type_code",
    "merchant_nm",
    "category_nm",
]

ARTIFACTS_DIR = ROOT / "artifacts"
INTERMEDIATE_DIR = ARTIFACTS_DIR / "intermediate"
FEATURES_DIR = INTERMEDIATE_DIR / "user_feature_stages"

PIPELINE_VERSION = "v2"

TX_LAYER_PATH = INTERMEDIATE_DIR / f"tx_enriched_data1_{PIPELINE_VERSION}.parquet"
BASE_FEATURES_PATH = FEATURES_DIR / f"base_activity_features_{PIPELINE_VERSION}.parquet"
DAILY_FEATURES_PATH = FEATURES_DIR / f"daily_features_{PIPELINE_VERSION}.parquet"
CATEGORY_FEATURES_PATH = FEATURES_DIR / f"category_features_{PIPELINE_VERSION}.parquet"
MCC_FEATURES_PATH = FEATURES_DIR / f"mcc_features_{PIPELINE_VERSION}.parquet"
MERCHANT_FEATURES_PATH = FEATURES_DIR / f"merchant_features_{PIPELINE_VERSION}.parquet"

FINAL_PARQUET_PATH = ARTIFACTS_DIR / "user_features_data1.parquet"
FINAL_CSV_PATH = ARTIFACTS_DIR / "user_features_data1.csv"

DT_FORMAT = "%Y-%m-%d %H:%M:%S"
TOP_CATEGORY_LIMIT = 30
TOP_MCC_LIMIT = 30

REBUILD_TX_LAYER = False
REBUILD_FEATURE_STAGES = False
REBUILD_FINAL_DATASET = True
DEDUPLICATE_BY_TRANSACTION_ID = False

SCHEMA_OVERRIDES = {
    "real_transaction_dttm": pl.Utf8,
    "party_rk": pl.Utf8,
    "transaction_rk": pl.Utf8,
    "merchant_type_code": pl.Int32,
    "merchant_nm": pl.Utf8,
    "category_nm": pl.Utf8,
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


def sanitize_token(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


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
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    # Собираем все CSV файлы из папок Data_1, Data_2, ..., Data_6
    csv_files = []
    for i in range(1, 7):
        folder = DATA_DIR / f"Data_{i}"
        if folder.exists():
            csv_files.extend(sorted(folder.glob("*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {DATA_DIR}")

    reference_columns = SCHEMA_OVERRIDES.keys()
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
    if input_columns != list(SCHEMA_OVERRIDES.keys()):
        raise ValueError(f"Unexpected input columns: {input_columns}")

    prepared_lf = raw_lf.select(list(SCHEMA_OVERRIDES.keys())).with_columns(
        pl.col("real_transaction_dttm").str.strptime(pl.Datetime, format=DT_FORMAT, strict=False),
        pl.col("party_rk").cast(pl.Utf8),
        pl.col("transaction_rk").cast(pl.Utf8),
        pl.coalesce([pl.col("merchant_nm").cast(pl.Utf8), pl.lit("unknown_merchant")]).alias("merchant_nm"),
        pl.coalesce([pl.col("category_nm").cast(pl.Utf8), pl.lit("unknown")]).alias("category_nm"),
        pl.col("merchant_type_code").cast(pl.Int32, strict=False).fill_null(-1).alias("merchant_type_code"),
    )

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
        )["cnt"][0]
        > 0
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
        )["cnt"][0]
        > 0
    )

    mcc_bucket_values = [str(value) for value in top_mcc_values]
    if mcc_unknown_present:
        mcc_bucket_values = extend_with_required(mcc_bucket_values, ["unknown"])

    category_share_name_map = build_unique_feature_name_map(top_category_values + ["other"], "cat_share_")
    mcc_share_name_map = build_unique_feature_name_map(mcc_bucket_values + ["other"], "mcc_share_")

    return top_category_values, top_mcc_values, mcc_bucket_values, category_share_name_map, mcc_share_name_map


def build_feature_stages(tx_lf: pl.LazyFrame) -> None:
    top_category_values, top_mcc_values, mcc_bucket_values, category_share_name_map, mcc_share_name_map = get_top_buckets(tx_lf)

    if not BASE_FEATURES_PATH.exists() or REBUILD_FEATURE_STAGES:
        sink_parquet(BASE_FEATURES_PATH, build_base_activity_lf(tx_lf))

    if not DAILY_FEATURES_PATH.exists() or REBUILD_FEATURE_STAGES:
        sink_parquet(DAILY_FEATURES_PATH, build_daily_features_lf(tx_lf))

    if not CATEGORY_FEATURES_PATH.exists() or REBUILD_FEATURE_STAGES:
        sink_parquet(
            CATEGORY_FEATURES_PATH,
            build_category_features_lf(
                tx_lf=tx_lf,
                top_category_values=top_category_values,
                category_share_name_map=category_share_name_map,
            ),
        )

    if not MCC_FEATURES_PATH.exists() or REBUILD_FEATURE_STAGES:
        sink_parquet(
            MCC_FEATURES_PATH,
            build_mcc_features_lf(
                tx_lf=tx_lf,
                top_mcc_values=top_mcc_values,
                mcc_bucket_values=mcc_bucket_values,
                mcc_share_name_map=mcc_share_name_map,
            ),
        )

    if not MERCHANT_FEATURES_PATH.exists() or REBUILD_FEATURE_STAGES:
        sink_parquet(MERCHANT_FEATURES_PATH, build_merchant_features_lf(tx_lf))


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
    save_final_outputs(user_features_pl)

    print(f"PARQUET saved: {FINAL_PARQUET_PATH.resolve()}")
    print(f"CSV saved: {FINAL_CSV_PATH.resolve()}")


if __name__ == "__main__":
    run()