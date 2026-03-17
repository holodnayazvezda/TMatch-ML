from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import polars as pl

from ml.featurize.transactions_features import add_deterministic_user_traits

_SUPPORTED_FORMATS = {"csv", "parquet"}
_REQUIRED_BASE_COLUMNS = ("txn_cnt", "active_days_cnt", "activity_span_days")


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic user trait scores/tags for a local user-level dataset "
            "(same logic as transaction rebuild)."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input dataset (CSV/Parquet).")
    parser.add_argument("--output", required=False, help="Path to output dataset.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite input file atomically instead of writing to a new output path.",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "csv", "parquet"),
        default="auto",
        help="Input format override (default: auto by extension).",
    )
    parser.add_argument(
        "--output-format",
        choices=("auto", "csv", "parquet"),
        default="auto",
        help="Output format override (default: auto by extension).",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_path}")

    if args.output and args.in_place:
        raise ValueError("Use either --output or --in-place, not both.")

    input_format = _resolve_format(path=input_path, requested_format=args.input_format, label="input")
    output_path, output_format = _resolve_output_spec(
        input_path=input_path,
        output_arg=args.output,
        in_place=bool(args.in_place),
        requested_output_format=args.output_format,
        fallback_format=input_format,
    )

    source_df = _read_dataset(path=input_path, file_format=input_format)
    _validate_required_columns(source_df.columns)

    enriched_df = add_deterministic_user_traits(source_df)
    _write_dataset_atomic(df=enriched_df, path=output_path, file_format=output_format)

    trait_score_columns = [
        column_name
        for column_name in enriched_df.columns
        if column_name.startswith("trait_score_")
    ]
    users_with_display_tags = int(
        enriched_df.select(
            (pl.col("trait_tags_display").fill_null("[]") != "[]").sum().alias("with_tags")
        ).item()
    )

    summary = {
        "status": "ok",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_format": input_format,
        "output_format": output_format,
        "row_count": int(enriched_df.height),
        "trait_score_columns": sorted(trait_score_columns),
        "users_with_display_tags": users_with_display_tags,
    }
    print(json.dumps(summary, ensure_ascii=False))


def _resolve_output_spec(
    *,
    input_path: Path,
    output_arg: str | None,
    in_place: bool,
    requested_output_format: str,
    fallback_format: str,
) -> tuple[Path, str]:
    if in_place:
        output_path = input_path
    elif output_arg:
        output_path = Path(output_arg).expanduser().resolve()
    else:
        suffix = ".parquet" if fallback_format == "parquet" else ".csv"
        output_path = input_path.with_name(f"{input_path.stem}_with_traits{suffix}")

    output_format = _resolve_format(
        path=output_path,
        requested_format=requested_output_format,
        label="output",
        fallback_format=fallback_format,
    )
    return output_path, output_format


def _resolve_format(
    *,
    path: Path,
    requested_format: str,
    label: str,
    fallback_format: str | None = None,
) -> str:
    if requested_format != "auto":
        return requested_format

    suffix = path.suffix.lower().lstrip(".")
    if suffix in _SUPPORTED_FORMATS:
        return suffix

    if fallback_format is not None:
        return fallback_format

    raise ValueError(
        f"Cannot detect {label} format from extension '{path.suffix}'. "
        "Use --input-format/--output-format."
    )


def _read_dataset(*, path: Path, file_format: str) -> pl.DataFrame:
    if file_format == "csv":
        return pl.read_csv(path)
    if file_format == "parquet":
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported input format: {file_format}")


def _write_dataset_atomic(*, df: pl.DataFrame, path: Path, file_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{uuid4().hex}")

    if file_format == "csv":
        df.write_csv(temp_path)
    elif file_format == "parquet":
        df.write_parquet(temp_path)
    else:
        raise ValueError(f"Unsupported output format: {file_format}")

    temp_path.replace(path)


def _validate_required_columns(columns: list[str]) -> None:
    missing_columns = [
        column_name
        for column_name in _REQUIRED_BASE_COLUMNS
        if column_name not in columns
    ]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            "Input dataset does not contain required base user-level columns: "
            f"{missing_text}"
        )


if __name__ == "__main__":
    run()
