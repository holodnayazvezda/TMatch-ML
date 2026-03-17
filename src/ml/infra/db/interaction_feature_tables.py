from __future__ import annotations

import polars as pl
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Table

from ml.featurize.interactions_features import get_user_interaction_feature_schema

from .base import Base

USER_INTERACTION_FEATURES_TABLE_NAME = "user_interaction_features"


def _sqlalchemy_type_from_polars(dtype: pl.DataType):
    if dtype == pl.Boolean:
        return Boolean
    if dtype in {pl.Float32, pl.Float64}:
        return Float
    if dtype in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }:
        return Integer
    if isinstance(dtype, pl.Datetime):
        return DateTime(timezone=False)
    return String


def _build_feature_columns() -> list[Column]:
    columns: list[Column] = []
    for column_name, dtype in get_user_interaction_feature_schema().items():
        sqlalchemy_type = _sqlalchemy_type_from_polars(dtype)
        if column_name == "user_id":
            columns.append(Column(column_name, String, primary_key=True))
            continue

        nullable = column_name != "snapshot_at"
        columns.append(Column(column_name, sqlalchemy_type, nullable=nullable))
    return columns


user_interaction_features_table = Table(
    USER_INTERACTION_FEATURES_TABLE_NAME,
    Base.metadata,
    *_build_feature_columns(),
)
