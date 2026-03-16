from __future__ import annotations

from typing import Any

import pandas as pd

from ml.candidates import transactions_candidates
from ml.explanation import get_transaction_trait_mapper
from ml.services.user_id import normalize_user_id

_TOP_TRAIT_COUNT = 3


def build_transaction_trait_reasons_response(raw_user_id: str) -> dict[str, object]:
    normalized_user_id = normalize_user_id(str(raw_user_id).strip(), allow_hash=True)
    if normalized_user_id is None:
        raise ValueError("Invalid user_id.")

    search_bundle = transactions_candidates.warmup_search_bundle()
    output_df = search_bundle.get("output_df")
    id_index = search_bundle.get("id_index")

    if not isinstance(output_df, pd.DataFrame) or id_index is None:
        raise RuntimeError("Transaction feature snapshot is unavailable.")
    if normalized_user_id not in id_index:
        raise LookupError("User not found in transaction feature snapshot.")

    user_row = _resolve_user_row(output_df, id_index, normalized_user_id)
    reasons = _build_top_transaction_traits(output_df, user_row, top_k=_TOP_TRAIT_COUNT)

    return {
        "user_id": normalized_user_id,
        "reasons": reasons,
    }


def _resolve_user_row(output_df: pd.DataFrame, id_index: Any, normalized_user_id: str) -> pd.Series:
    location = id_index.get_loc(normalized_user_id)
    if isinstance(location, slice):
        return output_df.iloc[int(location.start)]
    if isinstance(location, list):
        return output_df.iloc[int(location[0])]
    if hasattr(location, "tolist"):
        as_list = location.tolist()
        if isinstance(as_list, list):
            if as_list and isinstance(as_list[0], bool):
                true_positions = [position for position, is_true in enumerate(as_list) if is_true]
                if not true_positions:
                    raise LookupError("User not found in transaction feature snapshot.")
                return output_df.iloc[int(true_positions[0])]
            if as_list:
                return output_df.iloc[int(as_list[0])]
    return output_df.iloc[int(location)]


def _build_top_transaction_traits(
    output_df: pd.DataFrame,
    user_row: pd.Series,
    *,
    top_k: int,
) -> dict[str, float]:
    scored_traits: list[tuple[str, float, int]] = []
    trait_mapper = get_transaction_trait_mapper()

    for order_index, (column_name, trait_code) in enumerate(trait_mapper.items()):
        percentile = _compute_empirical_percentile(output_df, user_row, column_name)
        if percentile is None:
            continue
        scored_traits.append((trait_code, percentile, order_index))

    scored_traits.sort(key=lambda item: (-item[1], item[2]))
    limited_traits = scored_traits[: max(int(top_k), 0)]
    return {
        trait_code: percentile
        for trait_code, percentile, _ in limited_traits
    }


def _compute_empirical_percentile(
    output_df: pd.DataFrame,
    user_row: pd.Series,
    column_name: str,
) -> float | None:
    if column_name not in output_df.columns:
        return None

    population_values = pd.to_numeric(output_df[column_name], errors="coerce").dropna()
    if population_values.empty:
        return None

    user_value = pd.to_numeric(pd.Series([user_row.get(column_name)]), errors="coerce").iloc[0]
    if pd.isna(user_value):
        return None

    percentile = float((population_values <= float(user_value)).mean())
    return round(percentile, 4)
