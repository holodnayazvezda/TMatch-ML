from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any

import pandas as pd

PROBABILITY_TARGETS: tuple[str, ...] = (
    "block",
    "dislike",
    "skip",
    "like",
    "match",
    "chat_started",
)

PREDICTION_COLUMN_BY_TARGET = {
    target: f"y_pred_{target}"
    for target in PROBABILITY_TARGETS
}

SERVING_PROBABILITY_TARGETS: tuple[str, ...] = (
    "like",
    "dislike",
    "match",
    "block",
)

SERVING_PREDICTION_COLUMN_BY_TARGET = {
    target: f"y_pred_{target}"
    for target in SERVING_PROBABILITY_TARGETS
}


class RecommendationStage(str, Enum):
    STAGE_1 = "stage_1"
    STAGE_2 = "stage_2"
    STAGE_3 = "stage_3"
    STAGE_4 = "stage_4"


@dataclass(frozen=True)
class ScoreWeights:
    version: str = "v1.1"
    w_block: float = 1.0
    w_dislike: float = 1.0
    w_skip: float = 1.0
    w_like: float = 1.0
    w_match: float = 1.0
    w_chat_started: float = 1.0


DEFAULT_SCORE_WEIGHTS = ScoreWeights()


@dataclass(frozen=True)
class EnsembleWeights:
    version: str = "v1.0"
    w_like: float = 1.0
    w_dislike: float = -1.0
    w_match: float = 2.0
    w_block: float = -2.0


DEFAULT_ENSEMBLE_WEIGHTS = EnsembleWeights()


def compute_linear_score_frame(
    frame: pd.DataFrame,
    *,
    weights: ScoreWeights = DEFAULT_SCORE_WEIGHTS,
) -> pd.Series:
    return (
        frame[PREDICTION_COLUMN_BY_TARGET["like"]].fillna(0.0) * float(weights.w_like)
        + frame[PREDICTION_COLUMN_BY_TARGET["match"]].fillna(0.0) * float(weights.w_match)
        + frame[PREDICTION_COLUMN_BY_TARGET["chat_started"]].fillna(0.0) * float(weights.w_chat_started)
        - frame[PREDICTION_COLUMN_BY_TARGET["skip"]].fillna(0.0) * float(weights.w_skip)
        - frame[PREDICTION_COLUMN_BY_TARGET["dislike"]].fillna(0.0) * float(weights.w_dislike)
        - frame[PREDICTION_COLUMN_BY_TARGET["block"]].fillna(0.0) * float(weights.w_block)
    ).astype(float)


def get_score_weights_snapshot(weights: ScoreWeights = DEFAULT_SCORE_WEIGHTS) -> dict[str, Any]:
    snapshot = asdict(weights)
    snapshot["version"] = weights.version
    return snapshot


def compute_weighted_utility_score_frame(
    frame: pd.DataFrame,
    *,
    weights: EnsembleWeights = DEFAULT_ENSEMBLE_WEIGHTS,
) -> pd.Series:
    return (
        frame[SERVING_PREDICTION_COLUMN_BY_TARGET["like"]].fillna(0.0) * float(weights.w_like)
        + frame[SERVING_PREDICTION_COLUMN_BY_TARGET["match"]].fillna(0.0) * float(weights.w_match)
        + frame[SERVING_PREDICTION_COLUMN_BY_TARGET["dislike"]].fillna(0.0) * float(weights.w_dislike)
        + frame[SERVING_PREDICTION_COLUMN_BY_TARGET["block"]].fillna(0.0) * float(weights.w_block)
    ).astype(float)


def get_ensemble_weights_snapshot(weights: EnsembleWeights = DEFAULT_ENSEMBLE_WEIGHTS) -> dict[str, Any]:
    snapshot = asdict(weights)
    snapshot["version"] = weights.version
    return snapshot
