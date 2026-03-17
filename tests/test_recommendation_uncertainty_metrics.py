from __future__ import annotations

import pandas as pd
import pytest

from ml.services import recommendation_serving


def test_uncertainty_metrics_empty_frame_returns_none() -> None:
    mean_all_score, gap_score, finite_score_count = recommendation_serving._compute_uncertainty_metrics(pd.DataFrame())
    assert mean_all_score is None
    assert gap_score is None
    assert finite_score_count == 0


def test_uncertainty_metrics_all_nan_returns_none() -> None:
    candidate_df = pd.DataFrame({"rerank_score": [float("nan"), None, float("nan")]})
    mean_all_score, gap_score, finite_score_count = recommendation_serving._compute_uncertainty_metrics(candidate_df)
    assert mean_all_score is None
    assert gap_score is None
    assert finite_score_count == 0


def test_uncertainty_metrics_less_than_five_candidates() -> None:
    candidate_df = pd.DataFrame({"rerank_score": [0.3, 0.2, 0.1]})
    mean_all_score, gap_score, finite_score_count = recommendation_serving._compute_uncertainty_metrics(candidate_df)
    assert mean_all_score == pytest.approx(0.2)
    assert gap_score == pytest.approx(0.0)
    assert finite_score_count == 3


def test_uncertainty_metrics_more_than_five_candidates() -> None:
    candidate_df = pd.DataFrame({"rerank_score": [1.0, 0.9, 0.8, 0.7, 0.6, 0.1]})
    mean_all_score, gap_score, finite_score_count = recommendation_serving._compute_uncertainty_metrics(candidate_df)
    assert mean_all_score == pytest.approx(0.6833333333333332)
    assert gap_score == pytest.approx(0.7000000000000001)
    assert finite_score_count == 6
