from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ml.explanation import build_exact_reasons
from ml.services.recommendation_constraints import (
    UserTargetingProfile,
    filter_candidate_frame_by_targeting,
)
from ml.services.recommendation_serving import _ensure_candidate_ids, _return_top_n


class ExactReasonsTests(unittest.TestCase):
    def test_build_exact_reasons_backfills_to_three_unique_labels(self) -> None:
        raw_reasons = {
            "dominant_weekday_num": 0.91,
            "dominant_weekday": 0.82,
            "txn_cnt": 0.74,
        }

        reasons = build_exact_reasons(raw_reasons, exact_count=3)

        self.assertIsInstance(reasons, dict)
        self.assertEqual(len(reasons), 3)
        self.assertIn("PRIME_WEEKDAY", reasons)
        self.assertIn("ACTIVITY_FREQUENCY", reasons)


class TargetingFilterTests(unittest.TestCase):
    @patch("ml.services.recommendation_constraints.load_user_targeting_profiles")
    def test_filter_candidate_frame_by_targeting_uses_mutual_compatibility(
        self,
        mock_load_profiles,
    ) -> None:
        actor_user_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        candidate_df = pd.DataFrame(
            {
                "candidate_id": [
                    "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "cccccccc-cccc-cccc-cccc-cccccccccccc",
                    "dddddddd-dddd-dddd-dddd-dddddddddddd",
                    "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
                ]
            }
        )
        mock_load_profiles.return_value = {
            actor_user_id: UserTargetingProfile(
                user_id=actor_user_id,
                gender="male",
                age=30,
                gender_target="female",
                age_from=25,
                age_to=35,
            ),
            "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb": UserTargetingProfile(
                user_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                gender="female",
                age=25,
                gender_target="male",
                age_from=30,
                age_to=35,
            ),
            "cccccccc-cccc-cccc-cccc-cccccccccccc": UserTargetingProfile(
                user_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
                gender="female",
                age=35,
                gender_target="male",
                age_from=18,
                age_to=29,
            ),
            "dddddddd-dddd-dddd-dddd-dddddddddddd": UserTargetingProfile(
                user_id="dddddddd-dddd-dddd-dddd-dddddddddddd",
                gender="male",
                age=27,
                gender_target="female",
                age_from=20,
                age_to=40,
            ),
        }

        filtered_df = filter_candidate_frame_by_targeting(
            candidate_df,
            actor_user_id=actor_user_id,
        )

        self.assertEqual(
            filtered_df["candidate_id"].tolist(),
            ["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"],
        )


class RecommendationServingTests(unittest.TestCase):
    def test_ensure_candidate_ids_preserves_existing_index_alignment(self) -> None:
        candidate_df = pd.DataFrame(
            {
                "party_rk": [
                    "11111111-1111-1111-1111-111111111111",
                    "22222222-2222-2222-2222-222222222222",
                ]
            },
            index=[5, 7],
        )

        result = _ensure_candidate_ids(candidate_df)

        self.assertEqual(
            result.loc[5, "candidate_id"],
            "11111111-1111-1111-1111-111111111111",
        )
        self.assertEqual(
            result.loc[7, "candidate_id"],
            "22222222-2222-2222-2222-222222222222",
        )

    @patch("ml.services.recommendation_serving._annotate_shown_history", side_effect=lambda df, actor_user_id: df)
    @patch("ml.services.recommendation_serving._prepare_candidate_reasons", side_effect=lambda df: df)
    @patch(
        "ml.services.recommendation_serving._apply_hard_filters",
        side_effect=lambda df, actor_user_id: df,
    )
    def test_return_top_n_prefers_fresh_then_recycles_shown(
        self,
        mock_apply_filters,
        mock_prepare_reasons,
        mock_annotate_shown_history,
    ) -> None:
        candidate_df = pd.DataFrame(
            {
                "candidate_id": [
                    "11111111-1111-1111-1111-111111111111",
                    "22222222-2222-2222-2222-222222222222",
                    "33333333-3333-3333-3333-333333333333",
                ],
                "reasons": [
                    {"A": 0.9, "B": 0.8, "C": 0.7},
                    {"D": 0.9, "E": 0.8, "F": 0.7},
                    {"G": 0.9, "H": 0.8, "I": 0.7},
                ],
                "already_shown": [True, False, True],
            }
        )

        result = _return_top_n(
            candidate_df,
            actor_user_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            top_n=3,
        )

        self.assertEqual(
            result["candidate_id"].tolist(),
            [
                "22222222-2222-2222-2222-222222222222",
                "11111111-1111-1111-1111-111111111111",
                "33333333-3333-3333-3333-333333333333",
            ],
        )
        self.assertEqual(result["already_shown"].tolist(), [False, True, True])
        mock_apply_filters.assert_called()
        mock_prepare_reasons.assert_called()
        mock_annotate_shown_history.assert_called()


if __name__ == "__main__":
    unittest.main()
