from dataclasses import dataclass, field

from core.config import GlobalConfig
from core.data_config import InteractionWeightsMapper, Paths
from ml.ml_config import RerankConfig
from ml.services.recommendation_scoring import (
    DEFAULT_ENSEMBLE_WEIGHTS,
    EnsembleWeights,
    RecommendationStage,
)


@dataclass(frozen=True)
class RecommendationRuntimeConfig:
    recommendation_stage: RecommendationStage = RecommendationStage.STAGE_4
    rerank_mode: int = 0
    top_n: int = 150
    ratios: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    an_impactful_features: int = 6
    n_max_candidates: int | None = 300
    n_min_candidates: int | None = 100
    n_rec_users: int = 5
    ensemble_weights: EnsembleWeights = DEFAULT_ENSEMBLE_WEIGHTS
    popularity_action_weights: dict[str, float] = field(
        default_factory=lambda: dict(InteractionWeightsMapper().dict_to_map)
    )
    uncertainty_mean_score_threshold: float = 0.0
    uncertainty_gap_threshold: float = 0.05
    stage_23_model_dir: str = str(Paths.stage_23_model_dir)
    stage_4_model_dir: str = str(Paths.stage_4_model_dir)

    def build_configs(self, user_id: str) -> tuple[RerankConfig, GlobalConfig]:
        rerank_conf = RerankConfig(
            rerank_mode=self.rerank_mode,
            user_id=user_id,
            top_n=self.top_n,
            ratios=list(self.ratios),
            an_impactful_features=self.an_impactful_features,
            recommendation_stage=self.recommendation_stage.value,
            popularity_action_weights=dict(self.popularity_action_weights),
            uncertainty_mean_score_threshold=float(self.uncertainty_mean_score_threshold),
            uncertainty_gap_threshold=float(self.uncertainty_gap_threshold),
        )
        global_conf = GlobalConfig(
            n_max_candidates=self.n_max_candidates,
            n_min_candidates=self.n_min_candidates,
            n_rec_users=self.n_rec_users,
        )
        return rerank_conf, global_conf


DEFAULT_RECOMMENDATION_RUNTIME_CONFIG = RecommendationRuntimeConfig()
