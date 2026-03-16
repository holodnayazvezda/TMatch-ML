from dataclasses import dataclass

from core.config import GlobalConfig
from ml.ml_config import RerankConfig


@dataclass(frozen=True)
class RecommendationRuntimeConfig:
    rerank_mode: int = 0
    top_n: int = 100
    ratios: tuple[float, ...] = (1.0, 0.0, 0.0)
    an_impactful_features: int = 3
    n_max_candidates: int | None = 300
    n_min_candidates: int | None = 100
    n_rec_users: int = 5

    def build_configs(self, user_id: str) -> tuple[RerankConfig, GlobalConfig]:
        rerank_conf = RerankConfig(
            rerank_mode=self.rerank_mode,
            user_id=user_id,
            top_n=self.top_n,
            ratios=list(self.ratios),
            an_impactful_features=self.an_impactful_features,
        )
        global_conf = GlobalConfig(
            n_max_candidates=self.n_max_candidates,
            n_min_candidates=self.n_min_candidates,
            n_rec_users=self.n_rec_users,
        )
        return rerank_conf, global_conf


DEFAULT_RECOMMENDATION_RUNTIME_CONFIG = RecommendationRuntimeConfig()
