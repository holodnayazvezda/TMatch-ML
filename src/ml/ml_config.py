from dataclasses import dataclass, field
from pathlib import Path



@dataclass
class RerankConfig:
    rerank_mode: int = 3
    user_id: str | None = None
    top_n: int = 100
    ratios: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    an_impactful_features: int = 3
    retrieval_candidates_path: str | Path | None = None
    rerank_mode_1_model_path: str | Path | None = None
    rerank_mode_2_model_path: str | Path | None = None
    rerank_mode_3_model_path: str | Path | None = None
    rerank_model_paths: dict[int, str | Path] | None = None
