from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class InteractionWeightsMapper:
    dict_to_map: dict = field(default_factory=lambda: {
        "block": -15,
        "dislike": -6,
        "skip": -1,
        "like": 3,
        "chat_started": 4,
        "match": 8
    })


@dataclass(frozen=True)
class Paths:
    project_root: Path = PROJECT_ROOT
    out_data_path: Path = PROJECT_ROOT / "data_out"
    in_data_path: Path = PROJECT_ROOT / "data_in"
    transaction_user_features_parquet_path: Path = PROJECT_ROOT / "data_out" / "user_features_all.parquet"
    transaction_user_features_csv_path: Path = PROJECT_ROOT / "data_out" / "user_features_all.csv"
    merged_user_features_parquet_path: Path = PROJECT_ROOT / "data_out" / "user_features_tx_interaction.parquet"
    tx_interaction_hnsw_index_path: Path = PROJECT_ROOT / "data_out" / "tx_interaction_hnsw.index"
    model_artifacts_dir: Path = PROJECT_ROOT / "data_out" / "models"
    stage_23_model_dir: Path = PROJECT_ROOT / "data_out" / "models" / "stage_2_3"
    stage_4_model_dir: Path = PROJECT_ROOT / "data_out" / "models" / "stage_4"
    stage_23_model_metadata_path: Path = PROJECT_ROOT / "data_out" / "models" / "stage_2_3" / "metadata.json"
    stage_4_model_metadata_path: Path = PROJECT_ROOT / "data_out" / "models" / "stage_4" / "metadata.json"
