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