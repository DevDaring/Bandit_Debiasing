"""
Pre-computed steering vector paths and configurations.
"""

import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class SteeringVectorPaths:
    """Paths to pre-computed steering vectors."""

    base_dir: str = "data/steering_vectors"

    @property
    def gender_path(self) -> str:
        return os.path.join(self.base_dir, "gender_steering.pt")

    @property
    def race_path(self) -> str:
        return os.path.join(self.base_dir, "race_steering.pt")

    @property
    def religion_path(self) -> str:
        return os.path.join(self.base_dir, "religion_steering.pt")

    def get_path(self, bias_type: str) -> str:
        """Get path for a specific bias type."""
        paths = {"gender": self.gender_path, "race": self.race_path, "religion": self.religion_path}
        return paths.get(bias_type, "")

    def all_paths(self) -> Dict[str, str]:
        """Get all steering vector paths."""
        return {"gender": self.gender_path, "race": self.race_path, "religion": self.religion_path}


# Contrastive pairs data paths
CONTRASTIVE_PAIRS_DIR = "data/contrastive_pairs"
GENDER_PAIRS_PATH = os.path.join(CONTRASTIVE_PAIRS_DIR, "gender_pairs.json")
RACE_PAIRS_PATH = os.path.join(CONTRASTIVE_PAIRS_DIR, "race_pairs.json")
RELIGION_PAIRS_PATH = os.path.join(CONTRASTIVE_PAIRS_DIR, "religion_pairs.json")
