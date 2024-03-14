"""
Model registry and version management.

Tracks model versions, their artifacts, and performance metrics.
Supports loading different model versions for A/B experiments and canary rollouts.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from flickpick.models.ranker import RankingModel
from flickpick.models.two_tower import TwoTowerModel
from flickpick.serving.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    version_id: str
    model_type: str  # "two_tower" or "ranker"
    artifact_path: str
    metrics: dict = field(default_factory=dict)
    is_active: bool = False


class ModelRegistry:
    """Manages model versions and loading."""

    def __init__(self, base_path: str = "models/artifacts"):
        self.base_path = Path(base_path)
        self.versions: dict[str, ModelVersion] = {}
        self._loaded_two_towers: dict[str, TwoTowerModel] = {}
        self._loaded_rankers: dict[str, RankingModel] = {}
        self._loaded_indices: dict[str, FAISSIndex] = {}

    def register(self, version: ModelVersion):
        """Register a new model version."""
        self.versions[version.version_id] = version
        logger.info(f"Registered model version: {version.version_id} ({version.model_type})")

    def get_active_version(self, model_type: str) -> ModelVersion | None:
        """Get the currently active version of a model type."""
        for v in self.versions.values():
            if v.model_type == model_type and v.is_active:
                return v
        return None

    def activate(self, version_id: str):
        """Activate a model version (deactivates the previous active version of the same type)."""
        version = self.versions.get(version_id)
        if not version:
            raise ValueError(f"Unknown version: {version_id}")

        # Deactivate previous
        for v in self.versions.values():
            if v.model_type == version.model_type and v.is_active:
                v.is_active = False

        version.is_active = True
        logger.info(f"Activated model version: {version_id}")

    def load_two_tower(
        self,
        version_id: str,
        num_users: int,
        num_items: int,
        num_genres: int,
        embedding_dim: int = 64,
    ) -> TwoTowerModel:
        """Load a two-tower model version."""
        if version_id in self._loaded_two_towers:
            return self._loaded_two_towers[version_id]

        version = self.versions.get(version_id)
        if not version:
            raise ValueError(f"Unknown version: {version_id}")

        model = TwoTowerModel(num_users, num_items, num_genres, embedding_dim)
        checkpoint = torch.load(version.artifact_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self._loaded_two_towers[version_id] = model
        logger.info(f"Loaded two-tower model: {version_id}")
        return model

    def load_ranker(self, version_id: str) -> RankingModel:
        """Load a ranker model version."""
        if version_id in self._loaded_rankers:
            return self._loaded_rankers[version_id]

        version = self.versions.get(version_id)
        if not version:
            raise ValueError(f"Unknown version: {version_id}")

        ranker = RankingModel(model_path=version.artifact_path)
        self._loaded_rankers[version_id] = ranker
        logger.info(f"Loaded ranker model: {version_id}")
        return ranker

    def load_faiss_index(self, version_id: str, embedding_dim: int = 64) -> FAISSIndex:
        """Load the FAISS index for a two-tower model version."""
        if version_id in self._loaded_indices:
            return self._loaded_indices[version_id]

        version = self.versions.get(version_id)
        if not version:
            raise ValueError(f"Unknown version: {version_id}")

        index_path = str(Path(version.artifact_path).parent / "faiss_index")
        index = FAISSIndex(embedding_dim=embedding_dim)
        index.load(index_path)

        self._loaded_indices[version_id] = index
        logger.info(f"Loaded FAISS index: {version_id}")
        return index

    def list_versions(self, model_type: str | None = None) -> list[ModelVersion]:
        """List all registered versions, optionally filtered by type."""
        versions = list(self.versions.values())
        if model_type:
            versions = [v for v in versions if v.model_type == model_type]
        return sorted(versions, key=lambda v: v.version_id, reverse=True)
