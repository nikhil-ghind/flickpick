"""
End-to-end recommendation serving pipeline.

Flow:
  1. Fetch user features from feature store
  2. Encode user → embedding via user tower
  3. FAISS ANN search → top-k candidates
  4. Fetch candidate item features from feature store
  5. Assemble ranker feature vectors
  6. Score with LightGBM ranker
  7. Apply business rules (diversity, dedup)
  8. Return ranked recommendations
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from flickpick.features.feature_store import FeatureStore
from flickpick.models.ranker import RankingModel
from flickpick.models.two_tower import TwoTowerModel
from flickpick.serving.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


@dataclass
class RecommendationRequest:
    user_id: str
    num_results: int = 20
    num_candidates: int = 200
    exclude_item_ids: list[str] | None = None
    context: dict | None = None  # hour, dow, device


@dataclass
class RecommendationResult:
    items: list[dict]  # [{item_id, score, retrieval_score}]
    latency_ms: float
    candidates_retrieved: int
    model_version: str


class RecommendationPipeline:
    def __init__(
        self,
        two_tower: TwoTowerModel,
        ranker: RankingModel,
        faiss_index: FAISSIndex,
        feature_store: FeatureStore,
        model_version: str = "v1",
    ):
        self.two_tower = two_tower
        self.ranker = ranker
        self.faiss_index = faiss_index
        self.feature_store = feature_store
        self.model_version = model_version

        self.two_tower.eval()
        self.device = next(two_tower.parameters()).device

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        """Generate recommendations for a user."""
        start = time.monotonic()

        # 1. Fetch user features
        user_features = self.feature_store.get_user_features(request.user_id)
        watch_history = self.feature_store.get_watch_history(request.user_id)

        # 2. Encode user via user tower
        user_embedding = self._encode_user(user_features, watch_history, request.context)

        # 3. FAISS retrieval
        candidates = self.faiss_index.search(user_embedding, top_k=request.num_candidates)

        # Filter excluded items
        if request.exclude_item_ids:
            exclude_set = set(request.exclude_item_ids)
            candidates = [(iid, s) for iid, s in candidates if iid not in exclude_set]

        # 4. Fetch item features for candidates
        candidate_ids = [iid for iid, _ in candidates]
        retrieval_scores = {iid: s for iid, s in candidates}
        item_features = self.feature_store.get_batch_item_features(candidate_ids)

        # 5. Assemble ranker features
        ranker_input = self._build_ranker_features(
            user_features, item_features, retrieval_scores, candidate_ids,
        )

        # 6. Score with ranker
        ranked = self.ranker.rank(ranker_input, candidate_ids, top_k=request.num_results * 2)

        # 7. Apply diversity rules
        diversified = self._diversify(ranked, item_features, request.num_results)

        latency_ms = (time.monotonic() - start) * 1000

        return RecommendationResult(
            items=[
                {
                    "item_id": iid,
                    "score": score,
                    "retrieval_score": retrieval_scores.get(iid, 0.0),
                }
                for iid, score in diversified
            ],
            latency_ms=latency_ms,
            candidates_retrieved=len(candidates),
            model_version=self.model_version,
        )

    def _encode_user(
        self,
        user_features: dict,
        watch_history: list[str],
        context: dict | None,
    ) -> np.ndarray:
        """Encode user into embedding via user tower."""
        # Build watch history embeddings from FAISS index's stored embeddings
        history_embs = []
        for item_id in watch_history[:50]:
            emb = self.faiss_index.get_item_embedding(item_id)
            if emb is not None:
                history_embs.append(emb)

        if history_embs:
            history_tensor = torch.tensor(np.stack(history_embs), device=self.device).unsqueeze(0)
            history_mask = torch.ones(1, len(history_embs), device=self.device)
            # Pad to fixed length
            pad_len = 50 - len(history_embs)
            if pad_len > 0:
                history_tensor = torch.cat([
                    history_tensor,
                    torch.zeros(1, pad_len, history_tensor.shape[-1], device=self.device),
                ], dim=1)
                history_mask = torch.cat([
                    history_mask,
                    torch.zeros(1, pad_len, device=self.device),
                ], dim=1)
        else:
            history_tensor = torch.zeros(1, 50, 64, device=self.device)
            history_mask = torch.zeros(1, 50, device=self.device)

        ctx = context or {}
        genre_ids = user_features.get("genre_affinity", {})
        top_genres = sorted(genre_ids.keys(), key=lambda g: genre_ids[g], reverse=True)[:5]
        genre_tensor = torch.zeros(1, 5, dtype=torch.long, device=self.device)
        for i, g in enumerate(top_genres):
            genre_tensor[0, i] = int(g)

        with torch.no_grad():
            user_emb = self.two_tower.user_tower(
                user_ids=torch.tensor([int(hash(user_features.get("user_id", "0")) % 100000)], device=self.device),
                watch_history_embeds=history_tensor,
                watch_history_mask=history_mask,
                genre_ids=genre_tensor,
                hour=torch.tensor([ctx.get("hour", 12)], device=self.device),
                dow=torch.tensor([ctx.get("dow", 0)], device=self.device),
                device=torch.tensor([ctx.get("device", 0)], device=self.device),
            )

        return user_emb.cpu().numpy().flatten()

    def _build_ranker_features(
        self,
        user_features: dict,
        item_features: dict[str, dict],
        retrieval_scores: dict[str, float],
        candidate_ids: list[str],
    ) -> np.ndarray:
        """Build the feature matrix for the ranker model."""
        rows = []
        user_genre_affinity = user_features.get("genre_affinity", {})

        for item_id in candidate_ids:
            item = item_features.get(item_id, {})
            item_genres = item.get("genre_ids", [])

            # Genre match score
            genre_match = 0.0
            if item_genres and user_genre_affinity:
                genre_match = sum(
                    user_genre_affinity.get(str(g), 0) for g in item_genres
                ) / max(len(item_genres), 1)

            row = [
                retrieval_scores.get(item_id, 0.0),
                user_features.get("avg_watch_pct", 0.0),
                user_features.get("total_watches", 0),
                genre_match,  # user_genre_affinity for this item
                user_features.get("recency_days", 999),
                user_features.get("session_depth", 0),
                item.get("popularity_7d", 0.0),
                item.get("popularity_28d", 0.0),
                item.get("avg_watch_pct", 0.0),
                item.get("avg_rating", 0.0),
                item.get("release_recency", 0.0),
                item.get("duration_minutes", 0.0),
                genre_match,
                retrieval_scores.get(item_id, 0.0),  # collaborative_score proxy
                0.0,  # time_relevance placeholder
            ]
            rows.append(row)

        return np.array(rows, dtype=np.float32)

    def _diversify(
        self,
        ranked: list[tuple[str, float]],
        item_features: dict[str, dict],
        num_results: int,
    ) -> list[tuple[str, float]]:
        """Apply maximal marginal relevance (MMR) for genre diversity.

        Ensures the final list doesn't have too many items from the same genre.
        """
        if not ranked:
            return []

        selected = []
        genre_counts: dict[int, int] = {}
        max_per_genre = max(num_results // 3, 2)

        for item_id, score in ranked:
            if len(selected) >= num_results:
                break

            item = item_features.get(item_id, {})
            genres = item.get("genre_ids", [])
            primary_genre = genres[0] if genres else -1

            if genre_counts.get(primary_genre, 0) >= max_per_genre:
                continue

            selected.append((item_id, score))
            genre_counts[primary_genre] = genre_counts.get(primary_genre, 0) + 1

        return selected
