"""
FAISS-based approximate nearest neighbor index for candidate retrieval.

Precomputes item tower embeddings and indexes them. At serving time,
the user tower embedding is used to retrieve top-k candidate items.
"""

import logging
from pathlib import Path

import faiss
import numpy as np
import torch

from flickpick.models.two_tower import TwoTowerModel

logger = logging.getLogger(__name__)


class FAISSIndex:
    def __init__(self, embedding_dim: int = 64, use_gpu: bool = False):
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.index: faiss.Index | None = None
        self.item_ids: list[str] = []
        self.item_embeddings: np.ndarray | None = None

    def build_from_model(
        self,
        model: TwoTowerModel,
        item_features: list[dict],
        item_ids: list[str],
        batch_size: int = 1024,
    ):
        """Build the FAISS index from item tower embeddings.

        Args:
            model: Trained two-tower model
            item_features: List of item feature dicts for the item tower
            item_ids: Corresponding item IDs
            batch_size: Batch size for encoding
        """
        model.eval()
        device = next(model.parameters()).device

        all_embeddings = []
        for i in range(0, len(item_features), batch_size):
            batch = item_features[i : i + batch_size]
            batch_tensors = self._collate_item_features(batch, device)

            with torch.no_grad():
                emb = model.item_tower(**batch_tensors)
            all_embeddings.append(emb.cpu().numpy())

        self.item_embeddings = np.vstack(all_embeddings).astype(np.float32)
        self.item_ids = item_ids

        self._build_index()
        logger.info(f"Built FAISS index with {len(self.item_ids)} items")

    def build_from_embeddings(self, embeddings: np.ndarray, item_ids: list[str]):
        """Build index from pre-computed embeddings."""
        self.item_embeddings = embeddings.astype(np.float32)
        self.item_ids = item_ids
        self._build_index()

    def _build_index(self):
        """Build the FAISS index structure.

        Uses IVF (inverted file) with PQ (product quantization) for
        scalable ANN search. Falls back to flat index for small catalogs.
        """
        n = len(self.item_ids)
        d = self.embedding_dim

        if n < 10_000:
            # Small catalog: exact search is fast enough
            self.index = faiss.IndexFlatIP(d)
        else:
            # Large catalog: IVF + PQ for approximate search
            nlist = min(int(np.sqrt(n)), 256)  # number of clusters
            m = 8  # number of sub-quantizers
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
            self.index.train(self.item_embeddings)
            self.index.nprobe = min(nlist // 4, 32)

        self.index.add(self.item_embeddings)

        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)

    def search(self, user_embedding: np.ndarray, top_k: int = 100) -> list[tuple[str, float]]:
        """Retrieve top-k candidate items for a user embedding.

        Args:
            user_embedding: User tower output (1, embedding_dim) or (embedding_dim,)
            top_k: Number of candidates to retrieve

        Returns:
            List of (item_id, similarity_score) tuples
        """
        if self.index is None:
            raise RuntimeError("Index not built")

        query = user_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                results.append((self.item_ids[idx], float(score)))
        return results

    def batch_search(
        self, user_embeddings: np.ndarray, top_k: int = 100
    ) -> list[list[tuple[str, float]]]:
        """Batch retrieve candidates for multiple users."""
        if self.index is None:
            raise RuntimeError("Index not built")

        queries = user_embeddings.astype(np.float32)
        scores, indices = self.index.search(queries, top_k)

        results = []
        for user_scores, user_indices in zip(scores, indices):
            user_results = []
            for score, idx in zip(user_scores, user_indices):
                if idx >= 0:
                    user_results.append((self.item_ids[idx], float(score)))
            results.append(user_results)
        return results

    def save(self, path: str):
        """Save index and metadata to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)

        if self.use_gpu and faiss.get_num_gpus() > 0:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
        else:
            index_cpu = self.index

        faiss.write_index(index_cpu, f"{path}/index.faiss")
        np.save(f"{path}/item_ids.npy", np.array(self.item_ids))
        np.save(f"{path}/item_embeddings.npy", self.item_embeddings)
        logger.info(f"Saved FAISS index to {path}")

    def load(self, path: str):
        """Load index and metadata from disk."""
        self.index = faiss.read_index(f"{path}/index.faiss")
        self.item_ids = np.load(f"{path}/item_ids.npy", allow_pickle=True).tolist()
        self.item_embeddings = np.load(f"{path}/item_embeddings.npy")
        self.embedding_dim = self.item_embeddings.shape[1]

        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)

        logger.info(f"Loaded FAISS index with {len(self.item_ids)} items")

    def get_item_embedding(self, item_id: str) -> np.ndarray | None:
        """Get the stored embedding for an item."""
        try:
            idx = self.item_ids.index(item_id)
            return self.item_embeddings[idx]
        except ValueError:
            return None

    def _collate_item_features(self, batch: list[dict], device: torch.device) -> dict:
        """Collate a list of item feature dicts into batched tensors."""
        return {
            "item_ids": torch.tensor([f["item_id"] for f in batch], device=device),
            "genre_ids": torch.stack([f["genre_ids"] for f in batch]).to(device),
            "continuous_features": torch.stack([f["continuous"] for f in batch]).to(device),
        }
