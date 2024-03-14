"""
Two-Tower retrieval model for candidate generation.

User tower encodes user features + watch history into an embedding.
Item tower encodes item metadata into an embedding.
Trained with sampled softmax / in-batch negatives on implicit feedback.
At serving time, item embeddings are indexed in FAISS for ANN retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """Encodes user features into a dense embedding vector."""

    def __init__(
        self,
        num_users: int,
        num_genres: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        history_len: int = 50,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim // 4)

        # Watch history encoder: average pooling over recent item embeddings
        self.history_len = history_len
        self.history_proj = nn.Linear(embedding_dim, embedding_dim)

        # Context features: hour_of_day (24), day_of_week (7), device_type (5)
        self.hour_embedding = nn.Embedding(24, 8)
        self.dow_embedding = nn.Embedding(7, 4)
        self.device_embedding = nn.Embedding(5, 4)

        context_dim = 8 + 4 + 4
        genre_input = embedding_dim // 4  # multi-hot genre avg

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim + genre_input + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        user_ids: torch.Tensor,
        watch_history_embeds: torch.Tensor,
        watch_history_mask: torch.Tensor,
        genre_ids: torch.Tensor,
        hour: torch.Tensor,
        dow: torch.Tensor,
        device: torch.Tensor,
    ) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)

        # Average pooling over watch history with mask
        history_emb = self.history_proj(watch_history_embeds)
        mask = watch_history_mask.unsqueeze(-1).float()
        history_emb = (history_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Genre preferences (multi-hot average)
        genre_emb = self.genre_embedding(genre_ids).mean(dim=1)

        # Context
        ctx = torch.cat([
            self.hour_embedding(hour),
            self.dow_embedding(dow),
            self.device_embedding(device),
        ], dim=-1)

        x = torch.cat([user_emb, history_emb, genre_emb, ctx], dim=-1)
        x = self.mlp(x)
        x = self.layer_norm(x)
        return F.normalize(x, p=2, dim=-1)


class ItemTower(nn.Module):
    """Encodes item metadata into a dense embedding vector."""

    def __init__(
        self,
        num_items: int,
        num_genres: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim // 4)

        # Continuous features: duration_minutes, release_year, avg_rating, popularity_score
        self.continuous_proj = nn.Linear(4, 16)

        genre_input = embedding_dim // 4
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + genre_input + 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        item_ids: torch.Tensor,
        genre_ids: torch.Tensor,
        continuous_features: torch.Tensor,
    ) -> torch.Tensor:
        item_emb = self.item_embedding(item_ids)
        genre_emb = self.genre_embedding(genre_ids).mean(dim=1)
        cont = self.continuous_proj(continuous_features)

        x = torch.cat([item_emb, genre_emb, cont], dim=-1)
        x = self.mlp(x)
        x = self.layer_norm(x)
        return F.normalize(x, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Two-tower retrieval model trained with in-batch negatives.

    Loss: sampled softmax — for each (user, positive_item) pair in the batch,
    all other items in the batch serve as negatives.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_genres: int,
        embedding_dim: int = 64,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.user_tower = UserTower(num_users, num_genres, embedding_dim)
        self.item_tower = ItemTower(num_items, num_genres, embedding_dim)
        self.temperature = temperature

    def forward(self, user_features: dict, item_features: dict) -> tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_tower(
            user_ids=user_features["user_id"],
            watch_history_embeds=user_features["watch_history_embeds"],
            watch_history_mask=user_features["watch_history_mask"],
            genre_ids=user_features["genre_ids"],
            hour=user_features["hour"],
            dow=user_features["dow"],
            device=user_features["device"],
        )
        item_emb = self.item_tower(
            item_ids=item_features["item_id"],
            genre_ids=item_features["genre_ids"],
            continuous_features=item_features["continuous"],
        )
        return user_emb, item_emb

    def compute_loss(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """In-batch sampled softmax loss."""
        # Similarity matrix: (batch_size, batch_size)
        logits = torch.matmul(user_emb, item_emb.T) / self.temperature

        # Labels: diagonal (each user's positive item is at its own index)
        labels = torch.arange(logits.size(0), device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss
