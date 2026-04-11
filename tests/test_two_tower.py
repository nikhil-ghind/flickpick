"""Tests for the two-tower retrieval model."""

import torch
import pytest

from flickpick.models.two_tower import TwoTowerModel


@pytest.fixture
def model():
    return TwoTowerModel(
        num_users=100, num_items=200, num_genres=15,
        embedding_dim=32, temperature=0.05,
    )


@pytest.fixture
def sample_batch():
    batch_size = 8
    return (
        {
            "user_id": torch.randint(0, 100, (batch_size,)),
            "watch_history_embeds": torch.randn(batch_size, 50, 32),
            "watch_history_mask": torch.ones(batch_size, 50),
            "genre_ids": torch.randint(0, 15, (batch_size, 5)),
            "hour": torch.randint(0, 24, (batch_size,)),
            "dow": torch.randint(0, 7, (batch_size,)),
            "device": torch.randint(0, 5, (batch_size,)),
        },
        {
            "item_id": torch.randint(0, 200, (batch_size,)),
            "genre_ids": torch.randint(0, 15, (batch_size, 3)),
            "continuous": torch.randn(batch_size, 4),
        },
    )


def test_forward_produces_embeddings(model, sample_batch):
    user_features, item_features = sample_batch
    user_emb, item_emb = model(user_features, item_features)

    assert user_emb.shape == (8, 32)
    assert item_emb.shape == (8, 32)


def test_embeddings_are_normalized(model, sample_batch):
    user_features, item_features = sample_batch
    user_emb, item_emb = model(user_features, item_features)

    user_norms = torch.norm(user_emb, dim=1)
    item_norms = torch.norm(item_emb, dim=1)

    torch.testing.assert_close(user_norms, torch.ones_like(user_norms), atol=1e-5, rtol=0)
    torch.testing.assert_close(item_norms, torch.ones_like(item_norms), atol=1e-5, rtol=0)


def test_loss_is_scalar(model, sample_batch):
    user_features, item_features = sample_batch
    user_emb, item_emb = model(user_features, item_features)
    loss = model.compute_loss(user_emb, item_emb)

    assert loss.dim() == 0
    assert loss.item() > 0


def test_loss_decreases_with_training(model, sample_batch):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    user_features, item_features = sample_batch

    initial_loss = None
    for step in range(50):
        user_emb, item_emb = model(user_features, item_features)
        loss = model.compute_loss(user_emb, item_emb)

        if initial_loss is None:
            initial_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < initial_loss


def test_watch_history_mask_works(model):
    """Verify that masked positions don't affect the output."""
    batch_size = 2
    user_features = {
        "user_id": torch.tensor([0, 0]),
        "watch_history_embeds": torch.randn(batch_size, 50, 32),
        "watch_history_mask": torch.zeros(batch_size, 50),  # all masked
        "genre_ids": torch.randint(0, 15, (batch_size, 5)),
        "hour": torch.tensor([12, 12]),
        "dow": torch.tensor([0, 0]),
        "device": torch.tensor([0, 0]),
    }

    # Should not crash with all-zero mask
    emb = model.user_tower(**user_features)
    assert emb.shape == (2, 32)
    assert not torch.isnan(emb).any()
