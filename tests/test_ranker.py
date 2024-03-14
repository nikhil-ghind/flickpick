"""Tests for the LightGBM ranking model."""

import numpy as np
import pytest

from flickpick.models.ranker import RankingModel


@pytest.fixture
def training_data():
    """Generate synthetic training data for the ranker."""
    rng = np.random.default_rng(42)
    n = 5000
    n_features = 15

    X = rng.random((n, n_features)).astype(np.float32)
    # Label: item is engaging if retrieval_score (col 0) + genre_match (col 12) are high
    y = ((X[:, 0] + X[:, 12]) > 1.0).astype(np.float32)

    split = int(n * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


def test_train_and_predict(training_data):
    X_train, y_train, X_val, y_val = training_data
    model = RankingModel()
    metrics = model.train(X_train, y_train, X_val, y_val)

    assert "best_val_auc" in metrics
    assert metrics["best_val_auc"] > 0.5  # better than random

    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)
    assert all(0 <= p <= 1 for p in preds)


def test_rank_returns_ordered_results(training_data):
    X_train, y_train, X_val, y_val = training_data
    model = RankingModel()
    model.train(X_train, y_train, X_val, y_val)

    item_ids = [f"item_{i}" for i in range(len(X_val))]
    ranked = model.rank(X_val, item_ids, top_k=10)

    assert len(ranked) == 10
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_incremental_train(training_data):
    X_train, y_train, X_val, y_val = training_data
    model = RankingModel()
    model.train(X_train, y_train, X_val, y_val)

    initial_trees = model.model.num_trees()

    # Incremental update
    result = model.incremental_train(X_val, y_val, num_boost_round=10)
    assert result["total_trees"] > initial_trees


def test_save_and_load(training_data, tmp_path):
    X_train, y_train, X_val, y_val = training_data
    model = RankingModel()
    model.train(X_train, y_train, X_val, y_val)

    path = str(tmp_path / "ranker.joblib")
    model.save(path)

    loaded = RankingModel(model_path=path)
    original_preds = model.predict(X_val)
    loaded_preds = loaded.predict(X_val)

    np.testing.assert_array_almost_equal(original_preds, loaded_preds)
