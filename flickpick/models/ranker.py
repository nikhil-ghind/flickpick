"""
LightGBM ranking model that scores candidates retrieved by the two-tower model.

Takes candidate items + user context features and predicts engagement probability
(P(watch > 70% of content)). Trained on historical interaction data with
binary labels derived from watch-time thresholds.
"""

import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from dataclasses import dataclass


@dataclass
class RankerFeatures:
    """Features used by the ranking model.

    Retrieval features (from two-tower):
        - retrieval_score: cosine similarity from two-tower model

    User features (from feature store):
        - user_avg_watch_pct: average watch percentage across all content
        - user_total_watches: total items watched
        - user_genre_affinity: affinity score for this item's primary genre
        - user_recency_days: days since last watch
        - user_session_depth: number of items viewed in current session

    Item features (from feature store):
        - item_popularity_7d: view count in last 7 days (log-transformed)
        - item_popularity_28d: view count in last 28 days (log-transformed)
        - item_avg_watch_pct: average watch percentage across all users
        - item_avg_rating: average user rating
        - item_release_recency: days since release (log-transformed)
        - item_duration_minutes: content duration

    Cross features:
        - genre_match_score: overlap between user genre prefs and item genres
        - collaborative_score: user-item score from collaborative filtering
        - time_relevance: how well item matches user's typical watch time
    """

    feature_names: list[str] = None

    def __post_init__(self):
        self.feature_names = [
            "retrieval_score",
            "user_avg_watch_pct",
            "user_total_watches",
            "user_genre_affinity",
            "user_recency_days",
            "user_session_depth",
            "item_popularity_7d",
            "item_popularity_28d",
            "item_avg_watch_pct",
            "item_avg_rating",
            "item_release_recency",
            "item_duration_minutes",
            "genre_match_score",
            "collaborative_score",
            "time_relevance",
        ]


class RankingModel:
    """LightGBM-based ranking model for candidate scoring."""

    def __init__(self, model_path: str | None = None):
        self.feature_spec = RankerFeatures()
        self.model: lgb.Booster | None = None
        if model_path:
            self.load(model_path)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: dict | None = None,
    ) -> dict:
        """Train the ranking model.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Binary labels (1 = engaged, 0 = bounced)
            X_val: Validation features
            y_val: Validation labels
            params: LightGBM parameters override

        Returns:
            Dictionary of evaluation metrics
        """
        default_params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 7,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbose": -1,
        }
        if params:
            default_params.update(params)

        train_data = lgb.Dataset(
            X_train, label=y_train,
            feature_name=self.feature_spec.feature_names,
        )
        val_data = lgb.Dataset(
            X_val, label=y_val,
            feature_name=self.feature_spec.feature_names,
            reference=train_data,
        )

        callbacks = [
            lgb.early_stopping(50),
            lgb.log_evaluation(100),
        ]

        self.model = lgb.train(
            default_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        return {
            "best_iteration": self.model.best_iteration,
            "best_val_auc": self.model.best_score["val"]["auc"],
            "best_val_logloss": self.model.best_score["val"]["binary_logloss"],
            "feature_importance": dict(
                zip(
                    self.feature_spec.feature_names,
                    self.model.feature_importance(importance_type="gain").tolist(),
                )
            ),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Score candidates. Returns P(engagement) for each row."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def rank(self, X: np.ndarray, item_ids: list[str], top_k: int = 20) -> list[tuple[str, float]]:
        """Score and rank candidates, returning top-k (item_id, score) pairs."""
        scores = self.predict(X)
        ranked_indices = np.argsort(-scores)[:top_k]
        return [(item_ids[i], float(scores[i])) for i in ranked_indices]

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)

    def incremental_train(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        num_boost_round: int = 50,
    ) -> dict:
        """Incrementally update the model with new interaction data.

        Uses LightGBM's init_model to continue training from the current model
        on fresh data, enabling online learning without full retraining.
        """
        if self.model is None:
            raise RuntimeError("No base model to update")

        new_data = lgb.Dataset(
            X_new, label=y_new,
            feature_name=self.feature_spec.feature_names,
        )

        params = self.model.params.copy()
        params["learning_rate"] = 0.01  # lower LR for incremental updates

        self.model = lgb.train(
            params,
            new_data,
            num_boost_round=num_boost_round,
            init_model=self.model,
        )

        return {"new_trees": num_boost_round, "total_trees": self.model.num_trees()}
