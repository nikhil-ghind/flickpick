"""
Batch feature computation jobs.

Run periodically (e.g., daily) to compute and push features to the feature store.
These are features that are expensive to compute in real-time but change slowly.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from flickpick.features.feature_store import FeatureStore

logger = logging.getLogger(__name__)


class BatchFeatureComputer:
    def __init__(self, db_url: str, feature_store: FeatureStore):
        self.engine = create_engine(db_url)
        self.store = feature_store

    def compute_user_features(self):
        """Compute and store batch user features."""
        logger.info("Computing batch user features...")

        query = text("""
            SELECT
                u.user_id,
                COUNT(w.item_id) AS total_watches,
                AVG(w.watch_pct) AS avg_watch_pct,
                EXTRACT(EPOCH FROM (NOW() - MAX(w.watched_at))) / 86400 AS recency_days,
                ARRAY_AGG(DISTINCT g.genre_id) AS genre_ids
            FROM users u
            LEFT JOIN watch_events w ON u.user_id = w.user_id
                AND w.watched_at > NOW() - INTERVAL '90 days'
            LEFT JOIN item_genres g ON w.item_id = g.item_id
            GROUP BY u.user_id
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        for _, row in df.iterrows():
            user_id = str(row["user_id"])

            # Compute genre affinity vector
            genre_ids = row["genre_ids"] if row["genre_ids"] else []
            genre_counts = {}
            for gid in genre_ids:
                if gid is not None:
                    genre_counts[gid] = genre_counts.get(gid, 0) + 1
            total = max(sum(genre_counts.values()), 1)
            genre_affinity = {str(k): v / total for k, v in genre_counts.items()}

            features = {
                "total_watches": int(row["total_watches"]),
                "avg_watch_pct": float(row["avg_watch_pct"] or 0),
                "recency_days": float(row["recency_days"] or 999),
                "genre_affinity": genre_affinity,
            }
            self.store.set_user_batch_features(user_id, features)

        logger.info(f"Computed features for {len(df)} users")

    def compute_item_features(self):
        """Compute and store batch item features."""
        logger.info("Computing batch item features...")

        query = text("""
            SELECT
                i.item_id,
                i.title,
                i.duration_minutes,
                i.release_date,
                i.avg_rating,
                COUNT(CASE WHEN w.watched_at > NOW() - INTERVAL '7 days' THEN 1 END) AS views_7d,
                COUNT(CASE WHEN w.watched_at > NOW() - INTERVAL '28 days' THEN 1 END) AS views_28d,
                AVG(w.watch_pct) AS avg_watch_pct,
                ARRAY_AGG(DISTINCT g.genre_id) AS genre_ids
            FROM items i
            LEFT JOIN watch_events w ON i.item_id = w.item_id
            LEFT JOIN item_genres g ON i.item_id = g.item_id
            GROUP BY i.item_id, i.title, i.duration_minutes, i.release_date, i.avg_rating
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        for _, row in df.iterrows():
            item_id = str(row["item_id"])

            release_recency = 0.0
            if row["release_date"]:
                delta = datetime.now() - pd.Timestamp(row["release_date"]).to_pydatetime()
                release_recency = max(delta.days, 0)

            features = {
                "duration_minutes": float(row["duration_minutes"] or 0),
                "avg_rating": float(row["avg_rating"] or 0),
                "popularity_7d": np.log1p(float(row["views_7d"] or 0)),
                "popularity_28d": np.log1p(float(row["views_28d"] or 0)),
                "avg_watch_pct": float(row["avg_watch_pct"] or 0),
                "release_recency": np.log1p(release_recency),
                "genre_ids": [int(g) for g in (row["genre_ids"] or []) if g is not None],
            }
            self.store.set_item_batch_features(item_id, features)

        logger.info(f"Computed features for {len(df)} items")

    def compute_user_embeddings(self, model, item_embeddings: dict[str, np.ndarray]):
        """Compute and store user watch history embeddings for two-tower input.

        Averages the item embeddings of the user's recent watch history.
        """
        logger.info("Computing user history embeddings...")

        query = text("""
            SELECT user_id, ARRAY_AGG(item_id ORDER BY watched_at DESC) AS history
            FROM (
                SELECT user_id, item_id, watched_at,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY watched_at DESC) AS rn
                FROM watch_events
            ) ranked
            WHERE rn <= 50
            GROUP BY user_id
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)

        for _, row in df.iterrows():
            user_id = str(row["user_id"])
            history = row["history"] or []

            # Average item embeddings for watch history
            embs = [item_embeddings[str(iid)] for iid in history if str(iid) in item_embeddings]
            if embs:
                history_emb = np.mean(embs, axis=0).astype(np.float32)
            else:
                history_emb = np.zeros(64, dtype=np.float32)

            self.store.set_user_batch_features(user_id, {"history_embedding": history_emb})

        logger.info(f"Computed embeddings for {len(df)} users")

    def run_all(self):
        """Run all batch feature computations."""
        self.compute_user_features()
        self.compute_item_features()
        logger.info("Batch feature computation complete")
