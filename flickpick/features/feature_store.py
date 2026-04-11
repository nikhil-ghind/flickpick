"""
Redis-backed online feature store.

Two types of features:
- Batch features: computed offline (e.g., user watch history embeddings, item popularity decay).
  Written by batch jobs, read at serving time.
- Real-time features: computed on the fly from streaming events (e.g., session depth, recency).
  Updated via Kafka consumer, read at serving time.

Key schema:
  user:{user_id}:batch   → hash of batch-computed user features
  user:{user_id}:rt      → hash of real-time user features
  item:{item_id}:batch   → hash of batch-computed item features
  item:{item_id}:stats   → hash of real-time item statistics
"""

import json
import logging
from typing import Any

import numpy as np
import redis

logger = logging.getLogger(__name__)


class FeatureStore:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url, decode_responses=True)

    # --- Write methods (called by batch jobs and stream consumers) ---

    def set_user_batch_features(self, user_id: str, features: dict[str, Any], ttl: int = 86400):
        """Write batch-computed user features. Called by daily batch job."""
        key = f"user:{user_id}:batch"
        serialized = {k: self._serialize(v) for k, v in features.items()}
        pipe = self.redis.pipeline()
        pipe.hset(key, mapping=serialized)
        pipe.expire(key, ttl)
        pipe.execute()

    def set_item_batch_features(self, item_id: str, features: dict[str, Any], ttl: int = 86400):
        """Write batch-computed item features."""
        key = f"item:{item_id}:batch"
        serialized = {k: self._serialize(v) for k, v in features.items()}
        pipe = self.redis.pipeline()
        pipe.hset(key, mapping=serialized)
        pipe.expire(key, ttl)
        pipe.execute()

    def update_user_realtime(self, user_id: str, features: dict[str, Any]):
        """Update real-time user features from streaming events."""
        key = f"user:{user_id}:rt"
        serialized = {k: self._serialize(v) for k, v in features.items()}
        self.redis.hset(key, mapping=serialized)
        self.redis.expire(key, 3600)  # 1h TTL for session features

    def increment_item_view_count(self, item_id: str, window: str = "7d"):
        """Increment item view count for popularity tracking."""
        key = f"item:{item_id}:stats"
        self.redis.hincrby(key, f"views_{window}", 1)
        self.redis.expire(key, 604800)  # 7 day TTL

    # --- Read methods (called at serving time) ---

    def get_user_features(self, user_id: str) -> dict[str, Any]:
        """Get all user features (batch + real-time) for serving."""
        pipe = self.redis.pipeline()
        pipe.hgetall(f"user:{user_id}:batch")
        pipe.hgetall(f"user:{user_id}:rt")
        batch_raw, rt_raw = pipe.execute()

        features = {}
        for raw in [batch_raw, rt_raw]:
            for k, v in raw.items():
                features[k] = self._deserialize(v)
        return features

    def get_item_features(self, item_id: str) -> dict[str, Any]:
        """Get all item features for serving."""
        pipe = self.redis.pipeline()
        pipe.hgetall(f"item:{item_id}:batch")
        pipe.hgetall(f"item:{item_id}:stats")
        batch_raw, stats_raw = pipe.execute()

        features = {}
        for raw in [batch_raw, stats_raw]:
            for k, v in raw.items():
                features[k] = self._deserialize(v)
        return features

    def get_batch_user_features(self, user_ids: list[str]) -> dict[str, dict]:
        """Batch fetch user features for multiple users."""
        pipe = self.redis.pipeline()
        for uid in user_ids:
            pipe.hgetall(f"user:{uid}:batch")
            pipe.hgetall(f"user:{uid}:rt")

        results = pipe.execute()
        out = {}
        for i, uid in enumerate(user_ids):
            batch_raw = results[i * 2]
            rt_raw = results[i * 2 + 1]
            features = {}
            for raw in [batch_raw, rt_raw]:
                for k, v in raw.items():
                    features[k] = self._deserialize(v)
            out[uid] = features
        return out

    def get_batch_item_features(self, item_ids: list[str]) -> dict[str, dict]:
        """Batch fetch item features for multiple items."""
        pipe = self.redis.pipeline()
        for iid in item_ids:
            pipe.hgetall(f"item:{iid}:batch")
            pipe.hgetall(f"item:{iid}:stats")

        results = pipe.execute()
        out = {}
        for i, iid in enumerate(item_ids):
            batch_raw = results[i * 2]
            stats_raw = results[i * 2 + 1]
            features = {}
            for raw in [batch_raw, stats_raw]:
                for k, v in raw.items():
                    features[k] = self._deserialize(v)
            out[iid] = features
        return out

    # --- User watch history (for two-tower input) ---

    def append_watch_history(self, user_id: str, item_id: str, max_len: int = 50):
        """Append an item to user's recent watch history (capped list)."""
        key = f"user:{user_id}:history"
        pipe = self.redis.pipeline()
        pipe.lpush(key, item_id)
        pipe.ltrim(key, 0, max_len - 1)
        pipe.execute()

    def get_watch_history(self, user_id: str, max_len: int = 50) -> list[str]:
        """Get user's recent watch history (most recent first)."""
        return self.redis.lrange(f"user:{user_id}:history", 0, max_len - 1)

    # --- Serialization helpers ---

    def _serialize(self, value: Any) -> str:
        if isinstance(value, np.ndarray):
            return json.dumps({"__ndarray__": value.tolist()})
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)

    def _deserialize(self, value: str) -> Any:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict) and "__ndarray__" in parsed:
                return np.array(parsed["__ndarray__"], dtype=np.float32)
            return parsed
        except (json.JSONDecodeError, TypeError):
            try:
                return float(value)
            except ValueError:
                return value
