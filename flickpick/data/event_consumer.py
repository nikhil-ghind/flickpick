"""
Kafka event consumer for processing user interaction events in real-time.

Consumes events from the 'user-interactions' topic and:
1. Updates real-time features in the feature store
2. Records A/B experiment metric observations
3. Appends to watch history
4. Increments item popularity counters
"""

import json
import logging
from datetime import datetime

from kafka import KafkaConsumer

from flickpick.experiment.ab_framework import ExperimentRegistry, MetricObservation
from flickpick.features.feature_store import FeatureStore

logger = logging.getLogger(__name__)


class EventConsumer:
    def __init__(
        self,
        bootstrap_servers: str,
        feature_store: FeatureStore,
        experiment_registry: ExperimentRegistry,
        group_id: str = "flickpick-feature-updater",
    ):
        self.feature_store = feature_store
        self.experiment_registry = experiment_registry
        self.consumer = KafkaConsumer(
            "user-interactions",
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset="latest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            enable_auto_commit=True,
            auto_commit_interval_ms=5000,
        )

    def run(self):
        """Main consumer loop. Blocks forever."""
        logger.info("Event consumer started, listening for interactions...")

        for message in self.consumer:
            try:
                self._process_event(message.value)
            except Exception:
                logger.exception(f"Failed to process event: {message.value}")

    def _process_event(self, event: dict):
        """Process a single interaction event.

        Expected event schema:
        {
            "event_type": "watch" | "click" | "skip" | "rate",
            "user_id": "u123",
            "item_id": "i456",
            "timestamp": "2024-01-15T10:30:00Z",
            "watch_pct": 0.85,       # for watch events
            "watch_seconds": 2400,   # for watch events
            "rating": 4.5,           # for rate events
            "experiment_id": "exp-1", # optional
            "variant": "treatment",   # optional
            "session_id": "s789"
        }
        """
        event_type = event.get("event_type")
        user_id = event.get("user_id")
        item_id = event.get("item_id")
        timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))

        if not user_id or not item_id:
            return

        # Update real-time user features
        self._update_user_realtime(user_id, event)

        # Update item popularity
        if event_type in ("watch", "click"):
            self.feature_store.increment_item_view_count(item_id, "7d")

        # Append to watch history
        if event_type == "watch" and event.get("watch_pct", 0) > 0.1:
            self.feature_store.append_watch_history(user_id, item_id)

        # Record experiment observations
        self._record_experiment_metrics(event, timestamp)

    def _update_user_realtime(self, user_id: str, event: dict):
        """Update real-time user features based on the event."""
        current = self.feature_store.get_user_features(user_id)

        session_depth = current.get("session_depth", 0)
        if event.get("event_type") in ("watch", "click"):
            session_depth += 1

        rt_features = {
            "session_depth": session_depth,
            "last_event_type": event.get("event_type", ""),
            "last_item_id": event.get("item_id", ""),
        }
        self.feature_store.update_user_realtime(user_id, rt_features)

    def _record_experiment_metrics(self, event: dict, timestamp: datetime):
        """Record metric observations for active experiments."""
        experiment_id = event.get("experiment_id")
        variant = event.get("variant")

        if not experiment_id or not variant:
            return

        exp = self.experiment_registry.get(experiment_id)
        if not exp:
            return

        user_id = event["user_id"]
        event_type = event.get("event_type")

        # CTR: 1 if clicked/watched, 0 if shown but skipped
        if event_type in ("watch", "click"):
            exp.record_observation(MetricObservation(
                user_id=user_id, variant=variant,
                metric_name="ctr", value=1.0, timestamp=timestamp,
            ))
        elif event_type == "skip":
            exp.record_observation(MetricObservation(
                user_id=user_id, variant=variant,
                metric_name="ctr", value=0.0, timestamp=timestamp,
            ))

        # Watch time (seconds)
        if event_type == "watch":
            watch_seconds = event.get("watch_seconds", 0)
            exp.record_observation(MetricObservation(
                user_id=user_id, variant=variant,
                metric_name="watch_time", value=float(watch_seconds),
                timestamp=timestamp,
            ))

    def close(self):
        self.consumer.close()
