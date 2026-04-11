"""
Canary rollout manager for model deployments.

Gradually shifts traffic from old model to new model while monitoring
key metrics. Automatically rolls back on metric degradation.

Stages:
  1. Canary (5% traffic) — monitor for 1 hour
  2. Partial (25% traffic) — monitor for 2 hours
  3. Majority (50% traffic) — monitor for 4 hours
  4. Full (100% traffic) — complete rollout
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from flickpick.experiment.ab_framework import (
    ABExperiment,
    ExperimentConfig,
    ExperimentStatus,
    MetricConfig,
    Variant,
)

logger = logging.getLogger(__name__)


class RolloutStage(str, Enum):
    CANARY = "canary"
    PARTIAL = "partial"
    MAJORITY = "majority"
    FULL = "full"
    ROLLED_BACK = "rolled_back"


@dataclass
class RolloutConfig:
    rollout_id: str
    old_model_version: str
    new_model_version: str
    stages: dict[RolloutStage, float] = None  # stage → traffic percentage for new model
    stage_durations: dict[RolloutStage, timedelta] = None
    guardrail_metrics: list[str] = None

    def __post_init__(self):
        if self.stages is None:
            self.stages = {
                RolloutStage.CANARY: 0.05,
                RolloutStage.PARTIAL: 0.25,
                RolloutStage.MAJORITY: 0.50,
                RolloutStage.FULL: 1.0,
            }
        if self.stage_durations is None:
            self.stage_durations = {
                RolloutStage.CANARY: timedelta(hours=1),
                RolloutStage.PARTIAL: timedelta(hours=2),
                RolloutStage.MAJORITY: timedelta(hours=4),
            }
        if self.guardrail_metrics is None:
            self.guardrail_metrics = ["watch_time", "ctr"]


class CanaryRollout:
    """Manages progressive rollout of a new model version."""

    def __init__(self, config: RolloutConfig):
        self.config = config
        self.current_stage = RolloutStage.CANARY
        self.stage_start_time = datetime.now()
        self.experiment = self._create_experiment()

    def _create_experiment(self) -> ABExperiment:
        """Create an A/B experiment for the current stage."""
        new_pct = self.config.stages[self.current_stage]
        old_pct = 1.0 - new_pct

        exp_config = ExperimentConfig(
            experiment_id=f"canary-{self.config.rollout_id}-{self.current_stage.value}",
            name=f"Canary: {self.config.old_model_version} → {self.config.new_model_version}",
            variants=[
                Variant("control", self.config.old_model_version, old_pct),
                Variant("treatment", self.config.new_model_version, new_pct),
            ],
            metrics=[
                MetricConfig(name="ctr", is_guardrail=True, min_detectable_effect=0.01),
                MetricConfig(name="watch_time", is_guardrail=True, min_detectable_effect=0.02),
                MetricConfig(name="retention_1d", is_guardrail=False),
                MetricConfig(name="diversity_score", is_guardrail=False),
            ],
            status=ExperimentStatus.RUNNING,
            start_time=datetime.now(),
        )

        return ABExperiment(exp_config)

    def check_and_advance(self) -> RolloutStage:
        """Check metrics and advance to next stage if safe.

        Returns the current stage after evaluation.
        """
        # Check guardrails
        violations = self.experiment.check_guardrails()
        if violations:
            logger.error(
                f"Canary rollback triggered! Guardrail violations: {violations}"
            )
            self.current_stage = RolloutStage.ROLLED_BACK
            self.experiment.config.status = ExperimentStatus.ROLLED_BACK
            return self.current_stage

        # Check if enough time has passed for this stage
        elapsed = datetime.now() - self.stage_start_time
        stage_duration = self.config.stage_durations.get(self.current_stage)

        if stage_duration and elapsed < stage_duration:
            return self.current_stage  # not enough time yet

        # Advance to next stage
        stage_order = [RolloutStage.CANARY, RolloutStage.PARTIAL, RolloutStage.MAJORITY, RolloutStage.FULL]
        current_idx = stage_order.index(self.current_stage)

        if current_idx < len(stage_order) - 1:
            self.current_stage = stage_order[current_idx + 1]
            self.stage_start_time = datetime.now()
            self.experiment = self._create_experiment()
            logger.info(
                f"Advanced to {self.current_stage.value} "
                f"({self.config.stages[self.current_stage]*100:.0f}% traffic)"
            )
        else:
            self.experiment.config.status = ExperimentStatus.CONCLUDED
            logger.info("Rollout complete — 100% traffic on new model")

        return self.current_stage

    def get_model_version(self, user_id: str) -> str:
        """Get the model version to serve for a user."""
        if self.current_stage == RolloutStage.ROLLED_BACK:
            return self.config.old_model_version
        if self.current_stage == RolloutStage.FULL:
            return self.config.new_model_version

        variant = self.experiment.assign_variant(user_id)
        for v in self.experiment.config.variants:
            if v.name == variant:
                return v.model_version
        return self.config.old_model_version

    def get_status(self) -> dict:
        return {
            "rollout_id": self.config.rollout_id,
            "stage": self.current_stage.value,
            "old_model": self.config.old_model_version,
            "new_model": self.config.new_model_version,
            "traffic_pct": self.config.stages.get(self.current_stage, 0),
            "stage_start": self.stage_start_time.isoformat(),
            "results": [r.__dict__ for r in self.experiment.analyze_all()],
        }
