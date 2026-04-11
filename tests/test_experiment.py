"""Tests for the A/B experimentation framework."""

import pytest
from datetime import datetime

from flickpick.experiment.ab_framework import (
    ABExperiment,
    ExperimentConfig,
    ExperimentStatus,
    MetricConfig,
    MetricObservation,
    Variant,
)


@pytest.fixture
def experiment():
    config = ExperimentConfig(
        experiment_id="test-exp-1",
        name="Test Experiment",
        variants=[
            Variant("control", "v1", 0.5),
            Variant("treatment", "v2", 0.5),
        ],
        metrics=[
            MetricConfig(name="ctr", min_detectable_effect=0.02),
            MetricConfig(name="watch_time", is_guardrail=True, min_detectable_effect=0.05),
        ],
        status=ExperimentStatus.RUNNING,
    )
    return ABExperiment(config)


def test_deterministic_assignment(experiment):
    """Same user always gets the same variant."""
    v1 = experiment.assign_variant("user_123")
    v2 = experiment.assign_variant("user_123")
    assert v1 == v2


def test_traffic_split_is_roughly_even(experiment):
    """With 50/50 split, assignments should be roughly balanced."""
    counts = {"control": 0, "treatment": 0}
    for i in range(10000):
        variant = experiment.assign_variant(f"user_{i}")
        counts[variant] += 1

    # Should be within 5% of 50/50
    assert abs(counts["control"] - 5000) < 500
    assert abs(counts["treatment"] - 5000) < 500


def test_analyze_returns_none_with_insufficient_data(experiment):
    """Need at least 30 observations per variant."""
    result = experiment.analyze("ctr")
    assert result is None


def test_analyze_detects_significant_difference(experiment):
    """Large effect size should be detected as significant."""
    now = datetime.now()

    # Control: mean ~0.1
    for i in range(500):
        experiment.record_observation(MetricObservation(
            user_id=f"c_{i}", variant="control",
            metric_name="ctr", value=0.1, timestamp=now,
        ))

    # Treatment: mean ~0.3 (huge effect)
    for i in range(500):
        experiment.record_observation(MetricObservation(
            user_id=f"t_{i}", variant="treatment",
            metric_name="ctr", value=0.3, timestamp=now,
        ))

    result = experiment.analyze("ctr")
    assert result is not None
    assert result.is_significant
    assert result.relative_lift > 0


def test_guardrail_detects_degradation(experiment):
    """Guardrail metric should flag significant negative lift."""
    now = datetime.now()

    # Control: higher watch time
    for i in range(500):
        experiment.record_observation(MetricObservation(
            user_id=f"c_{i}", variant="control",
            metric_name="watch_time", value=120.0, timestamp=now,
        ))

    # Treatment: significantly lower watch time
    for i in range(500):
        experiment.record_observation(MetricObservation(
            user_id=f"t_{i}", variant="treatment",
            metric_name="watch_time", value=60.0, timestamp=now,
        ))

    violations = experiment.check_guardrails()
    assert "watch_time" in violations


def test_no_guardrail_violation_when_treatment_is_better(experiment):
    """No violation when treatment improves the guardrail metric."""
    now = datetime.now()

    for i in range(500):
        experiment.record_observation(MetricObservation(
            user_id=f"c_{i}", variant="control",
            metric_name="watch_time", value=100.0, timestamp=now,
        ))

    for i in range(500):
        experiment.record_observation(MetricObservation(
            user_id=f"t_{i}", variant="treatment",
            metric_name="watch_time", value=150.0, timestamp=now,
        ))

    violations = experiment.check_guardrails()
    assert len(violations) == 0
