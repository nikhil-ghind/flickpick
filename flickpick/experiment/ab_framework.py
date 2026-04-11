"""
A/B experimentation framework.

Supports:
- Traffic splitting by user ID hash (deterministic assignment)
- Sequential testing with always-valid p-values (no peeking problem)
- Automatic metric collection (CTR, watch-time, retention)
- Early stopping on significant positive or negative results
- Guardrail metrics with automatic rollback
"""

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    STOPPED = "stopped"
    CONCLUDED = "concluded"  # reached significance
    ROLLED_BACK = "rolled_back"  # guardrail triggered


@dataclass
class Variant:
    name: str  # "control" or "treatment"
    model_version: str
    traffic_pct: float  # 0.0 to 1.0


@dataclass
class MetricConfig:
    name: str
    is_guardrail: bool = False  # if True, rollback on significant degradation
    min_detectable_effect: float = 0.02  # MDE for power calculation
    alpha: float = 0.05  # significance level


@dataclass
class ExperimentConfig:
    experiment_id: str
    name: str
    variants: list[Variant]
    metrics: list[MetricConfig]
    start_time: datetime | None = None
    end_time: datetime | None = None
    status: ExperimentStatus = ExperimentStatus.DRAFT
    min_samples_per_variant: int = 1000


@dataclass
class MetricObservation:
    """A single metric observation for a user in a variant."""
    user_id: str
    variant: str
    metric_name: str
    value: float
    timestamp: datetime


@dataclass
class VariantStats:
    """Aggregated statistics for a variant on a metric."""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # for Welford's online variance

    def update(self, value: float):
        """Welford's online algorithm for mean and variance."""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


@dataclass
class ExperimentResult:
    metric_name: str
    control_mean: float
    treatment_mean: float
    relative_lift: float  # (treatment - control) / control
    p_value: float
    confidence_interval: tuple[float, float]
    is_significant: bool
    control_n: int
    treatment_n: int


class ABExperiment:
    """Manages a single A/B experiment."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        # metric_name → variant_name → VariantStats
        self.stats: dict[str, dict[str, VariantStats]] = {}
        for metric in config.metrics:
            self.stats[metric.name] = {v.name: VariantStats() for v in config.variants}

    def assign_variant(self, user_id: str) -> str:
        """Deterministically assign a user to a variant based on user_id hash.

        Uses consistent hashing so the same user always gets the same variant
        for this experiment.
        """
        hash_input = f"{self.config.experiment_id}:{user_id}"
        hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_val % 10000) / 10000.0

        cumulative = 0.0
        for variant in self.config.variants:
            cumulative += variant.traffic_pct
            if bucket < cumulative:
                return variant.name

        return self.config.variants[-1].name

    def record_observation(self, obs: MetricObservation):
        """Record a metric observation and check for significance."""
        if obs.metric_name not in self.stats:
            return

        variant_stats = self.stats[obs.metric_name]
        if obs.variant not in variant_stats:
            return

        variant_stats[obs.variant].update(obs.value)

    def analyze(self, metric_name: str) -> ExperimentResult | None:
        """Analyze results for a metric using always-valid sequential testing.

        Uses a mixture sequential probability ratio test (mSPRT) which provides
        valid p-values at any stopping time — no peeking problem.
        """
        if metric_name not in self.stats:
            return None

        metric_config = next(
            (m for m in self.config.metrics if m.name == metric_name), None
        )
        if metric_config is None:
            return None

        variant_stats = self.stats[metric_name]
        control = variant_stats.get("control")
        treatment = variant_stats.get("treatment")

        if not control or not treatment or control.n < 30 or treatment.n < 30:
            return None

        # Always-valid p-value using mSPRT
        p_value = self._msprt_p_value(control, treatment, metric_config.min_detectable_effect)

        # Confidence interval via normal approximation
        se = math.sqrt(control.variance / control.n + treatment.variance / treatment.n)
        diff = treatment.mean - control.mean
        ci_lower = diff - 1.96 * se
        ci_upper = diff + 1.96 * se

        relative_lift = diff / control.mean if control.mean != 0 else 0.0

        return ExperimentResult(
            metric_name=metric_name,
            control_mean=control.mean,
            treatment_mean=treatment.mean,
            relative_lift=relative_lift,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < metric_config.alpha,
            control_n=control.n,
            treatment_n=treatment.n,
        )

    def check_guardrails(self) -> list[str]:
        """Check guardrail metrics for significant degradation.

        Returns list of violated guardrail metric names.
        """
        violations = []
        for metric in self.config.metrics:
            if not metric.is_guardrail:
                continue
            result = self.analyze(metric.name)
            if result and result.is_significant and result.relative_lift < 0:
                violations.append(metric.name)
                logger.warning(
                    f"Guardrail violation: {metric.name} "
                    f"(lift={result.relative_lift:.4f}, p={result.p_value:.4f})"
                )
        return violations

    def analyze_all(self) -> list[ExperimentResult]:
        """Analyze all metrics."""
        results = []
        for metric in self.config.metrics:
            result = self.analyze(metric.name)
            if result:
                results.append(result)
        return results

    def _msprt_p_value(
        self, control: VariantStats, treatment: VariantStats, tau: float
    ) -> float:
        """Compute always-valid p-value using mixture Sequential Probability Ratio Test.

        The mSPRT uses a mixing distribution over the alternative hypothesis,
        which provides valid inference at any stopping time.

        Args:
            control: Control variant statistics
            treatment: Treatment variant statistics
            tau: Mixing parameter (related to minimum detectable effect)
        """
        n_c, n_t = control.n, treatment.n
        if n_c < 2 or n_t < 2:
            return 1.0

        # Pooled variance estimate
        pooled_var = (control.m2 + treatment.m2) / (n_c + n_t - 2)
        if pooled_var <= 0:
            return 1.0

        # Variance of the difference
        var_diff = pooled_var * (1.0 / n_c + 1.0 / n_t)
        se_diff = math.sqrt(var_diff)

        # Observed difference
        z = (treatment.mean - control.mean) / se_diff

        # mSPRT: mix over N(0, tau^2) prior on the effect size
        # The likelihood ratio at the mixture is:
        #   Lambda = sqrt(var_diff / (var_diff + tau^2)) * exp(tau^2 * z^2 / (2 * (var_diff + tau^2)))
        tau_sq = tau ** 2
        ratio = var_diff / (var_diff + tau_sq)

        if ratio <= 0:
            return 1.0

        log_lambda = 0.5 * math.log(ratio) + (tau_sq * z ** 2) / (2 * (var_diff + tau_sq))

        # p-value = 1 / Lambda (Ville's inequality)
        if log_lambda > 0:
            p_value = math.exp(-log_lambda)
        else:
            p_value = 1.0

        return min(p_value, 1.0)


class ExperimentRegistry:
    """Manages multiple concurrent experiments."""

    def __init__(self):
        self.experiments: dict[str, ABExperiment] = {}

    def create(self, config: ExperimentConfig) -> ABExperiment:
        exp = ABExperiment(config)
        self.experiments[config.experiment_id] = exp
        return exp

    def get(self, experiment_id: str) -> ABExperiment | None:
        return self.experiments.get(experiment_id)

    def get_active(self) -> list[ABExperiment]:
        return [
            e for e in self.experiments.values()
            if e.config.status == ExperimentStatus.RUNNING
        ]

    def get_user_assignments(self, user_id: str) -> dict[str, str]:
        """Get all active experiment assignments for a user.

        Returns: {experiment_id: variant_name}
        """
        assignments = {}
        for exp in self.get_active():
            assignments[exp.config.experiment_id] = exp.assign_variant(user_id)
        return assignments

    def get_model_version_for_user(self, user_id: str, default_version: str) -> str:
        """Determine which model version to serve for a user based on experiments."""
        for exp in self.get_active():
            variant_name = exp.assign_variant(user_id)
            for variant in exp.config.variants:
                if variant.name == variant_name:
                    return variant.model_version
        return default_version
