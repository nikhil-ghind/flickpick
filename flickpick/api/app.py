"""
FastAPI application for serving recommendations and managing experiments.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from flickpick.experiment.ab_framework import (
    ABExperiment,
    ExperimentConfig,
    ExperimentRegistry,
    ExperimentStatus,
    MetricConfig,
    MetricObservation,
    Variant,
)
from flickpick.experiment.canary import CanaryRollout, RolloutConfig
from flickpick.features.feature_store import FeatureStore

logger = logging.getLogger(__name__)

# Global state — initialized in lifespan
feature_store: FeatureStore | None = None
experiment_registry: ExperimentRegistry | None = None
canary_rollout: CanaryRollout | None = None
# recommendation_pipeline loaded separately based on model version


@asynccontextmanager
async def lifespan(app: FastAPI):
    global feature_store, experiment_registry
    feature_store = FeatureStore()
    experiment_registry = ExperimentRegistry()
    logger.info("Flickpick API started")
    yield
    logger.info("Flickpick API shutting down")


app = FastAPI(title="Flickpick", version="1.0.0", lifespan=lifespan)


# --- Request/Response models ---

class RecommendRequest(BaseModel):
    user_id: str
    num_results: int = 20
    exclude_item_ids: list[str] | None = None
    context: dict | None = None


class RecommendResponse(BaseModel):
    items: list[dict]
    latency_ms: float
    model_version: str
    experiment_variant: str | None = None


class CreateExperimentRequest(BaseModel):
    experiment_id: str
    name: str
    control_model: str
    treatment_model: str
    treatment_traffic_pct: float = 0.5
    metrics: list[dict] | None = None


class RecordMetricRequest(BaseModel):
    experiment_id: str
    user_id: str
    variant: str
    metric_name: str
    value: float


class StartCanaryRequest(BaseModel):
    rollout_id: str
    old_model_version: str
    new_model_version: str


# --- Recommendation endpoints ---

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """Get personalized recommendations for a user.

    If an A/B experiment is active, serves the appropriate model version.
    """
    # Determine model version from experiments
    variant = None
    model_version = "v1"  # default

    if experiment_registry:
        assignments = experiment_registry.get_user_assignments(req.user_id)
        if assignments:
            exp_id, variant_name = next(iter(assignments.items()))
            variant = variant_name
            model_version = experiment_registry.get_model_version_for_user(
                req.user_id, model_version
            )

    if canary_rollout:
        model_version = canary_rollout.get_model_version(req.user_id)

    # In production, we'd load the appropriate pipeline for the model version.
    # For now, return a structured response showing the routing.
    return RecommendResponse(
        items=[{"item_id": f"item_{i}", "score": 1.0 - i * 0.05} for i in range(req.num_results)],
        latency_ms=12.5,
        model_version=model_version,
        experiment_variant=variant,
    )


# --- Experiment endpoints ---

@app.post("/experiments")
async def create_experiment(req: CreateExperimentRequest):
    """Create a new A/B experiment."""
    metrics = [
        MetricConfig(name=m.get("name", "ctr"), is_guardrail=m.get("is_guardrail", False))
        for m in (req.metrics or [{"name": "ctr"}, {"name": "watch_time", "is_guardrail": True}])
    ]

    config = ExperimentConfig(
        experiment_id=req.experiment_id,
        name=req.name,
        variants=[
            Variant("control", req.control_model, 1.0 - req.treatment_traffic_pct),
            Variant("treatment", req.treatment_model, req.treatment_traffic_pct),
        ],
        metrics=metrics,
        status=ExperimentStatus.RUNNING,
        start_time=datetime.now(),
    )

    exp = experiment_registry.create(config)
    return {"experiment_id": config.experiment_id, "status": "running"}


@app.post("/experiments/{experiment_id}/observe")
async def record_observation(experiment_id: str, req: RecordMetricRequest):
    """Record a metric observation for an experiment."""
    exp = experiment_registry.get(experiment_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")

    exp.record_observation(MetricObservation(
        user_id=req.user_id,
        variant=req.variant,
        metric_name=req.metric_name,
        value=req.value,
        timestamp=datetime.now(),
    ))
    return {"ok": True}


@app.get("/experiments/{experiment_id}/results")
async def get_experiment_results(experiment_id: str):
    """Get current experiment results with statistical analysis."""
    exp = experiment_registry.get(experiment_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")

    results = exp.analyze_all()
    guardrail_violations = exp.check_guardrails()

    return {
        "experiment_id": experiment_id,
        "status": exp.config.status.value,
        "guardrail_violations": guardrail_violations,
        "metrics": [
            {
                "name": r.metric_name,
                "control_mean": r.control_mean,
                "treatment_mean": r.treatment_mean,
                "relative_lift": r.relative_lift,
                "p_value": r.p_value,
                "confidence_interval": list(r.confidence_interval),
                "is_significant": r.is_significant,
                "control_n": r.control_n,
                "treatment_n": r.treatment_n,
            }
            for r in results
        ],
    }


@app.get("/experiments")
async def list_experiments():
    """List all experiments."""
    return {
        "experiments": [
            {
                "experiment_id": exp.config.experiment_id,
                "name": exp.config.name,
                "status": exp.config.status.value,
                "variants": [
                    {"name": v.name, "model": v.model_version, "traffic": v.traffic_pct}
                    for v in exp.config.variants
                ],
            }
            for exp in experiment_registry.experiments.values()
        ]
    }


# --- Canary rollout endpoints ---

@app.post("/canary/start")
async def start_canary(req: StartCanaryRequest):
    """Start a canary rollout for a new model version."""
    global canary_rollout
    config = RolloutConfig(
        rollout_id=req.rollout_id,
        old_model_version=req.old_model_version,
        new_model_version=req.new_model_version,
    )
    canary_rollout = CanaryRollout(config)
    return canary_rollout.get_status()


@app.post("/canary/advance")
async def advance_canary():
    """Check metrics and advance canary to next stage."""
    if not canary_rollout:
        raise HTTPException(400, "No active canary rollout")
    canary_rollout.check_and_advance()
    return canary_rollout.get_status()


@app.get("/canary/status")
async def canary_status():
    """Get current canary rollout status."""
    if not canary_rollout:
        return {"status": "no active rollout"}
    return canary_rollout.get_status()


# --- Feature store endpoints ---

@app.get("/features/user/{user_id}")
async def get_user_features(user_id: str):
    """Get all features for a user (for debugging)."""
    features = feature_store.get_user_features(user_id)
    history = feature_store.get_watch_history(user_id)
    return {"user_id": user_id, "features": features, "watch_history": history}


@app.get("/features/item/{item_id}")
async def get_item_features(item_id: str):
    """Get all features for an item (for debugging)."""
    features = feature_store.get_item_features(item_id)
    return {"item_id": item_id, "features": features}


@app.get("/health")
async def health():
    return {"status": "ok"}
