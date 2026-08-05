"""Translate canonical benchmark jobs into NVIDIA AIC requests."""

from __future__ import annotations

from collections.abc import Mapping

from data_generation.methods.dynamo_aic.client import AicRequest
from data_generation.methods.dynamo_aic.settings import DynamoAicSettings
from data_generation.scheduling import HardwarePool, SchedulerJob


def build_aic_request(
    job: SchedulerJob,
    pool: HardwarePool,
    settings: DynamoAicSettings,
    available_devices: int,
) -> AicRequest:
    """Map one scheduler-visible job and hardware pool to AIC."""
    if job.kind not in {"online", "batch"}:
        raise ValueError(f"Unsupported job kind: {job.kind}")
    if pool.aic_system is None:
        raise ValueError(f"Pool {pool.pool_id} has no AIC system mapping")

    model = _text(job.spec.get("model_id"), "model_id")
    features_value = job.spec.get("job_features")
    features = _mapping(features_value, "job_features")
    optimization_goal = job.optimization_goal or settings.default_optimization_goal
    return AicRequest(
        model=model,
        system=pool.aic_system,
        available_devices=available_devices,
        backend=settings.backend,
        backend_version=settings.backend_versions.get(
            pool.pool_id,
            settings.backend_version,
        ),
        database_mode=settings.database_mode,
        serving_mode=settings.serving_mode,
        input_tokens=max(1, round(job.workload.input_tokens)),
        output_tokens=max(1, round(job.workload.output_tokens)),
        target_ttft_ms=_positive_number(features, "target_p99_ttft_ms"),
        target_tpot_ms=_positive_number(features, "target_p99_tpot_ms"),
        optimization_goal=optimization_goal,
        target_concurrency=None,
        target_request_rate=job.workload.request_rate,
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"job_features.{key} must be a positive number")
    return float(value)
