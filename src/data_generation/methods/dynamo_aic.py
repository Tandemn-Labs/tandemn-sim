"""Greedy per-job hardware selection using NVIDIA AIC."""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Protocol

from data_generation.scheduling import (
    CandidateEvaluation,
    CandidateStatus,
    DecisionFailureCategory,
    DecisionStatus,
    HardwarePool,
    OptimizationGoal,
    RegionalHardwarePool,
    ResourceInventory,
    SchedulerJob,
    SchedulingDecision,
    load_resource_inventory,
)

_AIC_CANDIDATE_LIMIT = 1000


class AicEvaluationError(RuntimeError):
    """Report an AIC dependency or evaluation failure."""


class AicUnavailableError(ValueError):
    """Report that the NVIDIA AIC package cannot run in this environment."""


@dataclass(frozen=True)
class AicRequest:
    """Inputs for one homogeneous AIC recommendation."""

    model: str
    system: str
    available_devices: int
    backend: str
    backend_version: str | None
    database_mode: str
    serving_mode: str
    input_tokens: int
    output_tokens: int
    target_ttft_ms: float
    target_tpot_ms: float
    optimization_goal: OptimizationGoal
    target_concurrency: float | None = None
    target_request_rate: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the request as a JSON object."""
        return asdict(self)


@dataclass(frozen=True)
class AicResult:
    """Normalized rank-one aggregate configuration returned by AIC."""

    devices: int
    replicas: int
    ttft_ms: float
    tpot_ms: float
    request_latency_ms: float
    throughput_tokens_per_second: float
    configuration: Mapping[str, object]


class AicClient(Protocol):
    """Evaluate one job against one homogeneous hardware system."""

    def evaluate(self, request: AicRequest) -> AicResult:
        """Return AIC's rank-one aggregate configuration."""


class NvidiaAicClient:
    """Call the public AIC recommendation API and retain aggregate serving."""

    def evaluate(self, request: AicRequest) -> AicResult:
        """Run AIC load matching for one hardware system."""
        if request.serving_mode != "agg":
            raise AicEvaluationError("Only aggregate serving is supported here")
        try:
            from aiconfigurator.cli.api import cli_recommend
            from aiconfigurator.sdk.picking import pick_optimization_type
        except ImportError as error:
            raise AicUnavailableError(
                "AIC is not installed; run `uv sync --group dynamo`"
            ) from error

        previous_logging_threshold = logging.root.manager.disable
        try:
            logging.disable(logging.CRITICAL)
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                result = cli_recommend(
                    model_path=request.model,
                    system=request.system,
                    target_request_rate=request.target_request_rate,
                    target_concurrency=request.target_concurrency,
                    backend=request.backend,
                    backend_version=request.backend_version,
                    database_mode=request.database_mode,
                    isl=request.input_tokens,
                    osl=request.output_tokens,
                    ttft=request.target_ttft_ms,
                    tpot=request.target_tpot_ms,
                    strict_sla=True,
                    top_n=_AIC_CANDIDATE_LIMIT,
                )
        except SystemExit as error:
            raise AicEvaluationError(str(error)) from error
        except Exception as error:
            raise AicEvaluationError(str(error)) from error
        finally:
            logging.disable(previous_logging_threshold)

        frame = result.best_configs.get("agg")
        if frame is None or frame.empty:
            raise AicEvaluationError("AIC returned no aggregate configuration")
        try:
            fitting_frame = frame[
                frame["total_gpus_needed"] <= request.available_devices
            ]
            objective_input = fitting_frame if not fitting_frame.empty else frame
            if request.optimization_goal == OptimizationGoal.LATENCY:
                frame = objective_input.sort_values(
                    by=["ttft", "tpot", "total_gpus_needed"],
                    ascending=[True, True, True],
                ).head(1)
            else:
                picked = pick_optimization_type(
                    pareto_df=objective_input,
                    optimization_type=request.optimization_goal.value,
                    total_gpus=request.available_devices,
                    serving_mode=request.serving_mode,
                    top_n=1,
                )
                frame = picked["best_config_df"]
        except Exception as error:
            raise AicEvaluationError(str(error)) from error
        if frame is None or frame.empty:
            raise AicEvaluationError(
                "AIC returned no configuration for the optimization goal"
            )
        row = frame.iloc[0]
        configuration = _json_value(row.to_dict())
        if not isinstance(configuration, Mapping):
            raise AicEvaluationError("AIC returned an invalid configuration")

        replicas = _aic_positive_int(row.get("replicas_needed", 1), "replicas")
        devices = _aic_positive_int(
            row.get("total_gpus_needed", row.get("num_total_gpus")),
            "total_gpus_needed",
        )
        per_replica_throughput = _aic_positive_float(row.get("tokens/s"), "tokens/s")
        ttft_ms = _aic_positive_float(row.get("ttft"), "ttft")
        tpot_ms = _aic_positive_float(row.get("tpot"), "tpot")
        request_latency_ms = float(
            row.get(
                "request_latency",
                ttft_ms + tpot_ms * max(request.output_tokens - 1, 0),
            )
        )
        return AicResult(
            devices=devices,
            replicas=replicas,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            request_latency_ms=request_latency_ms,
            throughput_tokens_per_second=(per_replica_throughput * replicas),
            configuration=configuration,
        )


@dataclass(frozen=True)
class DynamoAicSettings:
    """AIC engine settings loaded from the method catalog."""

    backend: str
    backend_version: str | None
    backend_versions: Mapping[str, str]
    database_mode: str
    serving_mode: str
    default_optimization_goal: OptimizationGoal


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

    features = _mapping(job.spec.get("job_features"), "job_features")
    optimization_goal = job.optimization_goal or settings.default_optimization_goal
    return AicRequest(
        model=_text(job.spec.get("model_id"), "model_id"),
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


class DynamoAicMethod:
    """Evaluate homogeneous pools independently and greedily pick one."""

    method_name = "dynamo_aic"

    def __init__(
        self,
        inventory: ResourceInventory,
        settings: DynamoAicSettings,
        client: AicClient,
        randomizer: random.Random | None = None,
    ) -> None:
        self.inventory = inventory
        self.settings = settings
        self.client = client
        self.randomizer = randomizer or random.Random()

    def schedule(self, job: SchedulerJob) -> SchedulingDecision:
        """Evaluate every hardware type, select a winner, and reserve it."""
        goal = job.optimization_goal or self.settings.default_optimization_goal
        before = self.inventory.snapshot()
        candidates = tuple(
            self._evaluate_pool(job, regional_pool)
            for regional_pool in self.inventory.regional_pools()
        )
        feasible = [
            candidate
            for candidate in candidates
            if candidate.status == CandidateStatus.FEASIBLE
        ]
        if not feasible:
            failure_category, reason = _failure_details(job, candidates)
            return SchedulingDecision(
                job_id=job.job_id,
                method=self.method_name,
                status=DecisionStatus.REJECTED,
                optimization_goal=goal,
                candidates=candidates,
                resource_snapshot_before=before,
                resource_snapshot_after=self.inventory.snapshot(),
                failure_category=failure_category,
                reason=reason,
            )

        best_score = min(_candidate_score(candidate, goal) for candidate in feasible)
        tied_candidates = [
            candidate
            for candidate in feasible
            if _candidate_score(candidate, goal) == best_score
        ]
        selected = self.randomizer.choice(tied_candidates)
        if selected.requested_devices is None:
            raise RuntimeError("Feasible candidate has no device count")
        markets = self.inventory.available_markets(
            selected.pool_id,
            selected.requested_devices,
        )
        if not markets:
            raise RuntimeError("Feasible candidate has no available market")
        market = self.randomizer.choice(markets)
        self.inventory.reserve(
            job.job_id,
            selected.pool_id,
            selected.region,
            market,
            selected.requested_devices,
        )
        return SchedulingDecision(
            job_id=job.job_id,
            method=self.method_name,
            status=DecisionStatus.ACCEPTED,
            optimization_goal=goal,
            candidates=candidates,
            resource_snapshot_before=before,
            resource_snapshot_after=self.inventory.snapshot(),
            pool_id=selected.pool_id,
            accelerator=selected.accelerator,
            region=selected.region,
            market=market,
            devices=selected.requested_devices,
            latency_ms=selected.latency_ms,
            ttft_ms=selected.ttft_ms,
            tpot_ms=selected.tpot_ms,
            throughput_tokens_per_second=(selected.throughput_tokens_per_second),
            configuration=selected.configuration,
        )

    def release(self, job_id: str) -> None:
        """Return a terminal job's devices to the benchmark inventory."""
        self.inventory.release(job_id)

    def resource_snapshot(self) -> dict[str, object]:
        """Return the current benchmark resource map."""
        return self.inventory.snapshot()

    def _evaluate_pool(
        self,
        job: SchedulerJob,
        regional_pool: RegionalHardwarePool,
    ) -> CandidateEvaluation:
        """Normalize AIC's result for one hardware-region candidate."""
        pool = regional_pool.pool
        region = regional_pool.region
        available = regional_pool.available_devices
        if available == 0:
            return CandidateEvaluation(
                pool_id=pool.pool_id,
                accelerator=pool.accelerator,
                region=region,
                status=CandidateStatus.INFEASIBLE,
                available_devices=0,
                reason="No devices are currently available",
            )
        if pool.aic_system is None:
            return CandidateEvaluation(
                pool_id=pool.pool_id,
                accelerator=pool.accelerator,
                region=region,
                status=CandidateStatus.UNSUPPORTED,
                available_devices=available,
                reason=pool.unsupported_reason or "No AIC system mapping",
            )

        try:
            request = build_aic_request(
                job,
                pool,
                self.settings,
                available,
            )
            result = self.client.evaluate(request)
        except AicUnavailableError:
            raise
        except (AicEvaluationError, ValueError) as error:
            return CandidateEvaluation(
                pool_id=pool.pool_id,
                accelerator=pool.accelerator,
                region=region,
                status=CandidateStatus.ERROR,
                available_devices=available,
                reason=str(error),
            )

        reason = _infeasible_reason(
            request,
            result.ttft_ms,
            result.tpot_ms,
        )
        if result.devices > available:
            reason = f"AIC requires {result.devices} devices; {available} are available"
        status = (
            CandidateStatus.FEASIBLE if reason is None else CandidateStatus.INFEASIBLE
        )
        return CandidateEvaluation(
            pool_id=pool.pool_id,
            accelerator=pool.accelerator,
            region=region,
            status=status,
            available_devices=available,
            requested_devices=result.devices,
            latency_ms=result.request_latency_ms,
            ttft_ms=result.ttft_ms,
            tpot_ms=result.tpot_ms,
            throughput_tokens_per_second=(result.throughput_tokens_per_second),
            configuration=result.configuration,
            aic_request=request.to_dict(),
            reason=reason,
        )


def load_dynamo_aic_method(
    benchmark_directory: Path,
    settings: Mapping[str, object],
    *,
    client: AicClient | None = None,
    randomizer: random.Random | None = None,
) -> DynamoAicMethod:
    """Build the method from benchmark resources and catalog settings."""
    inventory = load_resource_inventory(benchmark_directory / "resources.yaml")
    return DynamoAicMethod(
        inventory,
        _load_settings(settings),
        client or NvidiaAicClient(),
        randomizer,
    )


def _infeasible_reason(
    request: AicRequest,
    ttft_ms: float,
    tpot_ms: float,
) -> str | None:
    target_ttft_ms = request.target_ttft_ms
    target_tpot_ms = request.target_tpot_ms
    if ttft_ms > target_ttft_ms:
        return f"TTFT {ttft_ms:.3f} ms exceeds {target_ttft_ms:.3f} ms"
    if tpot_ms > target_tpot_ms:
        return f"TPOT {tpot_ms:.3f} ms exceeds {target_tpot_ms:.3f} ms"
    return None


def _failure_details(
    job: SchedulerJob,
    candidates: tuple[CandidateEvaluation, ...],
) -> tuple[DecisionFailureCategory, str]:
    """Summarize why no hardware candidate could place a job."""
    reasons = [candidate.reason or "" for candidate in candidates]
    architecture_marker = "The model's architecture "
    architecture_suffix = " is not supported"
    for reason in reasons:
        if architecture_marker in reason and architecture_suffix in reason:
            architecture = reason.split(architecture_marker, 1)[1].split(
                architecture_suffix,
                1,
            )[0]
            return (
                DecisionFailureCategory.MODEL_UNSUPPORTED,
                f"AIC does not support model architecture {architecture}",
            )

    if any("HTTP error 404" in reason for reason in reasons):
        model = job.spec.get("model_id", "unknown")
        return (
            DecisionFailureCategory.MODEL_UNAVAILABLE,
            f"model {model} was not found on Hugging Face",
        )

    capacity_candidates = [
        candidate
        for candidate in candidates
        if candidate.status == CandidateStatus.INFEASIBLE
        and (
            candidate.available_devices == 0
            or (
                candidate.requested_devices is not None
                and candidate.requested_devices > candidate.available_devices
            )
        )
    ]
    if capacity_candidates:
        selected = min(
            capacity_candidates,
            key=lambda candidate: (
                (candidate.requested_devices or 0) - candidate.available_devices,
                candidate.pool_id,
                candidate.region,
            ),
        )
        if selected.requested_devices is None:
            reason = (
                f"no {selected.accelerator} devices are available in {selected.region}"
            )
        else:
            reason = (
                f"{selected.accelerator} in {selected.region} requires "
                f"{selected.requested_devices} devices; "
                f"{selected.available_devices} available"
            )
        return DecisionFailureCategory.INSUFFICIENT_CAPACITY, reason

    slo_reason = next(
        (
            reason
            for reason in reasons
            if reason.startswith("TTFT ") or reason.startswith("TPOT ")
        ),
        None,
    )
    if slo_reason is not None:
        return DecisionFailureCategory.SLO_INFEASIBLE, slo_reason

    infeasible_messages = (
        "No feasible GPU configuration",
        "AIC returned no aggregate configuration",
    )
    if any(message in reason for reason in reasons for message in infeasible_messages):
        return (
            DecisionFailureCategory.LOAD_INFEASIBLE,
            "AIC found no configuration meeting the job load and SLO",
        )

    if all(candidate.status == CandidateStatus.UNSUPPORTED for candidate in candidates):
        return (
            DecisionFailureCategory.HARDWARE_UNSUPPORTED,
            "AIC supports none of the available hardware pools",
        )

    error_reason = next((reason for reason in reasons if reason), "unknown error")
    return (
        DecisionFailureCategory.METHOD_ERROR,
        f"AIC evaluation failed: {error_reason}",
    )


def _candidate_score(
    candidate: CandidateEvaluation,
    goal: OptimizationGoal,
) -> tuple[float, float, int]:
    latency = candidate.latency_ms
    ttft = candidate.ttft_ms
    tpot = candidate.tpot_ms
    throughput = candidate.throughput_tokens_per_second
    devices = candidate.requested_devices
    if (
        latency is None
        or ttft is None
        or tpot is None
        or throughput is None
        or devices is None
    ):
        raise RuntimeError("Feasible candidate is missing comparison metrics")
    if goal == OptimizationGoal.LATENCY:
        return (ttft, tpot, devices)
    return (-throughput, latency, devices)


def _load_settings(values: Mapping[str, object]) -> DynamoAicSettings:
    serving_mode = _text(values.get("serving_mode"), "serving_mode")
    if serving_mode != "agg":
        raise ValueError("Dynamo+AIC currently requires aggregate serving")
    version_value = values.get("backend_version")
    backend_version = (
        None if version_value is None else _text(version_value, "backend_version")
    )
    versions = _mapping(values.get("backend_versions", {}), "backend_versions")
    return DynamoAicSettings(
        backend=_text(values.get("backend"), "backend"),
        backend_version=backend_version,
        backend_versions={
            _text(pool_id, "backend_versions key"): _text(
                version,
                f"backend_versions.{pool_id}",
            )
            for pool_id, version in versions.items()
        },
        database_mode=_text(values.get("database_mode"), "database_mode"),
        serving_mode=serving_mode,
        default_optimization_goal=OptimizationGoal(
            _text(
                values.get("default_optimization_goal"),
                "default_optimization_goal",
            )
        ),
    )


def _mapping(value: object, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_number(values: Mapping[object, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"job_features.{key} must be a positive number")
    return float(value)


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return item_method()
    return value


def _aic_positive_int(value: object, name: str) -> int:
    converted = _json_value(value)
    if isinstance(converted, bool) or not isinstance(converted, int | float):
        raise AicEvaluationError(f"AIC returned invalid {name}")
    number = int(converted)
    if number <= 0:
        raise AicEvaluationError(f"AIC returned invalid {name}")
    return number


def _aic_positive_float(value: object, name: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise AicEvaluationError(f"AIC returned invalid {name}") from error
    if number <= 0:
        raise AicEvaluationError(f"AIC returned invalid {name}")
    return number
