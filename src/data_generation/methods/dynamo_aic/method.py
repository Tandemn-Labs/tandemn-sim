"""Greedy per-job hardware selection using NVIDIA AIC."""

from __future__ import annotations

from pathlib import Path

from data_generation.methods.dynamo_aic.adapter import build_aic_request
from data_generation.methods.dynamo_aic.client import (
    AicClient,
    AicEvaluationError,
    AicRequest,
    AicUnavailableError,
    NvidiaAicClient,
)
from data_generation.methods.dynamo_aic.settings import (
    DynamoAicSettings,
    load_dynamo_aic_settings,
)
from data_generation.scheduling import (
    CandidateEvaluation,
    CandidateStatus,
    DecisionFailureCategory,
    DecisionStatus,
    HardwarePool,
    OptimizationGoal,
    ResourceInventory,
    SchedulerJob,
    SchedulingDecision,
    load_resource_inventory,
)


class DynamoAicMethod:
    """Evaluate homogeneous pools independently and greedily pick one."""

    method_name = "dynamo_aic"

    def __init__(
        self,
        inventory: ResourceInventory,
        settings: DynamoAicSettings,
        client: AicClient,
    ) -> None:
        self.inventory = inventory
        self.settings = settings
        self.client = client

    def schedule(self, job: SchedulerJob) -> SchedulingDecision:
        """Evaluate every hardware type, select a winner, and reserve it."""
        goal = job.optimization_goal or self.settings.default_optimization_goal
        before = self.inventory.snapshot()
        candidates = tuple(
            self._evaluate_pool(job, pool) for pool in self.inventory.pools
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

        selected = min(feasible, key=lambda item: _candidate_key(item, goal))
        if selected.requested_devices is None:
            raise RuntimeError("Feasible candidate has no device count")
        self.inventory.reserve(
            job.job_id,
            selected.pool_id,
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
        pool: HardwarePool,
    ) -> CandidateEvaluation:
        """Normalize AIC's best result for one homogeneous pool."""
        available = self.inventory.available(pool.pool_id)
        if available == 0:
            return CandidateEvaluation(
                pool_id=pool.pool_id,
                accelerator=pool.accelerator,
                status=CandidateStatus.INFEASIBLE,
                available_devices=0,
                reason="No devices are currently available",
            )
        if pool.aic_system is None:
            return CandidateEvaluation(
                pool_id=pool.pool_id,
                accelerator=pool.accelerator,
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
    *,
    client: AicClient | None = None,
) -> DynamoAicMethod:
    """Build the method from benchmark-owned resources and settings."""
    inventory = load_resource_inventory(benchmark_directory / "resources.yaml")
    settings = load_dynamo_aic_settings(
        benchmark_directory / "methods" / "dynamo_aic.yaml"
    )
    return DynamoAicMethod(
        inventory,
        settings,
        client or NvidiaAicClient(),
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
            ),
        )
        if selected.requested_devices is None:
            reason = f"no {selected.accelerator} devices are available"
        else:
            reason = (
                f"{selected.accelerator} requires {selected.requested_devices} "
                f"devices; {selected.available_devices} available"
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

    if any("No feasible GPU configuration" in reason for reason in reasons):
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


def _candidate_key(
    candidate: CandidateEvaluation,
    goal: OptimizationGoal,
) -> tuple[float, float, int, str]:
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
        return (ttft, tpot, devices, candidate.pool_id)
    return (-throughput, latency, devices, candidate.pool_id)
