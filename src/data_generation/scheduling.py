"""Shared placement and resource contracts for benchmark scheduler methods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import yaml


class OptimizationGoal(StrEnum):
    """Metric used to compare feasible hardware candidates."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"


class CandidateStatus(StrEnum):
    """Outcome of evaluating one hardware type."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class DecisionStatus(StrEnum):
    """Final outcome of one scheduling attempt."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


class DecisionFailureCategory(StrEnum):
    """Why a scheduler could not place a job."""

    INSUFFICIENT_CAPACITY = "insufficient_capacity"
    SLO_INFEASIBLE = "slo_infeasible"
    LOAD_INFEASIBLE = "load_infeasible"
    MODEL_UNSUPPORTED = "model_unsupported"
    MODEL_UNAVAILABLE = "model_unavailable"
    HARDWARE_UNSUPPORTED = "hardware_unsupported"
    METHOD_ERROR = "method_error"


@dataclass(frozen=True)
class SchedulerWorkload:
    """Actual traffic shape emitted by one benchmark job."""

    input_tokens: float
    output_tokens: float
    request_rate: float
    max_concurrency: int | None = None


@dataclass(frozen=True)
class SchedulerJob:
    """Scheduler-visible job data, independent of its traffic generator."""

    job_id: str
    kind: str
    spec: Mapping[str, object]
    workload: SchedulerWorkload
    optimization_goal: OptimizationGoal | None = None


@dataclass(frozen=True)
class HardwarePool:
    """One homogeneous accelerator pool in a benchmark resource map."""

    pool_id: str
    accelerator: str
    total_devices: int
    devices_per_node: int
    memory_gb: int
    aic_system: str | None
    regions: Mapping[str, int]
    unsupported_reason: str | None = None


class ResourceInventory:
    """Mutable available capacity and per-job allocations for one run."""

    def __init__(self, pools: tuple[HardwarePool, ...]) -> None:
        self.pools = pools
        self._available = {pool.pool_id: pool.total_devices for pool in pools}
        self._allocations: dict[str, tuple[str, int]] = {}

    def available(self, pool_id: str) -> int:
        """Return currently unallocated devices in one pool."""
        return self._available[pool_id]

    def reserve(self, job_id: str, pool_id: str, devices: int) -> None:
        """Atomically reserve devices for one accepted job."""
        if job_id in self._allocations:
            raise ValueError(f"Job already has an allocation: {job_id}")
        if devices <= 0:
            raise ValueError("Allocation devices must be positive")
        available = self.available(pool_id)
        if devices > available:
            raise ValueError(f"Pool {pool_id} has {available} devices, not {devices}")
        self._available[pool_id] = available - devices
        self._allocations[job_id] = (pool_id, devices)

    def release(self, job_id: str) -> None:
        """Return a completed job's allocation to its pool."""
        allocation = self._allocations.pop(job_id, None)
        if allocation is None:
            return
        pool_id, devices = allocation
        self._available[pool_id] += devices

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serializable view of current cluster capacity."""
        allocations = {
            job_id: {"pool_id": pool_id, "devices": devices}
            for job_id, (pool_id, devices) in sorted(self._allocations.items())
        }
        pools = [
            {
                "pool_id": pool.pool_id,
                "accelerator": pool.accelerator,
                "total_devices": pool.total_devices,
                "available_devices": self.available(pool.pool_id),
                "allocated_devices": (
                    pool.total_devices - self.available(pool.pool_id)
                ),
            }
            for pool in self.pools
        ]
        return {"pools": pools, "allocations": allocations}


@dataclass(frozen=True)
class CandidateEvaluation:
    """Normalized AIC result for one hardware type."""

    pool_id: str
    accelerator: str
    status: CandidateStatus
    available_devices: int
    requested_devices: int | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    throughput_tokens_per_second: float | None = None
    configuration: Mapping[str, object] | None = None
    aic_request: Mapping[str, object] | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return this candidate as a JSON object."""
        return {
            "pool_id": self.pool_id,
            "accelerator": self.accelerator,
            "status": self.status.value,
            "available_devices": self.available_devices,
            "requested_devices": self.requested_devices,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "throughput_tokens_per_second": (self.throughput_tokens_per_second),
            "configuration": self.configuration,
            "aic_request": self.aic_request,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SchedulingDecision:
    """One scheduler method's accepted placement or rejection."""

    job_id: str
    method: str
    status: DecisionStatus
    optimization_goal: OptimizationGoal
    candidates: tuple[CandidateEvaluation, ...]
    resource_snapshot_before: Mapping[str, object]
    resource_snapshot_after: Mapping[str, object]
    pool_id: str | None = None
    accelerator: str | None = None
    devices: int | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    throughput_tokens_per_second: float | None = None
    configuration: Mapping[str, object] | None = None
    failure_category: DecisionFailureCategory | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return this decision as a JSON object."""
        return {
            "job_id": self.job_id,
            "method": self.method,
            "status": self.status.value,
            "optimization_goal": self.optimization_goal.value,
            "pool_id": self.pool_id,
            "accelerator": self.accelerator,
            "devices": self.devices,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "throughput_tokens_per_second": (self.throughput_tokens_per_second),
            "configuration": self.configuration,
            "failure_category": (
                None if self.failure_category is None else self.failure_category.value
            ),
            "reason": self.reason,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "resource_snapshot_before": self.resource_snapshot_before,
            "resource_snapshot_after": self.resource_snapshot_after,
        }


class SchedulingMethod(Protocol):
    """Controller boundary implemented by each scheduler method."""

    method_name: str

    def schedule(self, job: SchedulerJob) -> SchedulingDecision:
        """Choose and reserve a placement for one arriving job."""

    def release(self, job_id: str) -> None:
        """Release a terminal job's allocation."""

    def resource_snapshot(self) -> Mapping[str, object]:
        """Return the method's current cluster view."""


def load_resource_inventory(path: Path) -> ResourceInventory:
    """Load a benchmark's frozen hardware catalog and initial capacity."""
    document = _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        "resource map",
    )
    if document.get("schema_version") != "resource-map/v1":
        raise ValueError("Unsupported resource-map schema_version")
    resources = document.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ValueError("resource map resources must be a non-empty array")

    pools: list[HardwarePool] = []
    pool_ids: set[str] = set()
    for index, value in enumerate(resources):
        definition = _mapping(value, f"resources[{index}]")
        pool_id = _text(definition.get("id"), f"resources[{index}].id")
        if pool_id in pool_ids:
            raise ValueError(f"Duplicate resource pool: {pool_id}")
        pool_ids.add(pool_id)
        total_devices = _positive_int(
            definition.get("total_devices"),
            f"resources[{index}].total_devices",
        )
        regions_value = _mapping(
            definition.get("regions"),
            f"resources[{index}].regions",
        )
        regions = {
            _text(region, f"resources[{index}].regions key"): _nonnegative_int(
                count,
                f"resources[{index}].regions.{region}",
            )
            for region, count in regions_value.items()
        }
        if sum(regions.values()) != total_devices:
            raise ValueError(f"Regions do not total {total_devices} for {pool_id}")
        aic_system_value = definition.get("aic_system")
        aic_system = (
            None
            if aic_system_value is None
            else _text(aic_system_value, f"resources[{index}].aic_system")
        )
        reason_value = definition.get("unsupported_reason")
        unsupported_reason = (
            None
            if reason_value is None
            else _text(reason_value, f"resources[{index}].unsupported_reason")
        )
        pools.append(
            HardwarePool(
                pool_id=pool_id,
                accelerator=_text(
                    definition.get("accelerator"),
                    f"resources[{index}].accelerator",
                ),
                total_devices=total_devices,
                devices_per_node=_positive_int(
                    definition.get("devices_per_node"),
                    f"resources[{index}].devices_per_node",
                ),
                memory_gb=_positive_int(
                    definition.get("memory_gb"),
                    f"resources[{index}].memory_gb",
                ),
                aic_system=aic_system,
                regions=regions,
                unsupported_reason=unsupported_reason,
            )
        )
    return ResourceInventory(tuple(pools))


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value
