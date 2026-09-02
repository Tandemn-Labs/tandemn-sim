"""Data shapes used by the standalone cluster simulator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from llm_usage import LlmUsage


@dataclass(frozen=True)
class OnlineJob:
    """One continuous online workload from benchmark start through offer stop."""

    name: str
    store_job_id: str
    model_name: str
    starts_at_ns: int
    stops_at_ns: int
    workload: dict[str, Any]


@dataclass(frozen=True)
class BatchJob:
    """One finite batch backlog with a release, deadline, and hard stop."""

    name: str
    store_job_id: str
    model_name: str
    releases_at_ns: int
    deadline_at_ns: int
    hard_stop_at_ns: int
    workload: dict[str, Any]


@dataclass(frozen=True)
class Scenario:
    """One benchmark timeline containing online and batch simulation jobs."""

    benchmark_name: str
    tick_duration_ns: int
    drain_end_ns: int
    online_jobs: tuple[OnlineJob, ...]
    batch_jobs: tuple[BatchJob, ...]
    environment: str = "simulation"


@dataclass(frozen=True)
class Request:
    """One offered model request routed by its stable session name."""

    request_name: str
    session_name: str
    turn_number: int
    arrives_at_ns: int
    input_tokens: int
    output_tokens: int
    prefix_key: str | None


@dataclass
class BatchProgress:
    """Track completion and deadline state for one batch backlog."""

    total_requests: int
    completed_requests: int = 0
    next_request_number: int = 0
    deadline_missed: bool = False
    completed_at_ns: int | None = None


@dataclass(frozen=True)
class ChainBinding:
    """Bind one canonical Store rank replica to one local simulation instance."""

    rank_id: str
    chain_index: int
    gpu_count: int


@dataclass(frozen=True)
class RankDeployment:
    """One independently simulated rank deployment for a Store job."""

    store_job_id: str
    rank_name: str
    model_name: str
    shape: dict[str, Any]
    chains: tuple[ChainBinding, ...]


@dataclass(frozen=True)
class RankMetricSample:
    """One LLMServingSim metric payload for one rank instance."""

    sim_time_ns: int
    sample_interval_ns: int
    instance_id: int
    metrics: dict[str, float | None]
    sample_kind: str = "periodic"


@dataclass(frozen=True)
class CreatedPlan:
    """Reference one Store plan created during a logical control tick."""

    plan_id: str


@dataclass(frozen=True)
class PlannerResult:
    """Record plan creation and any degraded planner outcome for one tick."""

    tick_number: int
    plans: tuple[CreatedPlan, ...]
    degraded: bool = False
    error: str | None = None
    llm_usage: tuple[LlmUsage, ...] = ()


@dataclass(frozen=True)
class SimulationResult:
    """Summarize one completed cluster simulation run."""

    started_at: datetime
    finished_at: datetime
    tick_count: int
    summary: dict[str, Any]
