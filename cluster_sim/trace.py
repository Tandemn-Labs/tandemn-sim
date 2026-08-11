"""Write experiment trace artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def trace_path(
    root: Path,
    scenario: str,
    benchmark: str,
    method: str,
    run_id: str,
) -> Path:
    """Return the JSONL trace location for one experiment run."""
    return root / scenario / benchmark / method / f"{run_id}.jsonl"


def append_event(path: Path, event: dict[str, object]) -> None:
    """Append one JSON event to a trace, creating its parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def read_events(path: Path) -> tuple[dict[str, object], ...]:
    """Read every event from one JSONL trace in append order."""
    with path.open(encoding="utf-8") as trace_file:
        return tuple(json.loads(line) for line in trace_file if line.strip())


def run_started(
    run_id: str,
    scenario: str,
    benchmark: str,
    method: str,
    started_at: str,
    tick_duration_seconds: float,
) -> dict[str, object]:
    """Build the immutable first event for one experiment trace."""
    return {
        "kind": "run_started",
        "trace_version": 1,
        "run_id": run_id,
        "scenario": scenario,
        "benchmark": benchmark,
        "method": method,
        "started_at": started_at,
        "tick_duration_seconds": tick_duration_seconds,
        "admission_definition": "first_place_plan_created",
    }


def planner_finished(
    tick: int,
    started_at: str,
    finished_at: str,
    decision_time_ms: float,
    plans: list[dict[str, object]],
    planned_job_ids: tuple[str, ...],
    timed_out: bool | None,
    error: str | None,
) -> dict[str, object]:
    """Build one planner outcome event for a completed control tick."""
    return {
        "kind": "planner_finished",
        "tick": tick,
        "planner_started_at": started_at,
        "planner_finished_at": finished_at,
        "decision_time_ms": decision_time_ms,
        "plans": plans,
        "planned_job_ids": list(planned_job_ids),
        "timed_out": timed_out,
        "error": error,
    }


def job_admitted(
    job_id: str,
    plan_id: str,
    submitted_at: str,
    admitted_at: str,
    admission_source: str,
) -> dict[str, object]:
    """Build the first-PLACE admission event used for queue-time analysis."""
    return {
        "kind": "job_admitted",
        "job_id": job_id,
        "plan_id": plan_id,
        "submitted_at": submitted_at,
        "admitted_at": admitted_at,
        "admission_source": admission_source,
        "action_type": "place",
    }


def rank_window(
    tick: int,
    job_id: str,
    rank_id: str,
    chain_ids: tuple[str, ...],
    window_start: str,
    window_end: str,
    v_actual: dict[str, list[float | None]],
    y_actual: dict[str, list[float | None]],
    v_hat: dict[str, object] | None = None,
    y_hat: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one per-rank telemetry trajectory for a completed tick window."""
    return {
        "kind": "rank_window",
        "tick": tick,
        "job_id": job_id,
        "rank_id": rank_id,
        "chain_ids": list(chain_ids),
        "window_start": window_start,
        "window_end": window_end,
        "v_hat": v_hat,
        "y_hat": y_hat,
        "v_actual": v_actual,
        "y_actual": y_actual,
        "aggregation_policy_version": 1,
    }


def job_window(
    tick: int,
    job_id: str,
    job_kind: str,
    window_start: str,
    window_end: str,
    p99_ttft_ms: float | None,
    p99_tpot_ms: float | None,
    throughput_token_per_sec: float | None,
    cost_per_token: float | None,
    target_p99_ttft_ms: float | None,
    target_p99_tpot_ms: float | None,
    ttft_slo_met: bool | None,
    tpot_slo_met: bool | None,
) -> dict[str, object]:
    """Build one job-performance observation, derived from its rank telemetry."""
    return {
        "kind": "job_window",
        "tick": tick,
        "job_id": job_id,
        "job_kind": job_kind,
        "window_start": window_start,
        "window_end": window_end,
        "p99_ttft_ms": p99_ttft_ms,
        "p99_tpot_ms": p99_tpot_ms,
        "throughput_token_per_sec": throughput_token_per_sec,
        "cost_per_token": cost_per_token,
        "target_p99_ttft_ms": target_p99_ttft_ms,
        "target_p99_tpot_ms": target_p99_tpot_ms,
        "ttft_slo_met": ttft_slo_met,
        "tpot_slo_met": tpot_slo_met,
    }


def pool_window(
    tick: int,
    pool_id: str,
    expected_gpu_count: int,
    reported_gpu_count: int,
    allocated_gpu_count: int,
    physical_sm_utilization_percent: float,
    telemetry_coverage_percent: float,
) -> dict[str, object]:
    """Build one physical GPU-pool capacity and telemetry-coverage observation."""
    missing_gpu_count = expected_gpu_count - reported_gpu_count
    return {
        "kind": "pool_window",
        "tick": tick,
        "pool_id": pool_id,
        "expected_gpu_count": expected_gpu_count,
        "reported_gpu_count": reported_gpu_count,
        "missing_gpu_count": missing_gpu_count,
        "zero_filled_gpu_count": missing_gpu_count,
        "allocated_gpu_count": allocated_gpu_count,
        "available_gpu_count": expected_gpu_count - allocated_gpu_count,
        "physical_sm_utilization_percent": physical_sm_utilization_percent,
        "telemetry_coverage_percent": telemetry_coverage_percent,
    }


def cluster_window(
    tick: int,
    window_start: str,
    window_end: str,
    capacity_gpu_hours: float,
    allocated_gpu_hours: float,
    capacity_cost: float,
    allocated_cost: float,
) -> dict[str, object]:
    """Build aggregate GPU-hour and cost facts across every physical pool."""
    return {
        "kind": "cluster_window",
        "tick": tick,
        "window_start": window_start,
        "window_end": window_end,
        "capacity_gpu_hours": capacity_gpu_hours,
        "allocated_gpu_hours": allocated_gpu_hours,
        "capacity_cost": capacity_cost,
        "allocated_cost": allocated_cost,
    }


def job_submitted(
    job_id: str,
    job_kind: str,
    submitted_at: str,
    tenant_id: str | None,
    target_p99_ttft_ms: float | None,
    target_p99_tpot_ms: float | None,
    deadline_at: str | None,
) -> dict[str, object]:
    """Build immutable workload facts for one submitted online or batch job."""
    return {
        "kind": "job_submitted",
        "job_id": job_id,
        "job_kind": job_kind,
        "submitted_at": submitted_at,
        "tenant_id": tenant_id,
        "target_p99_ttft_ms": target_p99_ttft_ms,
        "target_p99_tpot_ms": target_p99_tpot_ms,
        "deadline_at": deadline_at,
    }


def job_finished(
    job_id: str,
    finished_at: str,
    completed_successfully: bool,
    finish_reason: str | None,
    deadline_slo_met: bool | None,
) -> dict[str, object]:
    """Build one terminal job fact for JCT and batch deadline derivation."""
    return {
        "kind": "job_finished",
        "job_id": job_id,
        "finished_at": finished_at,
        "completed_successfully": completed_successfully,
        "finish_reason": finish_reason,
        "deadline_slo_met": deadline_slo_met,
    }


def gpu_sample(
    tick: int,
    timestamp: str,
    gpu_uuid: str,
    job_id: str | None,
    rank_id: str | None,
    chain_id: str | None,
    node_name: str | None,
    instance_type: str | None,
    metrics: dict[str, float | None],
) -> dict[str, object]:
    """Build one raw Store GPU sample, including measured idle GPUs."""
    return {
        "kind": "gpu_sample",
        "tick": tick,
        "ts": timestamp,
        "gpu_uuid": gpu_uuid,
        "job_id": job_id,
        "rank_id": rank_id,
        "chain_id": chain_id,
        "node_name": node_name,
        "instance_type": instance_type,
        "metrics": metrics,
    }


def koi_llm_call(
    tick: int,
    call_index: int,
    elapsed_ms: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    error: str | None,
) -> dict[str, object]:
    """Build one Koi LLM-call fact using only provider-reported token usage."""
    return {
        "kind": "koi_llm_call",
        "tick": tick,
        "call_index": call_index,
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "error": error,
    }
