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
        "trace_version": 4,
        "run_id": run_id,
        "scenario": scenario,
        "benchmark": benchmark,
        "method": method,
        "started_at": started_at,
        "tick_duration_seconds": tick_duration_seconds,
        "admission_definition": "first_active_rank_materialized",
    }


def tick_started(tick: int, started_at: str) -> dict[str, object]:
    """Build the logical control-window start event."""
    return {
        "kind": "tick_started",
        "tick": tick,
        "started_at": started_at,
    }


def resource_snapshot(tick: int, pools: list[dict[str, object]]) -> dict[str, object]:
    """Build one immutable ResourceMap pool snapshot for a control tick."""
    return {
        "kind": "resource_snapshot",
        "tick": tick,
        "pools": pools,
    }


def placement_observed(
    tick: int,
    plan_id: str | None,
    job_id: str,
    rank_id: str,
    n_replicas: int,
    shape: dict[str, object],
) -> dict[str, object]:
    """Build one post-apply canonical rank placement observation."""
    return {
        "kind": "placement_observed",
        "tick": tick,
        "plan_id": plan_id,
        "job_id": job_id,
        "rank_id": rank_id,
        "n_replicas": n_replicas,
        "allocated_gpu_count": shape.get("count"),
        "shape": shape,
    }


def planner_finished(
    tick: int | None,
    started_at: str,
    finished_at: str,
    decision_time_ms: float | None,
    plans: list[dict[str, object]],
    planned_job_ids: tuple[str, ...],
    timed_out: bool,
    error: str | None,
) -> dict[str, object]:
    """Build one planner outcome event for a completed control tick."""
    suggested_action_count = 0
    for plan in plans:
        actions = plan.get("actions")
        if isinstance(actions, list):
            suggested_action_count += sum(
                action.get("type") in {"place", "swap"}
                for action in actions
                if isinstance(action, dict)
            )
    return {
        "kind": "planner_finished",
        "tick": tick,
        "planner_started_at": started_at,
        "planner_finished_at": finished_at,
        "decision_time_ms": decision_time_ms,
        "plans": plans,
        "planned_job_ids": list(planned_job_ids),
        "timed_out": timed_out,
        "suggested_action_count": suggested_action_count,
        "error": error,
    }


def plan_validation(
    tick: int,
    plan_id: str,
    applied_action_count: int,
    selected_plan_valid: bool | None,
) -> dict[str, object]:
    """Build one post-apply plan outcome after rank simulation."""
    return {
        "kind": "plan_validation",
        "tick": tick,
        "plan_id": plan_id,
        "applied_action_count": applied_action_count,
        "selected_plan_valid": selected_plan_valid,
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


def job_state_changed(
    job_id: str,
    from_status: str | None,
    to_status: str,
    at: str,
    reason: str,
) -> dict[str, object]:
    """Build one logical job lifecycle transition for queue-time accounting."""
    return {
        "kind": "job_state_changed",
        "job_id": job_id,
        "from_status": from_status,
        "to_status": to_status,
        "at": at,
        "reason": reason,
    }


def rank_failed(
    job_id: str,
    rank_id: str,
    failed_at: str,
    reason: str,
    released_gpu_count: int | None,
    error_detail: str,
) -> dict[str, object]:
    """Build one rank-local failure with its service and capacity effects."""
    return {
        "kind": "rank_failed",
        "job_id": job_id,
        "rank_id": rank_id,
        "failed_at": failed_at,
        "reason": reason,
        "service_impact": "degraded",
        "released_gpu_count": released_gpu_count,
        "error_detail": error_detail,
    }


def rank_window(
    tick: int,
    job_id: str,
    rank_id: str,
    chain_indices: tuple[int, ...],
    window_start: str,
    window_end: str,
    v_actual: dict[str, list[float | None]],
    y_actual: dict[str, list[float | None]],
    v_hat: dict[str, object] | None = None,
    y_hat: dict[str, object] | None = None,
    telemetry_scope: str = "observed_tick",
    projection_method: str | None = None,
    sample_bucket_starts: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build one per-rank telemetry trajectory for a completed tick window."""
    return {
        "kind": "rank_window",
        "tick": tick,
        "job_id": job_id,
        "rank_id": rank_id,
        "chain_indices": list(chain_indices),
        "window_start": window_start,
        "window_end": window_end,
        "v_hat": v_hat,
        "y_hat": y_hat,
        "v_actual": v_actual,
        "y_actual": y_actual,
        "telemetry_scope": telemetry_scope,
        "projection_method": projection_method,
        "sample_bucket_starts": list(sample_bucket_starts),
        "aggregation_policy_version": 2,
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
    attempted_request_count: int = 0,
    completed_request_count: int = 0,
    unfinished_request_count: int = 0,
    completion_ratio: float | None = None,
    latency_scope: str = "completed_requests",
    slo_pass_count: int | None = None,
    slo_fail_count: int | None = None,
    slo_count_scope: str = "completed_requests",
    ttft_pass_count: int | None = None,
    ttft_fail_count: int | None = None,
    tpot_pass_count: int | None = None,
    tpot_fail_count: int | None = None,
    offered_request_count: int | None = None,
    admitted_request_count: int | None = None,
    qualified_output_tokens: int | None = None,
    request_count_scope: str = "tick_requests",
    batch_latency_diagnostic: bool = False,
    probe_attempted_request_count: int | None = None,
    probe_completed_request_count: int | None = None,
    probe_unfinished_request_count: int | None = None,
    campaign_total_request_count: int | None = None,
    campaign_completed_in_window: int | None = None,
    campaign_completed_request_count: int | None = None,
    campaign_remaining_request_count: int | None = None,
    campaign_request_per_sec: float | None = None,
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
        "attempted_request_count": attempted_request_count,
        "completed_request_count": completed_request_count,
        "unfinished_request_count": unfinished_request_count,
        "completion_ratio": completion_ratio,
        "latency_scope": latency_scope,
        "slo_pass_count": slo_pass_count,
        "slo_fail_count": slo_fail_count,
        "slo_count_scope": slo_count_scope,
        "ttft_pass_count": ttft_pass_count,
        "ttft_fail_count": ttft_fail_count,
        "tpot_pass_count": tpot_pass_count,
        "tpot_fail_count": tpot_fail_count,
        "offered_request_count": offered_request_count,
        "admitted_request_count": admitted_request_count,
        "qualified_output_tokens": qualified_output_tokens,
        "request_count_scope": request_count_scope,
        "batch_latency_diagnostic": batch_latency_diagnostic,
        "probe_attempted_request_count": probe_attempted_request_count,
        "probe_completed_request_count": probe_completed_request_count,
        "probe_unfinished_request_count": probe_unfinished_request_count,
        "campaign_total_request_count": campaign_total_request_count,
        "campaign_completed_in_window": campaign_completed_in_window,
        "campaign_completed_request_count": campaign_completed_request_count,
        "campaign_remaining_request_count": campaign_remaining_request_count,
        "campaign_request_per_sec": campaign_request_per_sec,
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
    jobs_admitted_in_tick: int = 0,
    admission_count_so_far: int = 0,
    rolling_admission_time_ms: float | None = None,
) -> dict[str, object]:
    """Build aggregate GPU-hour and cost facts across every physical pool."""
    capacity_gpu_seconds = capacity_gpu_hours * 3_600
    allocated_gpu_seconds = allocated_gpu_hours * 3_600
    return {
        "kind": "cluster_window",
        "tick": tick,
        "window_start": window_start,
        "window_end": window_end,
        "capacity_gpu_hours": capacity_gpu_hours,
        "allocated_gpu_hours": allocated_gpu_hours,
        "capacity_gpu_seconds": capacity_gpu_seconds,
        "allocated_gpu_seconds": allocated_gpu_seconds,
        "busy_gpu_seconds": allocated_gpu_seconds,
        "capacity_cost": capacity_cost,
        "allocated_cost": allocated_cost,
        "jobs_admitted_in_tick": jobs_admitted_in_tick,
        "admission_count_so_far": admission_count_so_far,
        "rolling_admission_time_ms": rolling_admission_time_ms,
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
    unserved_request_count: int = 0,
    *,
    deadline_outcome: str | None = None,
    offered_request_count: int = 0,
    completed_request_count: int = 0,
    unfinished_request_count: int = 0,
    ttft_pass_count: int | None = None,
    tpot_pass_count: int | None = None,
    slo_pass_count: int | None = None,
    qualified_output_tokens: int | None = None,
    slo_count_scope: str = "all_offered_requests",
) -> dict[str, object]:
    """Build one terminal job fact for JCT and batch deadline derivation."""
    return {
        "kind": "job_finished",
        "job_id": job_id,
        "finished_at": finished_at,
        "completed_successfully": completed_successfully,
        "finish_reason": finish_reason,
        "deadline_slo_met": deadline_slo_met,
        "deadline_outcome": deadline_outcome,
        "unserved_request_count": unserved_request_count,
        "offered_request_count": offered_request_count,
        "completed_request_count": completed_request_count,
        "unfinished_request_count": unfinished_request_count,
        "ttft_pass_count": ttft_pass_count,
        "tpot_pass_count": tpot_pass_count,
        "slo_pass_count": slo_pass_count,
        "qualified_output_tokens": qualified_output_tokens,
        "slo_count_scope": slo_count_scope,
    }


def run_finished(
    finished_at: str,
    termination_reason: str | None = None,
    truncated: bool = False,
    status: str = "completed",
) -> dict[str, object]:
    """Build the terminal event for one completed experiment trace."""
    return {
        "kind": "run_finished",
        "finished_at": finished_at,
        "status": status,
        "termination_reason": termination_reason,
        "truncated": truncated,
    }


def gpu_sample(
    tick: int,
    timestamp: str,
    gpu_uuid: str,
    job_id: str | None,
    rank_id: str | None,
    chain_index: int | None,
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
        "chain_index": chain_index,
        "node_name": node_name,
        "instance_type": instance_type,
        "metrics": metrics,
    }
