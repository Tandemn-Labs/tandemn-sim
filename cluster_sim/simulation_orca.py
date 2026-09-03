"""Store-only job normalization and plan application for cluster simulation."""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from tandemn_system_data.clients import JobStore, PlanStore
from tandemn_system_data.models.enums import (
    ActionType,
    JobKind,
    JobStatus,
    RankRole,
    RankStatus,
    ReasonCode,
)
from tandemn_system_data.models.plan import Plan, PlanAction
from tandemn_system_data.models.rank import Rank

LOGGER = logging.getLogger(__name__)

WORKLOAD_DEFAULTS = {
    "isl_token_avg": 4000,
    "isl_token_min": 1,
    "isl_token_max": 8000,
    "isl_distribution_type": "LogNormal",
    "osl_token_avg": 1000,
    "osl_token_min": 1,
    "osl_token_max": 4000,
    "osl_distribution_type": "LogNormal",
    "request_arrival_rate": 1.0,
    "request_arrival_pattern": "Poisson",
    "peak_to_mean_ratio": 2,
    "workload_prefix_concentration": 0,
    "multi_turn_ratio": 0.5,
    "shared_prefix_length_avg": 500,
    "sliding_window_attention": 0,
    "is_session_affinity": True,
    "total_token_budget": 1_000_000,
    "priority_class": "STANDARD",
    "max_concurrent_streaming": 10,
    "preemption_policy": "lifo",
    "router_policy": "kv_router",
}
TRAFFIC_ALIASES = {
    "requestRate": "request_arrival_rate",
    "concurrency": "max_concurrent_streaming",
}
REQUEST_RATE_KEYS = frozenset({"requestRate", "request_arrival_rate"})
CONCURRENCY_KEYS = frozenset({"concurrency", "max_concurrent_streaming"})
USER_FEATURE_KEYS = frozenset(
    {
        *WORKLOAD_DEFAULTS,
        *TRAFFIC_ALIASES,
        "deadline_hrs",
        "target_p99_ttft_ms",
        "target_p99_tpot_ms",
    }
)
DIST_TYPES = {"Constant", "Uniform", "Normal", "LogNormal", "Exponential"}
ARRIVAL_PATTERNS = {"Poisson", "Constant", "Bursty", "Diurnal"}
PRIORITY_CLASSES = {"HIGH", "STANDARD", "LOW"}
RANK_ID_RE = re.compile(r"rank_[0-7][0-9A-HJKMNP-TV-Z]{25}\Z")


def normalize_job_spec(spec: dict[str, Any], kind: JobKind) -> dict[str, Any]:
    """Return Store-compatible workload metadata for one simulated job."""
    model_id = spec.get("model_id")
    if not model_id:
        raise ValueError("job spec must include model_id")

    provided_request_rate = _has_any_user_key(spec, REQUEST_RATE_KEYS)
    provided_concurrency = _has_any_user_key(spec, CONCURRENCY_KEYS)
    if provided_request_rate and provided_concurrency:
        raise ValueError("job spec must provide only one traffic mode")

    features: dict[str, Any] = dict(WORKLOAD_DEFAULTS)
    if isinstance(spec.get("job_features"), dict):
        features.update(_canonical_user_features(spec["job_features"]))
    features.update(_canonical_user_features(spec))
    features["model_id"] = model_id
    features["type"] = kind.value
    _canonicalize_runtime_policy(features)

    if kind is JobKind.ONLINE:
        for key in ("target_p99_ttft_ms", "target_p99_tpot_ms"):
            if features.get(key) is None:
                raise ValueError(f"job spec missing required field {key}")
        features.setdefault("deadline_hrs", 0)
        _derive_online_traffic(features, provided_request_rate, provided_concurrency)
    else:
        deadline = features.get("deadline_hrs")
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or deadline <= 0
        ):
            raise ValueError("job spec deadline_hrs must be a positive number")

    features["pd_ratio"] = float(features["isl_token_avg"]) / float(
        features["osl_token_avg"]
    )
    _validate_features(features)
    return {**spec, "model_id": model_id, "job_features": features}


def apply_pending_plans(user_id: str, postgres: object) -> int:
    """Apply pending Store plans without deployment or RPC side effects."""
    jobs = JobStore(postgres)
    plans = PlanStore(postgres)
    applied = 0
    for plan in plans.unapplied(user_id):
        for action in plan.actions:
            try:
                _apply_action(jobs, plan, action)
            except Exception:
                LOGGER.exception(
                    "simulation action %s for job %s failed (plan %s)",
                    action.type,
                    action.job_id,
                    plan.plan_id,
                )
        if plans.mark_applied(plan.plan_id):
            applied += 1
    return applied


def ladder_to_ranks(
    ladder: list[dict[str, Any]] | None,
    *,
    job_id: str,
    plan_id: str,
    target_p99_ttft_ms: float | None = None,
    target_p99_tpot_ms: float | None = None,
    job_spec: dict[str, Any] | None = None,
) -> list[Rank]:
    """Convert valid Store plan ladder entries to canonical rank rows."""
    job_spec = job_spec or {}
    ranks = []
    seen_rank_ids = set()
    for entry in ladder or []:
        if not isinstance(entry, dict):
            continue
        try:
            role = RankRole(entry.get("role"))
        except ValueError:
            continue
        config = entry.get("config")
        count = (
            config.get("gpu_count", config.get("count"))
            if isinstance(config, dict)
            else None
        )
        replicas = entry.get("n_replicas", entry.get("chains", 1))
        rank_id = entry.get("rank_id")
        if (
            type(count) is not int
            or count <= 0
            or type(replicas) is not int
            or replicas <= 0
            or not isinstance(rank_id, str)
            or RANK_ID_RE.fullmatch(rank_id) is None
        ):
            continue
        if rank_id in seen_rank_ids:
            raise ValueError(f"duplicate rank_id {rank_id}")
        seen_rank_ids.add(rank_id)

        shape = dict(config)
        shape["count"] = count
        shape.setdefault("sp", 1)
        shape.setdefault("ep", 1)
        shape.setdefault("cp", 1)
        env = entry.get("env")
        if env is not None:
            shape["env"] = list(env) if isinstance(env, (list, tuple)) else env
        if entry.get("mechanism_id") is not None:
            shape["mechanism_id"] = entry["mechanism_id"]
        for key in ("predicted_y", "predicted_v"):
            if isinstance(entry.get(key), dict):
                shape[key] = dict(entry[key])
        if target_p99_ttft_ms is not None:
            shape["target_p99_ttft_ms"] = target_p99_ttft_ms
        if target_p99_tpot_ms is not None:
            shape["target_p99_tpot_ms"] = target_p99_tpot_ms
        if (
            not shape.get("gpu_type")
            and isinstance(env, (list, tuple))
            and len(env) >= 5
        ):
            shape["gpu_type"] = str(env[4])
        if not shape.get("model_id") and job_spec.get("model_id"):
            shape["model_id"] = str(job_spec["model_id"])
        shape.setdefault("engine_name", "vllm")
        if not all(shape.get(key) for key in ("instance_type", "gpu_type", "model_id")):
            continue
        ranks.append(
            Rank(
                rank_id=rank_id,
                job_id=job_id,
                plan_id=plan_id,
                role=role,
                shape_json=shape,
                n_replicas=replicas,
            )
        )
    return ranks


def _apply_action(jobs: JobStore, plan: Plan, action: PlanAction) -> None:
    if action.type is ActionType.PLACE:
        _place(jobs, plan, action)
    elif action.type is ActionType.SWAP:
        _swap(jobs, plan, action)
    elif action.type is ActionType.PREEMPT:
        _preempt(jobs, action)


def _place(jobs: JobStore, plan: Plan, action: PlanAction) -> None:
    job = jobs.get(action.job_id)
    ranks = _materialize_ranks(plan, action, job.spec_json if job else None)
    previous_status = (
        job.status
        if job and job.status in (JobStatus.WAITING, JobStatus.PAUSED)
        else None
    )
    if not jobs.transition(
        action.job_id,
        JobStatus.RUNNING,
        [JobStatus.WAITING, JobStatus.PAUSED],
    ):
        raise ValueError(f"place: job {action.job_id} is not waiting or paused")
    try:
        jobs.launch_ranks(ranks)
    except Exception:
        if previous_status is not None:
            jobs.transition(action.job_id, previous_status, [JobStatus.RUNNING])
        raise


def _swap(jobs: JobStore, plan: Plan, action: PlanAction) -> None:
    job = jobs.get(action.job_id)
    if job is None or job.status is not JobStatus.RUNNING:
        raise ValueError(f"swap: job {action.job_id} is not running")
    old_ranks = jobs.active_ranks(action.job_id)
    old_by_id = {rank.rank_id: rank for rank in old_ranks}
    ranks = _materialize_ranks(plan, action, job.spec_json)
    ranks = [
        rank.model_copy(
            update={
                "status": old_by_id[rank.rank_id].status,
                "reason_code": old_by_id[rank.rank_id].reason_code,
            }
        )
        if rank.rank_id in old_by_id
        else rank
        for rank in ranks
    ]
    jobs.launch_ranks(ranks)
    _stop_ranks(jobs, action.job_id, set(old_by_id) - {rank.rank_id for rank in ranks})


def _preempt(jobs: JobStore, action: PlanAction) -> None:
    _stop_ranks(
        jobs,
        action.job_id,
        {rank.rank_id for rank in jobs.active_ranks(action.job_id)},
        ReasonCode.PREEMPTED,
    )
    jobs.transition(action.job_id, JobStatus.PAUSED, [JobStatus.RUNNING])


def _stop_ranks(
    jobs: JobStore,
    job_id: str,
    rank_ids: set[str],
    reason_code: str | None = None,
) -> None:
    # ponytail: ranks are Store intent only; ServingSim owns simulated deployment state.
    for rank_id in rank_ids:
        jobs.set_rank_status(
            rank_id,
            RankStatus.STOPPED,
            [RankStatus.LAUNCHING, RankStatus.RUNNING],
            reason_code=reason_code,
        )


def _materialize_ranks(
    plan: Plan, action: PlanAction, job_spec: dict[str, Any] | None
) -> list[Rank]:
    ranks = ladder_to_ranks(
        action.ladder,
        job_id=action.job_id,
        plan_id=plan.plan_id,
        target_p99_ttft_ms=action.target_p99_ttft_ms,
        target_p99_tpot_ms=action.target_p99_tpot_ms,
        job_spec=job_spec,
    )
    if not ranks:
        raise ValueError(
            f"no launchable ranks for job {action.job_id} (plan {plan.plan_id})"
        )
    return ranks


def _canonical_user_features(source: dict[str, Any]) -> dict[str, Any]:
    return {
        TRAFFIC_ALIASES.get(key, key): value
        for key, value in source.items()
        if key in USER_FEATURE_KEYS
    }


def _has_any_user_key(spec: dict[str, Any], keys: frozenset[str]) -> bool:
    if any(key in spec for key in keys):
        return True
    nested = spec.get("job_features")
    return isinstance(nested, dict) and any(key in nested for key in keys)


def _derive_online_traffic(
    features: dict[str, Any],
    provided_request_rate: bool,
    provided_concurrency: bool,
) -> None:
    if not provided_request_rate and not provided_concurrency:
        features["_traffic_mode"] = "request_rate"
        return
    service_time_s = _target_service_time_seconds(features)
    if provided_request_rate:
        features["_traffic_mode"] = "request_rate"
        request_rate = _positive_float(features, "request_arrival_rate")
        features["request_arrival_rate"] = request_rate
        features["max_concurrent_streaming"] = max(
            1, math.ceil(request_rate * service_time_s)
        )
        return
    features["_traffic_mode"] = "concurrency"
    concurrency = max(
        1, math.ceil(_positive_float(features, "max_concurrent_streaming"))
    )
    features["max_concurrent_streaming"] = concurrency
    features["request_arrival_rate"] = concurrency / service_time_s


def _target_service_time_seconds(features: dict[str, Any]) -> float:
    return (
        _positive_float(features, "target_p99_ttft_ms")
        + _positive_float(features, "target_p99_tpot_ms")
        * _positive_float(features, "osl_token_avg")
    ) / 1_000


def _positive_float(features: dict[str, Any], key: str) -> float:
    value = features.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"job spec field {key} must be a positive number")
    return float(value)


def _canonicalize_runtime_policy(features: dict[str, Any]) -> None:
    if features.get("preemption_policy") not in {"lifo", "fifo"}:
        features["preemption_policy"] = "lifo"
    if features.get("router_policy") != "kv_router":
        features["router_policy"] = "kv_router"


def _validate_features(features: dict[str, Any]) -> None:
    for key in (
        "isl_token_avg",
        "isl_token_min",
        "isl_token_max",
        "osl_token_avg",
        "osl_token_min",
        "osl_token_max",
        "request_arrival_rate",
        "max_concurrent_streaming",
    ):
        _positive_float(features, key)
    for key, allowed in (
        ("isl_distribution_type", DIST_TYPES),
        ("osl_distribution_type", DIST_TYPES),
        ("request_arrival_pattern", ARRIVAL_PATTERNS),
        ("priority_class", PRIORITY_CLASSES),
    ):
        if features[key] not in allowed:
            raise ValueError(f"job spec {key} must be one of {sorted(allowed)}")
