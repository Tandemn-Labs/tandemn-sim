"""Aggregate Store GPU telemetry into Koi-compatible rank trajectories."""

from __future__ import annotations

import math
from datetime import datetime
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Mapping

# TODO HUGE technical debt, will be one place for this telemetry function in the future.
DEFAULT_SAMPLE_BUCKET_SEC = 10
DEFAULT_POLICY = "latest_mean"
METRIC_POLICIES = {
    "throughput_token_per_sec": "sum",
    "live_batch_size": "sum",
    "depth_req_q": "sum",
    "prefill_iteration_counts_per_second": "sum",
    "decode_itr_counts_per_second": "sum",
    "p99_ttft_ms": "max",
    "p99_tpot_ms": "max",
    "slo_margin": "min",
    "cost_per_token": "weighted_by_throughput",
    "kv_cache_util": "mean",
    "activation_mem_pressure": "mean",
    "gpu_mem_used_fraction": "mean",
    "vram_headroom_gb": "mean",
    "nvlink_tput_observed": "mean",
    "pcie_tput_observed": "mean",
    "pd_inbalance": "mean",
    "expert_inbalance": "mean",
    "comm_overhead_pct": "mean",
    "pipeline_bubble_fraction": "mean",
    "per_tok_comm_bytes": "mean",
    "kv_pressure_score": "mean",
    "dispatch_overhead_ms": "mean",
    "input_length_observed": "mean",
    "output_length_observed": "mean",
    "kvcache_hit_rate": "mean",
}
OUTCOME_METRICS = frozenset(
    {
        "p99_ttft_ms",
        "p99_tpot_ms",
        "throughput_token_per_sec",
        "cost_per_token",
        "slo_margin",
    }
)


def _metric_value(row: Any, metric: str) -> float | None:
    """Read one finite metric value from a Store model or raw JSON dict."""
    value = getattr(row, metric, None)
    if value is None:
        metrics_json = getattr(row, "metrics_json", None)
        if isinstance(metrics_json, dict):
            value = metrics_json.get(metric)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _policy(metric: str) -> str:
    """Return Koi's aggregation policy for one telemetry metric."""
    return METRIC_POLICIES.get(metric, DEFAULT_POLICY)


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of at least one metric value."""
    return float(sum(values) / len(values))


def _chain_value(metric: str, rows: list[Any]) -> float | None:
    """Collapse one chain's GPU rows into one Koi telemetry bucket value."""
    samples = []
    for row in rows:
        value = _metric_value(row, metric)
        if value is not None:
            samples.append((row.ts, value))
    if not samples:
        return None
    values = [value for _, value in samples]
    if _policy(metric) == "mean":
        return _mean(values)
    return max(samples, key=lambda item: item[0])[1]


def _chain_values(metric: str, chain_rows: dict[int, list[Any]]) -> dict[int, float]:
    """Collapse each rank replica's GPU rows for one metric in one bucket."""
    values = {}
    for chain_id, rows in chain_rows.items():
        value = _chain_value(metric, rows)
        if value is not None:
            values[chain_id] = value
    return values


def _rank_value(
    metric: str,
    chain_values: dict[int, float],
    chain_throughput: dict[int, float],
) -> float | None:
    """Collapse one metric's chain values into one Koi-compatible rank value."""
    policy = _policy(metric)
    values = list(chain_values.values())
    if policy == "sum":
        return float(sum(values))
    if policy == "max":
        return max(values)
    if policy == "min":
        return min(values)
    if policy == "weighted_by_throughput" and chain_throughput:
        weighted = [
            (value, weight)
            for chain_id, value in chain_values.items()
            if (weight := chain_throughput.get(chain_id)) is not None
        ]
        total = sum(weight for _, weight in weighted)
        if total > 0:
            return sum(value * weight for value, weight in weighted) / total
    return _mean(values)


def _rank_sample(chain_rows: dict[int, list[Any]], metric_names: set[str]) -> dict[str, float]:
    """Build one rank sample from all chains in one telemetry bucket."""
    throughput = _chain_values("throughput_token_per_sec", chain_rows)
    sample = {}
    for metric in sorted(metric_names):
        values = _chain_values(metric, chain_rows)
        if not values:
            continue
        rank_value = _rank_value(metric, values, throughput)
        if rank_value is not None:
            sample[metric] = rank_value
    return sample


def aggregate_rank_trajectories(
    rows: Iterable[Any],
    metric_names: set[str],
    sample_bucket_sec: int = DEFAULT_SAMPLE_BUCKET_SEC,
) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Aggregate raw active-chain Store rows into rank metric trajectories."""
    rows_by_rank: dict[tuple[str, str], dict[int, dict[str, list[Any]]]] = {}
    for row in rows:
        job_id = getattr(row, "job_id", None)
        rank_id = getattr(row, "rank_id", None)
        chain_index = getattr(row, "chain_index", None)
        if job_id is None or rank_id is None or chain_index is None:
            continue
        if not isinstance(chain_index, int) or isinstance(chain_index, bool):
            continue
        bucket = int(row.ts.timestamp() // sample_bucket_sec)
        rank_rows = rows_by_rank.setdefault((str(job_id), str(rank_id)), {})
        bucket_rows = rank_rows.setdefault(bucket, {})
        bucket_rows.setdefault(chain_index, []).append(row)

    trajectories_by_rank = {}
    for rank, bucket_rows in rows_by_rank.items():
        trajectories: dict[str, list[float]] = defaultdict(list)
        for bucket in sorted(bucket_rows):
            sample = _rank_sample(bucket_rows[bucket], metric_names)
            for metric, value in sample.items():
                trajectories[metric].append(value)
        trajectories_by_rank[rank] = dict(trajectories)
    return trajectories_by_rank


def collect_rank_telemetry(
    store: Any,
    user_id: str,
    job_ids: Iterable[str],
    start: datetime,
    end: datetime,
) -> tuple[list[Any], dict[tuple[str, str], dict[str, list[float]]]]:
    """Read Store telemetry and aggregate it into Koi-compatible rank trajectories."""
    rows = [
        row
        for job_id in sorted(set(job_ids))
        for row in store.rows_for_job_window(user_id, job_id, start, end)
    ]
    return rows, aggregate_rank_trajectories(rows, set(METRIC_POLICIES))


def aggregate_job_metrics(
    trajectories: dict[tuple[str, str], dict[str, list[float]]]
) -> dict[str, dict[str, float]]:
    """Aggregate each job's latest rank values with Koi metric policies."""
    values_by_job: dict[str, dict[str, dict[str, float]]] = {}
    for (job_id, rank_id), trajectory in trajectories.items():
        values_by_job.setdefault(job_id, {})[rank_id] = {
            metric: values[-1] for metric, values in trajectory.items() if values
        }

    metrics_by_job = {}
    for job_id, rank_values in values_by_job.items():
        throughput = {
            rank_id: values["throughput_token_per_sec"]
            for rank_id, values in rank_values.items()
            if "throughput_token_per_sec" in values
        }
        metrics = {}
        for metric in METRIC_POLICIES:
            values = {
                rank_id: values[metric]
                for rank_id, values in rank_values.items()
                if metric in values
            }
            if not values:
                continue
            value = _rank_value(metric, values, throughput)
            if value is not None:
                metrics[metric] = value
        metrics_by_job[job_id] = metrics
    return metrics_by_job


def pool_telemetry(
    expected_gpu_count: int,
    allocated_gpu_count: int,
    rows: Iterable[Any],
) -> dict[str, int | float]:
    """Summarize one existing ResourceMap pool with absent GPUs counted as zero."""
    latest_by_gpu = {}
    for row in rows:
        gpu_uuid = getattr(row, "gpu_uuid", None)
        if gpu_uuid is None:
            continue
        previous = latest_by_gpu.get(gpu_uuid)
        if previous is None or row.ts > previous.ts:
            latest_by_gpu[gpu_uuid] = row
    reported_gpu_count = len(latest_by_gpu)
    sm_utilization_sum = sum(
        _metric_value(row, "sm_utilization") or 0.0 for row in latest_by_gpu.values()
    )
    return {
        "expected_gpu_count": expected_gpu_count,
        "reported_gpu_count": reported_gpu_count,
        "allocated_gpu_count": allocated_gpu_count,
        "available_gpu_count": expected_gpu_count - allocated_gpu_count,
        "physical_sm_utilization_percent": sm_utilization_sum / expected_gpu_count,
        "telemetry_coverage_percent": 100 * reported_gpu_count / expected_gpu_count,
    }


def cluster_accounting(
    pools: Iterable[Mapping[str, int | float | None]],
    duration_hours: float,
) -> dict[str, float]:
    """Calculate whole-cluster and allocated GPU-hour cost for one tick."""
    capacity_gpu_hours = 0.0
    allocated_gpu_hours = 0.0
    capacity_cost = 0.0
    allocated_cost = 0.0
    for pool in pools:
        gpu_count = int(pool["total"])
        allocated_gpu_count = int(pool["allocated_gpu_count"])
        gpus_per_instance = int(pool["gpus_per_instance"])
        price_per_instance_hour = pool["price_per_instance_hour"]
        capacity_gpu_hours += gpu_count * duration_hours
        allocated_gpu_hours += allocated_gpu_count * duration_hours
        if price_per_instance_hour is not None:
            price = float(price_per_instance_hour)
            capacity_cost += int(pool["total_instances"]) * price * duration_hours
            allocated_cost += allocated_gpu_count / gpus_per_instance * price * duration_hours
    return {
        "capacity_gpu_hours": capacity_gpu_hours,
        "allocated_gpu_hours": allocated_gpu_hours,
        "capacity_cost": capacity_cost,
        "allocated_cost": allocated_cost,
    }
