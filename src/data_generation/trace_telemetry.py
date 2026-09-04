"""Aggregate Store GPU telemetry into real-trace rank windows."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from tandemn_system_data.models.gpu_metric import GpuMetric

BatchChainSamples = dict[
    tuple[str, str, int],
    list[tuple[datetime, dict[str, float]]],
]

SUM_METRICS = frozenset(
    {
        "throughput_token_per_sec",
        "live_batch_size",
        "depth_req_q",
        "prefill_iteration_counts_per_second",
        "decode_itr_counts_per_second",
    }
)
MAX_METRICS = frozenset({"p99_ttft_ms", "p99_tpot_ms"})
MIN_METRICS = frozenset({"slo_margin"})
WEIGHTED_METRICS = frozenset({"cost_per_token"})
MEAN_METRICS = frozenset(
    {
        "activation_mem_pressure",
        "comm_overhead_pct",
        "dispatch_overhead_ms",
        "expert_inbalance",
        "gpu_mem_used_fraction",
        "input_length_observed",
        "kvcache_hit_rate",
        "kv_cache_util",
        "kv_pressure_score",
        "mem_bandwidth_utilization",
        "nvlink_tput_observed",
        "output_length_observed",
        "pcie_tput_observed",
        "pd_inbalance",
        "per_tok_comm_bytes",
        "pipeline_bubble_fraction",
        "sm_utilization",
        "vram_headroom_gb",
    }
)
RANK_METRICS = SUM_METRICS | MAX_METRICS | MIN_METRICS | WEIGHTED_METRICS | MEAN_METRICS
SAMPLE_BUCKET_SECONDS = 10
BATCH_COUNTER_METRICS = frozenset(
    {
        "batched_reqs_processed_total",
        "batched_chunks_input_pulled_total",
        "batched_chunks_output_written_total",
    }
)
BATCH_METRICS = BATCH_COUNTER_METRICS | frozenset({"batched_reqs_inflight"})


def rank_trajectories(
    rows: list[GpuMetric],
) -> dict[tuple[str, str], dict[str, object]]:
    """Build ten-second rank trajectories using simulator-compatible policies.

    Each Store row represents one physical GPU, while inference metrics repeat
    across every GPU in a chain. Rows are first collapsed to one chain value,
    then chains are combined into a rank value. This prevents TP/PP GPU width
    from multiplying worker-level throughput, latency, and queue metrics.
    """
    grouped: dict[tuple[str, str], dict[int, dict[int, list[GpuMetric]]]] = {}
    for row in rows:
        if row.job_id is None or row.rank_id is None or row.chain_index is None:
            continue
        bucket = int(row.ts.timestamp() // SAMPLE_BUCKET_SECONDS)
        rank = grouped.setdefault((row.job_id, row.rank_id), {})
        rank.setdefault(bucket, {}).setdefault(row.chain_index, []).append(row)
    return {
        rank: _rank_trajectory(bucket_chains)
        for rank, bucket_chains in grouped.items()
    }


def batch_chain_samples(
    rows: list[GpuMetric],
) -> BatchChainSamples:
    """Return ordered batch-driver samples once per job/rank/chain.

    Batch-driver values repeat on every physical GPU in a chain. Consecutive
    identical values are collapsed, which removes those GPU copies without
    discarding counter changes or resets observed later in the tick.
    """
    grouped: dict[tuple[str, str, int], list[GpuMetric]] = {}
    for row in rows:
        if row.job_id is None or row.rank_id is None or row.chain_index is None:
            continue
        key = (row.job_id, row.rank_id, row.chain_index)
        grouped.setdefault(key, []).append(row)

    samples: BatchChainSamples = {}
    for key, chain_rows in grouped.items():
        previous: dict[str, float] | None = None
        for row in sorted(chain_rows, key=lambda item: item.ts):
            values = {
                name: value
                for name in BATCH_METRICS
                if (value := _metric_value(row, name)) is not None
            }
            if not values or values == previous:
                continue
            samples.setdefault(key, []).append((row.ts, values))
            previous = values
    return samples


def _rank_trajectory(
    bucket_chains: dict[int, dict[int, list[GpuMetric]]],
) -> dict[str, object]:
    buckets = sorted(bucket_chains)
    metrics: dict[str, list[float | None]] = {}
    chain_indices: set[int] = set()
    for index, bucket in enumerate(buckets):
        chains = bucket_chains[bucket]
        chain_indices.update(chains)
        for metric, value in _rank_bucket(chains).items():
            metrics.setdefault(metric, [None] * len(buckets))[index] = value
    return {
        "chain_indices": sorted(chain_indices),
        "sample_bucket_starts": [
            datetime.fromtimestamp(bucket * SAMPLE_BUCKET_SECONDS, UTC).isoformat()
            for bucket in buckets
        ],
        "metrics": metrics,
    }


def _rank_bucket(chains: dict[int, list[GpuMetric]]) -> dict[str, float]:
    throughput = _chain_metric_values("throughput_token_per_sec", chains)
    values = {}
    for metric in RANK_METRICS:
        chain_values = _chain_metric_values(metric, chains)
        if chain_values:
            values[metric] = _aggregate_rank_metric(metric, chain_values, throughput)
    return values


def _chain_metric_values(
    metric: str, chains: dict[int, list[GpuMetric]]
) -> dict[int, float]:
    values = {}
    for chain_index, rows in chains.items():
        samples = [
            (row.ts, value)
            for row in rows
            if (value := _metric_value(row, metric)) is not None
        ]
        if not samples:
            continue
        if metric in MEAN_METRICS:
            values[chain_index] = sum(value for _, value in samples) / len(samples)
        else:
            values[chain_index] = max(samples, key=lambda sample: sample[0])[1]
    return values


def _aggregate_rank_metric(
    metric: str,
    values: dict[int, float],
    throughput: dict[int, float],
) -> float:
    samples = list(values.values())
    if metric in SUM_METRICS:
        return float(sum(samples))
    if metric in MAX_METRICS:
        return max(samples)
    if metric in MIN_METRICS:
        return min(samples)
    if metric in WEIGHTED_METRICS:
        weighted = [
            (value, throughput[chain])
            for chain, value in values.items()
            if chain in throughput
        ]
        total = sum(weight for _, weight in weighted)
        if total > 0:
            return sum(value * weight for value, weight in weighted) / total
    return sum(samples) / len(samples)


def _metric_value(metric: GpuMetric, name: str) -> float | None:
    value = getattr(metric, name)
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return parsed * 100 if name == "sm_utilization" else parsed
