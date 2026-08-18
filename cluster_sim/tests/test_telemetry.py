"""Test copied Koi rank-telemetry aggregation primitives."""

from __future__ import annotations

import math
import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from telemetry import (
    _metric_value,
    _policy,
    RankTrajectory,
    aggregate_job_metrics,
    aggregate_rank_trajectories,
    cluster_accounting,
    collect_rank_telemetry,
    pool_telemetry,
)


class MetricValueTest(unittest.TestCase):
    """Verify Store metric values are safe for aggregation."""

    def test_reads_only_finite_values(self) -> None:
        """Accept numeric attributes and reject absent or non-finite values."""
        row = SimpleNamespace(
            throughput_token_per_sec="12.5",
            metrics_json={"p99_ttft_ms": math.inf},
        )

        self.assertEqual(_metric_value(row, "throughput_token_per_sec"), 12.5)
        self.assertIsNone(_metric_value(row, "p99_ttft_ms"))
        self.assertIsNone(_metric_value(row, "missing"))

    def test_uses_koi_metric_policies(self) -> None:
        """Keep explicit Koi rules while defaulting unknown metrics safely."""
        self.assertEqual(_policy("p99_ttft_ms"), "max")
        self.assertEqual(_policy("cost_per_token"), "weighted_by_throughput")
        self.assertEqual(_policy("unknown"), "latest_mean")

    def test_aggregates_chain_rows_like_koi(self) -> None:
        """Avoid GPU double counting while retaining Koi's rank policies."""
        started_at = datetime(2026, 1, 1, tzinfo=UTC)
        rows = (
            _row(started_at, 0, 10, 80, 0.2, 0.4),
            _row(started_at, 0, 10, 80, 0.2, 0.8),
            _row(started_at, 1, 20, 120, 0.65, 0.6),
            _row(started_at, 1, 20, 120, 0.65, 0.8),
            _row(started_at + timedelta(seconds=10), 0, 15, 90, 0.3, 0.5),
            _row(started_at + timedelta(seconds=10), 1, 25, 110, 0.7, 0.7),
        )

        trajectories = aggregate_rank_trajectories(
            rows,
            {
                "throughput_token_per_sec",
                "p99_ttft_ms",
                "cost_per_token",
                "kv_cache_util",
            },
        )

        rank = trajectories[("job-a", "rank-0")]
        self.assertEqual(rank.metrics["throughput_token_per_sec"], [30.0, 40.0])
        self.assertEqual(rank.metrics["p99_ttft_ms"], [120.0, 110.0])
        self.assertEqual(rank.metrics["cost_per_token"], [0.5, 0.55])
        self.assertEqual(rank.metrics["kv_cache_util"], [0.65, 0.6])

    def test_keeps_sparse_metrics_aligned_to_bucket_starts(self) -> None:
        started_at = datetime(2026, 1, 1, tzinfo=UTC)
        rows = (
            SimpleNamespace(
                ts=started_at,
                job_id="job-a",
                rank_id="rank-0",
                chain_index=0,
                throughput_token_per_sec=10.0,
                p99_ttft_ms=None,
            ),
            SimpleNamespace(
                ts=started_at + timedelta(seconds=10),
                job_id="job-a",
                rank_id="rank-0",
                chain_index=0,
                throughput_token_per_sec=20.0,
                p99_ttft_ms=120.0,
            ),
        )

        trajectory = aggregate_rank_trajectories(
            rows,
            {"throughput_token_per_sec", "p99_ttft_ms"},
        )[("job-a", "rank-0")]

        self.assertEqual(
            trajectory.bucket_starts,
            (started_at, started_at + timedelta(seconds=10)),
        )
        self.assertEqual(
            trajectory.metrics["throughput_token_per_sec"], [10.0, 20.0])
        self.assertEqual(trajectory.metrics["p99_ttft_ms"], [None, 120.0])

    def test_collects_rows_through_koi_store_api(self) -> None:
        """Use the same per-job Store window query as Koi."""
        started_at = datetime(2026, 1, 1, tzinfo=UTC)
        rows = [_row(started_at, 0, 10, 80, 0.2, 0.4)]
        calls = []

        class Store:
            def rows_for_job_window(self, user_id, job_id, start, end):
                calls.append((user_id, job_id, start, end))
                return rows if job_id == "job-a" else []

        observed, trajectories = collect_rank_telemetry(
            Store(),
            "user-a",
            ("job-b", "job-a", "job-a"),
            started_at,
            started_at + timedelta(seconds=10),
        )

        self.assertEqual([call[1] for call in calls], ["job-a", "job-b"])
        self.assertEqual(observed, rows)
        self.assertEqual(
            trajectories[("job-a", "rank-0")].metrics["p99_ttft_ms"],
            [80.0],
        )

    def test_aggregates_latest_rank_values_per_job(self) -> None:
        """Use Koi's sum, max, and weighted-cost rules across job ranks."""
        metrics = aggregate_job_metrics(
            {
                ("job-a", "rank-0"): RankTrajectory(
                    (),
                    {
                        "throughput_token_per_sec": [10.0, 15.0],
                        "p99_ttft_ms": [80.0, 90.0],
                        "cost_per_token": [0.2, 0.3],
                    },
                ),
                ("job-a", "rank-1"): RankTrajectory(
                    (),
                    {
                        "throughput_token_per_sec": [20.0, 25.0],
                        "p99_ttft_ms": [120.0, 110.0],
                        "cost_per_token": [0.65, 0.7],
                    },
                ),
            }
        )

        job = metrics["job-a"]
        self.assertEqual(job["throughput_token_per_sec"], 40.0)
        self.assertEqual(job["p99_ttft_ms"], 110.0)
        self.assertEqual(job["cost_per_token"], 0.55)

    def test_zero_fills_missing_pool_gpu_telemetry(self) -> None:
        """Include unreported capacity in physical utilization and coverage."""
        started_at = datetime(2026, 1, 1, tzinfo=UTC)
        rows = [
            SimpleNamespace(
                gpu_uuid=f"GPU-{index}",
                ts=started_at,
                sm_utilization=20.0,
            )
            for index in range(6)
        ]

        observation = pool_telemetry(8, 3, rows)

        self.assertEqual(observation["available_gpu_count"], 5)
        self.assertEqual(observation["reported_gpu_count"], 6)
        self.assertEqual(observation["physical_sm_utilization_percent"], 15.0)
        self.assertEqual(observation["telemetry_coverage_percent"], 75.0)

    def test_uses_latest_non_null_sm_sample_per_gpu(self) -> None:
        """Do not let a later sparse cache summary erase an SM observation."""
        started_at = datetime(2026, 1, 1, tzinfo=UTC)
        rows = (
            SimpleNamespace(
                gpu_uuid="GPU-1",
                ts=started_at,
                sm_utilization=100.0,
            ),
            SimpleNamespace(
                gpu_uuid="GPU-1",
                ts=started_at + timedelta(seconds=10),
                sm_utilization=None,
                kvcache_hit_rate=0.5,
            ),
        )

        observation = pool_telemetry(8, 1, rows)

        self.assertEqual(observation["reported_gpu_count"], 1)
        self.assertEqual(observation["physical_sm_utilization_percent"], 12.5)

    def test_empty_existing_pool_has_zero_utilization(self) -> None:
        """Treat an existing unoccupied ResourceMap pool as zero utilization."""
        observation = pool_telemetry(8, 0, ())

        self.assertEqual(observation["physical_sm_utilization_percent"], 0.0)
        self.assertEqual(observation["telemetry_coverage_percent"], 0.0)

    def test_calculates_whole_and_allocated_pool_cost(self) -> None:
        """Price all capacity separately from the active GPU share."""
        accounting = cluster_accounting(
            (
                {
                    "total": 8,
                    "allocated_gpu_count": 3,
                    "gpus_per_instance": 8,
                    "total_instances": 1,
                    "price_per_instance_hour": 10.0,
                },
            ),
            0.5,
        )

        self.assertEqual(accounting["capacity_gpu_hours"], 4.0)
        self.assertEqual(accounting["allocated_gpu_hours"], 1.5)
        self.assertEqual(accounting["capacity_cost"], 5.0)
        self.assertEqual(accounting["allocated_cost"], 1.875)


def _row(
    timestamp: datetime,
    chain_index: int,
    throughput: float,
    ttft: float,
    cost_per_token: float,
    kv_cache_util: float,
) -> SimpleNamespace:
    """Build one minimal active-chain Store telemetry row."""
    return SimpleNamespace(
        ts=timestamp,
        job_id="job-a",
        rank_id="rank-0",
        chain_index=chain_index,
        throughput_token_per_sec=throughput,
        p99_ttft_ms=ttft,
        cost_per_token=cost_per_token,
        kv_cache_util=kv_cache_util,
    )


if __name__ == "__main__":
    unittest.main()
