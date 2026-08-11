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

from telemetry import _metric_value, _policy, aggregate_rank_trajectories


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
            _row(started_at, "chain-a", 10, 80, 0.2, 0.4),
            _row(started_at, "chain-a", 10, 80, 0.2, 0.8),
            _row(started_at, "chain-b", 20, 120, 0.65, 0.6),
            _row(started_at, "chain-b", 20, 120, 0.65, 0.8),
            _row(started_at + timedelta(seconds=10), "chain-a", 15, 90, 0.3, 0.5),
            _row(started_at + timedelta(seconds=10), "chain-b", 25, 110, 0.7, 0.7),
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
        self.assertEqual(rank["throughput_token_per_sec"], [30.0, 40.0])
        self.assertEqual(rank["p99_ttft_ms"], [120.0, 110.0])
        self.assertEqual(rank["cost_per_token"], [0.5, 0.55])
        self.assertEqual(rank["kv_cache_util"], [0.65, 0.6])


def _row(
    timestamp: datetime,
    chain_id: str,
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
        chain_id=chain_id,
        throughput_token_per_sec=throughput,
        p99_ttft_ms=ttft,
        cost_per_token=cost_per_token,
        kv_cache_util=kv_cache_util,
    )


if __name__ == "__main__":
    unittest.main()
