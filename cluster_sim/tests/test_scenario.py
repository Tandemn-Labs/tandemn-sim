"""Test deterministic YAML workload compilation for the cluster simulator."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import BatchProgress
from scenario import (
    _intensity,
    _prefix_keys,
    build_batch_probe,
    load_scenario,
    materialize_online_requests,
    requests_in_tick,
)


class ScenarioTest(unittest.TestCase):
    """Check workload generation against the executable simulation_3h YAML."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the benchmark once for deterministic workload checks."""
        cls.scenario = load_scenario(ROOT / "benchmarks/simulation_3h/benchmark.yaml")

    def test_loads_all_jobs(self) -> None:
        """Load the benchmark's complete online and batch workload set."""
        self.assertEqual(len(self.scenario.online_jobs), 18)
        self.assertEqual(len(self.scenario.batch_jobs), 7)
        self.assertEqual(self.scenario.tick_duration_ns, 300_000_000_000)

    def test_online_trace_is_deterministic_and_tick_partitioned(self) -> None:
        """Generate one continuous trace and recover it exactly from tick slices."""
        job = next(job for job in self.scenario.online_jobs if job.name.endswith("0-1"))
        requests = materialize_online_requests(job)

        self.assertEqual(requests, materialize_online_requests(job))
        self.assertTrue(requests)
        self.assertTrue(all(job.starts_at_ns <= request.arrives_at_ns < job.stops_at_ns for request in requests))

        sliced = []
        for start_ns in range(job.starts_at_ns, job.stops_at_ns, self.scenario.tick_duration_ns):
            sliced.extend(
                requests_in_tick(
                    requests,
                    start_ns,
                    min(start_ns + self.scenario.tick_duration_ns, job.stops_at_ns),
                )
            )
        self.assertEqual(tuple(sliced), requests)

    def test_online_tokens_stay_within_yaml_bounds(self) -> None:
        """Use the existing bounded LogNormal token ranges for every request."""
        job = next(job for job in self.scenario.online_jobs if job.name.endswith("0-1"))
        features = job.workload["features"]

        for request in materialize_online_requests(job):
            self.assertGreaterEqual(request.input_tokens, features["isl_token_min"])
            self.assertLessEqual(request.input_tokens, features["isl_token_max"])
            self.assertGreaterEqual(request.output_tokens, features["osl_token_min"])
            self.assertLessEqual(request.output_tokens, features["osl_token_max"])

    def test_hot_prefixes_cover_six_context_lengths(self) -> None:
        """Keep six stable prefix families around each job's configured average."""
        self.assertEqual(
            _prefix_keys("premium", 300),
            (
                "premium/prefix-75",
                "premium/prefix-150",
                "premium/prefix-225",
                "premium/prefix-300",
                "premium/prefix-375",
                "premium/prefix-450",
            ),
        )

    def test_batch_probe_has_variable_request_lengths(self) -> None:
        """Generate the next configured shard with independently sampled tokens."""
        job = next(job for job in self.scenario.batch_jobs if job.name.endswith("15-1"))
        progress = BatchProgress(total_requests=job.workload["record_count"])
        probe = build_batch_probe(job, progress)

        self.assertEqual(len(probe), job.workload["requests_per_shard"])
        self.assertGreater(len({request.input_tokens for request in probe}), 1)
        self.assertGreater(len({request.output_tokens for request in probe}), 1)

    def test_benchmark_azure_pools_use_cataloged_regions(self) -> None:
        """Keep Azure benchmark capacity aligned with the seeded catalog offerings."""
        resource_map = json.loads(
            (ROOT / "benchmarks/resource_map_benchmarking.jsonl").read_text()
        )["resource_map"]
        regions = resource_map["clouds"]["azure"]["regions"]

        nd = regions["westus2"]["zones"]["1"]["network_fabrics"]["infiniband"][
            "machine_pools"
        ]["Standard_ND96amsr_A100_v4"]
        nc = regions["westus2"]["zones"]["1"]["network_fabrics"]["default"][
            "machine_pools"
        ]["Standard_NC96ads_A100_v4"]

        self.assertEqual(
            (nd["gpu_type"], nd["gpu_memory_gb"], nd["gpus_per_instance"]),
            ("A100_80GB", 80, 8),
        )
        self.assertEqual(
            (nc["gpu_type"], nc["gpu_memory_gb"], nc["gpus_per_instance"]),
            ("A100_80GB", 80, 4),
        )

    def test_linear_ramp_uses_integrated_rate(self) -> None:
        """Treat rateRamp as a continuous increase, not a repeated tick pattern."""
        self.assertAlmostEqual(_intensity(300, 0.022, 10_800), 0.0916666667)
        self.assertAlmostEqual(_intensity(600, 0.022, 10_800), 0.3666666667)


if __name__ == "__main__":
    unittest.main()
