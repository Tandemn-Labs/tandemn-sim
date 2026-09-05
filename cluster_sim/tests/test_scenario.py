"""Test deterministic YAML workload compilation for the cluster simulator."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import yaml

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
        self.assertEqual(self.scenario.workload_end_ns, 10_800_000_000_000)

    def test_online_trace_is_deterministic_and_tick_partitioned(self) -> None:
        """Generate one continuous trace and recover it exactly from tick slices."""
        job = next(job for job in self.scenario.online_jobs if job.name.endswith("0-1"))
        requests = materialize_online_requests(job)

        self.assertEqual(requests, materialize_online_requests(job))
        self.assertTrue(requests)
        self.assertTrue(
            all(
                job.starts_at_ns <= request.arrives_at_ns < job.stops_at_ns
                for request in requests
            )
        )

        sliced = []
        for start_ns in range(
            job.starts_at_ns, job.stops_at_ns, self.scenario.tick_duration_ns
        ):
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

    def test_calibrated_load_envelope(self) -> None:
        """Keep the paper workload challenging without restoring the overload floor."""
        online_intensity = sum(
            _intensity(
                float(phase["duration"]),
                float(phase["rate"]),
                float(phase.get("rateRamp", 0)),
            )
            for job in self.scenario.online_jobs
            for phase in job.workload["phases"]
        )
        moderation = {
            job.name: job
            for job in self.scenario.online_jobs
            if "-0-4" in job.name or "-49-2" in job.name
        }

        self.assertAlmostEqual(online_intensity, 573_765.6)
        self.assertEqual(len(moderation), 2)
        for job in moderation.values():
            self.assertEqual(job.workload["features"]["max_concurrent_streaming"], 32)
            self.assertEqual(job.workload["phases"][0]["rate"], 28.0)
            self.assertEqual(job.workload["phases"][0]["concurrency"], 32)

        peak_minute = 70
        active_at_peak = [
            job
            for job in self.scenario.online_jobs
            if job.starts_at_ns <= peak_minute * 60_000_000_000 < job.stops_at_ns
        ]
        peak_concurrency = sum(
            job.workload["features"]["max_concurrent_streaming"]
            for job in active_at_peak
        )
        peak_rate = 0.0
        for job in active_at_peak:
            phase = job.workload["phases"][0]
            rate = float(phase["rate"])
            ramp = float(phase.get("rateRamp", 0))
            elapsed_seconds = (
                peak_minute * 60_000_000_000 - job.starts_at_ns
            ) / 1_000_000_000
            peak_rate += rate * min(elapsed_seconds / ramp, 1) if ramp else rate
        self.assertEqual(peak_concurrency, 520)
        self.assertAlmostEqual(peak_rate, 87.6256666667)

        batch_documents = []
        job_paths = (ROOT / "benchmarks/simulation_3h/jobs").glob("*.yaml")
        for path in sorted(job_paths):
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            if document["schedule"]["kind"] == "batch":
                batch_documents.append(document)
        expected_campaigns = {
            "job-benchmark_timing_3hr-15-1": (7_200, 36),
            "job-benchmark_timing_3hr-15-3": (9_000, 18),
            "job-benchmark_timing_3hr-85-1": (6_000, 12),
            "job-benchmark_timing_3hr-100-1": (4_000, 20),
            "job-benchmark_timing_3hr-125-2": (4_000, 40),
            "job-benchmark_timing_3hr-145-2": (3_200, 16),
            "job-benchmark_timing_3hr-160-1": (12_000, 24),
        }
        self.assertEqual(
            {
                document["job_id"]: (
                    document["workload"]["dataset"]["record_count"],
                    document["workload"]["dataset"]["partitioning"]["shard_count"],
                )
                for document in batch_documents
            },
            expected_campaigns,
        )
        self.assertEqual(
            sum(
                document["workload"]["dataset"]["record_count"]
                for document in batch_documents
            ),
            45_400,
        )
        self.assertEqual(
            sum(
                document["workload"]["dataset"]["partitioning"]["shard_count"]
                for document in batch_documents
            ),
            166,
        )
        for document in batch_documents:
            features = document["spec"]["features"]
            dataset = document["workload"]["dataset"]
            partitioning = dataset["partitioning"]
            self.assertEqual(
                dataset["record_count"],
                partitioning["shard_count"] * partitioning["requests_per_shard"],
            )
            self.assertEqual(
                features["total_token_budget"],
                dataset["record_count"]
                * (features["isl_token_avg"] + features["osl_token_avg"]),
            )

    def test_benchmark_azure_pools_use_cataloged_regions(self) -> None:
        """Keep Azure benchmark capacity aligned with the seeded catalog offerings."""
        rows = [
            json.loads(line)
            for line in (ROOT / "benchmarks/resource_map_benchmarking.jsonl")
            .read_text()
            .splitlines()
        ]
        resource_map = next(
            row["resource_map"]
            for row in rows
            if row["benchmark_id"] == "simulation_3h"
        )
        regions = resource_map["clouds"]["azure"]["regions"]

        nd = regions["westus2"]["zones"]["1"]["network_fabrics"]["infiniband"][
            "machine_pools"
        ]["Standard_ND96amsr_A100_v4"]
        nc = regions["westus2"]["zones"]["1"]["network_fabrics"]["default"][
            "machine_pools"
        ]["Standard_NC96ads_A100_v4"]

        self.assertEqual(
            (nd["gpu_type"], nd["gpu_memory_gb"], nd["gpus_per_instance"]),
            ("A100-80GB", 80, 8),
        )
        self.assertEqual(
            (nc["gpu_type"], nc["gpu_memory_gb"], nc["gpus_per_instance"]),
            ("A100-40GB", 40, 4),
        )

    def test_linear_ramp_uses_integrated_rate(self) -> None:
        """Treat rateRamp as a continuous increase, not a repeated tick pattern."""
        self.assertAlmostEqual(_intensity(300, 0.022, 10_800), 0.0916666667)
        self.assertAlmostEqual(_intensity(600, 0.022, 10_800), 0.3666666667)


if __name__ == "__main__":
    unittest.main()
