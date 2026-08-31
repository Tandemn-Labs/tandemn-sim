"""Validate the cheap real-hardware Orca smoke benchmark."""

# ruff: noqa: E402

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from scenario import load_scenario, materialize_online_requests


class FifteenMinuteRealBenchmarkTest(unittest.TestCase):
    """Keep the real run at 16 GPUs and roughly 15 minutes."""

    def test_real_orca_smoke_contract(self) -> None:
        """Validate the full benchmark path against simulation_3h structure."""
        root = ROOT / "benchmarks" / "15min_real"
        reference = ROOT / "benchmarks" / "simulation_3h"
        manifest = yaml.safe_load(
            (root / "benchmark.yaml").read_text(encoding="utf-8")
        )["benchmark"]
        reference_manifest = yaml.safe_load(
            (reference / "benchmark.yaml").read_text(encoding="utf-8")
        )["benchmark"]
        resources = yaml.safe_load(
            (root / "resources.yaml").read_text(encoding="utf-8")
        )["resources"]
        job_document = yaml.safe_load(
            (root / "jobs" / "code-chat.yaml").read_text(encoding="utf-8")
        )
        batch_document = yaml.safe_load(
            (root / "jobs" / "small-batch-eval.yaml").read_text(encoding="utf-8")
        )
        reference_job = yaml.safe_load(
            (reference / "jobs" / "small-model-core.yaml").read_text(encoding="utf-8")
        )
        scenario = load_scenario(root / "benchmark.yaml")
        job = scenario.online_jobs[0]

        self.assertEqual(set(manifest), set(reference_manifest))
        for path in (
            (),
            ("schedule",),
            ("spec",),
            ("spec", "features"),
            ("workload",),
            ("workload", "benchmark"),
            ("workload", "benchmark", "endpoint"),
            ("workload", "benchmark", "tokenizer"),
            ("workload", "benchmark", "dataset"),
        ):
            actual = job_document
            expected = reference_job
            for key in path:
                actual = actual[key]
                expected = expected[key]
            self.assertEqual(set(actual), set(expected))
        self.assertEqual(
            set(job_document["workload"]["benchmark"]["phases"][0]),
            set(reference_job["workload"]["benchmark"]["phases"][0]),
        )

        self.assertEqual(manifest["id"], "15min_real")
        self.assertEqual(manifest["environment"], "real_hardware")
        self.assertEqual(manifest["duration_minutes"], 15)
        self.assertEqual(manifest["hard_stop_at_offset_minutes"], 16)
        self.assertEqual(len(scenario.online_jobs), 1)
        self.assertEqual(len(scenario.batch_jobs), 1)
        self.assertEqual(scenario.online_jobs[0].starts_at_ns, 0)
        self.assertEqual(scenario.batch_jobs[0].releases_at_ns, 0)
        self.assertEqual(scenario.batch_jobs[0].workload["record_count"], 256)
        self.assertEqual(
            scenario.batch_jobs[0].model_name,
            "meta-llama/Llama-3.2-1B-Instruct",
        )
        batch_features = batch_document["spec"]["features"]
        batch_dataset = batch_document["workload"]["dataset"]
        partitioning = batch_dataset["partitioning"]
        self.assertEqual(
            batch_dataset["record_count"],
            partitioning["shard_count"] * partitioning["requests_per_shard"],
        )
        self.assertEqual(
            batch_features["total_token_budget"],
            batch_dataset["record_count"]
            * (batch_features["isl_token_avg"] + batch_features["osl_token_avg"]),
        )
        self.assertEqual(scenario.tick_duration_ns, 60_000_000_000)

        catalog = yaml.safe_load(
            (ROOT / "benchmarks" / "catalog.yaml").read_text(encoding="utf-8")
        )["benchmarks"][manifest["id"]]
        self.assertEqual(catalog["manifest"], "15min_real/benchmark.yaml")
        self.assertEqual(catalog["environment"], manifest["environment"])
        self.assertEqual(catalog["duration_hours"], 0.25)

        rows = [
            json.loads(line)
            for line in (ROOT / "benchmarks" / "resource_map_benchmarking.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        maps = [row for row in rows if row["benchmark_id"] == manifest["id"]]
        self.assertEqual(len(maps), 1)
        resource_map = maps[0]["resource_map"]
        self.assertEqual(resource_map["market"], ["on_demand"])
        pools = resource_map["clouds"]["gcp"]["regions"]["us-central1"][
            "zones"
        ]["us-central1-b"]["network_fabrics"]["default"]["machine_pools"]
        expected = {
            "g4-standard-384": ("RTXPRO6000", 96),
            "g2-standard-96": ("L4", 24),
        }
        self.assertEqual(set(pools), set(expected))
        self.assertEqual(sum(resource["total_devices"] for resource in resources), 16)
        for resource, (machine_type, (gpu_type, memory_gb)) in zip(
            resources, expected.items(), strict=True
        ):
            pool = pools[machine_type]
            self.assertEqual(resource["total_devices"], 8)
            self.assertEqual(resource["markets"], {"on_demand": 8})
            self.assertEqual(resource["devices_per_node"], pool["gpus_per_instance"])
            self.assertEqual(pool["total_instances"], 1)
            self.assertEqual(pool["gpu_type"], gpu_type)
            self.assertEqual(pool["gpu_memory_gb"], memory_gb)

        phase = job_document["workload"]["benchmark"]["phases"][0]
        features = job_document["spec"]["features"]
        dataset = job_document["workload"]["benchmark"]["dataset"]
        self.assertEqual(phase["duration"], manifest["duration_minutes"] * 60)
        self.assertEqual(phase["concurrency"], features["max_concurrent_streaming"])
        self.assertEqual(features["request_arrival_pattern"], "Poisson")
        self.assertEqual(phase["type"], "poisson")
        self.assertEqual(dataset["type"], "synthetic")
        self.assertEqual(dataset["sampling"], "random")
        self.assertGreater(dataset["entries"], 0)
        requests = materialize_online_requests(job)
        expected_requests = phase["rate"] * phase["duration"]
        self.assertAlmostEqual(len(requests), expected_requests, delta=90)
        intervals = {
            right.arrives_at_ns - left.arrives_at_ns
            for left, right in zip(requests, requests[1:], strict=False)
        }
        self.assertGreater(len(intervals), 1)
        self.assertLess(requests[-1].arrives_at_ns, job.stops_at_ns)
        projected_streams = phase["rate"] * (
            features["target_p99_ttft_ms"] / 1000
            + (features["osl_token_avg"] - 1) * features["target_p99_tpot_ms"] / 1000
        )
        self.assertLessEqual(
            projected_streams,
            0.7 * features["max_concurrent_streaming"],
        )

        with (ROOT / "benchmarks" / "model_catalogs.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            models = list(csv.DictReader(stream))
        self.assertEqual(
            sum(row["model_id"] == job.model_name for row in models),
            1,
        )


if __name__ == "__main__":
    unittest.main()
