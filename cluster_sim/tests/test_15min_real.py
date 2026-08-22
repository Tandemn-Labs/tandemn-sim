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

from bestfit import BEST_FIT_CONFIGURATIONS, _mapped_placement, _reserve
from policy import load_config_policy
from scenario import load_scenario, materialize_online_requests
from vram import VRAM_REQUIREMENTS_GIB


class FifteenMinuteRealBenchmarkTest(unittest.TestCase):
    """Keep the smoke run at 24 GPUs and roughly 15 minutes."""

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
        reference_job = yaml.safe_load(
            (reference / "jobs" / "small-model-core.yaml").read_text(encoding="utf-8")
        )
        scenario = load_scenario(root / "benchmark.yaml")
        job = scenario.online_jobs[0]
        placement = BEST_FIT_CONFIGURATIONS[manifest["id"]][job.name]
        configuration = placement["configuration"]
        resource = resources[0]

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
        self.assertFalse(scenario.batch_jobs)
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
        pool = maps[0]["resource_map"]["clouds"]["aws"]["regions"]["us-east-1"][
            "zones"
        ]["us-east-1a"]["network_fabrics"]["efa"]["machine_pools"]["g6e.48xlarge"]
        self.assertEqual(resource["total_devices"], 24)
        self.assertEqual(resource["devices_per_node"], pool["gpus_per_instance"])
        self.assertEqual(pool["total_instances"], 3)
        self.assertEqual(pool["gpu_type"], "L40S")
        self.assertEqual(
            pool["gpus_per_instance"] * pool["total_instances"],
            resource["total_devices"],
        )

        self.assertEqual(configuration, {"tp": 8, "pp": 1, "dp": 3})
        self.assertEqual(
            configuration["tp"] * configuration["pp"],
            pool["gpus_per_instance"],
        )
        self.assertEqual(
            configuration["dp"] * pool["gpus_per_instance"],
            resource["total_devices"],
        )
        planner_resources = {
            "on_demand|aws|us-east-1|us-east-1a|L40S": {
                "gpu_type": "L40S",
                "total": 24,
                "free": 24,
                "pools": [
                    {
                        **pool,
                        "instance_type": "g6e.48xlarge",
                        "allocation_kind": "instance",
                        "total": 24,
                        "free": 24,
                    }
                ],
            }
        }
        resolved = _mapped_placement(placement, planner_resources)
        self.assertIsNotNone(resolved)
        assert resolved is not None
        target, tp, pp, dp = resolved
        _reserve(target, tp * pp, dp)
        self.assertEqual(target.environment["free"], 0)
        assert target.pool is not None
        self.assertEqual(target.pool["free"], 0)
        rule = load_config_policy().rule_for(placement["hardware"], job.model_name)
        self.assertIsNotNone(rule)
        assert rule is not None
        self.assertIn(configuration["tp"], rule.allowed_tp)
        self.assertEqual(rule.precision, job_document["spec"]["precision"])
        self.assertLessEqual(
            VRAM_REQUIREMENTS_GIB[manifest["id"]][job.name],
            configuration["tp"] * resource["memory_gb"] * 0.9,
        )

        phase = job_document["workload"]["benchmark"]["phases"][0]
        features = job_document["spec"]["features"]
        dataset = job_document["workload"]["benchmark"]["dataset"]
        self.assertEqual(phase["duration"], manifest["duration_minutes"] * 60)
        self.assertEqual(phase["concurrency"], features["max_concurrent_streaming"])
        self.assertEqual(features["request_arrival_pattern"], "Constant")
        self.assertEqual(phase["type"], "constant")
        self.assertEqual(dataset["type"], "synthetic")
        self.assertEqual(dataset["sampling"], "random")
        self.assertGreater(dataset["entries"], 0)
        requests = materialize_online_requests(job)
        self.assertEqual(len(requests), int(phase["rate"] * phase["duration"]) - 1)
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
        with (ROOT / "benchmarks" / "model_catalog_gpu_profiles.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            profiles = list(csv.DictReader(stream))
        l40s_fields = {
            row["profile_field"]
            for row in profiles
            if row["model_id"] == job.model_name and row["gpu_type"] == "L40S"
        }
        self.assertTrue(
            {"max_num_seq", "max_num_batched_tokens", "block_size", "kvcache_dtype"}
            <= l40s_fields
        )


if __name__ == "__main__":
    unittest.main()
