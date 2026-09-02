"""Validate the executable twelve-hour real-hardware benchmark."""

# ruff: noqa: E402

from __future__ import annotations

import csv
import json
import re
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from scenario import load_scenario


class RealTwelveHourTest(unittest.TestCase):
    """Keep the executable files aligned with the authored real benchmark."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load benchmark artifacts once."""
        cls.root = ROOT / "benchmarks" / "real_12h"
        cls.manifest = yaml.safe_load(
            (cls.root / "benchmark.yaml").read_text(encoding="utf-8")
        )["benchmark"]
        cls.resources = yaml.safe_load(
            (cls.root / "resources.yaml").read_text(encoding="utf-8")
        )["resources"]
        overview = (cls.root / "benchmark.md").read_text(encoding="utf-8")
        events = [
            json.loads(block)
            for block in re.findall(r"```json\s*(.*?)```", overview, re.S)
        ]
        cls.overview_jobs = {
            job["job_id"]: job for event in events for job in event["jobs"]
        }
        cls.jobs = {}
        for path in sorted((cls.root / "jobs").glob("*.yaml")):
            job = yaml.safe_load(path.read_text(encoding="utf-8"))
            cls.jobs[job["job_id"]] = job

    def test_manifest_loads_the_authored_workload(self) -> None:
        """Expose all 28 allowed jobs through the executable manifest."""
        scenario = load_scenario(self.root / "benchmark.yaml")

        self.assertEqual(self.manifest["id"], "real_12h")
        self.assertEqual(self.manifest["environment"], "real_hardware")
        self.assertEqual(self.manifest["status"], "executable")
        self.assertEqual(self.manifest["hard_stop_at_offset_minutes"], 780)
        self.assertEqual(scenario.tick_duration_ns, 300_000_000_000)
        self.assertEqual(len(scenario.online_jobs), 16)
        self.assertEqual(len(scenario.batch_jobs), 12)
        self.assertEqual(
            set(self.jobs),
            {
                job_id
                for job_id, job in self.overview_jobs.items()
                if job["kind"] == "online"
            },
        )
        self.assertEqual(
            {job.name for job in scenario.online_jobs + scenario.batch_jobs},
            set(self.overview_jobs),
        )

    def test_online_service_curve_is_smooth_and_search_free(self) -> None:
        """Maintain enterprise overlap without embedding or search traffic."""
        online = [
            job for job in self.jobs.values() if job["schedule"]["kind"] == "online"
        ]
        forbidden = ("baai", "baig", "embed", "retriev", "rag", "search", "rank")
        for job in self.jobs.values():
            workload = job["workload"].get("benchmark", job["workload"].get("dataset"))
            identity = " ".join(
                (
                    job["job_id"],
                    job["spec"]["user_id"],
                    workload["model"],
                    str(workload["endpoint"]),
                )
            ).lower()
            self.assertFalse(any(term in identity for term in forbidden), identity)

        boundaries = sorted(
            {
                value
                for job in online
                for value in (
                    job["schedule"]["start_at_offset_minutes"],
                    job["schedule"]["stop_at_offset_minutes"],
                )
            }
        )
        active_counts = [
            sum(
                job["schedule"]["start_at_offset_minutes"]
                <= minute
                < job["schedule"]["stop_at_offset_minutes"]
                for job in online
            )
            for minute in boundaries[:-1]
        ]
        active_concurrency = [
            sum(
                job["spec"]["features"]["max_concurrent_streaming"]
                for job in online
                if job["schedule"]["start_at_offset_minutes"]
                <= minute
                < job["schedule"]["stop_at_offset_minutes"]
            )
            for minute in boundaries[:-1]
        ]
        self.assertEqual(min(active_counts), 6)
        self.assertEqual(max(active_counts), 12)
        self.assertEqual(max(active_concurrency), 560)
        self.assertLessEqual(
            max(
                abs(right - left)
                for left, right in zip(active_counts, active_counts[1:], strict=False)
            ),
            2,
        )

    def test_job_files_match_the_definition(self) -> None:
        """Keep identity, lifecycle, features, and workload sizes exact."""
        phase_types = {
            "Bursty": "gamma",
            "Constant": "constant",
            "Diurnal": "gamma",
            "Poisson": "poisson",
        }
        expanded_shards = 0

        for job_id, source in self.overview_jobs.items():
            with self.subTest(job_id=job_id):
                if source["kind"] == "batch":
                    source_schedule = source["schedule"]
                    input_source = source["input_source"]
                    expanded_shards += input_source["shard_count"]
                    self.assertEqual(
                        input_source["shard_count"], source_schedule["shard_count"]
                    )
                    self.assertEqual(
                        input_source["requests_per_shard"],
                        source_schedule["requests_per_shard"],
                    )
                    continue

                job = self.jobs[job_id]
                schedule = job["schedule"]
                source_schedule = source["schedule"]
                features = job["spec"]["features"]

                self.assertEqual(schedule["kind"], source["kind"])
                self.assertEqual(
                    schedule["start_at_offset_minutes"],
                    source_schedule["release_at_offset_minutes"],
                )
                self.assertEqual(job["spec"]["user_id"], source["user_id"])
                self.assertEqual(features, source["spec_json"]["job_features"])

                benchmark = job["workload"]["benchmark"]
                phase = benchmark["phases"][0]
                self.assertEqual(
                    schedule["stop_at_offset_minutes"],
                    source_schedule["active_end_at_offset_minutes"],
                )
                self.assertEqual(
                    benchmark["model"], source["spec_json"]["model_id"]
                )
                self.assertEqual(
                    phase["duration"], source_schedule["duration_minutes"] * 60
                )
                self.assertEqual(
                    phase["concurrency"], features["max_concurrent_streaming"]
                )
                self.assertEqual(
                    phase["type"],
                    phase_types[features["request_arrival_pattern"]],
                )
                self.assertGreater(phase["rate"], 0)

        self.assertEqual(expanded_shards, 678)

    def test_frozen_fleet_and_catalog_match(self) -> None:
        """Keep the real run at 100 GPUs and its configured benchmark ID."""
        self.assertEqual(sum(item["total_devices"] for item in self.resources), 100)
        self.assertEqual(
            {item["accelerator"] for item in self.resources},
            {
                "H100",
                "A100-80GB",
                "A100-40GB",
                "RTXPRO6000",
                "L4",
            },
        )
        self.assertEqual(
            {
                market: sum(item["markets"][market] for item in self.resources)
                for market in ("reserved", "on_demand")
            },
            {"reserved": 68, "on_demand": 32},
        )
        self.assertEqual(
            {
                region: sum(item["regions"][region] for item in self.resources)
                for region in ("central-1",)
            },
            {"central-1": 100},
        )

        catalogs = yaml.safe_load(
            (ROOT / "benchmarks" / "catalog.yaml").read_text(encoding="utf-8")
        )["benchmarks"]
        catalog = catalogs["real_12h"]
        self.assertEqual(catalog["manifest"], "real_12h/benchmark.yaml")
        self.assertEqual(catalog["status"], "executable")

        resource_maps = [
            json.loads(line)
            for line in (ROOT / "benchmarks" / "resource_map_benchmarking.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(
            {row["benchmark_id"] for row in resource_maps},
            {
                benchmark_id
                for benchmark_id, benchmark in catalogs.items()
                if benchmark["status"] == "executable"
            },
        )
        resource_map = next(
            row["resource_map"]
            for row in resource_maps
            if row["benchmark_id"] == "real_12h"
        )
        self.assertEqual(set(resource_map["clouds"]), {"gcp"})
        regions = resource_map["clouds"]["gcp"]["regions"]
        self.assertEqual(set(regions), {"central-1"})
        zone = regions["central-1"]["zones"]["default"]
        self.assertEqual(set(zone["network_fabrics"]), {"tcpip"})
        self.assertFalse(zone["network_fabrics"]["tcpip"]["gpu_direct_rdma"])
        pools = []
        for cloud in resource_map["clouds"].values():
            for region in cloud["regions"].values():
                for zone in region["zones"].values():
                    for fabric in zone["network_fabrics"].values():
                        pools.extend(fabric["machine_pools"].values())
        self.assertEqual(len(pools), 5)
        self.assertEqual(
            {pool["gpu_type"] for pool in pools},
            {"H100", "A100-80GB", "A100-40GB", "RTXPRO6000", "L4"},
        )
        self.assertEqual(
            sum(pool["total_instances"] * pool["gpus_per_instance"] for pool in pools),
            100,
        )

    def test_models_stay_within_the_available_set(self) -> None:
        """Keep every authored job on an approved deployable model."""
        allowed_models = {
            "meta-llama/Llama-3.2-1B-Instruct",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "microsoft/phi-4",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen3-Next-80B-A3B-Thinking",
            "zai-org/GLM-4.5-Air",
            "QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ",
        }

        used_models = {
            job["spec_json"]["model_id"] for job in self.overview_jobs.values()
        }
        self.assertTrue(used_models.issubset(allowed_models))

        with (ROOT / "benchmarks" / "model_catalog_gpu_profiles.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            profiles = list(csv.DictReader(stream))
        expected_fields = {
            "max_num_seq",
            "max_num_batched_tokens",
            "block_size",
            "kvcache_dtype",
        }
        for model_id in used_models:
            self.assertEqual(
                {
                    row["profile_field"]
                    for row in profiles
                    if row["model_id"] == model_id
                    and row["gpu_type"] == "RTXPRO6000"
                },
                expected_fields,
            )


if __name__ == "__main__":
    unittest.main()
