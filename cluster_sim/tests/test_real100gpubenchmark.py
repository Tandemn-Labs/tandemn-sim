"""Validate the executable three-hour 96-GPU real benchmark."""

# ruff: noqa: E402

from __future__ import annotations

import csv
import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from bestfit import BEST_FIT_CONFIGURATIONS
from policy import configure_config_policy, load_config_policy
from scenario import load_scenario
from vram import VRAM_REQUIREMENTS_GIB

BENCHMARK_ID = "real100gpubenchmark"
ALLOWED_MODELS = {
    "meta-llama/Llama-3.2-1B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "microsoft/phi-4",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "zai-org/GLM-4.5-Air",
    "QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ",
}
ONLINE_RATES = {
    "job-benchmark_timing_real100gpu-0-1": (16, 0.65),
    "job-benchmark_timing_real100gpu-0-2": (6, 0.2125),
    "job-benchmark_timing_real100gpu-0-3": (8, 1.0),
    "job-benchmark_timing_real100gpu-7-1": (6, 0.3),
    "job-benchmark_timing_real100gpu-7-3": (8, 0.3),
    "job-benchmark_timing_real100gpu-67-2": (8, 0.04),
    "job-benchmark_timing_real100gpu-125-1": (8, 0.06),
}
BATCH_FILES = {
    "batch-mini-nightly-eval.yaml",
    "batch-phi4-quant-probe.yaml",
    "batch-calibration-probe.yaml",
    "batch-sales-reports.yaml",
}


class RealHundredGpuBenchmarkTest(unittest.TestCase):
    """Keep the real benchmark inside its workload and fleet envelope."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load benchmark artifacts once."""
        cls.root = ROOT / "benchmarks" / BENCHMARK_ID
        cls.manifest = yaml.safe_load(
            (cls.root / "benchmark.yaml").read_text(encoding="utf-8")
        )["benchmark"]
        cls.resources = yaml.safe_load(
            (cls.root / "resources.yaml").read_text(encoding="utf-8")
        )["resources"]
        cls.jobs = {}
        for path in sorted((cls.root / "jobs").glob("*.yaml")):
            job = yaml.safe_load(path.read_text(encoding="utf-8"))
            cls.jobs[job["job_id"]] = job

    def test_manifest_loads_selected_workload(self) -> None:
        """Expose the intended seven online and four batch jobs."""
        scenario = load_scenario(self.root / "benchmark.yaml")

        self.assertEqual(self.manifest["id"], BENCHMARK_ID)
        self.assertEqual(self.manifest["environment"], "real_hardware")
        self.assertEqual(self.manifest["duration_minutes"], 180)
        self.assertEqual(self.manifest["drain_minutes"], 30)
        self.assertEqual(self.manifest["hard_stop_at_offset_minutes"], 210)
        self.assertEqual(self.manifest["reoptimization_tick_minutes"], 5)
        self.assertEqual(self.manifest["status"], "executable")
        self.assertEqual(scenario.workload_end_ns, 180 * 60 * 1_000_000_000)
        self.assertEqual(scenario.drain_end_ns, 210 * 60 * 1_000_000_000)
        self.assertEqual(len(scenario.online_jobs), 7)
        self.assertEqual(len(scenario.batch_jobs), 4)
        self.assertEqual(
            {job.name for job in scenario.online_jobs}, set(ONLINE_RATES)
        )
        self.assertEqual(
            {job.name for job in scenario.batch_jobs},
            {
                "job-benchmark_timing_3hr-15-1",
                "job-benchmark_timing_3hr-15-3",
                "job-benchmark_timing_3hr-85-1",
                "job-benchmark_timing_3hr-125-2",
            },
        )

    def test_batch_jobs_are_unchanged_simulation_campaigns(self) -> None:
        """Keep every retained batch definition byte-for-byte unchanged."""
        source = ROOT / "benchmarks" / "simulation_3h" / "jobs"
        records = 0
        shards = 0

        for filename in BATCH_FILES:
            actual = self.root / "jobs" / filename
            job = yaml.safe_load(actual.read_text(encoding="utf-8"))
            source_job = yaml.safe_load(
                (source / filename).read_text(encoding="utf-8")
            )
            self.assertEqual(job, source_job)
            dataset = job["workload"]["dataset"]
            records += dataset["record_count"]
            shards += dataset["partitioning"]["shard_count"]

        self.assertEqual(records, 26_200)
        self.assertEqual(shards, 106)

    def test_online_pressure_is_frozen(self) -> None:
        """Preserve per-replica load and the intended overlap peak."""
        online = [
            job for job in self.jobs.values() if job["schedule"]["kind"] == "online"
        ]
        boundaries = sorted(
            {
                offset
                for job in online
                for offset in (
                    job["schedule"]["start_at_offset_minutes"],
                    job["schedule"]["stop_at_offset_minutes"],
                )
            }
        )
        active_counts = []
        active_concurrency = []

        for job in online:
            job_id = job["job_id"]
            schedule = job["schedule"]
            features = job["spec"]["features"]
            phase = job["workload"]["benchmark"]["phases"][0]
            expected_concurrency, expected_rate = ONLINE_RATES[job_id]
            self.assertEqual(
                phase["duration"],
                60
                * (
                    schedule["stop_at_offset_minutes"]
                    - schedule["start_at_offset_minutes"]
                ),
            )
            self.assertEqual(features["max_concurrent_streaming"], phase["concurrency"])
            self.assertEqual(phase["concurrency"], expected_concurrency)
            self.assertEqual(phase["rate"], expected_rate)
            self.assertGreaterEqual(
                schedule["stop_at_offset_minutes"]
                - schedule["start_at_offset_minutes"],
                45,
            )

        for minute in boundaries[:-1]:
            active = [
                job
                for job in online
                if job["schedule"]["start_at_offset_minutes"]
                <= minute
                < job["schedule"]["stop_at_offset_minutes"]
            ]
            active_counts.append(len(active))
            active_concurrency.append(
                sum(
                    job["spec"]["features"]["max_concurrent_streaming"]
                    for job in active
                )
            )

        self.assertEqual(max(active_counts), 7)
        self.assertEqual(max(active_concurrency), 60)

    def test_frozen_fleet_and_resource_map_match(self) -> None:
        """Keep all 96 GPUs in the exact two-region whole-node layout."""
        self.assertEqual(sum(item["total_devices"] for item in self.resources), 96)
        self.assertEqual(
            {item["accelerator"]: item["total_devices"] for item in self.resources},
            {"RTXPRO6000": 40, "H100": 16, "A100-80GB": 16, "L4": 24},
        )
        self.assertTrue(all(item["devices_per_node"] == 8 for item in self.resources))
        self.assertEqual(
            {
                region: sum(item["regions"][region] for item in self.resources)
                for region in ("us-central1", "europe-west2")
            },
            {"us-central1": 56, "europe-west2": 40},
        )

        catalogs = yaml.safe_load(
            (ROOT / "benchmarks" / "catalog.yaml").read_text(encoding="utf-8")
        )["benchmarks"]
        self.assertEqual(
            catalogs[BENCHMARK_ID],
            {
                "manifest": "real100gpubenchmark/benchmark.yaml",
                "environment": "real_hardware",
                "duration_hours": 3,
                "status": "executable",
            },
        )
        rows = [
            json.loads(line)
            for line in (ROOT / "benchmarks" / "resource_map_benchmarking.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        matching = [row for row in rows if row["benchmark_id"] == BENCHMARK_ID]
        self.assertEqual(len(matching), 1)
        resource_map = matching[0]["resource_map"]
        self.assertEqual(resource_map["market"], ["reserved"])
        regions = resource_map["clouds"]["gcp"]["regions"]
        self.assertEqual(set(regions), {"us-central1", "europe-west2"})

        actual_instances = {}
        pools = []
        for region_name, region in regions.items():
            fabric = region["zones"]["default"]["network_fabrics"]["tcpip"]
            self.assertFalse(fabric["gpu_direct_rdma"])
            for pool in fabric["machine_pools"].values():
                pools.append(pool)
                actual_instances[(region_name, pool["gpu_type"])] = pool[
                    "total_instances"
                ]

        self.assertEqual(len(pools), 8)
        self.assertTrue(all("allocation_kind" not in pool for pool in pools))
        self.assertEqual(
            sum(pool["total_instances"] * pool["gpus_per_instance"] for pool in pools),
            96,
        )
        self.assertEqual(
            actual_instances,
            {
                ("us-central1", "RTXPRO6000"): 3,
                ("us-central1", "H100"): 1,
                ("us-central1", "A100-80GB"): 1,
                ("us-central1", "L4"): 2,
                ("europe-west2", "RTXPRO6000"): 2,
                ("europe-west2", "H100"): 1,
                ("europe-west2", "A100-80GB"): 1,
                ("europe-west2", "L4"): 1,
            },
        )

    def test_models_placements_and_peak_nodes_are_deployable(self) -> None:
        """Keep model policy, VRAM fit, and whole-node pressure consistent."""
        self.addCleanup(configure_config_policy, "simulation")
        configure_config_policy("real_hardware")
        policy = load_config_policy()
        mappings = BEST_FIT_CONFIGURATIONS[BENCHMARK_ID]
        requirements = VRAM_REQUIREMENTS_GIB[BENCHMARK_ID]
        resources = {
            item["accelerator"]: item for item in self.resources
        }
        self.assertEqual(set(mappings), set(self.jobs))
        self.assertEqual(set(requirements), set(self.jobs))

        used_models = set()
        windows = {}
        for job_id, job in self.jobs.items():
            workload = job["workload"]
            source = workload.get("benchmark") or workload["dataset"]
            model_id = source["model"]
            used_models.add(model_id)
            mapping = mappings[job_id]
            configuration = mapping["configuration"]
            hardware = mapping["hardware"]
            rule = policy.rule_for(hardware, model_id)
            self.assertIsNotNone(rule, job_id)
            assert rule is not None
            self.assertTrue(
                rule.allows_parallelism(configuration["tp"], configuration["pp"]),
                job_id,
            )
            per_replica_gpus = configuration["tp"] * configuration["pp"]
            self.assertLessEqual(per_replica_gpus, 8, job_id)
            self.assertEqual(configuration["dp"], 1, job_id)
            self.assertGreaterEqual(
                per_replica_gpus * resources[hardware]["memory_gb"],
                requirements[job_id],
                job_id,
            )
            if job["schedule"]["kind"] == "online":
                self.assertEqual(job["spec"]["precision"], rule.precision)
                end = job["schedule"]["stop_at_offset_minutes"]
            else:
                end = job["schedule"]["deadline_at_offset_minutes"]
            windows[job_id] = (job["schedule"]["start_at_offset_minutes"], end)

        self.assertTrue(used_models.issubset(ALLOWED_MODELS))
        self.assertEqual(len(used_models), 7)
        node_limits = {
            item["accelerator"]: item["total_devices"] // item["devices_per_node"]
            for item in self.resources
        }
        boundaries = sorted(
            {offset for window in windows.values() for offset in window}
        )
        peak_nodes = 0
        for minute in boundaries[:-1]:
            active_by_hardware: dict[str, int] = defaultdict(int)
            for job_id, (start, end) in windows.items():
                if start <= minute < end:
                    mapping = mappings[job_id]
                    active_by_hardware[mapping["hardware"]] += mapping[
                        "configuration"
                    ]["dp"]
            peak_nodes = max(peak_nodes, sum(active_by_hardware.values()))
            for hardware, nodes in active_by_hardware.items():
                self.assertLessEqual(nodes, node_limits[hardware])
        self.assertEqual(peak_nodes, 10)

        with (ROOT / "benchmarks" / "model_catalog_gpu_profiles.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            profiles = list(csv.DictReader(stream))
        required_fields = {
            "max_num_seq",
            "max_num_batched_tokens",
            "block_size",
            "kvcache_dtype",
        }
        for job_id, job in self.jobs.items():
            workload = job["workload"]
            source = workload.get("benchmark") or workload["dataset"]
            hardware = mappings[job_id]["hardware"]
            self.assertEqual(
                {
                    row["profile_field"]
                    for row in profiles
                    if row["model_id"] == source["model"]
                    and row["gpu_type"] == hardware
                },
                required_fields,
                job_id,
            )


if __name__ == "__main__":
    unittest.main()
