"""Validate the executable homogeneous H100 benchmark."""

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

from bestfit import BEST_FIT_CONFIGURATIONS
from policy import load_config_policy
from scenario import load_scenario
from vram import MODEL_ARCHITECTURES, VRAM_REQUIREMENTS_GIB


class SimulationTwelveHourHomogenousTest(unittest.TestCase):
    """Keep the 12-hour simulation executable and H100-only."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the benchmark documents once for validation."""
        cls.benchmark_root = ROOT / "benchmarks" / "simulation_12h_homogenous"
        cls.manifest = yaml.safe_load(
            (cls.benchmark_root / "benchmark.yaml").read_text(encoding="utf-8")
        )["benchmark"]
        cls.resources = yaml.safe_load(
            (cls.benchmark_root / "resources.yaml").read_text(encoding="utf-8")
        )["resources"]
        resource_maps = (
            ROOT / "benchmarks" / "resource_map_benchmarking.jsonl"
        ).read_text(encoding="utf-8")
        cls.store_resource_map = next(
            json.loads(line)["resource_map"]
            for line in resource_maps.splitlines()
            if json.loads(line)["benchmark_id"] == "simulation_12h_homogenous"
        )
        cls.jobs = {}
        for path in sorted((cls.benchmark_root / "jobs").glob("*.yaml")):
            job = yaml.safe_load(path.read_text(encoding="utf-8"))
            cls.jobs[job["job_id"]] = job

    def test_manifest_loads_complete_twelve_hour_workload(self) -> None:
        """Expose the calibrated online and batch mix for 12 hours."""
        scenario = load_scenario(self.benchmark_root / "benchmark.yaml")

        self.assertEqual(self.manifest["id"], "simulation_12h_homogenous")
        self.assertEqual(self.manifest["environment"], "simulation")
        self.assertEqual(self.manifest["duration_minutes"], 720)
        self.assertEqual(self.manifest["hard_stop_at_offset_minutes"], 780)
        self.assertEqual(self.manifest["status"], "executable")
        self.assertEqual(len(scenario.online_jobs), 23)
        self.assertEqual(len(scenario.batch_jobs), 14)
        self.assertEqual(
            sum(
                job["workload"]
                .get("dataset", {})
                .get("partitioning", {})
                .get("shard_count", 0)
                for job in self.jobs.values()
            ),
            1592,
        )
        self.assertTrue(
            all("benchmark_timing_12hr_homogenous" in job_id for job_id in self.jobs)
        )

    def test_resource_map_is_400_on_demand_h100s(self) -> None:
        """Exclude reserved, heterogeneous, and cross-region capacity."""
        self.assertEqual(len(self.resources), 1)
        resource = self.resources[0]
        self.assertEqual(resource["accelerator"], "H100-SXM-80GB")
        self.assertEqual(resource["total_devices"], 400)
        self.assertEqual(resource["devices_per_node"], 8)
        self.assertEqual(resource["allocation_kind"], "gpu")
        self.assertEqual(resource["markets"], {"on_demand": 400})
        self.assertEqual(resource["regions"], {"us-east": 400})
        pool = self.store_resource_map["clouds"]["aws"]["regions"]["us-east-1"][
            "zones"
        ]["us-east-1a"]["network_fabrics"]["efa"]["machine_pools"]["p5.48xlarge"]
        self.assertEqual(pool["total_instances"] * pool["gpus_per_instance"], 400)
        self.assertEqual(pool["allocation_kind"], "gpu")

    def test_models_fit_context_policy_and_h100_memory(self) -> None:
        """Reject unsupported contexts, precisions, and per-replica shapes."""
        mappings = BEST_FIT_CONFIGURATIONS["simulation_12h_homogenous"]
        self.assertEqual(set(mappings), set(self.jobs))
        policy = load_config_policy()
        with (ROOT / "benchmarks" / "model_catalogs.csv").open(
            encoding="utf-8", newline=""
        ) as catalog_file:
            contexts = {
                row["model_id"]: int(row["max_model_len"])
                for row in csv.DictReader(catalog_file)
            }
        with (ROOT / "benchmarks" / "model_catalog_gpu_profiles.csv").open(
            encoding="utf-8", newline=""
        ) as profile_file:
            h100_profiles = {}
            for row in csv.DictReader(profile_file):
                if row["gpu_type"] == "H100":
                    h100_profiles.setdefault(row["model_id"], set()).add(
                        row["profile_field"]
                    )

        requirements = VRAM_REQUIREMENTS_GIB["simulation_12h_homogenous"]
        required_profiles = {
            "block_size",
            "kvcache_dtype",
            "max_num_batched_tokens",
            "max_num_seq",
        }
        for job_id, job in self.jobs.items():
            source = job["workload"].get("benchmark") or job["workload"]["dataset"]
            model_id = source["model"]
            features = job["spec"]["features"]
            self.assertLessEqual(
                features["isl_token_min"],
                features["isl_token_avg"],
                job_id,
            )
            self.assertLessEqual(
                features["isl_token_avg"],
                features["isl_token_max"],
                job_id,
            )
            self.assertLessEqual(
                features["osl_token_min"],
                features["osl_token_avg"],
                job_id,
            )
            self.assertLessEqual(
                features["osl_token_avg"],
                features["osl_token_max"],
                job_id,
            )
            self.assertLessEqual(
                features["isl_token_max"] + features["osl_token_max"],
                contexts[model_id],
                job_id,
            )
            self.assertTrue(
                required_profiles.issubset(h100_profiles[model_id]),
                job_id,
            )

            if job["schedule"]["kind"] == "batch":
                dataset = job["workload"]["dataset"]
                partitioning = dataset["partitioning"]
                self.assertEqual(
                    dataset["record_count"],
                    partitioning["shard_count"]
                    * partitioning["requests_per_shard"],
                    job_id,
                )
                self.assertEqual(
                    features["total_token_budget"],
                    dataset["record_count"]
                    * (features["isl_token_avg"] + features["osl_token_avg"]),
                    job_id,
                )

            rule = policy.rule_for("H100-SXM-80GB", model_id)
            self.assertIsNotNone(rule, f"unsupported H100 model for {job_id}")
            assert rule is not None
            self.assertEqual(job["spec"]["precision"], rule.precision)

            configuration = mappings[job_id]["configuration"]
            self.assertIn(configuration["tp"], rule.allowed_tp)
            architecture = MODEL_ARCHITECTURES[model_id]
            weight_bytes = 1 if rule.precision == "bf16-fp8" else 2
            weight_gib = architecture.parameter_count * weight_bytes / 2**30
            replica_vram_gib = weight_gib + (
                requirements[job_id] - weight_gib
            ) / configuration["dp"]
            usable_vram_gib = (
                configuration["tp"]
                * configuration["pp"]
                * self.resources[0]["memory_gb"]
                * 0.9
            )
            self.assertLessEqual(replica_vram_gib, usable_vram_gib, job_id)

    def test_fixed_placements_fit_capacity_envelopes(self) -> None:
        """Fit nominal work on 400 GPUs and missed-deadline backlog under 500."""
        mappings = BEST_FIT_CONFIGURATIONS["simulation_12h_homogenous"]
        footprints = {}
        boundaries = set()
        for job_id, job in self.jobs.items():
            mapping = mappings[job_id]
            self.assertEqual(mapping["hardware"], "H100-SXM-80GB")
            configuration = mapping["configuration"]
            footprints[job_id] = (
                configuration["tp"] * configuration["pp"] * configuration["dp"]
            )

            schedule = job["schedule"]
            start = schedule["start_at_offset_minutes"]
            stop = (
                schedule.get("stop_at_offset_minutes")
                or schedule["deadline_at_offset_minutes"]
            )
            boundaries.update((start, stop))

        peak = 0
        for offset in sorted(boundaries):
            active = 0
            for job_id, job in self.jobs.items():
                schedule = job["schedule"]
                start = schedule["start_at_offset_minutes"]
                stop = (
                    schedule.get("stop_at_offset_minutes")
                    or schedule["deadline_at_offset_minutes"]
                )
                if start <= offset < stop:
                    active += footprints[job_id]
            peak = max(peak, active)

        self.assertEqual(peak, 286)
        self.assertLessEqual(peak, self.resources[0]["total_devices"])

        hard_stop_peak = 0
        for offset in range(0, 781, 5):
            active = 0
            for job_id, job in self.jobs.items():
                schedule = job["schedule"]
                stop = (
                    schedule.get("stop_at_offset_minutes")
                    or schedule["hard_stop_at_offset_minutes"]
                )
                if schedule["start_at_offset_minutes"] <= offset < stop:
                    active += footprints[job_id]
            hard_stop_peak = max(hard_stop_peak, active)
        self.assertEqual(hard_stop_peak, 472)
        self.assertLessEqual(hard_stop_peak, 500)

    def test_replacement_has_no_additive_overlap(self) -> None:
        """End the superseded premium service when its replacement starts."""
        replacement = self.jobs["job-benchmark_timing_12hr_homogenous-465-1"]
        original = self.jobs["job-benchmark_timing_12hr_homogenous-0-3"]

        self.assertEqual(replacement["schedule"]["operation"], "replace_service")
        self.assertEqual(
            replacement["schedule"]["supersedes_service_id"],
            original["schedule"]["service_id"],
        )
        self.assertEqual(
            original["schedule"]["stop_at_offset_minutes"],
            replacement["schedule"]["start_at_offset_minutes"],
        )


if __name__ == "__main__":
    unittest.main()
