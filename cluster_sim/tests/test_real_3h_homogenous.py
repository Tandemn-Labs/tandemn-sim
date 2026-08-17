"""Validate the executable homogeneous L40S benchmark."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from bestfit import BEST_FIT_CONFIGURATIONS
from policy import load_config_policy
from scenario import load_scenario


class RealThreeHourHomogenousTest(unittest.TestCase):
    """Keep the real benchmark executable and L40S-only."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the benchmark documents once for validation."""
        cls.benchmark_root = ROOT / "benchmarks" / "real_3h_homogenous"
        cls.manifest = yaml.safe_load(
            (cls.benchmark_root / "benchmark.yaml").read_text(encoding="utf-8")
        )["benchmark"]
        cls.resources = yaml.safe_load(
            (cls.benchmark_root / "resources.yaml").read_text(encoding="utf-8")
        )["resources"]
        cls.jobs = {}
        for path in sorted((cls.benchmark_root / "jobs").glob("*.yaml")):
            job = yaml.safe_load(path.read_text(encoding="utf-8"))
            cls.jobs[job["job_id"]] = job

    def test_manifest_loads_complete_enterprise_workload(self) -> None:
        """Expose the inherited online and batch mix as a real benchmark."""
        scenario = load_scenario(self.benchmark_root / "benchmark.yaml")

        self.assertEqual(self.manifest["id"], "real_3h_homogenous")
        self.assertEqual(self.manifest["environment"], "real_hardware")
        self.assertEqual(self.manifest["status"], "executable")
        self.assertEqual(len(scenario.online_jobs), 18)
        self.assertEqual(len(scenario.batch_jobs), 7)
        self.assertTrue(
            all(
                "benchmark_timing_real_3hr_homogenous" in job_id for job_id in self.jobs
            )
        )

    def test_resource_map_is_on_demand_l40s_only(self) -> None:
        """Exclude reserved, heterogeneous, and cross-region capacity."""
        self.assertEqual(len(self.resources), 1)
        resource = self.resources[0]
        self.assertEqual(resource["accelerator"], "L40S-48GB")
        self.assertEqual(resource["total_devices"], 128)
        self.assertEqual(resource["devices_per_node"], 8)
        self.assertEqual(resource["markets"], {"on_demand": 128})
        self.assertEqual(resource["regions"], {"us-east": 128})

    def test_fixed_placements_are_supported_and_peak_at_119_gpus(self) -> None:
        """Fit every preferred placement inside the homogeneous fleet."""
        mappings = BEST_FIT_CONFIGURATIONS["real_3h_homogenous"]
        self.assertEqual(set(mappings), set(self.jobs))

        policy = load_config_policy()
        footprints = {}
        boundaries = set()
        for job_id, job in self.jobs.items():
            mapping = mappings[job_id]
            self.assertEqual(mapping["hardware"], "L40S-48GB")
            configuration = mapping["configuration"]
            workload = job["workload"]
            source = workload.get("benchmark") or workload["dataset"]
            rule = policy.rule_for("L40S-48GB", source["model"])
            self.assertIsNotNone(rule, f"unsupported L40S model for {job_id}")
            assert rule is not None
            self.assertIn(configuration["tp"], rule.allowed_tp)
            footprints[job_id] = (
                configuration["tp"] * configuration["pp"] * configuration["dp"]
            )

            schedule = job["schedule"]
            start = schedule["start_at_offset_minutes"]
            stop = (
                schedule.get("stop_at_offset_minutes")
                or schedule["hard_stop_at_offset_minutes"]
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
                    or schedule["hard_stop_at_offset_minutes"]
                )
                if start <= offset < stop:
                    active += footprints[job_id]
            peak = max(peak, active)

        self.assertEqual(peak, 119)
        self.assertLessEqual(peak, self.resources[0]["total_devices"])


if __name__ == "__main__":
    unittest.main()
