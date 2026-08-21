"""Validate the executable twelve-hour simulation benchmark."""

# ruff: noqa: E402

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from scenario import load_scenario


class SimulationTwelveHourTest(unittest.TestCase):
    """Keep the overview, jobs, traffic phases, and resources aligned."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the benchmark artifacts once."""
        cls.root = ROOT / "benchmarks" / "simulation_12h"
        cls.manifest = yaml.safe_load(
            (cls.root / "benchmark.yaml").read_text(encoding="utf-8")
        )["benchmark"]
        cls.resources = yaml.safe_load(
            (cls.root / "resources.yaml").read_text(encoding="utf-8")
        )["resources"]
        overview_text = (cls.root / "benchmark.md").read_text(encoding="utf-8")
        events = [
            json.loads(block)
            for block in re.findall(r"```json\s*(.*?)```", overview_text, re.S)
        ]
        cls.overview_jobs = {
            job["job_id"]: job for event in events for job in event["jobs"]
        }
        cls.jobs = {}
        for path in sorted((cls.root / "jobs").glob("*.yaml")):
            job = yaml.safe_load(path.read_text(encoding="utf-8"))
            cls.jobs[job["job_id"]] = job

    def test_manifest_loads_every_workload(self) -> None:
        """Expose all overview jobs through the executable manifest."""
        scenario = load_scenario(self.root / "benchmark.yaml")

        self.assertEqual(self.manifest["status"], "executable")
        self.assertEqual(self.manifest["hard_stop_at_offset_minutes"], 780)
        self.assertEqual(len(scenario.online_jobs), 23)
        self.assertEqual(len(scenario.batch_jobs), 14)
        self.assertEqual(set(self.jobs), set(self.overview_jobs))

    def test_job_yamls_match_the_overview(self) -> None:
        """Keep identity, schedule, model, and features identical."""
        for job_id, overview in self.overview_jobs.items():
            with self.subTest(job_id=job_id):
                job = self.jobs[job_id]
                schedule = job["schedule"]
                source_schedule = overview["schedule"]
                spec = job["spec"]
                source_spec = overview["spec_json"]

                self.assertEqual(schedule["kind"], overview["kind"])
                self.assertEqual(
                    schedule["start_at_offset_minutes"],
                    source_schedule["release_at_offset_minutes"],
                )
                self.assertEqual(spec["user_id"], overview["user_id"])
                self.assertEqual(spec["precision"], source_spec["precision"])
                self.assertEqual(spec["features"], source_spec["job_features"])

                if overview["kind"] == "online":
                    self._assert_online_parity(job, overview)
                else:
                    self._assert_batch_parity(job, overview)

    def test_aiperf_phase_notation_matches_traffic_intent(self) -> None:
        """Use one canonical AIPerf representation for each pattern."""
        expected_types = {
            "Constant": "constant",
            "Poisson": "poisson",
            "Bursty": "gamma",
            "Diurnal": "gamma",
        }
        for job_id, job in self.jobs.items():
            if job["schedule"]["kind"] != "online":
                continue
            with self.subTest(job_id=job_id):
                pattern = job["spec"]["features"]["request_arrival_pattern"]
                phase = job["workload"]["benchmark"]["phases"][0]
                self.assertEqual(phase["type"], expected_types[pattern])
                self.assertGreater(phase["rate"], 0)
                self.assertEqual(
                    phase["concurrency"],
                    job["spec"]["features"]["max_concurrent_streaming"],
                )
                if pattern in {"Bursty", "Diurnal"}:
                    self.assertGreaterEqual(phase["smoothness"], 0.001)
                else:
                    self.assertNotIn("smoothness", phase)
                if pattern == "Diurnal":
                    self.assertEqual(phase["rateRamp"], phase["duration"])
                else:
                    self.assertNotIn("rateRamp", phase)

    def test_resources_match_the_frozen_overview(self) -> None:
        """Expose the declared 400-device heterogeneous fleet."""
        self.assertEqual(
            sum(resource["total_devices"] for resource in self.resources), 400
        )
        for resource in self.resources:
            self.assertEqual(
                sum(resource["markets"].values()), resource["total_devices"]
            )
            self.assertEqual(
                sum(resource["regions"].values()), resource["total_devices"]
            )

    def _assert_online_parity(
        self,
        job: dict[str, object],
        overview: dict[str, object],
    ) -> None:
        schedule = job["schedule"]
        source_schedule = overview["schedule"]
        benchmark = job["workload"]["benchmark"]
        source_spec = overview["spec_json"]
        phase = benchmark["phases"][0]

        self.assertEqual(schedule["driver"], "aiperf")
        self.assertEqual(schedule["service_id"], source_schedule["service_id"])
        self.assertEqual(schedule["operation"], source_schedule["operation"])
        self.assertEqual(
            schedule["stop_at_offset_minutes"],
            source_schedule["active_end_at_offset_minutes"],
        )
        self.assertEqual(benchmark["model"], source_spec["model_id"])
        self.assertEqual(
            benchmark["endpoint"]["path"], overview["input_source"]["endpoint"]
        )
        request_rate = overview["input_source"].get("request_rate_per_second")
        if request_rate is not None:
            self.assertEqual(phase["rate"], request_rate)
        self.assertEqual(
            phase["duration"],
            (schedule["stop_at_offset_minutes"] - schedule["start_at_offset_minutes"])
            * 60,
        )

    def _assert_batch_parity(
        self,
        job: dict[str, object],
        overview: dict[str, object],
    ) -> None:
        schedule = job["schedule"]
        source_schedule = overview["schedule"]
        dataset = job["workload"]["dataset"]
        source = overview["input_source"]
        partitioning = dataset["partitioning"]

        self.assertEqual(schedule["driver"], "external_mock_data")
        self.assertEqual(schedule["campaign_id"], source_schedule["campaign_id"])
        self.assertEqual(
            schedule["deadline_at_offset_minutes"],
            source_schedule["deadline_at_offset_minutes"],
        )
        self.assertEqual(dataset["id"], source["dataset_id"])
        self.assertEqual(dataset["model"], overview["spec_json"]["model_id"])
        self.assertEqual(partitioning["shard_count"], source_schedule["shard_count"])
        self.assertEqual(
            partitioning["requests_per_shard"],
            source_schedule["requests_per_shard"],
        )
        self.assertEqual(
            dataset["record_count"],
            partitioning["shard_count"] * partitioning["requests_per_shard"],
        )


if __name__ == "__main__":
    unittest.main()
