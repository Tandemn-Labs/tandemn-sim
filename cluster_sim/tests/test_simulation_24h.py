"""Validate the executable twenty-four-hour simulation benchmark."""

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

from scenario import _intensity, load_scenario


class SimulationTwentyFourHourTest(unittest.TestCase):
    """Keep the overview, jobs, traces, models, and resources aligned."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the benchmark artifacts once."""
        cls.root = ROOT / "benchmarks" / "simulation_24h"
        cls.manifest = yaml.safe_load(
            (cls.root / "benchmark.yaml").read_text(encoding="utf-8")
        )["benchmark"]
        cls.resources = yaml.safe_load(
            (cls.root / "resources.yaml").read_text(encoding="utf-8")
        )["resources"]
        cls.trace_root = ROOT / "datasets" / "simulation_24h"
        cls.trace_mapping = yaml.safe_load(
            (cls.trace_root / "trace_mapping.yaml").read_text(encoding="utf-8")
        )
        overview_text = (cls.root / "benchmark.md").read_text(encoding="utf-8")
        cls.events = [
            json.loads(block)
            for block in re.findall(r"```json\s*(.*?)```", overview_text, re.S)
        ]
        cls.overview_jobs = {
            job["job_id"]: job for event in cls.events for job in event["jobs"]
        }
        cls.jobs = cls._load_jobs(cls.root)

    def test_manifest_loads_every_workload(self) -> None:
        """Expose all overview jobs through the executable manifest."""
        scenario = load_scenario(self.root / "benchmark.yaml")

        self.assertEqual(self.manifest["status"], "executable")
        self.assertEqual(self.manifest["hard_stop_at_offset_minutes"], 1560)
        self.assertEqual(len(scenario.online_jobs), 24)
        self.assertEqual(len(scenario.batch_jobs), 24)
        self.assertEqual(set(self.jobs), set(self.overview_jobs))

    def test_overview_layout_and_distribution(self) -> None:
        """Keep all releases, windows, campaigns, and totals well formed."""
        expected_offsets = [
            0,
            10,
            30,
            60,
            120,
            180,
            240,
            300,
            420,
            540,
            600,
            690,
            720,
            840,
            930,
            960,
            1020,
            1080,
            1140,
            1200,
            1320,
            1410,
        ]
        self.assertEqual(
            [event["submit_at_offset_minutes"] for event in self.events],
            expected_offsets,
        )

        online_count = 0
        batch_count = 0
        shard_count = 0
        record_count = 0
        for event in self.events:
            release = event["submit_at_offset_minutes"]
            for job in event["jobs"]:
                schedule = job["schedule"]
                self.assertEqual(schedule["release_at_offset_minutes"], release)
                if job["kind"] == "online":
                    online_count += 1
                    stop = schedule["active_end_at_offset_minutes"]
                    self.assertLessEqual(stop, 1440)
                    self.assertGreater(stop, release)
                    self.assertEqual(schedule["duration_minutes"], stop - release)
                    continue

                batch_count += 1
                self.assertLessEqual(schedule["deadline_at_offset_minutes"], 1440)
                self.assertGreater(schedule["deadline_at_offset_minutes"], release)
                self.assertEqual(
                    schedule["shard_count"], job["input_source"]["shard_count"]
                )
                self.assertEqual(
                    schedule["requests_per_shard"],
                    job["input_source"]["requests_per_shard"],
                )
                shard_count += schedule["shard_count"]
                record_count += schedule["shard_count"] * schedule["requests_per_shard"]

        self.assertEqual((online_count, batch_count), (24, 24))
        self.assertEqual(shard_count, 4704)
        self.assertEqual(record_count, 1_426_000)
        releases_by_quarter = []
        for start in range(0, 1440, 360):
            quarter = [
                job
                for event in self.events
                if start <= event["submit_at_offset_minutes"] < start + 360
                for job in event["jobs"]
            ]
            releases_by_quarter.append(
                (
                    sum(job["kind"] == "online" for job in quarter),
                    sum(job["kind"] == "batch" for job in quarter),
                )
            )
        self.assertEqual(releases_by_quarter, [(10, 12), (6, 1), (4, 6), (4, 5)])
        for tick in range(0, 1440, 5):
            active_online = sum(
                job["kind"] == "online"
                and job["schedule"]["release_at_offset_minutes"]
                <= tick
                < job["schedule"]["active_end_at_offset_minutes"]
                for job in self.overview_jobs.values()
            )
            self.assertGreaterEqual(active_online, 8)

    def test_models_and_hardware_match_shorter_simulations(self) -> None:
        """Use exactly the executable 3/12-hour model and accelerator types."""
        models = {self._job_model(job) for job in self.jobs.values()}
        accelerators = {resource["accelerator"] for resource in self.resources}

        for benchmark in ("simulation_3h", "simulation_12h"):
            root = ROOT / "benchmarks" / benchmark
            reference_jobs = self._load_jobs(root)
            reference_resources = yaml.safe_load(
                (root / "resources.yaml").read_text(encoding="utf-8")
            )["resources"]
            self.assertEqual(
                models,
                {self._job_model(job) for job in reference_jobs.values()},
            )
            self.assertEqual(
                accelerators,
                {resource["accelerator"] for resource in reference_resources},
            )

        with (ROOT / "benchmarks" / "model_catalogs.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            catalog = {row["model_id"]: row for row in csv.DictReader(stream)}
        self.assertLessEqual(models, set(catalog))
        for model in models:
            self.assertEqual(catalog[model]["engine_name"], "vllm")
            self.assertGreater(float(catalog[model]["model_size_gb"]), 0)

    def test_all_simulation_traffic_and_distribution(self) -> None:
        """Validate traffic notation and normalized coverage across all horizons."""
        expected_types = {
            "Constant": "constant",
            "Poisson": "poisson",
            "Bursty": "gamma",
            "Diurnal": "gamma",
        }
        for benchmark in ("simulation_3h", "simulation_12h", "simulation_24h"):
            with self.subTest(benchmark=benchmark):
                root = ROOT / "benchmarks" / benchmark
                manifest = yaml.safe_load(
                    (root / "benchmark.yaml").read_text(encoding="utf-8")
                )["benchmark"]
                horizon = manifest["duration_minutes"]
                jobs = self._load_jobs(root)

                for quarter in range(4):
                    start = quarter * horizon // 4
                    stop = (quarter + 1) * horizon // 4
                    releases = [
                        job
                        for job in jobs.values()
                        if start <= job["schedule"]["start_at_offset_minutes"] < stop
                    ]
                    self.assertEqual(
                        {job["schedule"]["kind"] for job in releases},
                        {"online", "batch"},
                    )

                for tick in range(0, horizon, manifest["reoptimization_tick_minutes"]):
                    self.assertTrue(
                        any(
                            job["schedule"]["kind"] == "online"
                            and job["schedule"]["start_at_offset_minutes"]
                            <= tick
                            < job["schedule"]["stop_at_offset_minutes"]
                            for job in jobs.values()
                        )
                    )

                for job in jobs.values():
                    schedule = job["schedule"]
                    if schedule["kind"] == "batch":
                        self.assertEqual(schedule["driver"], "external_mock_data")
                        self.assertLessEqual(
                            schedule["deadline_at_offset_minutes"], horizon
                        )
                        self.assertEqual(
                            schedule["hard_stop_at_offset_minutes"],
                            manifest["hard_stop_at_offset_minutes"],
                        )
                        continue

                    phase = job["workload"]["benchmark"]["phases"][0]
                    features = job["spec"]["features"]
                    pattern = features["request_arrival_pattern"]
                    self.assertEqual(schedule["driver"], "aiperf")
                    self.assertLessEqual(schedule["stop_at_offset_minutes"], horizon)
                    self.assertEqual(phase["type"], expected_types[pattern])
                    self.assertGreater(phase["rate"], 0)
                    self.assertEqual(
                        phase["concurrency"], features["max_concurrent_streaming"]
                    )
                    self.assertEqual(
                        phase["duration"],
                        (
                            schedule["stop_at_offset_minutes"]
                            - schedule["start_at_offset_minutes"]
                        )
                        * 60,
                    )
                    if pattern == "Diurnal":
                        self.assertEqual(phase["rateRamp"], phase["duration"])
                    else:
                        self.assertNotIn("rateRamp", phase)
                    if phase["type"] == "gamma":
                        self.assertGreater(phase["smoothness"], 0)
                    else:
                        self.assertNotIn("smoothness", phase)

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

    def test_online_drivers_have_real_trace_provenance(self) -> None:
        """Tie every AIPerf phase to its real dataset and calibrated profile."""
        mappings = {
            mapping["job_id"]: mapping for mapping in self.trace_mapping["jobs"]
        }
        online_jobs = {
            job_id: job
            for job_id, job in self.jobs.items()
            if job["schedule"]["kind"] == "online"
        }
        self.assertEqual(set(mappings), set(online_jobs))
        self.assertEqual(
            set(self.trace_mapping["sources"]),
            {mapping["source"] for mapping in mappings.values()},
        )
        expected_types = {
            "Constant": "constant",
            "Poisson": "poisson",
            "Bursty": "gamma",
            "Diurnal": "gamma",
        }
        for job_id, job in online_jobs.items():
            with self.subTest(job_id=job_id):
                mapping = mappings[job_id]
                source = (
                    self.trace_root / self.trace_mapping["sources"][mapping["source"]]
                )
                references = self._load_jobs(
                    ROOT / "benchmarks" / mapping["reference_benchmark"]
                )
                reference = references[mapping["reference_job_id"]]
                pattern = job["spec"]["features"]["request_arrival_pattern"]
                phase = job["workload"]["benchmark"]["phases"][0]
                self.assertTrue(source.is_file())
                self.assertEqual(job["schedule"]["driver"], "aiperf")
                self.assertEqual(
                    pattern,
                    reference["spec"]["features"]["request_arrival_pattern"],
                )
                self.assertEqual(phase["type"], expected_types[pattern])
                self.assertEqual(
                    self._phase_profile(job), self._phase_profile(reference)
                )
                self.assertEqual(
                    phase["concurrency"],
                    job["spec"]["features"]["max_concurrent_streaming"],
                )
                if pattern == "Diurnal":
                    self.assertEqual(phase["rateRamp"], phase["duration"])
                else:
                    self.assertNotIn("rateRamp", phase)

    def test_batch_drivers_are_finite_dataset_workloads(self) -> None:
        """Keep every batch campaign finite, partitioned, and externally driven."""
        batch_jobs = [
            job for job in self.jobs.values() if job["schedule"]["kind"] == "batch"
        ]
        self.assertEqual(len(batch_jobs), 24)
        for job in batch_jobs:
            with self.subTest(job_id=job["job_id"]):
                dataset = job["workload"]["dataset"]
                partitioning = dataset["partitioning"]
                self.assertEqual(job["schedule"]["driver"], "external_mock_data")
                self.assertEqual(
                    job["spec"]["features"]["request_arrival_pattern"], "Constant"
                )
                self.assertGreater(partitioning["shard_count"], 0)
                self.assertGreater(partitioning["requests_per_shard"], 0)
                self.assertEqual(
                    dataset["record_count"],
                    partitioning["shard_count"] * partitioning["requests_per_shard"],
                )

    def test_calibrated_load_envelope(self) -> None:
        """Keep the daily trace representative rather than saturating clients."""
        scenario = load_scenario(self.root / "benchmark.yaml")
        online_intensity = sum(
            _intensity(
                float(phase["duration"]),
                float(phase["rate"]),
                float(phase.get("rateRamp", 0)),
            )
            for job in scenario.online_jobs
            for phase in job.workload["phases"]
        )
        boundaries = {
            boundary
            for job in scenario.online_jobs
            for boundary in (job.starts_at_ns, job.stops_at_ns)
        }
        peak_concurrency = max(
            sum(
                int(job.workload["features"]["max_concurrent_streaming"])
                for job in scenario.online_jobs
                if job.starts_at_ns <= boundary < job.stops_at_ns
            )
            for boundary in boundaries
        )
        moderation = [
            job for job in scenario.online_jobs if job.name.endswith(("-0-6", "-840-2"))
        ]

        self.assertAlmostEqual(online_intensity, 4_192_512.6)
        self.assertEqual(peak_concurrency, 624)
        self.assertEqual(len(moderation), 2)
        for job in moderation:
            phase = job.workload["phases"][0]
            self.assertEqual(phase["rate"], 28.0)
            self.assertEqual(phase["concurrency"], 32)
        self.assertLessEqual(
            max(
                job.workload["features"]["max_concurrent_streaming"]
                for job in scenario.online_jobs
            ),
            96,
        )

    def test_resources_match_the_frozen_overview(self) -> None:
        """Expose the declared fleet and the same machine pools as 3/12 hours."""
        self.assertEqual(
            sum(resource["total_devices"] for resource in self.resources), 800
        )
        for resource in self.resources:
            self.assertEqual(
                sum(resource["markets"].values()), resource["total_devices"]
            )
            self.assertEqual(
                sum(resource["regions"].values()), resource["total_devices"]
            )

        rows = {
            row["benchmark_id"]: row["resource_map"]
            for row in map(
                json.loads,
                (ROOT / "benchmarks" / "resource_map_benchmarking.jsonl")
                .read_text(encoding="utf-8")
                .splitlines(),
            )
        }
        expected_shape = self._resource_pool_shape(rows["simulation_24h"])
        self.assertEqual(len(expected_shape), 9)
        self.assertEqual(sum(pool[-1] for pool in expected_shape), 800)
        devices_by_machine = {}
        for pool in expected_shape:
            devices_by_machine[pool[4]] = devices_by_machine.get(pool[4], 0) + pool[-1]
        self.assertEqual(
            devices_by_machine,
            {
                "p5.48xlarge": 160,
                "g5.48xlarge": 192,
                "g6e.48xlarge": 160,
                "Standard_ND96amsr_A100_v4": 128,
                "Standard_NC96ads_A100_v4": 64,
                "Standard_ND96isr_MI300X_v5": 96,
            },
        )
        for benchmark in ("simulation_3h", "simulation_12h"):
            self.assertEqual(
                {pool[:-1] for pool in expected_shape},
                {pool[:-1] for pool in self._resource_pool_shape(rows[benchmark])},
            )

    @staticmethod
    def _load_jobs(root: Path) -> dict[str, dict[str, object]]:
        """Load job documents keyed by benchmark job ID."""
        jobs = {}
        for path in sorted((root / "jobs").glob("*.yaml")):
            job = yaml.safe_load(path.read_text(encoding="utf-8"))
            jobs[job["job_id"]] = job
        return jobs

    @staticmethod
    def _job_model(job: dict[str, object]) -> str:
        """Return the model from either executable workload shape."""
        workload = job["workload"]
        source = workload.get("benchmark") or workload["dataset"]
        return str(source["model"])

    @staticmethod
    def _phase_profile(job: dict[str, object]) -> tuple[str, float, float | None]:
        """Return the trace-derived AIPerf fields independent of duration."""
        phase = job["workload"]["benchmark"]["phases"][0]
        smoothness = phase.get("smoothness")
        return (
            str(phase["type"]),
            float(phase["rate"]),
            float(smoothness) if smoothness is not None else None,
        )

    @staticmethod
    def _resource_pool_shape(
        resource_map: dict[str, object],
    ) -> list[tuple[str, str, str, str, str, str, int, int, int]]:
        """Flatten Store resource pools into comparable hardware identities."""
        pools = []
        for cloud, cloud_body in resource_map["clouds"].items():
            for region, region_body in cloud_body["regions"].items():
                for zone, zone_body in region_body["zones"].items():
                    for fabric, fabric_body in zone_body["network_fabrics"].items():
                        for machine, pool in fabric_body["machine_pools"].items():
                            pools.append(
                                (
                                    cloud,
                                    region,
                                    zone,
                                    fabric,
                                    machine,
                                    pool["gpu_type"],
                                    pool["gpu_memory_gb"],
                                    pool["gpus_per_instance"],
                                    pool["gpus_per_instance"] * pool["total_instances"],
                                )
                            )
        return pools

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
