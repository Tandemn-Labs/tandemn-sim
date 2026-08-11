"""Test the cluster coordinator's simulated online data-plane wiring."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import BatchJob, BatchProgress, CreatedPlan, PlannerResult, RankMetricSample, Scenario
from routing import StickyRoundRobinRouter
from runner import ClusterSimulator, LogicalClock, advance_batch_progress
from scenario import load_scenario
from serving import RankResult


class RunnerDataPlaneTest(unittest.TestCase):
    """Verify the coordinator serializes jobs and stages one metric commit."""

    def test_trace_configuration_writes_run_metadata(self) -> None:
        """Start one trace before the first simulation tick."""
        scenario = Scenario("trace-smoke", 300_000_000_000, 300_000_000_000, (), ())

        class JobStore:
            def __init__(self, postgres) -> None:
                pass

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("runner.verify_store_user"),
                patch("tandemn_system_data.clients.JobStore", JobStore),
            ):
                simulator = ClusterSimulator(
                    scenario,
                    "user_1",
                    object(),
                    SimpleNamespace(method_name="first-fit"),
                    LogicalClock(datetime(2026, 1, 1, tzinfo=UTC)),
                    trace_root=Path(directory),
                    trace_run_id="run-1",
                )

            self.assertIsNotNone(simulator.trace_file)
            event = json.loads(simulator.trace_file.read_text(encoding="utf-8"))

        self.assertEqual(event["run_id"], "run-1")
        self.assertEqual(event["method"], "first-fit")

    def test_tick_trace_includes_planner_time_and_canonical_plan(self) -> None:
        """Write one planner outcome before Noop-Orca applies its plan."""
        action = SimpleNamespace(
            model_dump=lambda mode: {"job_id": "job-a", "type": "place"}
        )
        plan = SimpleNamespace(
            plan_id="plan-1",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            status="created",
            actions=[action],
        )
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario(
            "trace-smoke", 300_000_000_000, 300_000_000_000, (), ()
        )
        simulator.user_id = "user_1"
        simulator.postgres = object()
        simulator.planner = SimpleNamespace(
            create_plans=lambda tick, start: PlannerResult(
                tick, (CreatedPlan("plan-1"),)
            )
        )
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator._submitted_jobs = set()
        simulator._admitted_jobs = set()
        simulator.jobs = SimpleNamespace(
            get=lambda job_id: SimpleNamespace(created_at=datetime(2026, 1, 1, tzinfo=UTC))
        )

        class PlanStore:
            def __init__(self, postgres) -> None:
                pass

            def get(self, plan_id):
                return plan

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with (
                patch("tandemn_system_data.clients.PlanStore", PlanStore),
                patch("runner.apply_pending_plans"),
                patch.object(simulator, "_simulate_tick"),
            ):
                simulator.run_tick(0)
                simulator.run_tick(1)

            events = [
                json.loads(line)
                for line in simulator.trace_file.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(events[0]["kind"], "planner_finished")
        self.assertEqual(events[0]["plans"][0]["actions"][0]["job_id"], "job-a")
        self.assertEqual(events[0]["planned_job_ids"], ["job-a"])
        self.assertEqual(events[1]["kind"], "job_admitted")
        self.assertEqual(events[1]["admitted_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(sum(event["kind"] == "job_admitted" for event in events), 1)

    def test_submit_trace_keeps_batch_deadline_and_slo_inputs(self) -> None:
        """Write immutable workload facts at the exact logical release time."""
        simulator = object.__new__(ClusterSimulator)
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.user_id = "user_1"
        simulator.jobs = object()
        simulator._submitted_jobs = set()
        job = BatchJob(
            "batch-a",
            "job-a",
            "model-a",
            60_000_000_000,
            3_660_000_000_000,
            7_260_000_000_000,
            {
                "features": {
                    "tenant_id": "tenant-a",
                    "target_p99_ttft_ms": 1000,
                    "target_p99_tpot_ms": 50,
                }
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with patch("runner.submit_waiting_job"):
                simulator._submit(job)

            event = json.loads(simulator.trace_file.read_text(encoding="utf-8"))

        self.assertEqual(event["job_kind"], "batch")
        self.assertEqual(event["submitted_at"], "2026-01-01T00:01:00+00:00")
        self.assertEqual(event["deadline_at"], "2026-01-01T01:01:00+00:00")

    def test_finish_trace_records_batch_deadline_outcome(self) -> None:
        """Use the simulation clock, not Store wall time, for batch completion."""
        calls = []
        simulator = object.__new__(ClusterSimulator)
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.jobs = SimpleNamespace(
            set_chain_status=lambda *args: calls.append(args),
            transition=lambda *args: calls.append(args),
        )
        simulator._batch_jobs = {
            "job-a": BatchJob(
                "batch-a",
                "job-a",
                "model-a",
                0,
                60_000_000_000,
                120_000_000_000,
                {"features": {}},
            )
        }
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job-a"),
            chains=(SimpleNamespace(chain_id="chain-1"),),
        )

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            simulator._finish(running)

            event = json.loads(simulator.trace_file.read_text(encoding="utf-8"))

        self.assertTrue(event["completed_successfully"])
        self.assertTrue(event["deadline_slo_met"])
        self.assertEqual(len(calls), 2)

    def test_hard_stop_trace_marks_batch_as_unsuccessful(self) -> None:
        """Record the logical drain-boundary failure for an unfinished batch job."""
        simulator = object.__new__(ClusterSimulator)
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job-a"),
            chains=(SimpleNamespace(chain_id="chain-1"),),
        )
        simulator.jobs = SimpleNamespace(
            running_jobs=lambda user_id: (running,),
            set_chain_status=lambda *args: None,
            transition=lambda *args, **kwargs: None,
        )
        simulator.user_id = "user_1"
        simulator._batch_jobs = {"job-a": object()}

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            simulator._hard_stop()

            event = json.loads(simulator.trace_file.read_text(encoding="utf-8"))

        self.assertFalse(event["completed_successfully"])
        self.assertFalse(event["deadline_slo_met"])
        self.assertEqual(event["finish_reason"], "SIMULATION_HARD_STOP")

    def test_active_online_job_runs_rank_and_commits_tick_metrics(self) -> None:
        """Route one admitted job through the ServingSim adapter boundary."""
        source = load_scenario(ROOT / "benchmarks/simulation_3h/benchmark.yaml")
        job = replace(source.online_jobs[0], store_job_id="job_1", stops_at_ns=900_000_000_000)
        scenario = Scenario("runner-smoke", 300_000_000_000, 300_000_000_000, (job,), ())
        chain = SimpleNamespace(
            chain_id="chain_1",
            shape_json={
                "rank_id": "rank_0",
                "env": ["reserved", "sim", "region-1", "zone-1", "H100-SXM-80GB"],
                "count": 1,
                "tp": 1,
                "pp": 1,
                "ep": 1,
                "instance_type": "sim.h100.8gpu",
                "target_p99_ttft_ms": 600,
            },
        )
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job_1", spec_json={"model_id": job.model_name}),
            chains=(chain,),
        )
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = scenario
        simulator.user_id = "user_1"
        simulator.postgres = object()
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.jobs = SimpleNamespace(running_jobs=lambda user_id: (running,))
        simulator.llmservingsim_root = Path(tempfile.gettempdir())
        simulator.container = "servingsim_docker"
        simulator.run_name = "runner-smoke"
        simulator.parallel_ranks_per_job = 1
        simulator.rank_timeout_seconds = 30
        simulator._traffic_starts = {}
        simulator._requests = {job.store_job_id: __import__("scenario").materialize_online_requests(job)}
        simulator._online_jobs = {job.store_job_id: job}
        simulator._batch_jobs = {}
        simulator._batch_progress = {}
        simulator._router = StickyRoundRobinRouter()

        catalog = {
            "regions": [
                {
                    "cloud": "sim",
                    "region": "region-1",
                    "instance_types": [
                        {
                            "instance_type": "sim.h100.8gpu",
                            "accelerators": [
                                {
                                    "name": "H100-SXM-80GB",
                                    "canonical_gpu_name": "H100-SXM-80GB",
                                    "memory_mib_each": 81_920,
                                    "gpu_bandwidth_gbps": 3_350,
                                    "nvlink_bandwidth_gbps": 900,
                                }
                            ],
                            "network": {"network_cards": [{"peak_bandwidth_gbps": 400}]},
                        }
                    ],
                }
            ]
        }
        calls = []

        class HardwareCatalogStore:
            def __init__(self, postgres) -> None:
                pass

            def get(self):
                return SimpleNamespace(catalog=catalog)

        class ResourceMapStore:
            def __init__(self, postgres, user_id) -> None:
                pass

            def get(self):
                return SimpleNamespace(
                    scheduling_summary=lambda: {
                        "reserved|sim|region-1|zone-1|H100-SXM-80GB": {
                            "pools": [
                                {
                                    "instance_type": "sim.h100.8gpu",
                                    "price_per_instance_hour": 10.0,
                                }
                            ]
                        }
                    }
                )

        class GpuMetricStore:
            def __init__(self, postgres) -> None:
                pass

        def prepare(*args):
            deployment = args[0]
            return SimpleNamespace(deployment=deployment)

        def run(rank_runs, max_parallel):
            deployment = rank_runs[0].deployment
            calls.append(("run", max_parallel))
            return (
                RankResult(
                    deployment,
                    (
                        {
                            "instance id": 0,
                            "input": 100,
                            "output": 10,
                            "end_time": 1_000_000_000,
                            "TTFT": 1_000_000,
                            "TPOT": 100_000,
                        },
                    ),
                    (
                        RankMetricSample(
                            10_000_000_000,
                            10_000_000_000,
                            0,
                            {"kv_cache_util": 0.5},
                        ),
                    ),
                    "",
                    "",
                ),
            )

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with (
                patch("tandemn_system_data.clients.HardwareCatalogStore", HardwareCatalogStore),
                patch("tandemn_system_data.clients.ResourceMapStore", ResourceMapStore),
                patch("tandemn_system_data.clients.GpuMetricStore", GpuMetricStore),
                patch("runner.prepare_rank_run", prepare),
                patch("runner.run_ranks", run),
                patch(
                    "runner.commit_tick_metrics",
                    lambda store, metrics: calls.append(("commit", metrics)),
                ),
            ):
                simulator._simulate_tick(0, 0, 300_000_000_000)

            event = json.loads(simulator.trace_file.read_text(encoding="utf-8"))

        self.assertEqual(calls[0], ("run", 1))
        self.assertEqual(calls[1][0], "commit")
        self.assertEqual(len(calls[1][1]), 1)
        self.assertEqual(event["kind"], "gpu_sample")
        self.assertEqual(event["metrics"]["kv_cache_util"], 0.5)

    def test_batch_probe_advances_full_backlog_by_observed_rate(self) -> None:
        """Scale a two-second, twenty-request probe across one five-minute tick."""
        result = RankResult(
            deployment=SimpleNamespace(),
            request_rows=tuple(
                {
                    "instance id": 0,
                    "input": 100,
                    "output": 10,
                    "end_time": 2_000_000_000,
                    "TTFT": 1_000_000,
                    "TPOT": 100_000,
                }
                for _ in range(20)
            ),
            metric_samples=(),
            stdout="",
            stderr="",
        )
        progress = BatchProgress(total_requests=10_000)

        completed = advance_batch_progress(progress, (result,), 300_000_000_000)

        self.assertEqual(completed, 3_000)
        self.assertEqual(progress.completed_requests, 3_000)
        self.assertEqual(progress.next_request_number, 20)

        progress.completed_requests = 9_500
        completed = advance_batch_progress(progress, (result,), 300_000_000_000)

        self.assertEqual(completed, 500)
        self.assertEqual(progress.completed_requests, 10_000)

    def test_finish_stops_chains_before_finishing_job(self) -> None:
        """Apply terminal Store transitions after simulated work completes."""
        calls = []
        simulator = object.__new__(ClusterSimulator)
        simulator.jobs = SimpleNamespace(
            set_chain_status=lambda *args: calls.append(("chain", *args)),
            transition=lambda *args: calls.append(("job", *args)),
        )
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job_1"),
            chains=(SimpleNamespace(chain_id="chain_0"), SimpleNamespace(chain_id="chain_1")),
        )

        simulator._finish(running)

        self.assertEqual([call[0] for call in calls], ["chain", "chain", "job"])
        self.assertEqual(calls[0][1], "chain_0")
        self.assertEqual(calls[1][1], "chain_1")
        self.assertEqual(calls[2][1], "job_1")


if __name__ == "__main__":
    unittest.main()
