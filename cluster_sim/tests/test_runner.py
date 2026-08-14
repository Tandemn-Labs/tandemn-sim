"""Test the cluster coordinator's simulated online data-plane wiring."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import (
    BatchJob,
    BatchProgress,
    CreatedPlan,
    OnlineJob,
    PlannerResult,
    RankMetricSample,
    Request,
    Scenario,
)
from routing import StickyRoundRobinRouter
from runner import ClusterSimulator, LogicalClock, advance_batch_progress
from scenario import load_scenario
from serving import RankResult


class RunnerDataPlaneTest(unittest.TestCase):
    """Verify the coordinator runs jobs and stages one metric commit."""

    def test_run_trace_ends_after_hard_stop(self) -> None:
        """Write the terminal trace event after every running job is stopped."""
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario("trace-smoke", 1, 0, (), ())
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with patch.object(simulator, "_hard_stop") as hard_stop:
                result = simulator.run()

            event = json.loads(simulator.trace_file.read_text(encoding="utf-8").splitlines()[0])

        hard_stop.assert_called_once_with()
        self.assertEqual(result, ())
        self.assertEqual(event["kind"], "run_finished")

    def test_run_honors_tick_limit_before_drain(self) -> None:
        """Hard-stop after the requested benchmark tick count."""
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario("trace-smoke", 1, 10, (), ())
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.run_tick = lambda tick: SimpleNamespace(tick_number=tick)

        with patch.object(simulator, "_hard_stop"):
            result = simulator.run(2)

        self.assertEqual([item.tick_number for item in result], [0, 1])

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
            event = json.loads(simulator.trace_file.read_text(encoding="utf-8").splitlines()[0])

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
        simulator._admission_latencies_ms = []
        simulator._admissions_in_tick = 0
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job-a"),
            ranks=(
                SimpleNamespace(
                    plan_id="plan-1",
                    rank_id="rank-a",
                    n_replicas=1,
                    shape_json={"count": 1},
                ),
            ),
        )
        simulator.jobs = SimpleNamespace(
            get=lambda job_id: SimpleNamespace(created_at=datetime(2026, 1, 1, tzinfo=UTC)),
            running_jobs=lambda user_id: (running,),
        )

        class PlanStore:
            def __init__(self, postgres) -> None:
                pass

            def get(self, plan_id):
                return plan

        class ResourceMapStore:
            def __init__(self, postgres, user_id) -> None:
                pass

            def get(self):
                return SimpleNamespace(
                    scheduling_summary=lambda: {
                        "reserved|sim|region-1|zone-1|H100": {
                            "pools": [
                                {
                                    "fabric_id": "default",
                                    "instance_type": "sim.h100.8gpu",
                                    "total_instances": 1,
                                    "gpus_per_instance": 8,
                                    "total": 8,
                                    "price_per_instance_hour": 10.0,
                                }
                            ]
                        },
                        "reserved|sim|region-2|zone-1|H100": {
                            "pools": [
                                {
                                    "fabric_id": "default",
                                    "instance_type": "sim.h100.8gpu",
                                    "total_instances": 1,
                                    "gpus_per_instance": 8,
                                    "total": 8,
                                    "price_per_instance_hour": 10.0,
                                }
                            ]
                        }
                    }
                )

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with (
                patch("tandemn_system_data.clients.PlanStore", PlanStore),
                patch("tandemn_system_data.clients.ResourceMapStore", ResourceMapStore),
                patch("runner.apply_pending_plans"),
                patch.object(simulator, "_simulate_tick"),
            ):
                simulator.run_tick(0)
                simulator.run_tick(1)

            events = [
                json.loads(line)
                for line in simulator.trace_file.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(events[0]["kind"], "tick_started")
        self.assertEqual(events[1]["kind"], "resource_snapshot")
        self.assertEqual(events[2]["kind"], "planner_finished")
        self.assertEqual(events[2]["plans"][0]["actions"][0]["job_id"], "job-a")
        self.assertEqual(events[2]["planned_job_ids"], ["job-a"])
        self.assertEqual(events[3]["kind"], "job_admitted")
        self.assertEqual(events[3]["admitted_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(sum(event["kind"] == "job_admitted" for event in events), 1)
        self.assertEqual(sum(event["kind"] == "placement_observed" for event in events), 2)
        self.assertEqual(simulator._admission_latencies_ms, [0.0])

    def test_tick_trace_appends_new_koi_debug_facts(self) -> None:
        """Copy Koi JSONL facts once after each simulator planner tick."""
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario("trace-smoke", 300_000_000_000, 300_000_000_000, (), ())
        simulator.user_id = "user_1"
        simulator.postgres = object()
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator._submitted_jobs = set()
        simulator._admitted_jobs = set()
        simulator.jobs = SimpleNamespace(running_jobs=lambda user_id: ())

        with tempfile.TemporaryDirectory() as directory:
            koi_events_path = Path(directory) / "koi-events.jsonl"
            koi_events_path.write_text(
                '{"kind":"state","tick":0,"ts":"2026-01-01T00:00:00+00:00",'
                '"payload":{"state":"S1_OBSERVE","duration_ms":1}}\n',
                encoding="utf-8",
            )
            simulator.planner = SimpleNamespace(
                create_plans=lambda tick, start: PlannerResult(tick, ()),
                debug_logger=SimpleNamespace(events_path=koi_events_path),
            )
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with (
                patch(
                    "tandemn_system_data.clients.ResourceMapStore",
                    return_value=SimpleNamespace(
                        get=lambda: SimpleNamespace(scheduling_summary=lambda: {})
                    ),
                ),
                patch("runner.apply_pending_plans"),
                patch.object(simulator, "_simulate_tick"),
            ):
                simulator.run_tick(0)
                simulator.run_tick(1)

            events = [
                json.loads(line)
                for line in simulator.trace_file.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(sum(event["kind"] == "koi_state" for event in events), 1)

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

            event = json.loads(simulator.trace_file.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(event["job_kind"], "batch")
        self.assertEqual(event["submitted_at"], "2026-01-01T00:01:00+00:00")
        self.assertEqual(event["deadline_at"], "2026-01-01T01:01:00+00:00")

    def test_finish_trace_records_batch_deadline_outcome(self) -> None:
        """Use the simulation clock, not Store wall time, for batch completion."""
        calls = []
        simulator = object.__new__(ClusterSimulator)
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.jobs = SimpleNamespace(
            set_rank_status=lambda *args: calls.append(args),
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
            ranks=(SimpleNamespace(rank_id="rank-1"),),
        )

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            simulator._finish(running)

            event = json.loads(simulator.trace_file.read_text(encoding="utf-8").splitlines()[-1])

        self.assertTrue(event["completed_successfully"])
        self.assertTrue(event["deadline_slo_met"])
        self.assertEqual(len(calls), 2)

    def test_hard_stop_trace_marks_batch_as_unsuccessful(self) -> None:
        """Record the logical drain-boundary failure for an unfinished batch job."""
        simulator = object.__new__(ClusterSimulator)
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job-a"),
            ranks=(SimpleNamespace(rank_id="rank-1"),),
        )
        simulator.jobs = SimpleNamespace(
            running_jobs=lambda user_id: (running,),
            set_rank_status=lambda *args: None,
            transition=lambda *args, **kwargs: None,
        )
        simulator.user_id = "user_1"
        simulator._online_jobs = {}
        simulator._online_backlog = {}
        simulator._batch_jobs = {"job-a": object()}
        simulator._batch_progress = {
            "job-a": BatchProgress(total_requests=10_000, completed_requests=3_000)
        }

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            simulator._hard_stop()

            event = json.loads(simulator.trace_file.read_text(encoding="utf-8").splitlines()[-1])

        self.assertFalse(event["completed_successfully"])
        self.assertFalse(event["deadline_slo_met"])
        self.assertEqual(event["finish_reason"], "SIMULATION_HARD_STOP")
        self.assertEqual(event["unserved_request_count"], 7_000)

    def test_hard_stop_trace_counts_online_backlog(self) -> None:
        """Report every online request still waiting at the hard stop."""
        simulator = object.__new__(ClusterSimulator)
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job-a"),
            ranks=(SimpleNamespace(rank_id="rank-1"),),
        )
        simulator.jobs = SimpleNamespace(
            running_jobs=lambda user_id: (running,),
            set_rank_status=lambda *args: None,
            transition=lambda *args, **kwargs: None,
        )
        simulator.user_id = "user_1"
        simulator._online_jobs = {"job-a": object()}
        simulator._online_backlog = {"job-a": (object(), object(), object())}
        simulator._batch_jobs = {}
        simulator._batch_progress = {}

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            simulator._hard_stop()

            event = json.loads(simulator.trace_file.read_text(encoding="utf-8").splitlines()[-1])

        self.assertEqual(event["unserved_request_count"], 3)

    def test_hard_stop_censors_waiting_batch(self) -> None:
        """Record full batch backlog when the campaign never reaches a rank."""
        simulator = object.__new__(ClusterSimulator)
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        waiting = SimpleNamespace(job_id="job-a")
        calls = []
        simulator.jobs = SimpleNamespace(
            running_jobs=lambda user_id: (),
            waiting_jobs=lambda user_id: (waiting,),
            transition=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        simulator.user_id = "user_1"
        simulator._online_jobs = {}
        simulator._online_backlog = {}
        simulator._batch_jobs = {"job-a": object()}
        simulator._batch_progress = {"job-a": BatchProgress(total_requests=10)}

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            simulator._hard_stop()
            event = json.loads(simulator.trace_file.read_text(encoding="utf-8").splitlines()[-1])

        self.assertEqual(event["unserved_request_count"], 10)
        self.assertEqual(len(calls), 1)

    def test_active_jobs_share_one_global_rank_pool(self) -> None:
        """Launch ranks from different jobs in one bounded worker pool."""
        jobs = tuple(
            OnlineJob(name, job_id, "model", 0, 600_000_000_000, {})
            for name, job_id in (("online-a", "job-a"), ("online-b", "job-b"))
        )
        running_jobs = tuple(
            SimpleNamespace(job=SimpleNamespace(job_id=job.store_job_id), ranks=())
            for job in jobs
        )
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario(
            "global-pool", 300_000_000_000, 300_000_000_000, jobs, ()
        )
        simulator.user_id = "user_1"
        simulator.postgres = object()
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.jobs = SimpleNamespace(running_jobs=lambda user_id: running_jobs)
        simulator.llmservingsim_root = Path(tempfile.gettempdir())
        simulator.parallel_simulators = 5
        simulator._traffic_starts = {}
        simulator._requests = {
            job.store_job_id: (
                Request(f"{job.name}/request", "session", 0, 0, 1, 1, None),
            )
            for job in jobs
        }
        simulator._online_jobs = {job.store_job_id: job for job in jobs}
        simulator._online_backlog = {job.store_job_id: () for job in jobs}
        simulator._batch_jobs = {}
        simulator._batch_progress = {}
        simulator.trace_file = None

        rank_runs = {
            "job-a": (SimpleNamespace(deployment="deployment-a", request_count=1),),
            "job-b": (SimpleNamespace(deployment="deployment-b", request_count=1),),
        }
        result_a = SimpleNamespace(completed_requests=(), request_rows=())
        result_b = SimpleNamespace(completed_requests=(), request_rows=())
        launched = []
        finalized = []

        class HardwareCatalogStore:
            def __init__(self, postgres) -> None:
                pass

            def get(self):
                return SimpleNamespace(catalog={})

        class ResourceMapStore:
            def __init__(self, postgres, user_id) -> None:
                pass

            def get(self):
                return SimpleNamespace(scheduling_summary=lambda: {})

        def prepare(running, *args):
            return rank_runs[running.job.job_id]

        def run(runs, max_parallel, deadline_epoch_seconds, progress=None):
            launched.append((runs, max_parallel, deadline_epoch_seconds))
            return result_a, result_b

        def metrics(job, deployments, results, *args):
            finalized.append((job.job_id, deployments, results))
            return []

        with (
            patch(
                "tandemn_system_data.clients.HardwareCatalogStore",
                HardwareCatalogStore,
            ),
            patch("tandemn_system_data.clients.ResourceMapStore", ResourceMapStore),
            patch.object(simulator, "_prepare_job_runs", side_effect=prepare),
            patch("runner.run_ranks", side_effect=run),
            patch("runner.gpu_metrics_from_results", side_effect=metrics),
            patch("runner.time", return_value=1_000),
        ):
            simulator._simulate_tick(0, 0, 300_000_000_000)

        self.assertEqual(
            launched,
            [((*rank_runs["job-a"], *rank_runs["job-b"]), 5, 1_300)],
        )
        self.assertEqual(
            finalized,
            [
                ("job-a", ("deployment-a",), (result_a,)),
                ("job-b", ("deployment-b",), (result_b,)),
            ],
        )

    def test_online_backlog_retries_before_fresh_requests_and_drains(self) -> None:
        """Retry unfinished traffic each tick until the stopped source drains."""
        job = OnlineJob("online", "job", "model", 0, 600_000_000_000, {})
        running = SimpleNamespace(job=SimpleNamespace(job_id="job"), ranks=())
        requests = (
            Request("r0", "s0", 0, 0, 1, 1, None),
            Request("r1", "s1", 0, 1, 1, 1, None),
            Request("r2", "s2", 0, 300_000_000_000, 1, 1, None),
        )
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario(
            "backlog", 300_000_000_000, 900_000_000_000, (job,), ()
        )
        simulator.user_id = "user_1"
        simulator.postgres = object()
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.jobs = SimpleNamespace(running_jobs=lambda user_id: (running,))
        simulator.llmservingsim_root = Path(tempfile.gettempdir())
        simulator.parallel_simulators = 1
        simulator._traffic_starts = {}
        simulator._requests = {job.store_job_id: requests}
        simulator._online_jobs = {job.store_job_id: job}
        simulator._online_backlog = {job.store_job_id: ()}
        simulator._batch_jobs = {}
        simulator._batch_progress = {}
        simulator.trace_file = None

        attempts = []
        completed_names = iter(("r1", "r2", "r0"))

        class HardwareCatalogStore:
            def __init__(self, postgres) -> None:
                pass

            def get(self):
                return SimpleNamespace(catalog={})

        class ResourceMapStore:
            def __init__(self, postgres, user_id) -> None:
                pass

            def get(self):
                return SimpleNamespace(scheduling_summary=lambda: {})

        def prepare(running, job_name, tick_requests, *args):
            attempts.append(tick_requests)
            return (
                SimpleNamespace(
                    deployment="deployment",
                    requests=tick_requests,
                    request_count=len(tick_requests),
                ),
            )

        def run(rank_runs, max_parallel, deadline_epoch_seconds, progress=None):
            name = next(completed_names)
            completed = next(
                request
                for request in rank_runs[0].requests
                if request.request_name == name
            )
            return (
                SimpleNamespace(
                    completed_requests=(completed,),
                    request_rows=({"TTFT": 1, "TPOT": 1},),
                ),
            )

        with (
            patch(
                "tandemn_system_data.clients.HardwareCatalogStore",
                HardwareCatalogStore,
            ),
            patch("tandemn_system_data.clients.ResourceMapStore", ResourceMapStore),
            patch.object(simulator, "_prepare_job_runs", side_effect=prepare),
            patch("runner.run_ranks", side_effect=run),
            patch("runner.gpu_metrics_from_results", return_value=[]),
            patch.object(simulator, "_finish") as finish,
        ):
            simulator._simulate_tick(0, 0, 300_000_000_000)
            self.assertEqual(
                [request.request_name for request in simulator._online_backlog["job"]],
                ["r0"],
            )
            simulator.clock.advance(timedelta(minutes=5))
            simulator._simulate_tick(1, 300_000_000_000, 600_000_000_000)
            self.assertEqual(
                [request.request_name for request in simulator._online_backlog["job"]],
                ["r0"],
            )
            finish.assert_not_called()
            simulator.clock.advance(timedelta(minutes=5))
            simulator._simulate_tick(2, 600_000_000_000, 900_000_000_000)

        self.assertEqual(
            [[request.request_name for request in tick] for tick in attempts],
            [["r0", "r1"], ["r0", "r2"], ["r0"]],
        )
        self.assertEqual(attempts[1][0].arrives_at_ns, 300_000_000_000)
        self.assertEqual(attempts[2][0].arrives_at_ns, 600_000_000_000)
        self.assertEqual(simulator._online_backlog["job"], ())
        finish.assert_called_once_with(
            running,
            datetime(2026, 1, 1, 0, 15, tzinfo=UTC),
        )

    def test_failed_metric_commit_keeps_request_state_unchanged(self) -> None:
        """Do not apply staged online or batch progress after commit failure."""
        online = OnlineJob("online", "online-job", "model", 0, 300_000_000_000, {})
        batch = BatchJob(
            "batch",
            "batch-job",
            "model",
            0,
            300_000_000_000,
            600_000_000_000,
            {"features": {}},
        )
        online_request = Request("online-r0", "session", 0, 0, 1, 1, None)
        fresh_request = Request("online-r1", "fresh-session", 0, 1, 1, 1, None)
        batch_request = Request("batch-r0", "batch-session", 0, 0, 1, 1, None)
        running_jobs = (
            SimpleNamespace(job=SimpleNamespace(job_id="online-job"), ranks=()),
            SimpleNamespace(job=SimpleNamespace(job_id="batch-job"), ranks=()),
        )
        original_progress = BatchProgress(total_requests=100)
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario(
            "atomic", 300_000_000_000, 600_000_000_000, (online,), (batch,)
        )
        simulator.user_id = "user_1"
        simulator.postgres = object()
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.jobs = SimpleNamespace(running_jobs=lambda user_id: running_jobs)
        simulator.llmservingsim_root = Path(tempfile.gettempdir())
        simulator.parallel_simulators = 2
        simulator._traffic_starts = {}
        simulator._requests = {"online-job": (fresh_request,)}
        simulator._online_jobs = {"online-job": online}
        simulator._online_backlog = {"online-job": (online_request,)}
        simulator._batch_jobs = {"batch-job": batch}
        simulator._batch_progress = {"batch-job": original_progress}
        simulator.trace_file = None

        rank_runs = iter(
            (
                (
                    SimpleNamespace(
                        deployment="online-deployment", request_count=2
                    ),
                ),
                (
                    SimpleNamespace(
                        deployment="batch-deployment", request_count=1
                    ),
                ),
            )
        )
        result_row = {
            "TTFT": 1_000_000,
            "TPOT": 100_000,
            "end_time": 1_000_000_000,
        }
        results = (
            SimpleNamespace(
                completed_requests=(online_request,),
                request_rows=(result_row,),
                status="completed",
                sim_time_ns=1_000_000_000,
            ),
            SimpleNamespace(
                completed_requests=(batch_request,),
                request_rows=(result_row,),
                status="completed",
                sim_time_ns=1_000_000_000,
            ),
        )

        class HardwareCatalogStore:
            def __init__(self, postgres) -> None:
                pass

            def get(self):
                return SimpleNamespace(catalog={})

        class ResourceMapStore:
            def __init__(self, postgres, user_id) -> None:
                pass

            def get(self):
                return SimpleNamespace(scheduling_summary=lambda: {})

        class GpuMetricStore:
            def __init__(self, postgres) -> None:
                pass

        with (
            patch(
                "tandemn_system_data.clients.HardwareCatalogStore",
                HardwareCatalogStore,
            ),
            patch("tandemn_system_data.clients.ResourceMapStore", ResourceMapStore),
            patch("tandemn_system_data.clients.GpuMetricStore", GpuMetricStore),
            patch.object(simulator, "_prepare_job_runs", side_effect=lambda *args: next(rank_runs)),
            patch("runner.build_batch_probe", return_value=(batch_request,)),
            patch("runner.run_ranks", return_value=results),
            patch("runner.gpu_metrics_from_results", return_value=[object()]),
            patch("runner.commit_tick_metrics", side_effect=RuntimeError("commit failed")),
            patch.object(simulator, "_finish") as finish,
        ):
            with self.assertRaisesRegex(RuntimeError, "commit failed"):
                simulator._simulate_tick(0, 0, 300_000_000_000)

        self.assertEqual(
            simulator._online_backlog["online-job"], (online_request,)
        )
        self.assertIs(simulator._batch_progress["batch-job"], original_progress)
        self.assertEqual(original_progress.completed_requests, 0)
        self.assertEqual(original_progress.next_request_number, 0)
        finish.assert_not_called()

    def test_active_online_job_runs_rank_and_commits_tick_metrics(self) -> None:
        """Route one admitted job through the ServingSim adapter boundary."""
        source = load_scenario(ROOT / "benchmarks/simulation_3h/benchmark.yaml")
        job = replace(source.online_jobs[0], store_job_id="job_1", stops_at_ns=900_000_000_000)
        scenario = Scenario("runner-smoke", 300_000_000_000, 300_000_000_000, (job,), ())
        rank = SimpleNamespace(
            rank_id="rank_1",
            n_replicas=1,
            shape_json={
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
            ranks=(rank,),
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
        simulator.parallel_simulators = 1
        simulator.rank_timeout_seconds = 30
        simulator._traffic_starts = {}
        simulator._requests = {job.store_job_id: __import__("scenario").materialize_online_requests(job)}
        simulator._online_jobs = {job.store_job_id: job}
        simulator._online_backlog = {job.store_job_id: ()}
        simulator._batch_jobs = {}
        simulator._batch_progress = {}
        simulator._admission_latencies_ms = [12_000.0]
        simulator._admissions_in_tick = 2
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
                                    "fabric_id": "default",
                                    "instance_type": "sim.h100.8gpu",
                                    "total": 8,
                                    "total_instances": 1,
                                    "gpus_per_instance": 8,
                                    "price_per_instance_hour": 10.0,
                                }
                            ]
                        },
                        "reserved|sim|region-2|zone-1|H100-SXM-80GB": {
                            "pools": [
                                {
                                    "fabric_id": "default",
                                    "instance_type": "sim.h100.8gpu",
                                    "total": 8,
                                    "total_instances": 1,
                                    "gpus_per_instance": 8,
                                    "price_per_instance_hour": 10.0,
                                }
                            ]
                        }
                    }
                )

        staged = []

        class GpuMetricStore:
            def __init__(self, postgres) -> None:
                pass

            def rows_for_job_window(self, user_id, job_id, start, end):
                return [metric for metric in staged if metric.job_id == job_id]

        class ModelCatalogStore:
            def __init__(self, postgres) -> None:
                pass

            def get(self, model_id):
                return SimpleNamespace(
                    weight_dtype="bfloat16",
                    weight_quantization_method="none",
                )

        def prepare(*args):
            deployment = args[0]
            return SimpleNamespace(deployment=deployment, request_count=1)

        def run(rank_runs, max_parallel, deadline_epoch_seconds, progress=None):
            deployment = rank_runs[0].deployment
            calls.append(("run", max_parallel, deadline_epoch_seconds))
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
                patch("tandemn_system_data.clients.ModelCatalogStore", ModelCatalogStore),
                patch("runner.prepare_rank_run", prepare),
                patch("runner.run_ranks", run),
                patch("runner.time", return_value=1_000),
                patch(
                    "runner.commit_tick_metrics",
                    lambda store, metrics: calls.append(("commit", metrics))
                    or staged.extend(metrics),
                ),
            ):
                simulator._simulate_tick(0, 0, 300_000_000_000)

            events = [
                json.loads(line)
                for line in simulator.trace_file.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(calls[0], ("run", 1, 1_300))
        self.assertEqual(calls[1][0], "commit")
        self.assertEqual(len(calls[1][1]), 1)
        gpu_event = next(event for event in events if event["kind"] == "gpu_sample")
        rank_event = next(event for event in events if event["kind"] == "rank_window")
        job_event = next(event for event in events if event["kind"] == "job_window")
        pool_event = next(event for event in events if event["kind"] == "pool_window")
        cluster_event = next(event for event in events if event["kind"] == "cluster_window")
        self.assertEqual(gpu_event["metrics"]["kv_cache_util"], 0.5)
        self.assertEqual(rank_event["v_actual"]["kv_cache_util"], [0.5])
        self.assertEqual(job_event["job_id"], "job_1")
        self.assertTrue(job_event["ttft_slo_met"])
        self.assertEqual(job_event["attempted_request_count"], 1)
        self.assertEqual(job_event["completed_request_count"], 1)
        self.assertEqual(job_event["unfinished_request_count"], 0)
        self.assertEqual(job_event["completion_ratio"], 1.0)
        self.assertEqual(job_event["latency_scope"], "completed_requests")
        self.assertEqual(pool_event["expected_gpu_count"], 8)
        self.assertEqual(pool_event["allocated_gpu_count"], 1)
        self.assertEqual(cluster_event["capacity_gpu_hours"], 16 * 300 / 3600)
        self.assertEqual(cluster_event["jobs_admitted_in_tick"], 2)
        self.assertEqual(cluster_event["rolling_admission_time_ms"], 12_000.0)
        second_pool = next(
            event
            for event in events
            if event["kind"] == "pool_window"
            and event["pool_id"].startswith("reserved|sim|region-2")
        )
        self.assertEqual(second_pool["reported_gpu_count"], 0)

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

    def test_batch_cutoff_scales_each_rank_by_simulated_observation_time(self) -> None:
        """Project partial rank probes using their reported simulated horizons."""
        row = {
            "instance id": 0,
            "input": 100,
            "output": 10,
            "end_time": 1_000_000_000,
            "TTFT": 1_000_000,
            "TPOT": 100_000,
        }
        results = (
            RankResult(
                SimpleNamespace(),
                tuple(dict(row) for _ in range(20)),
                (),
                "",
                "",
                "wall_time_limit",
                (),
                10_000_000_000,
            ),
            RankResult(
                SimpleNamespace(),
                tuple(dict(row) for _ in range(10)),
                (),
                "",
                "",
                "wall_time_limit",
                (),
                5_000_000_000,
            ),
        )
        progress = BatchProgress(total_requests=100_000)

        completed = advance_batch_progress(
            progress,
            results,
            300_000_000_000,
            attempted_requests=200,
        )

        self.assertEqual(completed, 1_200)
        self.assertEqual(progress.completed_requests, 1_200)
        self.assertEqual(progress.next_request_number, 200)

    def test_batch_completion_uses_observed_in_tick_time(self) -> None:
        """Record batch completion at its replay-derived time inside the tick."""
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
        progress = BatchProgress(total_requests=10_000, completed_requests=9_500)

        advance_batch_progress(progress, (result,), 300_000_000_000, 600_000_000_000)

        self.assertEqual(progress.completed_at_ns, 650_000_000_000)

    def test_finish_stops_ranks_before_finishing_job(self) -> None:
        """Apply terminal Store transitions after simulated work completes."""
        calls = []
        simulator = object.__new__(ClusterSimulator)
        simulator.jobs = SimpleNamespace(
            set_rank_status=lambda *args: calls.append(("rank", *args)),
            transition=lambda *args: calls.append(("job", *args)),
        )
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job_1"),
            ranks=(SimpleNamespace(rank_id="rank_0"), SimpleNamespace(rank_id="rank_1")),
        )

        simulator._finish(running)

        self.assertEqual([call[0] for call in calls], ["rank", "rank", "job"])
        self.assertEqual(calls[0][1], "rank_0")
        self.assertEqual(calls[1][1], "rank_1")
        self.assertEqual(calls[2][1], "job_1")


if __name__ == "__main__":
    unittest.main()
