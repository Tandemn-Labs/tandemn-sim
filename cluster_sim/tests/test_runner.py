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

from llm_usage import LlmUsage, usage_path
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
from runner import (
    ClusterSimulator,
    LogicalClock,
    _batch_outcome_rates,
    _batch_request_rate_per_second,
    advance_batch_progress,
    submit_waiting_job,
)
from scenario import load_scenario
from serving import (
    BatchProbeMeasurement,
    RankResult,
    batch_probe_measurements,
    gpu_metrics_from_results,
)


class RunnerDataPlaneTest(unittest.TestCase):
    """Verify the coordinator runs jobs and stages one metric commit."""

    def test_run_trace_ends_after_hard_stop(self) -> None:
        """Write the terminal trace event after every running job is stopped."""
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario("trace-smoke", 1, 0, (), ())
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.user_id = "user-a"
        simulator.postgres = object()

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with patch.object(simulator, "_hard_stop") as hard_stop:
                result = simulator.run()

            event = json.loads(simulator.trace_file.read_text(encoding="utf-8").splitlines()[0])

        hard_stop.assert_called_once_with("SIMULATION_HARD_STOP")
        self.assertEqual(result, ())
        self.assertEqual(event["kind"], "run_finished")
        self.assertEqual(event["termination_reason"], "SIMULATION_HARD_STOP")
        self.assertFalse(event["truncated"])
        self.assertEqual(event["status"], "completed")

    def test_run_trace_records_failure_before_reraising(self) -> None:
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario("trace-smoke", 1, 10, (), ())
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.run_tick = lambda tick: (_ for _ in ()).throw(RuntimeError("boom"))

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with self.assertRaisesRegex(RuntimeError, "boom"):
                simulator.run()
            event = json.loads(
                simulator.trace_file.read_text(encoding="utf-8").splitlines()[-1]
            )

        self.assertEqual(event["kind"], "run_finished")
        self.assertEqual(event["status"], "failed")
        self.assertEqual(event["termination_reason"], "RuntimeError: boom")

    def test_run_trace_records_hard_stop_failure(self) -> None:
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario("trace-smoke", 1, 0, (), ())
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with (
                patch.object(
                    simulator, "_hard_stop", side_effect=RuntimeError("cleanup failed")
                ),
                self.assertRaisesRegex(RuntimeError, "cleanup failed"),
            ):
                simulator.run()
            event = json.loads(
                simulator.trace_file.read_text(encoding="utf-8").splitlines()[-1]
            )

        self.assertEqual(event["status"], "failed")
        self.assertEqual(event["termination_reason"], "RuntimeError: cleanup failed")

    def test_run_honors_tick_limit_before_drain(self) -> None:
        """Hard-stop after the requested benchmark tick count."""
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario("trace-smoke", 1, 10, (), ())
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.run_tick = lambda tick: SimpleNamespace(tick_number=tick)

        with patch.object(simulator, "_hard_stop") as hard_stop:
            result = simulator.run(2)

        self.assertEqual([item.tick_number for item in result], [0, 1])
        hard_stop.assert_called_once_with("SIMULATION_TICK_LIMIT")

    def test_run_stops_at_workload_end_before_drain(self) -> None:
        """Do not simulate the manifest's post-workload drain period."""
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario(
            "strict-end",
            1_000_000_000,
            3_000_000_000,
            (),
            (),
            workload_end_ns=2_000_000_000,
        )
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        ticks = []

        def run_tick(tick_number):
            ticks.append(tick_number)
            simulator.clock.advance(timedelta(seconds=1))
            return SimpleNamespace(tick_number=tick_number)

        simulator.run_tick = run_tick

        with patch.object(simulator, "_hard_stop") as hard_stop:
            simulator.run()

        self.assertEqual(ticks, [0, 1])
        hard_stop.assert_called_once_with("SIMULATION_HARD_STOP")

    def test_run_tick_finishes_due_jobs_before_planning(self) -> None:
        """Do not expose jobs at their deadline to the planner snapshot."""
        from tandemn_system_data.models.enums import JobStatus

        class Jobs:
            def __init__(self) -> None:
                self.states = {
                    "online-id": JobStatus.RUNNING,
                    "waiting-id": JobStatus.WAITING,
                    "paused-id": JobStatus.PAUSED,
                }
                self.running = SimpleNamespace(
                    job=SimpleNamespace(job_id="online-id"),
                    ranks=(SimpleNamespace(rank_id="rank-1"),),
                )
                self.stopped_ranks = []

            def running_jobs(self, _user_id):
                return (
                    (self.running,)
                    if self.states["online-id"] is JobStatus.RUNNING
                    else ()
                )

            def waiting_jobs(self, _user_id):
                return tuple(
                    SimpleNamespace(job_id=job_id)
                    for job_id, status in self.states.items()
                    if status is JobStatus.WAITING
                )

            def paused_jobs(self, _user_id):
                return tuple(
                    SimpleNamespace(job_id=job_id)
                    for job_id, status in self.states.items()
                    if status is JobStatus.PAUSED
                )

            def set_rank_status(self, rank_id, status, expected):
                self.stopped_ranks.append((rank_id, status, expected))

            def transition(self, job_id, to, expected, **_kwargs):
                if self.states[job_id] not in expected:
                    return False
                self.states[job_id] = to
                return True

        deadline_ns = 5_000_000_000
        online = OnlineJob("online", "online-id", "model", 0, deadline_ns, {})
        waiting = BatchJob(
            "waiting", "waiting-id", "model", 0, deadline_ns, deadline_ns, {}
        )
        paused = BatchJob(
            "paused", "paused-id", "model", 0, deadline_ns, deadline_ns, {}
        )
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario(
            "deadline", deadline_ns, deadline_ns * 2, (online,), (waiting, paused)
        )
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.user_id = "user-a"
        simulator.postgres = object()
        simulator.jobs = Jobs()
        simulator._submitted_jobs = {"online", "waiting", "paused"}
        simulator._online_jobs = {"online-id": online}
        simulator._batch_jobs = {"waiting-id": waiting, "paused-id": paused}
        simulator._online_backlog = {"online-id": (object(),)}
        simulator._batch_progress = {
            "waiting-id": BatchProgress(total_requests=1),
            "paused-id": BatchProgress(total_requests=1),
        }
        simulator._admissions_in_tick = 0
        simulator.llmservingsim_root = None
        simulator.trace_file = None
        seen = []

        def create_plans(_tick, _time):
            seen.append(
                {
                    "running": simulator.jobs.running_jobs(simulator.user_id),
                    "waiting": simulator.jobs.waiting_jobs(simulator.user_id),
                    "paused": simulator.jobs.paused_jobs(simulator.user_id),
                }
            )
            return SimpleNamespace(plans=(), llm_usage=(), error=None)

        simulator.planner = SimpleNamespace(create_plans=create_plans)
        outcomes = []

        with (
            patch("runner.apply_pending_plans"),
            patch(
                "tandemn_system_data.clients.event_log.PostgresEventLog"
            ) as event_log,
        ):
            event_log.return_value.append.side_effect = outcomes.append
            simulator.run_tick(0)

            self.assertEqual(simulator.jobs.states, {
                "online-id": JobStatus.FINISHED,
                "waiting-id": JobStatus.FINISHED,
                "paused-id": JobStatus.FINISHED,
            })

            simulator.run_tick(1)

        self.assertEqual(seen[1], {"running": (), "waiting": (), "paused": ()})
        self.assertEqual(simulator.jobs.stopped_ranks[0][0], "rank-1")
        self.assertEqual(simulator._online_backlog["online-id"], ())
        self.assertEqual(
            [(event.job_id, event.payload_json) for event in outcomes],
            [
                ("waiting-id", {"deadline_met": False}),
                ("paused-id", {"deadline_met": False}),
            ],
        )

    def test_due_jobs_write_terminal_trace_events(self) -> None:
        """Record exact logical deadline outcomes for terminal online and batch jobs."""
        from tandemn_system_data.models.enums import JobStatus

        class Jobs:
            def __init__(self) -> None:
                self.states = {
                    "online-id": JobStatus.RUNNING,
                    "batch-id": JobStatus.WAITING,
                }
                self.running = SimpleNamespace(
                    job=SimpleNamespace(job_id="online-id"),
                    ranks=(SimpleNamespace(rank_id="rank-1"),),
                )

            def running_jobs(self, _user_id):
                return (
                    (self.running,)
                    if self.states["online-id"] is JobStatus.RUNNING
                    else ()
                )

            def waiting_jobs(self, _user_id):
                return (
                    (SimpleNamespace(job_id="batch-id"),)
                    if self.states["batch-id"] is JobStatus.WAITING
                    else ()
                )

            def paused_jobs(self, _user_id):
                return ()

            def set_rank_status(self, *_args):
                pass

            def transition(self, job_id, to, expected, **_kwargs):
                if self.states[job_id] not in expected:
                    return False
                self.states[job_id] = to
                return True

        deadline_ns = 5_000_000_000
        online = OnlineJob("online", "online-id", "model", 0, deadline_ns, {})
        batch = BatchJob(
            "batch", "batch-id", "model", 0, deadline_ns, deadline_ns, {}
        )
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario("deadline", deadline_ns, deadline_ns, (online,), (batch,))
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.user_id = "user-a"
        simulator.postgres = object()
        simulator.jobs = Jobs()
        simulator._submitted_jobs = {"online", "batch"}
        simulator._online_jobs = {"online-id": online}
        simulator._batch_jobs = {"batch-id": batch}
        simulator._online_backlog = {"online-id": ()}
        simulator._online_admitted_at_ns = {"online-id": 0}
        simulator._requests = {
            "online-id": (Request("request", "session", 0, 0, 1, 1, None),)
        }
        simulator._batch_progress = {
            "batch-id": BatchProgress(total_requests=5, completed_requests=2)
        }

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with patch("tandemn_system_data.clients.event_log.PostgresEventLog"):
                simulator._finish_jobs_due(deadline_ns)
            events = [
                json.loads(line)
                for line in simulator.trace_file.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["kind"] == "job_finished"
            ]

        self.assertEqual(len(events), 2)
        self.assertEqual(
            {(event["job_id"], event["finished_at"]) for event in events},
            {
                ("online-id", "2026-01-01T00:00:05+00:00"),
                ("batch-id", "2026-01-01T00:00:05+00:00"),
            },
        )
        online_event = next(event for event in events if event["job_id"] == "online-id")
        batch_event = next(event for event in events if event["job_id"] == "batch-id")
        self.assertEqual(online_event["unfinished_request_count"], 1)
        self.assertFalse(online_event["completed_successfully"])
        self.assertFalse(batch_event["deadline_slo_met"])
        self.assertEqual(batch_event["unfinished_request_count"], 3)

    def test_prices_keep_region_specific_instance_rates(self) -> None:
        resource_map = SimpleNamespace(
            scheduling_summary=lambda: {
                "reserved|aws|us-east-1|zone-a|H100": {
                    "pools": [
                        {
                            "instance_type": "p5.48xlarge",
                            "gpus_per_instance": 8,
                            "price_per_instance_hour": 55.04,
                        }
                    ]
                },
                "reserved|aws|ap-northeast-1|zone-a|H100": {
                    "pools": [
                        {
                            "instance_type": "p5.48xlarge",
                            "gpus_per_instance": 8,
                            "price_per_instance_hour": 68.8,
                        }
                    ]
                },
            }
        )

        prices = ClusterSimulator._prices(resource_map)

        self.assertEqual(
            prices[("reserved|aws|us-east-1|zone-a|H100", "p5.48xlarge")],
            (55.04, 8),
        )
        self.assertEqual(
            prices[("reserved|aws|ap-northeast-1|zone-a|H100", "p5.48xlarge")],
            (68.8, 8),
        )

    def test_request_outcome_projects_probe_counts_to_full_campaign(self) -> None:
        simulator = object.__new__(ClusterSimulator)
        simulator._request_outcomes = {}

        simulator._record_request_outcome(
            "batch-a",
            offered_request_count=10_000,
            projected_completed_request_count=3_000,
            request_summary={
                "completed_request_count": 20,
                "ttft_pass_count": 15,
                "tpot_pass_count": 14,
                "slo_pass_count": 12,
                "qualified_output_tokens": 240,
            },
        )

        self.assertEqual(
            simulator._terminal_request_outcome("batch-a"),
            {
                "offered_request_count": 10_000,
                "completed_request_count": 3_000,
                "unfinished_request_count": 7_000,
                "ttft_pass_count": 2_250,
                "tpot_pass_count": 2_100,
                "slo_pass_count": 1_800,
                "qualified_output_tokens": 36_000,
                "slo_count_scope": "all_offered_requests",
            },
        )

    def test_batch_outcomes_weight_each_rank_by_observed_horizon(self) -> None:
        fast = SimpleNamespace(rank_name="fast")
        slow = SimpleNamespace(rank_name="slow")
        passing_row = {"TTFT": 1, "TPOT": 1, "output": 1}
        failing_row = {"TTFT": 2_000_000, "TPOT": 2_000_000, "output": 1}
        results = (
            SimpleNamespace(deployment=fast, request_rows=(passing_row,) * 2),
            SimpleNamespace(deployment=slow, request_rows=(failing_row,) * 10),
        )
        measurements = (
            BatchProbeMeasurement(fast, 2_000_000_000, 2, 2),
            BatchProbeMeasurement(slow, 10_000_000_000, 10, 10),
        )

        rates = _batch_outcome_rates(results, measurements, 1, 1, "generate")
        simulator = object.__new__(ClusterSimulator)
        simulator._request_outcomes = {}
        simulator._record_request_outcome(
            "batch-a",
            offered_request_count=100,
            projected_completed_request_count=100,
            request_summary=rates,
        )
        outcome = simulator._terminal_request_outcome("batch-a")

        self.assertEqual(outcome["completed_request_count"], 100)
        self.assertEqual(outcome["ttft_pass_count"], 50)
        self.assertEqual(outcome["tpot_pass_count"], 50)
        self.assertEqual(outcome["slo_pass_count"], 50)
        self.assertEqual(outcome["qualified_output_tokens"], 50)

    def test_plan_validation_requires_every_selected_rank_to_materialize(self) -> None:
        plan = {
            "status": "applied",
            "actions": [
                {
                    "type": "place",
                    "ladder": [{"rank_id": "rank-a"}, {"rank_id": "rank-b"}],
                },
                {"type": "defer"},
            ],
        }

        self.assertEqual(
            ClusterSimulator._plan_validation_facts(
                plan, {"rank-a", "rank-b"}, set()
            ),
            (1, True),
        )
        self.assertEqual(
            ClusterSimulator._plan_validation_facts(plan, {"rank-a"}, set()),
            (0, False),
        )
        self.assertEqual(
            ClusterSimulator._plan_validation_facts(
                plan, {"rank-a", "rank-b"}, {"rank-b"}
            ),
            (1, False),
        )
        self.assertEqual(
            ClusterSimulator._plan_validation_facts(
                {**plan, "status": "failed"}, {"rank-a", "rank-b"}, set()
            ),
            (0, False),
        )

    def test_plan_validation_counts_only_fully_materialized_actions(self) -> None:
        plan = {
            "status": "applied",
            "actions": [
                {"type": "place", "ladder": [{"rank_id": "rank-a"}]},
                {"type": "swap", "ladder": [{"rank_id": "rank-b"}]},
                {"type": "keep"},
            ],
        }

        self.assertEqual(
            ClusterSimulator._plan_validation_facts(plan, {"rank-a"}, set()),
            (1, False),
        )

    def test_keep_only_plan_validation_is_not_applicable(self) -> None:
        plan = {
            "status": "applied",
            "actions": [{"type": "keep"}, {"type": "defer"}],
        }

        self.assertEqual(
            ClusterSimulator._plan_validation_facts(plan, set(), set()),
            (0, None),
        )

    def test_rank_execution_error_marks_applied_selected_plan_invalid(self) -> None:
        simulator = object.__new__(ClusterSimulator)
        plan = {
            "plan_id": "plan-a",
            "status": "applied",
            "actions": [
                {"type": "place", "ladder": [{"rank_id": "rank-a"}]}
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            simulator._trace_plan_validations(
                0,
                [plan],
                {"rank-a"},
                selected_plan_error=True,
            )
            event = json.loads(
                simulator.trace_file.read_text(encoding="utf-8").splitlines()[-1]
            )

        self.assertEqual(event["applied_action_count"], 1)
        self.assertFalse(event["selected_plan_valid"])

    def test_missing_materialized_rank_marks_selected_plan_invalid(self) -> None:
        simulator = object.__new__(ClusterSimulator)
        simulator._tick_failed_rank_ids = set()
        plan = {
            "plan_id": "plan-a",
            "status": "applied",
            "actions": [
                {"type": "place", "ladder": [{"rank_id": "rank-a"}]}
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            simulator._trace_plan_validations(0, [plan], set())
            event = json.loads(
                simulator.trace_file.read_text(encoding="utf-8").splitlines()[-1]
            )

        self.assertEqual(event["applied_action_count"], 0)
        self.assertFalse(event["selected_plan_valid"])

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
            model_dump=lambda mode: {
                "job_id": "job-a",
                "type": "place",
                "ladder": [{"rank_id": "rank-a"}],
            }
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
                tick,
                (CreatedPlan("plan-1"),),
                llm_usage=(LlmUsage(tick, 100, 20, "model-used"),),
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
            wall_times = (
                datetime(2026, 8, 19, 21, 30, 1, tzinfo=UTC),
                datetime(2026, 8, 19, 21, 30, 15, tzinfo=UTC),
                datetime(2026, 8, 19, 21, 30, 16, tzinfo=UTC),
                datetime(2026, 8, 19, 21, 30, 16, 10_000, tzinfo=UTC),
            )
            with (
                patch("tandemn_system_data.clients.PlanStore", PlanStore),
                patch("tandemn_system_data.clients.ResourceMapStore", ResourceMapStore),
                patch(
                    "runner.apply_pending_plans",
                    side_effect=lambda *args: setattr(plan, "status", "applied"),
                ),
                patch.object(simulator, "_simulate_tick"),
                patch("runner.datetime") as wall_datetime,
                patch("runner.perf_counter", side_effect=(100, 114, 200, 200.01)),
            ):
                wall_datetime.now.side_effect = wall_times
                simulator.run_tick(0)
                simulator.run_tick(1)

            events = [
                json.loads(line)
                for line in simulator.trace_file.read_text(encoding="utf-8").splitlines()
            ]
            usage_events = [
                json.loads(line)
                for line in usage_path(simulator.trace_file)
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(events[0]["kind"], "tick_started")
        self.assertEqual(events[1]["kind"], "resource_snapshot")
        self.assertEqual(events[2]["kind"], "planner_finished")
        self.assertEqual(events[2]["logical_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(
            events[2]["planner_started_at"], "2026-08-19T21:30:01+00:00"
        )
        self.assertEqual(
            events[2]["planner_finished_at"], "2026-08-19T21:30:15+00:00"
        )
        self.assertEqual(events[2]["decision_time_ms"], 14_000)
        self.assertEqual(events[2]["plans"][0]["actions"][0]["job_id"], "job-a")
        self.assertEqual(events[2]["planned_job_ids"], ["job-a"])
        self.assertEqual(events[3]["kind"], "job_admitted")
        self.assertEqual(events[3]["admitted_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(sum(event["kind"] == "job_admitted" for event in events), 1)
        self.assertEqual(sum(event["kind"] == "placement_observed" for event in events), 2)
        self.assertEqual(
            usage_events,
            [
                {
                    "tick_number": 0,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "model": "model-used",
                },
                {
                    "tick_number": 1,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "model": "model-used",
                },
            ],
        )
        validation = next(
            event for event in events if event["kind"] == "plan_validation"
        )
        self.assertEqual(validation["applied_action_count"], 1)
        self.assertTrue(validation["selected_plan_valid"])
        self.assertEqual(simulator._admission_latencies_ms, [0.0])
        self.assertEqual(
            simulator.clock.now(),
            datetime(2026, 1, 1, 0, 10, tzinfo=UTC),
        )

    def test_tick_admits_only_materialized_place(self) -> None:
        """Wait for a replacement rank before recording the first admission."""
        def plan(plan_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                plan_id=plan_id,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                status="applied",
                actions=[
                    SimpleNamespace(
                        model_dump=lambda mode: {"job_id": "job-a", "type": "place"}
                    )
                ],
            )

        plans = {plan_id: plan(plan_id) for plan_id in ("plan-failed", "plan-retry")}
        state = {"active_plan": None}
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario(
            "trace-smoke", 300_000_000_000, 600_000_000_000, (), ()
        )
        simulator.user_id = "user_1"
        simulator.postgres = object()
        simulator.planner = SimpleNamespace(
            create_plans=lambda tick, start: PlannerResult(
                tick,
                (CreatedPlan("plan-failed" if tick == 0 else "plan-retry"),),
            )
        )
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator._submitted_jobs = set()
        simulator._admitted_jobs = set()
        simulator._admission_latencies_ms = []
        simulator._admissions_in_tick = 0

        def running_jobs(user_id):
            ranks = ()
            if state["active_plan"] is not None:
                ranks = (
                    SimpleNamespace(
                        plan_id=state["active_plan"],
                        rank_id="rank-a",
                        n_replicas=1,
                        shape_json={"count": 1},
                    ),
                )
            return (SimpleNamespace(job=SimpleNamespace(job_id="job-a"), ranks=ranks),)

        simulator.jobs = SimpleNamespace(
            get=lambda job_id: SimpleNamespace(created_at=datetime(2026, 1, 1, tzinfo=UTC)),
            running_jobs=running_jobs,
        )

        class PlanStore:
            def __init__(self, postgres) -> None:
                pass

            def get(self, plan_id):
                return plans[plan_id]

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with (
                patch("tandemn_system_data.clients.PlanStore", PlanStore),
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
                state["active_plan"] = "plan-retry"
                simulator.run_tick(1)

            events = [
                json.loads(line)
                for line in simulator.trace_file.read_text(encoding="utf-8").splitlines()
            ]

        admissions = [event for event in events if event["kind"] == "job_admitted"]
        self.assertEqual(len(admissions), 1)
        self.assertEqual(admissions[0]["plan_id"], "plan-retry")
        self.assertEqual(admissions[0]["admission_source"], "active_rank_materialized")
        self.assertEqual(admissions[0]["admitted_at"], "2026-01-01T00:05:00+00:00")

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
        simulator.postgres = object()
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
            with (
                patch(
                    "tandemn_system_data.clients.ModelCatalogStore",
                    return_value=SimpleNamespace(get=lambda _: None),
                ),
                patch("runner.submit_waiting_job"),
            ):
                simulator._submit(job)

            event = json.loads(simulator.trace_file.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(event["job_kind"], "batch")
        self.assertEqual(event["submitted_at"], "2026-01-01T00:01:00+00:00")
        self.assertEqual(event["deadline_at"], "2026-01-01T01:01:00+00:00")

    def test_submit_uses_model_catalog_prefix_cache_setting(self) -> None:
        """Pass the catalog's prefix-cache setting into the submitted job spec."""
        job = OnlineJob(
            "online-a",
            "job-a",
            "model-a",
            0,
            60_000_000_000,
            {"features": {}},
        )
        simulator = object.__new__(ClusterSimulator)
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.user_id = "user-a"
        simulator.postgres = object()
        simulator.jobs = object()
        simulator._submitted_jobs = set()

        for catalog_value, expected in ((True, True), (False, False), (None, False)):
            with self.subTest(catalog_value=catalog_value):
                catalog = (
                    None
                    if catalog_value is None
                    else SimpleNamespace(prefix_cache_enabled=catalog_value)
                )
                with (
                    patch(
                        "tandemn_system_data.clients.ModelCatalogStore",
                        return_value=SimpleNamespace(get=lambda _: catalog),
                    ),
                    patch("runner.submit_waiting_job") as submit,
                ):
                    simulator._submit(job)

                self.assertEqual(submit.call_args.kwargs["prefix_cache_enabled"], expected)
                simulator._submitted_jobs.clear()

    def test_submit_preserves_catalog_prefix_cache_setting(self) -> None:
        """Keep the catalog-derived setting in canonical Store job metadata."""
        job = OnlineJob(
            "online-a",
            "job-a",
            "model-a",
            0,
            60_000_000_000,
            {
                "features": {
                    "target_p99_ttft_ms": 100,
                    "target_p99_tpot_ms": 10,
                }
            },
        )
        submitted = []

        submit_waiting_job(
            job,
            "user-a",
            SimpleNamespace(submit=submitted.append),
            datetime(2026, 1, 1, tzinfo=UTC),
            prefix_cache_enabled=True,
        )

        self.assertTrue(submitted[0].spec_json["prefix_cache_enabled"])

    def test_finish_trace_records_batch_deadline_outcome(self) -> None:
        """Use the simulation clock, not Store wall time, for batch completion."""
        calls = []
        simulator = object.__new__(ClusterSimulator)
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.user_id = "user-a"
        simulator.postgres = object()
        simulator.jobs = SimpleNamespace(
            set_rank_status=lambda *args: calls.append(args),
            transition=lambda *args: calls.append(args) or True,
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

        outcomes = []
        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with patch("tandemn_system_data.clients.event_log.PostgresEventLog") as event_log:
                event_log.return_value.append.side_effect = outcomes.append
                simulator._finish(running)

            event = json.loads(simulator.trace_file.read_text(encoding="utf-8").splitlines()[-1])

        self.assertTrue(event["completed_successfully"])
        self.assertTrue(event["deadline_slo_met"])
        self.assertEqual(event["deadline_outcome"], "met")
        self.assertEqual(len(calls), 2)
        self.assertTrue(outcomes[0].payload_json["deadline_met"])

    def test_fail_oom_keeps_siblings_and_job_running(self) -> None:
        """Mark only the OOM rank failed so Koi can repair the job."""
        rank_calls = []
        job_calls = []
        removed_assignments = []
        simulator = object.__new__(ClusterSimulator)
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.jobs = SimpleNamespace(
            set_rank_status=lambda *args, **kwargs: rank_calls.append((args, kwargs)),
            fail=lambda *args, **kwargs: job_calls.append((args, kwargs)),
        )
        simulator._router = SimpleNamespace(
            remove_rank=lambda job_name, rank_name: removed_assignments.append(
                (job_name, rank_name)
            )
        )
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job-a"),
            ranks=(
                SimpleNamespace(
                    rank_id="rank-oom",
                    n_replicas=4,
                    shape_json={"count": 2},
                ),
                SimpleNamespace(rank_id="rank-healthy"),
            ),
        )
        results = (
            SimpleNamespace(
                status="oom",
                deployment=SimpleNamespace(rank_name="rank-oom"),
                stdout="",
                stderr="Model size exceeds total NPU memory",
            ),
            SimpleNamespace(
                status="completed",
                deployment=SimpleNamespace(rank_name="rank-healthy"),
                stdout="",
                stderr="",
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            simulator._fail_oom(running, "online", results)
            events = [
                json.loads(line)
                for line in simulator.trace_file.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(rank_calls), 1)
        self.assertEqual(rank_calls[0][0][0], "rank-oom")
        self.assertEqual(rank_calls[0][0][1], "failed")
        self.assertEqual(rank_calls[0][1]["reason_code"], "OOM")
        self.assertEqual(job_calls, [])
        self.assertEqual(removed_assignments, [("online", "rank-oom")])
        self.assertEqual(events[0]["kind"], "rank_failed")
        self.assertEqual(events[0]["service_impact"], "degraded")
        self.assertEqual(events[0]["released_gpu_count"], 8)

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
        simulator._batch_jobs = {
            "job-a": BatchJob(
                "batch-a",
                "job-a",
                "model-a",
                0,
                60_000_000_000,
                180_000_000_000,
                {"features": {}},
            )
        }
        simulator._batch_progress = {
            "job-a": BatchProgress(total_requests=10_000, completed_requests=3_000)
        }

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            simulator._hard_stop()

            event = json.loads(simulator.trace_file.read_text(encoding="utf-8").splitlines()[-1])

        self.assertFalse(event["completed_successfully"])
        self.assertIs(event["deadline_slo_met"], False)
        self.assertEqual(event["deadline_outcome"], "missed")
        self.assertEqual(event["finish_reason"], "SIMULATION_HARD_STOP")
        self.assertEqual(event["unserved_request_count"], 7_000)

    def test_tick_limit_censors_future_batch_deadline(self) -> None:
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
        simulator._batch_jobs = {
            "job-a": BatchJob(
                "batch-a",
                "job-a",
                "model-a",
                0,
                60_000_000_000,
                180_000_000_000,
                {"features": {}},
            )
        }
        simulator._batch_progress = {
            "job-a": BatchProgress(total_requests=10_000, completed_requests=3_000)
        }

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            simulator._hard_stop("SIMULATION_TICK_LIMIT")
            event = json.loads(
                simulator.trace_file.read_text(encoding="utf-8").splitlines()[-1]
            )

        self.assertIsNone(event["deadline_slo_met"])
        self.assertEqual(event["deadline_outcome"], "censored")
        self.assertEqual(event["finish_reason"], "SIMULATION_TICK_LIMIT")

    def test_tick_limit_marks_past_batch_deadline_as_failed(self) -> None:
        simulator = object.__new__(ClusterSimulator)
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.clock.advance(timedelta(minutes=2))
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
        simulator._batch_jobs = {
            "job-a": BatchJob(
                "batch-a",
                "job-a",
                "model-a",
                0,
                60_000_000_000,
                180_000_000_000,
                {"features": {}},
            )
        }
        simulator._batch_progress = {
            "job-a": BatchProgress(total_requests=10_000, completed_requests=3_000)
        }

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            simulator._hard_stop("SIMULATION_TICK_LIMIT")
            event = json.loads(
                simulator.trace_file.read_text(encoding="utf-8").splitlines()[-1]
            )

        self.assertIs(event["deadline_slo_met"], False)
        self.assertEqual(event["deadline_outcome"], "missed")

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

    def test_hard_stop_finishes_paused_batch_with_lifetime_outcome(self) -> None:
        simulator = object.__new__(ClusterSimulator)
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        paused = SimpleNamespace(job_id="job-a")
        simulator.jobs = SimpleNamespace(
            running_jobs=lambda user_id: (),
            waiting_jobs=lambda user_id: (),
            paused_jobs=lambda user_id: (paused,),
            transition=lambda *args, **kwargs: None,
        )
        simulator.user_id = "user_1"
        simulator._online_jobs = {}
        simulator._online_backlog = {}
        simulator._batch_jobs = {"job-a": object()}
        simulator._batch_progress = {
            "job-a": BatchProgress(total_requests=10, completed_requests=4)
        }

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            simulator._hard_stop()
            event = json.loads(
                simulator.trace_file.read_text(encoding="utf-8").splitlines()[-1]
            )

        self.assertEqual(event["completed_request_count"], 4)
        self.assertEqual(event["unfinished_request_count"], 6)

    def test_hard_stop_counts_online_source_offers_without_active_ranks(self) -> None:
        simulator = object.__new__(ClusterSimulator)
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.clock.advance(timedelta(minutes=5))
        running = SimpleNamespace(job=SimpleNamespace(job_id="job-a"), ranks=())
        simulator.jobs = SimpleNamespace(
            running_jobs=lambda user_id: (running,),
            set_rank_status=lambda *args: None,
            transition=lambda *args, **kwargs: None,
        )
        simulator.user_id = "user_1"
        simulator._online_jobs = {"job-a": object()}
        simulator._online_backlog = {"job-a": ()}
        simulator._requests = {
            "job-a": (
                Request("r0", "s0", 0, 0, 1, 1, None),
                Request("r1", "s1", 0, 60_000_000_000, 1, 1, None),
                Request("r2", "s2", 0, 600_000_000_000, 1, 1, None),
            )
        }
        simulator._online_admitted_at_ns = {"job-a": 0}
        simulator._batch_jobs = {}
        simulator._batch_progress = {}

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            simulator._hard_stop()
            event = json.loads(
                simulator.trace_file.read_text(encoding="utf-8").splitlines()[-1]
            )

        self.assertEqual(event["offered_request_count"], 2)
        self.assertEqual(event["unfinished_request_count"], 2)

    def test_active_jobs_share_one_global_rank_pool(self) -> None:
        """Launch ranks from different jobs in one bounded worker pool."""
        jobs = tuple(
            OnlineJob(name, job_id, "model", 0, 600_000_000_000, {})
            for name, job_id in (("online-a", "job-a"), ("online-b", "job-b"))
        )
        running_jobs = tuple(
            SimpleNamespace(
                job=SimpleNamespace(job_id=job.store_job_id),
                ranks=(SimpleNamespace(rank_id=f"rank-{job.store_job_id}"),),
            )
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
                return SimpleNamespace(catalog={"regions": [{"cloud": "aws", "region": "test", "instance_types": []}]})

        class ResourceMapStore:
            def __init__(self, postgres, user_id) -> None:
                pass

            def get(self):
                return SimpleNamespace(scheduling_summary=lambda: {})

        def prepare(running, *args):
            return rank_runs[running.job.job_id], ()

        def run(runs, max_parallel, progress=None):
            launched.append((runs, max_parallel))
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
        ):
            simulator._simulate_tick(0, 0, 300_000_000_000)

        self.assertEqual(
            launched,
            [((*rank_runs["job-a"], *rank_runs["job-b"]), 5)],
        )
        self.assertEqual(
            finalized,
            [
                ("job-a", ("deployment-a",), (result_a,)),
                ("job-b", ("deployment-b",), (result_b,)),
            ],
        )

    def test_idle_tick_still_records_capacity_and_zero_allocation_seconds(self) -> None:
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario(
            "idle", 300_000_000_000, 300_000_000_000, (), ()
        )
        simulator.user_id = "user_1"
        simulator.postgres = object()
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.jobs = SimpleNamespace(running_jobs=lambda user_id: ())
        simulator.llmservingsim_root = Path(tempfile.gettempdir())
        simulator.parallel_simulators = 1
        simulator._online_jobs = {}
        simulator._online_backlog = {}
        simulator._batch_jobs = {}
        simulator._batch_progress = {}
        simulator._request_outcomes = {}

        resource_map = SimpleNamespace(
            scheduling_summary=lambda: {
                "reserved|sim|region-1|zone-1|H100": {
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

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with (
                patch(
                    "tandemn_system_data.clients.HardwareCatalogStore",
                    return_value=SimpleNamespace(
                        all=lambda: [SimpleNamespace(catalog={"regions": [{}]})]
                    ),
                ),
                patch(
                    "tandemn_system_data.clients.ResourceMapStore",
                    return_value=SimpleNamespace(get=lambda: resource_map),
                ),
            ):
                simulator._simulate_tick(0, 0, 300_000_000_000)
            events = [
                json.loads(line)
                for line in simulator.trace_file.read_text(encoding="utf-8").splitlines()
            ]

        cluster_event = next(event for event in events if event["kind"] == "cluster_window")
        self.assertEqual(cluster_event["capacity_gpu_seconds"], 8 * 300)
        self.assertEqual(cluster_event["allocated_gpu_seconds"], 0)
        self.assertEqual(cluster_event["busy_gpu_seconds"], 0)

    def test_rank_failures_requeue_requests_and_preserve_siblings(self) -> None:
        """Keep OOM, timeout, and catalog failures local to their ranks."""
        job = OnlineJob("online", "job-a", "model", 0, 300_000_000_000, {"features": {}})
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job-a", spec_json={"model_id": "model"}),
            ranks=(
                SimpleNamespace(
                    rank_id="rank-oom",
                    n_replicas=1,
                    shape_json={
                        "env": ["reserved", "aws", "us-east-1", "us-east-1a", "L40S"],
                        "instance_type": "g6e.48xlarge",
                        "count": 3,
                        "gpu_count": 3,
                    },
                ),
                SimpleNamespace(
                    rank_id="rank-timeout",
                    n_replicas=1,
                    shape_json={
                        "env": ["reserved", "aws", "us-east-1", "us-east-1a", "L40S"],
                        "instance_type": "g6e.48xlarge",
                        "count": 3,
                        "gpu_count": 3,
                    },
                ),
                SimpleNamespace(
                    rank_id="rank-catalog",
                    n_replicas=1,
                    shape_json={
                        "env": ["reserved", "aws", "us-east-1", "us-east-1a", "L40S"],
                        "instance_type": "g6e.48xlarge",
                        "count": 3,
                        "gpu_count": 3,
                    },
                ),
                SimpleNamespace(
                    rank_id="rank-healthy",
                    n_replicas=1,
                    shape_json={
                        "env": ["reserved", "aws", "us-east-1", "us-east-1a", "L40S"],
                        "instance_type": "g6e.48xlarge",
                        "count": 2,
                        "gpu_count": 2,
                    },
                ),
            ),
        )
        rank_calls = []
        transitions = []
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario(
            "oom",
            300_000_000_000,
            300_000_000_000,
            (job,),
            (),
        )
        simulator.user_id = "user-a"
        simulator.postgres = object()
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.jobs = SimpleNamespace(
            running_jobs=lambda user_id: (running,),
            set_rank_status=lambda *args, **kwargs: rank_calls.append((args, kwargs)),
            transition=lambda *args, **kwargs: transitions.append((args, kwargs)),
        )
        simulator.llmservingsim_root = Path(tempfile.gettempdir())
        simulator.parallel_simulators = 1
        simulator._router = StickyRoundRobinRouter()
        simulator._traffic_starts = {}
        simulator._requests = {
            "job-a": (
                Request("request-oom", "oom-session", 0, 0, 1, 1, None),
                Request("request-timeout", "timeout-session", 0, 0, 1, 1, None),
                Request("request-catalog", "catalog-session", 0, 0, 1, 1, None),
                Request("request-healthy", "healthy-session", 0, 0, 1, 1, None),
            ),
        }
        simulator._online_jobs = {"job-a": job}
        simulator._online_backlog = {"job-a": ()}
        simulator._batch_jobs = {}
        simulator._batch_progress = {}
        metric_calls = []

        class HardwareCatalogStore:
            def __init__(self, postgres) -> None:
                pass

            def get(self):
                return SimpleNamespace(
                    catalog={"regions": [{"cloud": "aws", "region": "test", "instance_types": []}]}
                )

        class ResourceMapStore:
            def __init__(self, postgres, user_id) -> None:
                pass

            def get(self):
                return SimpleNamespace(
                    scheduling_summary=lambda: {
                        "reserved|aws|us-east-1|us-east-1a|L40S": {
                            "pools": [
                                {
                                    "fabric_id": "efa",
                                    "instance_type": "g6e.48xlarge",
                                    "total": 16,
                                    "total_instances": 2,
                                    "gpus_per_instance": 8,
                                    "price_per_instance_hour": 10.0,
                                }
                            ]
                        }
                    }
                )

        class GpuMetricStore:
            def __init__(self, postgres) -> None:
                pass

        oom_rank_run = SimpleNamespace(
            deployment=SimpleNamespace(rank_name="rank-oom"),
            request_count=1,
        )
        healthy_rank_run = SimpleNamespace(
            deployment=SimpleNamespace(rank_name="rank-healthy"),
            request_count=1,
        )
        timeout_rank_run = SimpleNamespace(
            deployment=SimpleNamespace(rank_name="rank-timeout"),
            request_count=1,
        )
        catalog_rank_run = SimpleNamespace(
            deployment=SimpleNamespace(rank_name="rank-catalog"),
            request_count=1,
        )
        oom_result = SimpleNamespace(
            status="oom",
            deployment=oom_rank_run.deployment,
            request_rows=(),
            completed_requests=(),
            stdout="",
            stderr="[MemoryModel] exceeds total NPU memory",
        )
        healthy_result = SimpleNamespace(
            status="completed",
            deployment=healthy_rank_run.deployment,
            request_rows=(
                {"TTFT": 1, "TPOT": 1, "output": 1, "end_time": 1},
            ),
            completed_requests=(simulator._requests["job-a"][3],),
            stdout="",
            stderr="",
        )
        timeout_result = SimpleNamespace(
            status="process_crash",
            deployment=timeout_rank_run.deployment,
            request_rows=(),
            completed_requests=(),
            stdout="",
            stderr="rank exceeded wall time",
        )
        catalog_result = SimpleNamespace(
            status="model_catalog_invalid",
            deployment=catalog_rank_run.deployment,
            request_rows=(),
            completed_requests=(),
            stdout="",
            stderr="max_num_batched_tokens must be positive",
        )

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with (
                patch("tandemn_system_data.clients.HardwareCatalogStore", HardwareCatalogStore),
                patch("tandemn_system_data.clients.ResourceMapStore", ResourceMapStore),
                patch("tandemn_system_data.clients.GpuMetricStore", GpuMetricStore),
                patch.object(
                    simulator,
                    "_prepare_job_runs",
                    return_value=(
                        (oom_rank_run, timeout_rank_run, catalog_rank_run, healthy_rank_run),
                        (),
                    ),
                ),
                patch(
                    "runner.run_ranks",
                    return_value=(oom_result, timeout_result, catalog_result, healthy_result),
                ),
                patch("runner.collect_rank_telemetry", return_value=([], {})),
                patch("runner.commit_tick_metrics"),
                patch(
                    "runner.gpu_metrics_from_results",
                    side_effect=lambda *args: metric_calls.append(args)
                    or [SimpleNamespace(job_id="job-a")],
                ),
            ):
                simulator._simulate_tick(0, 0, 300_000_000_000)
            events = [
                json.loads(line)
                for line in simulator.trace_file.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            [call[0][0] for call in rank_calls],
            ["rank-oom", "rank-timeout", "rank-catalog"],
        )
        self.assertEqual(
            [call[1]["reason_code"] for call in rank_calls],
            ["OOM", "PROCESS_CRASH", "MODEL_CATALOG_INVALID"],
        )
        self.assertEqual([call[0].job_id for call in metric_calls], ["job-a"])
        self.assertEqual(metric_calls[0][1], (healthy_rank_run.deployment,))
        self.assertEqual(
            [request.request_name for request in simulator._online_backlog["job-a"]],
            ["request-oom", "request-timeout", "request-catalog"],
        )
        pool_event = next(event for event in events if event["kind"] == "pool_window")
        cluster_event = next(event for event in events if event["kind"] == "cluster_window")
        self.assertEqual(pool_event["allocated_gpu_count"], 2)
        self.assertEqual(cluster_event["allocated_gpu_hours"], 2 * 300 / 3600)
        self.assertEqual(transitions, [])

    def test_process_crash_requeues_all_dead_job(self) -> None:
        """Return a crashed single-rank job to the next planner tick."""
        job = OnlineJob(
            "online", "job-a", "model", 0, 300_000_000_000, {"features": {}}
        )
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job-a", spec_json={"model_id": "model"}),
            ranks=(
                SimpleNamespace(
                    rank_id="rank-crashed",
                    shape_json={
                        "env": ["reserved", "aws", "us-east-1", "us-east-1a", "L40S"],
                        "instance_type": "g6e.48xlarge",
                        "count": 1,
                        "gpu_count": 1,
                    },
                ),
            ),
        )
        rank_calls = []
        transitions = []

        def transition(*args, **kwargs):
            transitions.append((args, kwargs))
            return True

        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario(
            "process-crash",
            300_000_000_000,
            300_000_000_000,
            (job,),
            (),
        )
        simulator.user_id = "user-a"
        simulator.postgres = object()
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.jobs = SimpleNamespace(
            running_jobs=lambda user_id: (running,),
            set_rank_status=lambda *args, **kwargs: rank_calls.append((args, kwargs)),
            transition=transition,
        )
        simulator.llmservingsim_root = Path(tempfile.gettempdir())
        simulator.parallel_simulators = 1
        simulator._router = StickyRoundRobinRouter()
        simulator._traffic_starts = {}
        simulator._requests = {
            "job-a": (Request("request-a", "session", 0, 0, 1, 1, None),)
        }
        simulator._online_jobs = {"job-a": job}
        simulator._online_backlog = {"job-a": ()}
        simulator._batch_jobs = {}
        simulator._batch_progress = {}
        simulator._request_outcomes = {}
        simulator.trace_file = None

        rank_run = SimpleNamespace(
            deployment=SimpleNamespace(rank_name="rank-crashed"),
            request_count=1,
        )
        crashed = SimpleNamespace(
            status="process_crash",
            deployment=rank_run.deployment,
            request_rows=(),
            completed_requests=(),
            stdout="",
            stderr="ASTRA-Sim exited unexpectedly with code -6",
        )
        catalog = {"regions": [{"cloud": "aws", "region": "test", "instance_types": []}]}

        with (
            patch(
                "tandemn_system_data.clients.HardwareCatalogStore",
                return_value=SimpleNamespace(
                    get=lambda: SimpleNamespace(catalog=catalog)
                ),
            ),
            patch(
                "tandemn_system_data.clients.ResourceMapStore",
                return_value=SimpleNamespace(
                    get=lambda: SimpleNamespace(scheduling_summary=lambda: {})
                ),
            ),
            patch.object(
                simulator, "_prepare_job_runs", return_value=((rank_run,), ())
            ),
            patch("runner.run_ranks", return_value=(crashed,)),
            patch("runner.gpu_metrics_from_results", return_value=[]),
        ):
            simulator._simulate_tick(0, 0, 300_000_000_000)

        self.assertEqual(rank_calls[0][0][0], "rank-crashed")
        self.assertEqual(rank_calls[0][1]["reason_code"], "PROCESS_CRASH")
        self.assertEqual(transitions[0][0], ("job-a", "waiting", ["running"]))
        self.assertEqual(
            [request.request_name for request in simulator._online_backlog["job-a"]],
            ["request-a"],
        )

    def test_oom_requeues_batch_without_advancing_progress(self) -> None:
        """Return an all-OOM batch to the planner without losing progress."""
        batch = BatchJob(
            "batch",
            "job-a",
            "model",
            0,
            300_000_000_000,
            600_000_000_000,
            {"record_count": 10, "features": {}},
        )
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job-a"),
            ranks=(SimpleNamespace(rank_id="rank-oom", shape_json={}),),
        )
        rank_calls = []
        transitions = []
        progress = BatchProgress(
            total_requests=10,
            completed_requests=3,
            next_request_number=2,
        )
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario(
            "oom-batch", 300_000_000_000, 600_000_000_000, (), (batch,)
        )
        simulator.user_id = "user-a"
        simulator.postgres = object()
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.jobs = SimpleNamespace(
            running_jobs=lambda user_id: (running,),
            set_rank_status=lambda *args, **kwargs: rank_calls.append((args, kwargs)),
            transition=lambda *args, **kwargs: transitions.append((args, kwargs)) or True,
        )
        simulator.llmservingsim_root = Path(tempfile.gettempdir())
        simulator.parallel_simulators = 1
        simulator._router = StickyRoundRobinRouter()
        simulator._online_jobs = {}
        simulator._online_backlog = {}
        simulator._batch_jobs = {"job-a": batch}
        simulator._batch_progress = {"job-a": progress}
        simulator._request_outcomes = {}
        simulator.trace_file = None

        deployment = SimpleNamespace(rank_name="rank-oom")
        rank_run = SimpleNamespace(deployment=deployment, request_count=1)
        oom = SimpleNamespace(
            status="oom",
            deployment=deployment,
            request_rows=(),
            completed_requests=(),
            stdout="",
            stderr="[MemoryModel] exceeds total NPU memory",
        )
        request = Request("request-a", "session", 0, 0, 1, 1, None)

        with (
            patch(
                "tandemn_system_data.clients.HardwareCatalogStore",
                return_value=SimpleNamespace(
                    all=lambda: [SimpleNamespace(catalog={"regions": [{}]})]
                ),
            ),
            patch(
                "tandemn_system_data.clients.ResourceMapStore",
                return_value=SimpleNamespace(
                    get=lambda: SimpleNamespace(scheduling_summary=lambda: {})
                ),
            ),
            patch.object(
                simulator, "_prepare_job_runs", return_value=((rank_run,), ())
            ),
            patch("runner.build_batch_probe", return_value=(request,)),
            patch("runner.run_ranks", return_value=(oom,)),
            patch("runner.gpu_metrics_from_results", return_value=[]),
        ):
            simulator._simulate_tick(0, 0, 300_000_000_000)

        self.assertEqual(rank_calls[0][1]["reason_code"], "OOM")
        self.assertEqual(transitions[0][0], ("job-a", "waiting", ["running"]))
        self.assertEqual(simulator._batch_progress["job-a"].completed_requests, 3)
        self.assertEqual(simulator._batch_progress["job-a"].next_request_number, 2)

    def test_profile_rejection_requeues_and_reports_unfinished_requests(self) -> None:
        """Return a profile-rejected job to the planner with its backlog."""
        job = OnlineJob(
            "online", "job-a", "model", 0, 300_000_000_000, {"features": {}}
        )
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job-a", spec_json={"model_id": "model"}),
            ranks=(SimpleNamespace(rank_id="rank-a", shape_json={}),),
        )
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario(
            "zero-rank-runs", 300_000_000_000, 300_000_000_000, (job,), ()
        )
        simulator.user_id = "user-a"
        simulator.postgres = object()
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.jobs = SimpleNamespace(
            running_jobs=lambda user_id: (running,),
            set_rank_status=lambda *args, **kwargs: None,
            transition=lambda *args, **kwargs: transitions.append((args, kwargs)) or True,
        )
        simulator.llmservingsim_root = Path(tempfile.gettempdir())
        simulator.parallel_simulators = 1
        simulator._router = StickyRoundRobinRouter()
        simulator._traffic_starts = {}
        simulator._requests = {
            "job-a": tuple(
                Request(f"request-{index}", "session", 0, 0, 1, 1, None)
                for index in range(3)
            )
        }
        simulator._online_jobs = {"job-a": job}
        simulator._online_backlog = {"job-a": ()}
        simulator._batch_jobs = {}
        simulator._batch_progress = {}
        simulator._request_outcomes = {}
        transitions = []
        profile_failure = SimpleNamespace(
            rank_id="rank-a",
            reason_code="SOFTWARE_STACK_RANK_FAILURE",
            error_detail="PROFILE_UNAVAILABLE",
        )

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with (
                patch(
                    "tandemn_system_data.clients.HardwareCatalogStore",
                    return_value=SimpleNamespace(
                        all=lambda: [SimpleNamespace(catalog={"regions": [{}]})]
                    ),
                ),
                patch(
                    "tandemn_system_data.clients.ResourceMapStore",
                    return_value=SimpleNamespace(
                        get=lambda: SimpleNamespace(scheduling_summary=lambda: {})
                    ),
                ),
                patch.object(
                    simulator,
                    "_prepare_job_runs",
                    return_value=((), (profile_failure,)),
                ),
                patch("runner.run_ranks") as run,
                patch("runner.gpu_metrics_from_results", return_value=[]),
            ):
                simulator._simulate_tick(0, 0, 300_000_000_000)
            events = [
                json.loads(line)
                for line in simulator.trace_file.read_text(encoding="utf-8").splitlines()
            ]

        run.assert_not_called()
        event = next(event for event in events if event["kind"] == "job_window")
        self.assertEqual(event["offered_request_count"], 3)
        self.assertEqual(event["attempted_request_count"], 0)
        self.assertEqual(event["completed_request_count"], 0)
        self.assertEqual(event["unfinished_request_count"], 3)
        self.assertEqual(event["slo_fail_count"], 0)
        self.assertEqual(transitions[0][0], ("job-a", "waiting", ["running"]))
        self.assertEqual(len(simulator._online_backlog["job-a"]), 3)

    def test_rankless_running_job_is_skipped(self) -> None:
        """Wait for a planner replacement instead of routing into no ranks."""
        job = OnlineJob("online", "job-a", "model", 0, 300_000_000_000, {"features": {}})
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job-a", spec_json={"model_id": "model"}),
            ranks=(),
        )
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario("rankless", 300_000_000_000, 300_000_000_000, (job,), ())
        simulator.user_id = "user-a"
        simulator.postgres = object()
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.jobs = SimpleNamespace(running_jobs=lambda user_id: (running,))
        simulator.llmservingsim_root = Path(tempfile.gettempdir())
        simulator.parallel_simulators = 1
        simulator._traffic_starts = {}
        simulator._requests = {
            "job-a": (Request("request-a", "session", 0, 0, 1, 1, None),)
        }
        simulator._online_jobs = {"job-a": job}
        simulator._online_backlog = {"job-a": ()}
        simulator._batch_jobs = {}
        simulator._batch_progress = {}
        simulator.trace_file = None

        class HardwareCatalogStore:
            def __init__(self, postgres) -> None:
                pass

            def get(self):
                return SimpleNamespace(
                    catalog={"regions": [{"cloud": "aws", "region": "test", "instance_types": []}]}
                )

        class ResourceMapStore:
            def __init__(self, postgres, user_id) -> None:
                pass

            def get(self):
                return SimpleNamespace(scheduling_summary=lambda: {})

        with (
            patch("tandemn_system_data.clients.HardwareCatalogStore", HardwareCatalogStore),
            patch("tandemn_system_data.clients.ResourceMapStore", ResourceMapStore),
            patch.object(simulator, "_prepare_job_runs") as prepare,
            patch("runner.run_ranks") as run,
        ):
            simulator._simulate_tick(0, 0, 300_000_000_000)

        prepare.assert_not_called()
        run.assert_not_called()

    def test_online_backlog_retries_before_fresh_requests_and_drains(self) -> None:
        """Retry unfinished traffic each tick until the stopped source drains."""
        job = OnlineJob(
            "online",
            "job",
            "model",
            0,
            600_000_000_000,
            {
                "features": {
                    "target_p99_ttft_ms": 1,
                    "target_p99_tpot_ms": 1,
                }
            },
        )
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job"),
            ranks=(SimpleNamespace(rank_id="rank-job"),),
        )
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
        simulator._request_outcomes = {}
        simulator.trace_file = None

        attempts = []
        completed_names = iter(("r1", "r2", "r0"))

        class HardwareCatalogStore:
            def __init__(self, postgres) -> None:
                pass

            def get(self):
                return SimpleNamespace(catalog={"regions": [{"cloud": "aws", "region": "test", "instance_types": []}]})

        class ResourceMapStore:
            def __init__(self, postgres, user_id) -> None:
                pass

            def get(self):
                return SimpleNamespace(scheduling_summary=lambda: {})

        def prepare(running, job_name, tick_requests, *args):
            attempts.append(tick_requests)
            return (
                (
                    SimpleNamespace(
                        deployment="deployment",
                        requests=tick_requests,
                        request_count=len(tick_requests),
                    ),
                ),
                (),
            )

        def run(rank_runs, max_parallel, progress=None):
            name = next(completed_names)
            completed = next(
                request
                for request in rank_runs[0].requests
                if request.request_name == name
            )
            return (
                SimpleNamespace(
                    completed_requests=(completed,),
                    request_rows=({"TTFT": 1, "TPOT": 1, "output": 1},),
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
        self.assertEqual(
            simulator._terminal_request_outcome("job"),
            {
                "offered_request_count": 3,
                "completed_request_count": 3,
                "unfinished_request_count": 0,
                "ttft_pass_count": 3,
                "tpot_pass_count": 3,
                "slo_pass_count": 3,
                "qualified_output_tokens": 3,
                "slo_count_scope": "all_offered_requests",
            },
        )
        finish.assert_called_once_with(
            running,
            datetime(2026, 1, 1, 0, 10, tzinfo=UTC),
        )

    def test_online_job_uses_its_authored_window_after_late_admission(self) -> None:
        """Do not replay requests from before a late placement."""
        job = OnlineJob(
            "online",
            "job",
            "model",
            0,
            400_000_000_000,
            {"features": {"target_p99_ttft_ms": 1, "target_p99_tpot_ms": 1}},
        )
        running = SimpleNamespace(
            job=SimpleNamespace(job_id="job"),
            ranks=(SimpleNamespace(rank_id="rank-job"),),
        )
        before_admission = Request("before", "before", 0, 0, 1, 1, None)
        during_window = Request("during", "during", 0, 300_000_000_000, 1, 1, None)
        at_stop = Request("at-stop", "at-stop", 0, 400_000_000_000, 1, 1, None)
        simulator = object.__new__(ClusterSimulator)
        simulator.scenario = Scenario(
            "late-admission", 300_000_000_000, 600_000_000_000, (job,), ()
        )
        simulator.user_id = "user_1"
        simulator.postgres = object()
        simulator.clock = LogicalClock(datetime(2026, 1, 1, tzinfo=UTC))
        simulator.clock.advance(timedelta(minutes=5))
        simulator.jobs = SimpleNamespace(running_jobs=lambda _user_id: (running,))
        simulator.llmservingsim_root = Path(tempfile.gettempdir())
        simulator.parallel_simulators = 1
        simulator._traffic_starts = {}
        simulator._requests = {
            "job": (before_admission, during_window, at_stop),
        }
        simulator._online_jobs = {"job": job}
        simulator._online_backlog = {"job": ()}
        simulator._batch_jobs = {}
        simulator._batch_progress = {}
        simulator._request_outcomes = {}
        simulator.trace_file = None
        prepared = []
        deployment = SimpleNamespace(rank_name="rank-job")

        def prepare(_running, _job_name, requests, *_args):
            prepared.extend(requests)
            return (
                (
                    SimpleNamespace(
                        deployment=deployment,
                        requests=requests,
                        request_count=len(requests),
                    ),
                ),
                (),
            )

        result = SimpleNamespace(
            deployment=deployment,
            completed_requests=(during_window,),
            request_rows=(
                {"TTFT": 1, "TPOT": 1, "output": 1, "end_time": 200_000_000_000},
            ),
        )
        with (
            patch(
                "tandemn_system_data.clients.HardwareCatalogStore",
                return_value=SimpleNamespace(
                    get=lambda: SimpleNamespace(
                        catalog={"regions": [{"cloud": "aws", "region": "test", "instance_types": []}]}
                    )
                ),
            ),
            patch(
                "tandemn_system_data.clients.ResourceMapStore",
                return_value=SimpleNamespace(
                    get=lambda: SimpleNamespace(scheduling_summary=lambda: {})
                ),
            ),
            patch.object(simulator, "_prepare_job_runs", side_effect=prepare),
            patch("runner.run_ranks", return_value=(result,)),
            patch("runner.gpu_metrics_from_results", return_value=[]) as gpu_metrics,
            patch.object(simulator, "_finish") as finish,
        ):
            simulator._simulate_tick(1, 300_000_000_000, 600_000_000_000)

        self.assertEqual(prepared, [during_window])
        self.assertEqual(gpu_metrics.call_args.kwargs["online_active_ns"], 100_000_000_000)
        self.assertEqual(simulator._online_backlog["job"], (during_window,))
        simulator._ensure_online_offers_through(
            "job", datetime(2026, 1, 1, 0, 10, tzinfo=UTC)
        )
        self.assertEqual(
            simulator._terminal_request_outcome("job"),
            {
                "offered_request_count": 1,
                "completed_request_count": 0,
                "unfinished_request_count": 1,
                "ttft_pass_count": 0,
                "tpot_pass_count": 0,
                "slo_pass_count": 0,
                "qualified_output_tokens": 0,
                "slo_count_scope": "all_offered_requests",
            },
        )
        finish.assert_not_called()

    def test_failed_metric_commit_keeps_request_state_unchanged(self) -> None:
        """Do not apply staged online or batch progress after commit failure."""
        online = OnlineJob("online", "online-job", "model", 0, 300_000_000_000, {})
        batch = BatchJob(
            "batch",
            "batch-job",
            "model",
            0,
            100_000_000_000,
            600_000_000_000,
            {"record_count": 1_000, "features": {}},
        )
        online_request = Request("online-r0", "session", 0, 0, 1, 1, None)
        fresh_request = Request("online-r1", "fresh-session", 0, 1, 1, 1, None)
        batch_request = Request("batch-r0", "batch-session", 0, 0, 1, 1, None)
        running_jobs = (
            SimpleNamespace(
                job=SimpleNamespace(job_id="online-job"),
                ranks=(SimpleNamespace(rank_id="rank-online"),),
            ),
            SimpleNamespace(
                job=SimpleNamespace(job_id="batch-job"),
                ranks=(SimpleNamespace(rank_id="rank-batch"),),
            ),
        )
        original_progress = BatchProgress(total_requests=1_000)
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

        online_deployment = SimpleNamespace(rank_name="online-rank")
        batch_deployment = SimpleNamespace(rank_name="batch-rank")
        rank_runs = iter(
            (
                (
                    SimpleNamespace(
                        deployment=online_deployment, request_count=2
                    ),
                ),
                (
                    SimpleNamespace(
                        deployment=batch_deployment, request_count=1
                    ),
                ),
            )
        )
        result_row = {
            "TTFT": 1_000_000,
            "TPOT": 100_000,
            "output": 10,
            "end_time": 1_000_000_000,
        }
        results = (
            SimpleNamespace(
                deployment=online_deployment,
                completed_requests=(online_request,),
                request_rows=(result_row,),
                status="completed",
                sim_time_ns=1_000_000_000,
            ),
            SimpleNamespace(
                deployment=batch_deployment,
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
                return SimpleNamespace(catalog={"regions": [{"cloud": "aws", "region": "test", "instance_types": []}]})

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
            patch.object(
                simulator,
                "_prepare_job_runs",
                side_effect=lambda *args: (next(rank_runs), ()),
            ),
            patch("runner.build_batch_probe", return_value=(batch_request,)),
            patch("runner.run_ranks", return_value=results),
            patch(
                "runner.gpu_metrics_from_results", return_value=[object()]
            ) as gpu_metrics,
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
        self.assertEqual(gpu_metrics.call_args_list[0].kwargs, {})
        batch_call = gpu_metrics.call_args_list[1]
        self.assertEqual(batch_call.kwargs["batch_active_ns"], 100_000_000_000)
        self.assertEqual(
            tuple(batch_call.kwargs["batch_measurements"]),
            ("batch-rank",),
        )
        finish.assert_not_called()

    def test_active_online_job_runs_rank_and_commits_tick_metrics(self) -> None:
        """Route one admitted job through the ServingSim adapter boundary."""
        source = load_scenario(ROOT / "benchmarks/simulation_3h/benchmark.yaml")
        job = replace(source.online_jobs[0], store_job_id="job_1", stops_at_ns=900_000_000_000)
        scenario = Scenario("runner-smoke", 300_000_000_000, 300_000_000_000, (job,), ())
        rank = SimpleNamespace(
            rank_id="rank_1",
            n_replicas=4,
            shape_json={
                "env": ["reserved", "sim", "region-1", "zone-1", "H100-SXM-80GB"],
                "count": 2,
                "tp": 2,
                "pp": 1,
                "dp": 1,
                "gpu_count": 2,
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
        request = Request("request-0", "session", 0, 0, 100, 10, None)
        simulator._requests = {job.store_job_id: (request,)}
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

        def prepare(*args):
            deployment = args[0]
            return SimpleNamespace(deployment=deployment, request_count=1)

        def run(rank_runs, max_parallel, progress=None):
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
                    completed_requests=(request,),
                ),
            )

        with tempfile.TemporaryDirectory() as directory:
            simulator.trace_file = Path(directory) / "trace.jsonl"
            with (
                patch("tandemn_system_data.clients.HardwareCatalogStore", HardwareCatalogStore),
                patch("tandemn_system_data.clients.ResourceMapStore", ResourceMapStore),
                patch("tandemn_system_data.clients.GpuMetricStore", GpuMetricStore),
                patch("runner.profile_coverage_error", return_value=None),
                patch("runner.prepare_rank_run", prepare),
                patch("runner.run_ranks", run),
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

        self.assertEqual(calls[0], ("run", 1))
        self.assertEqual(calls[1][0], "commit")
        self.assertEqual(len(calls[1][1]), 2)
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
        self.assertEqual(pool_event["allocated_gpu_count"], 8)
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

    def test_runner_does_not_stamp_store_quantization(self) -> None:
        """Keep simulation quantization independent from the Store model catalog."""
        source = load_scenario(ROOT / "benchmarks/simulation_3h/benchmark.yaml")
        job = replace(source.online_jobs[0], store_job_id="job_1")
        simulator = object.__new__(ClusterSimulator)
        simulator.postgres = object()
        simulator.llmservingsim_root = Path(tempfile.gettempdir())
        simulator.container = "servingsim_docker"
        simulator.run_name = "runner-dtype"
        simulator.rank_timeout_seconds = 30
        simulator._online_jobs = {job.store_job_id: job}
        simulator._batch_jobs = {}
        request = Request("request-0", "session", 0, 0, 100, 10, None)
        shape = {
            "env": ["reserved", "sim", "region-1", "zone-1", "H100-SXM-80GB"],
            "count": 2,
            "tp": 2,
            "pp": 1,
            "gpu_count": 2,
            "instance_type": "sim.h100.8gpu",
        }
        captured_shapes = []

        def prepare(deployment, *args):
            captured_shapes.append(dict(deployment.shape))
            return SimpleNamespace()

        def run_with_shape(rank_shape):
            simulator._router = StickyRoundRobinRouter()
            running = SimpleNamespace(
                job=SimpleNamespace(
                    job_id=job.store_job_id,
                    spec_json={"model_id": job.model_name},
                ),
                ranks=(
                    SimpleNamespace(
                        rank_id="rank_1",
                        n_replicas=1,
                        shape_json=rank_shape,
                    ),
                ),
            )
            simulator._prepare_job_runs(
                running,
                job.name,
                (request,),
                0,
                0,
                {},
            )

        with (
            patch("runner.profile_coverage_error", return_value=None),
            patch("runner.resolve_rank_hardware", return_value={}),
            patch("runner.prepare_rank_run", side_effect=prepare),
        ):
            run_with_shape(
                {
                    **shape,
                    "weight_dtype": "bfloat16",
                    "weight_quantization_method": "none",
                }
            )
            run_with_shape(shape)

        self.assertEqual(captured_shapes[0]["weight_dtype"], "bfloat16")
        self.assertEqual(captured_shapes[0]["weight_quantization_method"], "none")
        self.assertNotIn("weight_dtype", captured_shapes[1])
        self.assertNotIn("weight_quantization_method", captured_shapes[1])

    def test_batch_probe_advances_full_backlog_by_observed_rate(self) -> None:
        """Scale a two-second, twenty-request probe across one five-minute tick."""
        measurement = BatchProbeMeasurement(
            SimpleNamespace(),
            2_000_000_000,
            20,
            200,
        )
        progress = BatchProgress(total_requests=10_000)

        completed = advance_batch_progress(
            progress,
            (measurement,),
            300_000_000_000,
        )

        self.assertEqual(completed, 3_000)
        self.assertEqual(progress.completed_requests, 3_000)
        self.assertEqual(progress.next_request_number, 20)

        progress.completed_requests = 9_500
        completed = advance_batch_progress(
            progress,
            (measurement,),
            300_000_000_000,
        )

        self.assertEqual(completed, 500)
        self.assertEqual(progress.completed_requests, 10_000)

    def test_batch_rate_is_unavailable_without_measurements(self) -> None:
        progress = BatchProgress(total_requests=10)

        completed = advance_batch_progress(
            progress,
            (),
            300_000_000_000,
            attempted_requests=2,
        )

        self.assertIsNone(_batch_request_rate_per_second(()))
        self.assertEqual(completed, 0)
        self.assertEqual(progress.completed_requests, 0)
        self.assertEqual(progress.next_request_number, 2)

    def test_batch_probe_telemetry_matches_campaign_rate(self) -> None:
        deployment = SimpleNamespace(
            rank_name="rank_a",
            model_name="model",
            chains=(SimpleNamespace(rank_id="rank_a", chain_index=0, gpu_count=1),),
            shape={"instance_type": "sim", "target_p99_ttft_ms": 100},
        )
        result = RankResult(
            deployment,
            (
                {
                    "instance id": 0,
                    "input": 100,
                    "output": 10,
                    "end_time": 10_000_000_000,
                    "TTFT": 10_000_000,
                    "TPOT": 1_000_000,
                },
                {
                    "instance id": 0,
                    "input": 100,
                    "output": 20,
                    "end_time": 20_000_000_000,
                    "TTFT": 20_000_000,
                    "TPOT": 2_000_000,
                },
            ),
            (
                RankMetricSample(
                    10_000_000_000,
                    10_000_000_000,
                    0,
                    {"kv_cache_util": 0.2},
                ),
                RankMetricSample(
                    20_000_000_000,
                    10_000_000_000,
                    0,
                    {"kv_cache_util": 0.6},
                ),
            ),
            "",
            "",
        )
        measurements = batch_probe_measurements((result,))
        progress = BatchProgress(total_requests=100)

        completed = advance_batch_progress(
            progress,
            measurements,
            300_000_000_000,
        )
        start = datetime(2026, 1, 1, tzinfo=UTC)
        metrics = gpu_metrics_from_results(
            SimpleNamespace(job_id="job_a"),
            (deployment,),
            (result,),
            start,
            300_000_000_000,
            {},
            batch_measurements={"rank_a": measurements[0]},
            batch_active_ns=300_000_000_000,
        )

        self.assertEqual(completed, 30)
        self.assertEqual(progress.completed_requests, 30)
        self.assertEqual(len(metrics), 30)
        self.assertEqual(
            [metric.throughput_token_per_sec for metric in metrics[:3]],
            [1.0, 2.0, 1.5],
        )
        self.assertEqual(
            [metric.kv_cache_util for metric in metrics[:3]],
            [0.2, 0.6, 0.6],
        )
        self.assertEqual(metrics[-1].ts, start + timedelta(seconds=295))

    def test_batch_cutoff_scales_each_rank_by_simulated_observation_time(self) -> None:
        """Project partial rank probes using their reported simulated horizons."""
        measurements = (
            BatchProbeMeasurement(SimpleNamespace(), 10_000_000_000, 20, 200),
            BatchProbeMeasurement(SimpleNamespace(), 5_000_000_000, 10, 100),
        )
        progress = BatchProgress(total_requests=100_000)

        completed = advance_batch_progress(
            progress,
            measurements,
            300_000_000_000,
            attempted_requests=200,
        )

        self.assertEqual(completed, 1_200)
        self.assertEqual(progress.completed_requests, 1_200)
        self.assertEqual(progress.next_request_number, 200)

    def test_batch_completion_uses_observed_in_tick_time(self) -> None:
        """Record batch completion at its replay-derived time inside the tick."""
        measurement = BatchProbeMeasurement(
            SimpleNamespace(),
            2_000_000_000,
            20,
            200,
        )
        progress = BatchProgress(total_requests=10_000, completed_requests=9_500)

        advance_batch_progress(
            progress,
            (measurement,),
            300_000_000_000,
            600_000_000_000,
        )

        self.assertEqual(progress.completed_at_ns, 650_000_000_000)

    def test_batch_capacity_rounds_after_combining_rank_rates(self) -> None:
        measurements = (
            BatchProbeMeasurement(SimpleNamespace(), 1_500_000_000, 1, 1),
            BatchProbeMeasurement(SimpleNamespace(), 1_500_000_000, 1, 1),
        )
        progress = BatchProgress(total_requests=10)

        completed = advance_batch_progress(
            progress,
            measurements,
            1_000_000_000,
        )

        self.assertEqual(completed, 1)

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
