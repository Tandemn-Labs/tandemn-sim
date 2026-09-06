"""Tests for the real-hardware trace collector skeleton."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tandemn_system_data.models.causal_graph import CausalNode
from tandemn_system_data.models.enums import JobKind, RankRole, RankStatus
from tandemn_system_data.models.event import Event
from tandemn_system_data.models.evidence import EvidenceRow
from tandemn_system_data.models.gpu_metric import GpuMetric
from tandemn_system_data.models.job import Job
from tandemn_system_data.models.plan import Plan, PlanAction
from tandemn_system_data.models.rank import Rank
from tandemn_system_data.models.resource_map import ResourceMap

from data_generation.traces import (
    LIFECYCLE_EVENT_TYPES,
    AIPerfRequest,
    RealTraceCollector,
    summarize_aiperf,
)


class FakeEventLog:
    """Expose a fixed EventLog batch and record acknowledgements."""

    def __init__(self, events: list[Event]) -> None:
        self.events = events
        self.reads: list[tuple[str, int, set[str] | None, str | None]] = []
        self.acknowledged: list[tuple[str, str]] = []

    def read_for_consumer(
        self,
        consumer_name: str,
        *,
        limit: int,
        types: set[str] | None = None,
        user_id: str | None = None,
    ) -> list[Event]:
        self.reads.append((consumer_name, limit, types, user_id))
        return self.events[:limit]

    def ack(self, consumer_name: str, event_id: str) -> None:
        self.acknowledged.append((consumer_name, event_id))


class FakeJobStore:
    """Return fixed Store job and rank rows without a database."""

    def __init__(self, jobs: list[Job], ranks_by_job: dict[str, list[Rank]]) -> None:
        self.jobs = jobs
        self.ranks_by_job = ranks_by_job
        self.user_ids: list[str] = []

    def list_jobs(self, user_id: str) -> list[Job]:
        self.user_ids.append(user_id)
        return self.jobs

    def ranks(self, job_id: str) -> list[Rank]:
        return self.ranks_by_job.get(job_id, [])


class FakeResourceMapStore:
    """Return the current mutable resource snapshot."""

    def __init__(self, resource_map: ResourceMap) -> None:
        self.resource_map = resource_map
        self.calls = 0

    def get(self) -> ResourceMap:
        self.calls += 1
        return self.resource_map


class FakeEvidenceStore:
    """Return fixed Evidence rows and record requested owners."""

    def __init__(self, rows: list[EvidenceRow]) -> None:
        self.rows = rows
        self.user_ids: list[str] = []

    def latest(self, user_id: str) -> list[EvidenceRow]:
        self.user_ids.append(user_id)
        return self.rows


class FakeCausalGraphStore:
    """Expose the three direct CausalGraphStore reads used by traces."""

    def __init__(self) -> None:
        self.calls = 0

    def load_nodes(self) -> dict[str, CausalNode]:
        self.calls += 1
        return {"throughput": CausalNode("throughput", "metric")}

    def load_edges(self) -> tuple[dict[str, object], dict[str, object]]:
        self.calls += 1
        return {}, {}

    def load_mechanisms(self) -> tuple[dict[str, object], dict[str, object]]:
        self.calls += 1
        return {}, {}


class FakeGpuMetricStore:
    """Return fixed GPU telemetry rows and record the canonical Store query."""

    def __init__(self, rows: list[GpuMetric]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, str, datetime, datetime]] = []

    def rows_for_job_window(
        self,
        user_id: str,
        job_id: str,
        start: datetime,
        end: datetime,
    ) -> list[GpuMetric]:
        self.calls.append((user_id, job_id, start, end))
        return self.rows


class FakePlanStore:
    """Return canonical Plan rows by ID without a database."""

    def __init__(self, plans: list[Plan]) -> None:
        self.plans = {plan.plan_id: plan for plan in plans}

    def get(self, plan_id: str) -> Plan | None:
        return self.plans.get(plan_id)


def store_event(
    event_id: str,
    user_id: str | None,
    job_id: str | None,
    rank_id: str | None,
    event_type: str,
    payload_json: dict[str, object],
    created_at: datetime,
    ) -> Event:
    """Build a real Store Event without repeating Pydantic keyword arguments."""
    return Event(
        event_id=event_id,
        user_id=user_id,
        job_id=job_id,
        rank_id=rank_id,
        type=event_type,
        payload_json=payload_json,
        created_at=created_at,
    )


def evidence_row(job_id: str = "job_01J") -> EvidenceRow:
    """Build the smallest valid Evidence row for trace snapshot tests."""
    return EvidenceRow(
        row_id=f"1_{job_id}_rank_01J",
        tick=1,
        deploy_timestamp_utc=1.0,
        job_id=job_id,
        rank_id="rank_01J",
        env_label=("reserved", "aws", "us-east-1", "use1-az1", "L40S"),
        X={},
        V_observed_trajectory={},
        V_predicted_trajectory={},
        y_observed_trajectory={},
        y_predicted={"throughput_token_per_sec": 10.0},
        y_observed_mean={},
        residuals_per_v={},
        residuals_per_y={},
    )


def aiperf_record(
    request_id: str,
    *,
    start_ns: int,
    end_ns: int,
    ttft: float | None = None,
    itl: float | None = None,
    latency: float | None = None,
    input_length: int | None = None,
    output_length: int | None = None,
    error: dict[str, object] | None = None,
    cancelled: bool = False,
) -> dict[str, object]:
    """Build one pinned-AIPerf JSONL record using its exported metric tags."""
    metrics: dict[str, dict[str, object]] = {}
    for name, value, unit in (
        ("time_to_first_token", ttft, "ms"),
        ("inter_token_latency", itl, "ms"),
        ("request_latency", latency, "ms"),
        ("input_sequence_length", input_length, "tokens"),
        ("output_sequence_length", output_length, "tokens"),
    ):
        if value is not None:
            metrics[name] = {"value": value, "unit": unit}
    return {
        "metadata": {
            "x_request_id": request_id,
            "request_start_ns": start_ns,
            "request_end_ns": end_ns,
            "was_cancelled": cancelled,
        },
        "metrics": metrics,
        "error": error,
    }


class RealTraceCollectorTests(unittest.TestCase):
    """Verify run metadata, bindings, and the fixed collection cadence."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.directory.name) / "run"
        self.started_at = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)
        self.collector = RealTraceCollector(
            run_dir=self.run_dir,
            run_id="run-1",
            benchmark_id="15min_real",
            method="koi",
            user_id="tenant",
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_start_and_close_write_terminal_jsonl_events(self) -> None:
        self.collector.start_run(self.started_at)
        self.collector.close_run(
            self.started_at + timedelta(minutes=15),
            status="completed",
        )

        events = self.read_events()
        self.assertEqual(
            [event["kind"] for event in events], ["run_started", "run_finished"]
        )
        self.assertEqual(events[0]["run_id"], "run-1")
        self.assertEqual(events[0]["benchmark"], "15min_real")
        self.assertEqual(events[0]["method"], "koi")
        self.assertEqual(events[0]["user_id"], "tenant")
        self.assertEqual(events[0]["collection_interval_seconds"], 60)
        self.assertEqual(events[0]["tick_holdback_seconds"], 60)
        self.assertFalse(events[0]["evidence_collection"])
        self.assertFalse(events[0]["causal_state_collection"])
        self.assertEqual(events[1]["status"], "completed")

    def test_bind_job_records_authored_and_store_ids(self) -> None:
        self.collector.start_run(self.started_at)
        self.collector.bind_job(
            authored_job_id="benchmark-job",
            store_job_id="job_01J",
            job_kind="online",
            benchmark_job_path=Path("benchmarks/15min_real/jobs/code-chat.yaml"),
        )

        binding = self.collector.job_binding("benchmark-job")
        self.assertEqual(binding.store_job_id, "job_01J")
        self.assertEqual(binding.job_kind, "online")
        self.assertEqual(
            binding.aiperf_profile_path,
            self.run_dir / "benchmark-job" / "artifacts" / "profile_export.jsonl",
        )

        with self.assertRaisesRegex(ValueError, "already bound"):
            self.collector.bind_job(
                authored_job_id="benchmark-job",
                store_job_id="job_02J",
                job_kind="online",
                benchmark_job_path=Path("other.yaml"),
            )

    def test_collect_is_due_once_per_collection_interval(self) -> None:
        self.collector.start_run(self.started_at)

        self.assertFalse(self.collector.collect(self.started_at))
        self.assertFalse(
            self.collector.collect(self.started_at + timedelta(seconds=59))
        )
        self.assertTrue(
            self.collector.collect(self.started_at + timedelta(seconds=60))
        )
        self.assertFalse(
            self.collector.collect(self.started_at + timedelta(seconds=61))
        )
        self.assertTrue(
            self.collector.collect(self.started_at + timedelta(seconds=120))
        )

    def test_causal_state_collection_is_opt_in(self) -> None:
        enabled = RealTraceCollector(
            run_dir=self.run_dir,
            run_id="run-2",
            benchmark_id="15min_real",
            method="koi",
            user_id="tenant",
            collect_causal_state=True,
        )
        enabled.start_run(self.started_at)

        self.assertTrue(self.read_events()[0]["causal_state_collection"])

    def test_store_events_create_ticks_and_forward_lifecycle(self) -> None:
        self.collector.start_run(self.started_at)
        self.collector.bind_job(
            authored_job_id="benchmark-job",
            store_job_id="job_01J",
            job_kind="online",
            benchmark_job_path=Path("job.yaml"),
        )
        log = FakeEventLog(
            [
                store_event(
                    "evt-1",
                    "tenant",
                    "job_01J",
                    None,
                    "job.submitted",
                    {"job_id": "job_01J"},
                    self.started_at,
                ),
                store_event(
                    "evt-2",
                    "tenant",
                    None,
                    None,
                    "plan.created",
                    {"plan_id": "plan_01J"},
                    self.started_at + timedelta(minutes=1),
                ),
                store_event(
                    "evt-3",
                    "tenant",
                    None,
                    None,
                    "plan.applied",
                    {"plan_id": "plan_01J"},
                    self.started_at + timedelta(minutes=1, seconds=1),
                ),
                store_event(
                    "evt-4",
                    "tenant",
                    "job_01J",
                    "rank_01J",
                    "rank.running",
                    {"rank_id": "rank_01J"},
                    self.started_at + timedelta(minutes=1, seconds=2),
                ),
                store_event(
                    "evt-5",
                    "other-tenant",
                    "job_01J",
                    None,
                    "job.finished",
                    {},
                    self.started_at + timedelta(minutes=1, seconds=3),
                ),
                store_event(
                    "evt-6",
                    "tenant",
                    "job_other",
                    None,
                    "job.finished",
                    {},
                    self.started_at + timedelta(minutes=1, seconds=4),
                ),
            ]
        )

        self.assertEqual(self.collector.collect_store_events(log), 6)

        events = self.read_events()
        self.assertEqual(
            [event["kind"] for event in events],
            ["run_started", "job.submitted", "plan_created", "rank.running"],
        )
        self.assertIsNone(events[1]["tick"])
        self.assertEqual(events[2]["tick"], 0)
        self.assertEqual(events[2]["plan_id"], "plan_01J")
        self.assertEqual(events[3]["tick"], 0)
        self.assertEqual(events[3]["source_event_id"], "evt-4")
        self.assertEqual(
            log.acknowledged,
            [("real-trace:run-1", f"evt-{index}") for index in range(1, 7)],
        )
        self.assertEqual(
            log.reads,
            [
                (
                    "real-trace:run-1",
                    100,
                    set(LIFECYCLE_EVENT_TYPES | {"plan.created"}),
                    "tenant",
                )
            ],
        )

    def test_plan_ticks_follow_event_order(self) -> None:
        self.collector.start_run(self.started_at)
        plans = FakePlanStore(
            [
                Plan(
                    plan_id="plan_1",
                    user_id="tenant",
                    actions=[PlanAction(job_id="job_01J", type="keep")],
                ),
                Plan(plan_id="plan_2", user_id="tenant"),
            ]
        )
        log = FakeEventLog(
            [
                store_event(
                    "evt-1",
                    "tenant",
                    None,
                    None,
                    "plan.created",
                    {"plan_id": "plan_1"},
                    self.started_at,
                ),
                store_event(
                    "evt-2",
                    "tenant",
                    None,
                    None,
                    "plan.created",
                    {"plan_id": "plan_2"},
                    self.started_at + timedelta(minutes=1),
                ),
            ]
        )

        self.collector.collect_store_events(log, plans=plans)

        plan_events = [
            event for event in self.read_events() if event["kind"] == "plan_created"
        ]
        self.assertEqual([event["tick"] for event in plan_events], [0, 1])
        self.assertEqual(plan_events[0]["plan"]["actions"][0]["job_id"], "job_01J")

    def test_closed_ticks_use_a_fixed_holdback_and_do_not_reopen(self) -> None:
        self.collector.start_run(self.started_at)
        self.collector.collect_store_events(
            FakeEventLog(
                [
                    store_event(
                        "evt-1",
                        "tenant",
                        None,
                        None,
                        "plan.created",
                        {"plan_id": "plan_1"},
                        self.started_at,
                    ),
                    store_event(
                        "evt-2",
                        "tenant",
                        None,
                        None,
                        "plan.created",
                        {"plan_id": "plan_2"},
                        self.started_at + timedelta(minutes=1),
                    ),
                    store_event(
                        "evt-3",
                        "tenant",
                        None,
                        None,
                        "plan.created",
                        {"plan_id": "plan_3"},
                        self.started_at + timedelta(minutes=2),
                    ),
                ]
            )
        )

        self.assertEqual(
            self.collector.tick_for(self.started_at + timedelta(seconds=59)),
            0,
        )
        self.assertEqual(
            self.collector.tick_for(self.started_at + timedelta(minutes=1)),
            1,
        )
        self.assertEqual(self.collector.finalize_closed_ticks(self.started_at), ())
        self.assertEqual(
            [tick.tick for tick in self.collector.finalize_closed_ticks(
                self.started_at + timedelta(minutes=2)
            )],
            [0],
        )
        self.assertIsNone(
            self.collector.tick_for(self.started_at + timedelta(seconds=30))
        )
        self.assertEqual(
            [tick.tick for tick in self.collector.finalize_closed_ticks(
                self.started_at + timedelta(minutes=3)
            )],
            [1],
        )

    def test_close_open_tick_uses_the_same_holdback(self) -> None:
        self.collector.start_run(self.started_at)
        self.collector.collect_store_events(
            FakeEventLog(
                [
                    store_event(
                        "evt-1",
                        "tenant",
                        None,
                        None,
                        "plan.created",
                        {"plan_id": "plan_1"},
                        self.started_at,
                    )
                ]
            )
        )
        self.collector.close_open_tick(self.started_at + timedelta(minutes=10))

        self.assertEqual(
            self.collector.finalize_closed_ticks(
                self.started_at + timedelta(minutes=10, seconds=59)
            ),
            (),
        )
        self.assertEqual(
            [tick.tick for tick in self.collector.finalize_closed_ticks(
                self.started_at + timedelta(minutes=11)
            )],
            [0],
        )

    def test_aiperf_reader_keeps_partial_lines_and_deduplicates_requests(self) -> None:
        self.collector.start_run(self.started_at)
        self.collector.bind_job(
            authored_job_id="benchmark-job",
            store_job_id="job_01J",
            job_kind="online",
            benchmark_job_path=Path("job.yaml"),
        )
        path = self.collector.job_binding("benchmark-job").aiperf_profile_path
        path.parent.mkdir(parents=True)
        first = aiperf_record(
            "request-1",
            start_ns=1_725_000_000_000_000_000,
            end_ns=1_725_000_000_100_000_000,
            ttft=0.2,
            itl=0.01,
            latency=0.3,
            input_length=10,
            output_length=20,
        )
        second = aiperf_record(
            "request-2",
            start_ns=1_725_000_000_200_000_000,
            end_ns=1_725_000_000_300_000_000,
            ttft=0.4,
            itl=0.02,
            latency=0.5,
            input_length=15,
            output_length=25,
        )
        second_json = json.dumps(second)
        path.write_text(json.dumps(first) + "\n" + second_json[:20], encoding="utf-8")

        first_read = self.collector.read_aiperf_records("benchmark-job")
        self.assertEqual([record.request_id for record in first_read], ["request-1"])
        self.assertEqual(first_read[0].ttft_ms, 0.2)
        self.assertEqual(first_read[0].output_tokens, 20)

        with path.open("a", encoding="utf-8") as artifact:
            artifact.write(second_json[20:] + "\n" + json.dumps(first) + "\n")

        second_read = self.collector.read_aiperf_records("benchmark-job")
        self.assertEqual([record.request_id for record in second_read], ["request-2"])
        self.assertEqual(self.collector.read_aiperf_records("benchmark-job"), ())

    def test_aiperf_summary_uses_completed_requests_for_latency_and_slos(self) -> None:
        records = (
            AIPerfRequest(
                request_id="request-1",
                started_at=self.started_at,
                ended_at=self.started_at,
                cancelled_at=None,
                failed=False,
                cancelled=False,
                ttft_ms=50.0,
                tpot_ms=10.0,
                latency_ms=200.0,
                input_length=10,
                output_tokens=20,
            ),
            AIPerfRequest(
                request_id="request-2",
                started_at=self.started_at,
                ended_at=self.started_at,
                cancelled_at=None,
                failed=False,
                cancelled=False,
                ttft_ms=150.0,
                tpot_ms=30.0,
                latency_ms=300.0,
                input_length=20,
                output_tokens=30,
            ),
            AIPerfRequest(
                request_id="request-3",
                started_at=self.started_at,
                ended_at=self.started_at,
                cancelled_at=None,
                failed=True,
                cancelled=False,
                ttft_ms=None,
                tpot_ms=None,
                latency_ms=None,
                input_length=None,
                output_tokens=None,
            ),
            AIPerfRequest(
                request_id="request-4",
                started_at=self.started_at,
                ended_at=None,
                cancelled_at=self.started_at,
                failed=False,
                cancelled=True,
                ttft_ms=None,
                tpot_ms=None,
                latency_ms=None,
                input_length=None,
                output_tokens=None,
            ),
        )

        summary = summarize_aiperf(
            records,
            ttft_target_ms=100.0,
            tpot_target_ms=20.0,
        )

        self.assertEqual(summary.attempted_request_count, 4)
        self.assertEqual(summary.completed_request_count, 2)
        self.assertEqual(summary.failed_request_count, 1)
        self.assertEqual(summary.cancelled_request_count, 1)
        self.assertEqual(summary.p90_ttft_ms, 140.0)
        self.assertEqual(summary.p99_tpot_ms, 29.8)
        self.assertEqual(summary.ttft_pass_count, 1)
        self.assertEqual(summary.tpot_pass_count, 1)
        self.assertEqual(summary.slo_pass_count, 1)
        self.assertEqual(summary.qualified_output_tokens, 20)

    def test_aiperf_records_are_assigned_by_end_time_and_emitted_per_tick(self) -> None:
        self.collector.start_run(self.started_at)
        self.collector.bind_job(
            authored_job_id="benchmark-job",
            store_job_id="job_01J",
            job_kind="online",
            benchmark_job_path=Path("job.yaml"),
        )
        self.collector.collect_store_events(
            FakeEventLog(
                [
                    store_event(
                        "evt-1",
                        "tenant",
                        None,
                        None,
                        "plan.created",
                        {"plan_id": "plan_1"},
                        self.started_at,
                    ),
                    store_event(
                        "evt-2",
                        "tenant",
                        None,
                        None,
                        "plan.created",
                        {"plan_id": "plan_2"},
                        self.started_at + timedelta(minutes=1),
                    ),
                ]
            )
        )
        path = self.collector.job_binding("benchmark-job").aiperf_profile_path
        path.parent.mkdir(parents=True)
        path.write_text(
            "\n".join(
                (
                    json.dumps(
                        aiperf_record(
                            "tick-0",
                            start_ns=epoch_ns(self.started_at + timedelta(seconds=20)),
                            end_ns=epoch_ns(self.started_at + timedelta(seconds=30)),
                            ttft=50,
                            itl=10,
                            latency=100,
                            output_length=20,
                        )
                    ),
                    json.dumps(
                        aiperf_record(
                            "tick-1",
                            start_ns=epoch_ns(self.started_at + timedelta(seconds=50)),
                            end_ns=epoch_ns(self.started_at + timedelta(minutes=1)),
                            ttft=150,
                            itl=30,
                            latency=200,
                            output_length=30,
                        )
                    ),
                )
            )
            + "\n",
            encoding="utf-8",
        )

        self.assertEqual(self.collector.collect_aiperf_records("benchmark-job"), 2)
        tick = self.collector.finalize_closed_ticks(
            self.started_at + timedelta(minutes=2)
        )[0]
        summary = self.collector.emit_online_job_window(
            tick,
            "benchmark-job",
            ttft_target_ms=100,
            tpot_target_ms=20,
        )

        self.assertEqual(summary.completed_request_count, 1)
        self.assertEqual(summary.qualified_output_tokens, 20)
        windows = [
            event
            for event in self.read_events()
            if event["kind"] == "online_job_window"
        ]
        self.assertEqual(windows[0]["tick"], 0)
        self.assertEqual(windows[0]["job_id"], "job_01J")
        self.assertEqual(windows[0]["completed_request_count"], 1)

    def test_gpu_samples_use_store_rows_inside_the_finalized_tick(self) -> None:
        self.collector.start_run(self.started_at)
        self.collector.bind_job(
            authored_job_id="benchmark-job",
            store_job_id="job_01J",
            job_kind="online",
            benchmark_job_path=Path("job.yaml"),
        )
        self.collector.collect_store_events(
            FakeEventLog(
                [
                    store_event(
                        "evt-1",
                        "tenant",
                        None,
                        None,
                        "plan.created",
                        {"plan_id": "plan_1"},
                        self.started_at,
                    ),
                    store_event(
                        "evt-2",
                        "tenant",
                        None,
                        None,
                        "plan.created",
                        {"plan_id": "plan_2"},
                        self.started_at + timedelta(minutes=1),
                    ),
                ]
            )
        )
        inside = GpuMetric(
            metric_id="gpum_inside",
            ts=self.started_at + timedelta(seconds=30),
            job_id="job_01J",
            gpu_uuid="GPU-inside",
            rank_id="rank_01J",
            chain_index=0,
            sm_utilization=0.4,
            throughput_token_per_sec=12.0,
        )
        boundary = GpuMetric(
            metric_id="gpum_boundary",
            ts=self.started_at + timedelta(minutes=1),
            job_id="job_01J",
            gpu_uuid="GPU-boundary",
            rank_id="rank_01J",
            chain_index=0,
        )
        metrics = FakeGpuMetricStore([inside, boundary])
        tick = self.collector.finalize_closed_ticks(
            self.started_at + timedelta(minutes=2)
        )[0]

        self.assertEqual(self.collector.emit_gpu_samples(tick, metrics), 1)
        self.assertEqual(self.collector.emit_gpu_samples(tick, metrics), 0)

        samples = [
            event for event in self.read_events() if event["kind"] == "gpu_sample"
        ]
        self.assertEqual(samples[0]["metric_id"], "gpum_inside")
        self.assertEqual(samples[0]["metrics"]["sm_utilization"], 40.0)
        self.assertEqual(samples[0]["metrics"]["throughput_token_per_sec"], 12.0)
        self.assertEqual(
            metrics.calls,
            [
                (
                    "tenant",
                    "job_01J",
                    self.started_at,
                    self.started_at + timedelta(minutes=1),
                ),
                (
                    "tenant",
                    "job_01J",
                    self.started_at,
                    self.started_at + timedelta(minutes=1),
                ),
            ],
        )

    def test_tick_telemetry_reuses_one_capped_store_read_per_job(self) -> None:
        self.collector.start_run(self.started_at)
        batch_path = self.run_dir / "batch.yaml"
        batch_path.write_text(
            """workload:
  dataset:
    record_count: 1000
    partitioning:
      shard_count: 10
      requests_per_shard: 100
""",
            encoding="utf-8",
        )
        self.collector.bind_job(
            authored_job_id="batch-job",
            store_job_id="job_batch",
            job_kind="batch",
            benchmark_job_path=batch_path,
        )
        self.collector.collect_store_events(
            FakeEventLog(
                [
                    store_event(
                        "evt-1",
                        "tenant",
                        None,
                        None,
                        "plan.created",
                        {"plan_id": "plan_1"},
                        self.started_at,
                    ),
                    store_event(
                        "evt-2",
                        "tenant",
                        None,
                        None,
                        "plan.created",
                        {"plan_id": "plan_2"},
                        self.started_at + timedelta(minutes=6),
                    ),
                ]
            )
        )
        metrics = FakeGpuMetricStore(
            [
                GpuMetric(
                    metric_id="before_window",
                    ts=self.started_at + timedelta(seconds=30),
                    job_id="job_batch",
                    gpu_uuid="GPU-before",
                    rank_id="rank_01J",
                    chain_index=0,
                ),
                GpuMetric(
                    metric_id="inside_window",
                    ts=self.started_at + timedelta(minutes=2),
                    job_id="job_batch",
                    gpu_uuid="GPU-inside",
                    rank_id="rank_01J",
                    chain_index=0,
                    throughput_token_per_sec=10,
                    batched_reqs_processed_total=100,
                ),
            ]
        )
        tick = self.collector.finalize_closed_ticks(
            self.started_at + timedelta(minutes=7)
        )[0]

        self.collector.emit_tick_telemetry(tick, metrics)

        window_start = self.started_at + timedelta(minutes=1)
        self.assertEqual(
            metrics.calls,
            [
                (
                    "tenant",
                    "job_batch",
                    window_start,
                    self.started_at + timedelta(minutes=6),
                )
            ],
        )
        samples = [
            event for event in self.read_events() if event["kind"] == "gpu_sample"
        ]
        self.assertEqual([event["metric_id"] for event in samples], ["inside_window"])
        rank_window = next(
            event for event in self.read_events() if event["kind"] == "rank_window"
        )
        batch_window = next(
            event for event in self.read_events() if event["kind"] == "batch_job_window"
        )
        self.assertEqual(rank_window["window_start"], window_start.isoformat())
        self.assertEqual(batch_window["window_start"], window_start.isoformat())

    def test_rank_windows_collapse_gpu_rows_before_aggregating_chains(self) -> None:
        self.collector.start_run(self.started_at)
        self.collector.bind_job(
            authored_job_id="benchmark-job",
            store_job_id="job_01J",
            job_kind="online",
            benchmark_job_path=Path("job.yaml"),
        )
        self.collector.collect_store_events(
            FakeEventLog(
                [
                    store_event(
                        "evt-1",
                        "tenant",
                        None,
                        None,
                        "plan.created",
                        {"plan_id": "plan_1"},
                        self.started_at,
                    ),
                    store_event(
                        "evt-2",
                        "tenant",
                        None,
                        None,
                        "plan.created",
                        {"plan_id": "plan_2"},
                        self.started_at + timedelta(minutes=1),
                    ),
                ]
            )
        )
        metrics = FakeGpuMetricStore(
            [
                GpuMetric(
                    metric_id="gpum_chain_0_a",
                    ts=self.started_at + timedelta(seconds=2),
                    job_id="job_01J",
                    gpu_uuid="GPU-0a",
                    rank_id="rank_01J",
                    chain_index=0,
                    sm_utilization=0.4,
                    throughput_token_per_sec=10,
                    p99_ttft_ms=100,
                    cost_per_token=0.1,
                ),
                GpuMetric(
                    metric_id="gpum_chain_0_b",
                    ts=self.started_at + timedelta(seconds=4),
                    job_id="job_01J",
                    gpu_uuid="GPU-0b",
                    rank_id="rank_01J",
                    chain_index=0,
                    sm_utilization=0.6,
                    throughput_token_per_sec=10,
                    p99_ttft_ms=100,
                    cost_per_token=0.1,
                ),
                GpuMetric(
                    metric_id="gpum_chain_1",
                    ts=self.started_at + timedelta(seconds=5),
                    job_id="job_01J",
                    gpu_uuid="GPU-1",
                    rank_id="rank_01J",
                    chain_index=1,
                    sm_utilization=0.8,
                    throughput_token_per_sec=20,
                    p99_ttft_ms=200,
                    cost_per_token=0.2,
                ),
            ]
        )
        tick = self.collector.finalize_closed_ticks(
            self.started_at + timedelta(minutes=2)
        )[0]

        self.assertEqual(self.collector.emit_rank_windows(tick, metrics), 1)

        window = next(
            event for event in self.read_events() if event["kind"] == "rank_window"
        )
        self.assertEqual(window["chain_indices"], [0, 1])
        self.assertEqual(window["metrics"]["throughput_token_per_sec"], [30.0])
        self.assertEqual(window["metrics"]["p99_ttft_ms"], [200.0])
        self.assertEqual(window["metrics"]["sm_utilization"], [65.0])
        self.assertAlmostEqual(window["metrics"]["cost_per_token"][0], 1 / 6)

    def test_batch_window_uses_only_batch_driver_metrics(self) -> None:
        self.collector.start_run(self.started_at)
        batch_path = self.run_dir / "batch.yaml"
        batch_path.write_text(
            """workload:
  dataset:
    record_count: 1000
    partitioning:
      shard_count: 10
      requests_per_shard: 100
""",
            encoding="utf-8",
        )
        self.collector.bind_job(
            authored_job_id="batch-job",
            store_job_id="job_batch",
            job_kind="batch",
            benchmark_job_path=batch_path,
        )
        self.collector.collect_store_events(
            FakeEventLog(
                [
                    store_event(
                        "evt-1",
                        "tenant",
                        None,
                        None,
                        "plan.created",
                        {"plan_id": "plan_1"},
                        self.started_at,
                    ),
                    store_event(
                        "evt-2",
                        "tenant",
                        None,
                        None,
                        "plan.created",
                        {"plan_id": "plan_2"},
                        self.started_at + timedelta(minutes=1),
                    ),
                ]
            )
        )
        metrics = FakeGpuMetricStore(
            [
                GpuMetric(
                    metric_id="gpu_chain_0_a",
                    ts=self.started_at + timedelta(seconds=2),
                    job_id="job_batch",
                    gpu_uuid="GPU-0a",
                    rank_id="rank_01J",
                    chain_index=0,
                    batched_reqs_inflight=3,
                    batched_reqs_processed_total=100,
                    batched_chunks_input_pulled_total=4,
                    batched_chunks_output_written_total=3,
                ),
                GpuMetric(
                    metric_id="gpu_chain_0_b",
                    ts=self.started_at + timedelta(seconds=4),
                    job_id="job_batch",
                    gpu_uuid="GPU-0b",
                    rank_id="rank_01J",
                    chain_index=0,
                    batched_reqs_inflight=3,
                    batched_reqs_processed_total=100,
                    batched_chunks_input_pulled_total=4,
                    batched_chunks_output_written_total=3,
                ),
                GpuMetric(
                    metric_id="gpu_chain_1",
                    ts=self.started_at + timedelta(seconds=5),
                    job_id="job_batch",
                    gpu_uuid="GPU-1",
                    rank_id="rank_01J",
                    chain_index=1,
                    batched_reqs_inflight=2,
                    batched_reqs_processed_total=50,
                    batched_chunks_input_pulled_total=2,
                    batched_chunks_output_written_total=1,
                ),
                GpuMetric(
                    metric_id="gpu_chain_0_later",
                    ts=self.started_at + timedelta(seconds=15),
                    job_id="job_batch",
                    gpu_uuid="GPU-0a",
                    rank_id="rank_01J",
                    chain_index=0,
                    batched_reqs_inflight=4,
                    batched_reqs_processed_total=140,
                    batched_chunks_input_pulled_total=5,
                    batched_chunks_output_written_total=4,
                ),
                GpuMetric(
                    metric_id="gpu_chain_0_reset",
                    ts=self.started_at + timedelta(seconds=25),
                    job_id="job_batch",
                    gpu_uuid="GPU-0a",
                    rank_id="rank_01J",
                    chain_index=0,
                    batched_reqs_inflight=2,
                    batched_reqs_processed_total=10,
                    batched_chunks_input_pulled_total=1,
                    batched_chunks_output_written_total=0,
                ),
                GpuMetric(
                    metric_id="gpu_chain_0_final",
                    ts=self.started_at + timedelta(seconds=35),
                    job_id="job_batch",
                    gpu_uuid="GPU-0a",
                    rank_id="rank_01J",
                    chain_index=0,
                    batched_reqs_inflight=1,
                    batched_reqs_processed_total=40,
                    batched_chunks_input_pulled_total=2,
                    batched_chunks_output_written_total=1,
                ),
            ]
        )
        tick = self.collector.finalize_closed_ticks(
            self.started_at + timedelta(minutes=2)
        )[0]

        self.assertEqual(self.collector.emit_batch_job_window(tick, metrics), 1)

        window = next(
            event for event in self.read_events() if event["kind"] == "batch_job_window"
        )
        self.assertEqual(window["requests_inflight"], 3.0)
        self.assertEqual(window["requests_processed_total"], 230.0)
        self.assertEqual(window["requests_processed_in_tick"], 230.0)
        self.assertEqual(window["request_completion_percent"], 23.0)
        self.assertEqual(window["pending_requests"], 767.0)
        self.assertEqual(window["chunks_input_pulled_total"], 9.0)
        self.assertEqual(window["chunks_output_written_total"], 6.0)
        self.assertEqual(window["chunk_completion_percent"], 60.0)
        self.assertEqual(
            window["rank_metrics"]["rank_01J"]["requests_processed_total"],
            230.0,
        )

    def test_batch_rank_totals_survive_a_tick_without_fresh_samples(self) -> None:
        definition = (1000, 10, 100)
        first = self.collector._batch_window(
            "job_batch",
            {
                ("job_batch", "rank_a", 0): [
                    (
                        self.started_at,
                        {
                            "batched_reqs_inflight": 2.0,
                            "batched_reqs_processed_total": 100.0,
                        },
                    )
                ],
                ("job_batch", "rank_b", 0): [
                    (
                        self.started_at,
                        {
                            "batched_reqs_inflight": 1.0,
                            "batched_reqs_processed_total": 50.0,
                        },
                    )
                ],
            },
            definition,
        )
        second = self.collector._batch_window(
            "job_batch",
            {
                ("job_batch", "rank_a", 0): [
                    (
                        self.started_at + timedelta(minutes=1),
                        {
                            "batched_reqs_inflight": 1.0,
                            "batched_reqs_processed_total": 120.0,
                        },
                    )
                ]
            },
            definition,
        )

        self.assertEqual(first["requests_processed_total"], 150.0)
        self.assertEqual(second["requests_processed_total"], 170.0)
        self.assertEqual(
            second["rank_metrics"]["rank_b"]["requests_processed_total"],
            50.0,
        )
        self.assertNotIn("requests_inflight", second["rank_metrics"]["rank_b"])

    def test_run_summary_uses_rank_running_for_admission(self) -> None:
        self.collector.start_run(self.started_at)
        self.collector.bind_job(
            authored_job_id="online-job",
            store_job_id="job_online",
            job_kind="online",
            benchmark_job_path=Path("online.yaml"),
        )
        self.collector.bind_job(
            authored_job_id="batch-job",
            store_job_id="job_batch",
            job_kind="batch",
            benchmark_job_path=Path("batch.yaml"),
        )
        self.collector.collect_store_events(
            FakeEventLog(
                [
                    store_event(
                        "evt-1",
                        "tenant",
                        "job_online",
                        "rank_online",
                        "rank.running",
                        {},
                        self.started_at,
                    ),
                    store_event(
                        "evt-2",
                        "tenant",
                        "job_batch",
                        "rank_batch",
                        "rank.running",
                        {},
                        self.started_at,
                    ),
                ]
            )
        )

        self.collector.emit_run_summary(duration_seconds=12_600)

        summary = next(
            event for event in self.read_events() if event["kind"] == "run_summary"
        )
        self.assertEqual(summary["online_admission_percent"], 100.0)
        self.assertEqual(summary["batch_admission_percent"], 100.0)
        self.assertIsNone(summary["batch_goodput_token_per_sec"])
        self.assertEqual(
            summary["batch_goodput_status"],
            "requires_output_token_counter",
        )

    def test_store_snapshots_include_only_bound_jobs_and_changed_resources(
        self,
    ) -> None:
        self.collector.start_run(self.started_at)
        self.collector.bind_job(
            authored_job_id="benchmark-job",
            store_job_id="job_01J",
            job_kind="online",
            benchmark_job_path=Path("job.yaml"),
        )
        bound = Job(job_id="job_01J", user_id="tenant", kind=JobKind.ONLINE)
        unbound = Job(job_id="job_other", user_id="tenant", kind=JobKind.BATCH)
        rank = Rank(
            rank_id="rank_01J",
            job_id=bound.job_id,
            role=RankRole.AGGREGATE,
            n_replicas=1,
        )
        jobs = FakeJobStore([bound, unbound], {bound.job_id: [rank]})
        resources = FakeResourceMapStore(ResourceMap(version=1))
        self.collector.collect_store_snapshots(
            jobs=jobs,
            resources=resources,
            observed_at=self.started_at,
        )
        self.collector.collect_store_snapshots(
            jobs=jobs,
            resources=resources,
            observed_at=self.started_at + timedelta(minutes=1),
        )
        rank.status = RankStatus.RUNNING
        rank.updated_at = self.started_at + timedelta(minutes=2)
        resources.resource_map = ResourceMap(version=2)
        self.collector.collect_store_snapshots(
            jobs=jobs,
            resources=resources,
            observed_at=self.started_at + timedelta(minutes=2),
        )

        events = self.read_events()
        self.assertEqual(
            [event["kind"] for event in events],
            [
                "run_started",
                "job_snapshot",
                "rank_snapshot",
                "resource_snapshot",
                "job_snapshot",
                "job_snapshot",
                "rank_snapshot",
                "resource_snapshot",
            ],
        )
        self.assertEqual(jobs.user_ids, ["tenant", "tenant", "tenant"])
        self.assertEqual(events[1]["job"]["job_id"], "job_01J")
        self.assertEqual(events[2]["rank"]["status"], "launching")
        self.assertEqual(events[6]["rank"]["status"], "running")
        self.assertEqual(events[7]["resource_map"]["version"], 2)

    def test_evidence_snapshots_emit_only_changed_bound_rows(self) -> None:
        collector = RealTraceCollector(
            run_dir=self.run_dir / "evidence",
            run_id="run-evidence",
            benchmark_id="15min_real",
            method="koi",
            user_id="tenant",
            collect_evidence=True,
        )
        collector.start_run(self.started_at)
        collector.bind_job(
            authored_job_id="benchmark-job",
            store_job_id="job_01J",
            job_kind="online",
            benchmark_job_path=Path("job.yaml"),
        )
        jobs = FakeJobStore([], {})
        resources = FakeResourceMapStore(ResourceMap(version=1))
        row = evidence_row()
        evidence = FakeEvidenceStore([row, evidence_row("job_other")])

        collector.collect_store_snapshots(
            jobs=jobs,
            resources=resources,
            evidence=evidence,
            observed_at=self.started_at,
        )
        collector.collect_store_snapshots(
            jobs=jobs,
            resources=resources,
            evidence=evidence,
            observed_at=self.started_at + timedelta(minutes=1),
        )
        row.y_predicted["throughput_token_per_sec"] = 20.0
        collector.collect_store_snapshots(
            jobs=jobs,
            resources=resources,
            evidence=evidence,
            observed_at=self.started_at + timedelta(minutes=2),
        )

        events = self.read_events(self.run_dir / "evidence")
        snapshots = [event for event in events if event["kind"] == "evidence_snapshot"]
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0]["evidence"]["job_id"], "job_01J")
        self.assertEqual(
            snapshots[1]["evidence"]["y_predicted"]["throughput_token_per_sec"],
            20.0,
        )

    def test_causal_state_is_read_only_when_enabled(self) -> None:
        jobs = FakeJobStore([], {})
        resources = FakeResourceMapStore(ResourceMap(version=1))
        causal = FakeCausalGraphStore()
        self.collector.start_run(self.started_at)

        self.collector.collect_store_snapshots(
            jobs=jobs,
            resources=resources,
            causal=causal,
            observed_at=self.started_at,
        )

        self.assertEqual(causal.calls, 0)

        enabled = RealTraceCollector(
            run_dir=self.run_dir / "causal",
            run_id="run-causal",
            benchmark_id="15min_real",
            method="koi",
            user_id="tenant",
            collect_causal_state=True,
        )
        enabled.start_run(self.started_at)
        enabled.collect_store_snapshots(
            jobs=jobs,
            resources=resources,
            causal=causal,
            observed_at=self.started_at,
        )

        self.assertEqual(causal.calls, 3)
        causal_events = [
            event
            for event in self.read_events(self.run_dir / "causal")
            if event["kind"] == "causal_state_snapshot"
        ]
        self.assertEqual(
            causal_events[0]["nodes"]["throughput"]["node_id"],
            "throughput",
        )

    def read_events(self, run_dir: Path | None = None) -> list[dict[str, object]]:
        path = (run_dir or self.run_dir) / "trace.jsonl"
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]


def epoch_ns(timestamp: datetime) -> int:
    """Convert one UTC test timestamp to AIPerf's epoch-nanosecond format."""
    return int(timestamp.timestamp() * 1_000_000_000)


if __name__ == "__main__":
    unittest.main()
