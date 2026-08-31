"""Write timestamped, real-hardware benchmark traces beside run artifacts.

This module deliberately observes experiments without controlling them.
``real_scenario`` owns workload processes; the collector owns only the
``trace.jsonl`` file and the in-memory state needed to read it consistently.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from tandemn_system_data.clients import (
    CausalGraphStore,
    EvidenceStore,
    GpuMetricStore,
    JobStore,
    ResourceMapStore,
)
from tandemn_system_data.clients.event_log import PostgresEventLog
from tandemn_system_data.models.event import Event
from tandemn_system_data.models.gpu_metric import GpuMetric, gpu_metric_to_metrics
from tandemn_system_data.models.rank import Rank

from data_generation.trace_telemetry import rank_trajectories

DEFAULT_COLLECTION_INTERVAL_SECONDS = 60
DEFAULT_TICK_HOLDBACK_SECONDS = 60
LIFECYCLE_EVENT_TYPES = frozenset(
    {
        "job.submitted",
        "job.place",
        "job.placed",
        "job.resumed",
        "job.paused",
        "job.finished",
        "rank.launching",
        "rank.running",
        "rank.failed",
        "rank.stopped",
    }
)


@dataclass(frozen=True)
class JobBinding:
    """
    Connect one authored benchmark job to its canonical Store job.

    Benchmark files name jobs with stable authored IDs while Tandemn Store uses
    generated IDs. Both are required: the authored ID locates local artifacts
    and the Store ID identifies telemetry and lifecycle records.

    authored_job_id
    = ID in benchmark YAML
    = artifact folder name

    store_job_id
    = generated canonical Store ID
    = ID used by ranks, GPU telemetry, and Orca events

    Example:
    benchmark YAML:
    job-benchmark_timing_15min_real-0-1

    local AIPerf artifact:
    runs/<run>/job-benchmark_timing_15min_real-0-1/artifacts/profile_export.jsonl

    Store:
    job_01J...
    
    bind_job() stores that relationship once, so traces.py knows:
    read AIPerf from the authored-job directory
    read Store events/GPU metrics/ranks using the Store job ID
    write both identities into trace records where needed
    """

    authored_job_id: str
    store_job_id: str
    job_kind: str
    benchmark_job_path: Path
    aiperf_profile_path: Path


@dataclass(frozen=True)
class TraceTick:
    """One plan-created trace window awaiting final telemetry aggregation."""

    tick: int
    plan_id: str
    started_at: datetime
    ended_at: datetime | None = None
    finalizes_at: datetime | None = None

    def contains(self, timestamp: datetime) -> bool:
        """Return whether a source timestamp belongs to this half-open window."""
        return self.started_at <= timestamp and (
            self.ended_at is None or timestamp < self.ended_at
        )


@dataclass(frozen=True)
class AIPerfRequest:
    """One terminal AIPerf request record normalized for trace aggregation."""

    request_id: str
    started_at: datetime | None
    ended_at: datetime | None
    cancelled_at: datetime | None
    failed: bool
    cancelled: bool
    ttft_ms: float | None
    tpot_ms: float | None
    latency_ms: float | None
    input_length: int | None
    output_tokens: int | None

    @property
    def completed(self) -> bool:
        """Return whether AIPerf recorded a successful terminal response."""
        return self.ended_at is not None and not self.failed and not self.cancelled


@dataclass(frozen=True)
class AIPerfSummary:
    """Client-observed request outcomes and latency distribution for one window."""

    attempted_request_count: int
    completed_request_count: int
    failed_request_count: int
    cancelled_request_count: int
    mean_ttft_ms: float | None
    p90_ttft_ms: float | None
    p99_ttft_ms: float | None
    mean_tpot_ms: float | None
    p90_tpot_ms: float | None
    p99_tpot_ms: float | None
    mean_latency_ms: float | None
    p90_latency_ms: float | None
    p99_latency_ms: float | None
    output_tokens: int
    ttft_pass_count: int
    tpot_pass_count: int
    slo_pass_count: int
    qualified_output_tokens: int


def summarize_aiperf(
    records: tuple[AIPerfRequest, ...],
    *,
    ttft_target_ms: float | None,
    tpot_target_ms: float | None,
) -> AIPerfSummary:
    """Summarize completed AIPerf records using AIPerf's linear percentiles.

    Failed and cancelled requests remain visible in their own counts but never
    contribute latency distributions or SLO-qualified output tokens. AIPerf
    calls its per-output-token metric ``inter_token_latency``; this is the
    TPOT metric used by the benchmark and trace.
    """
    completed = tuple(record for record in records if record.completed)
    ttft = tuple(record.ttft_ms for record in completed if record.ttft_ms is not None)
    tpot = tuple(record.tpot_ms for record in completed if record.tpot_ms is not None)
    latency = tuple(
        record.latency_ms for record in completed if record.latency_ms is not None
    )
    ttft_pass = 0
    tpot_pass = 0
    slo_pass = 0
    qualified_tokens = 0
    for record in completed:
        passed_ttft = (
            ttft_target_ms is not None
            and record.ttft_ms is not None
            and record.ttft_ms <= ttft_target_ms
        )
        passed_tpot = (
            tpot_target_ms is not None
            and record.tpot_ms is not None
            and record.tpot_ms <= tpot_target_ms
        )
        ttft_pass += int(passed_ttft)
        tpot_pass += int(passed_tpot)
        if passed_ttft and passed_tpot:
            slo_pass += 1
            qualified_tokens += record.output_tokens or 0
    return AIPerfSummary(
        attempted_request_count=sum(
            record.started_at is not None for record in records
        ),
        completed_request_count=len(completed),
        failed_request_count=sum(record.failed for record in records),
        cancelled_request_count=sum(record.cancelled for record in records),
        mean_ttft_ms=_mean(ttft),
        p90_ttft_ms=_percentile(ttft, 0.9),
        p99_ttft_ms=_percentile(ttft, 0.99),
        mean_tpot_ms=_mean(tpot),
        p90_tpot_ms=_percentile(tpot, 0.9),
        p99_tpot_ms=_percentile(tpot, 0.99),
        mean_latency_ms=_mean(latency),
        p90_latency_ms=_percentile(latency, 0.9),
        p99_latency_ms=_percentile(latency, 0.99),
        output_tokens=sum(record.output_tokens or 0 for record in completed),
        ttft_pass_count=ttft_pass,
        tpot_pass_count=tpot_pass,
        slo_pass_count=slo_pass,
        qualified_output_tokens=qualified_tokens,
    )


class RealTraceCollector:
    """Manage a trace file and fixed collection schedule for one benchmark run.

    This first phase does not query Store or AIPerf. It establishes the stable
    run contract those readers will use: one run file, one benchmark user,
    explicit authored-to-Store job bindings, and a fixed one-minute cadence.
    No method accepts a process handle, so this class cannot manage AIPerf.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        run_id: str,
        benchmark_id: str,
        method: str,
        user_id: str,
        collection_interval_seconds: int = DEFAULT_COLLECTION_INTERVAL_SECONDS,
        tick_holdback_seconds: int = DEFAULT_TICK_HOLDBACK_SECONDS,
        collect_evidence: bool = False,
        collect_causal_state: bool = False,
    ) -> None:
        """Configure one collector before a benchmark run starts.

        ``collection_interval_seconds`` controls only observation latency. Later
        phases will assign Store events and metrics by their own timestamps,
        never by this polling schedule.
        """
        if collection_interval_seconds <= 0 or tick_holdback_seconds < 0:
            raise ValueError(
                "collection interval must be positive and holdback nonnegative"
            )
        self.run_dir = run_dir
        self.run_id = run_id
        self.benchmark_id = benchmark_id
        self.method = method
        self.user_id = user_id
        self.collection_interval_seconds = collection_interval_seconds
        self.tick_holdback_seconds = tick_holdback_seconds
        self.collect_evidence = collect_evidence
        self.collect_causal_state = collect_causal_state
        self._bindings: dict[str, JobBinding] = {}
        self._next_collection_at: datetime | None = None
        self._open_tick: TraceTick | None = None
        self._closed_ticks: list[TraceTick] = []
        self._next_tick = 0
        self._rank_updated_at: dict[str, datetime] = {}
        self._resource_version: int | None = None
        self._evidence_digests: dict[str, str] = {}
        self._aiperf_offsets: dict[str, int] = {}
        self._aiperf_tails: dict[str, bytes] = {}
        self._aiperf_request_ids: dict[str, set[str]] = {}
        self._aiperf_records_by_window: dict[tuple[int, str], list[AIPerfRequest]] = {}
        self._gpu_metric_ids: set[str] = set()

    def start_run(self, started_at: datetime) -> None:
        """Create the trace file and record immutable run collection settings."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._append(
            {
                "kind": "run_started",
                "trace_version": 1,
                "run_id": self.run_id,
                "benchmark": self.benchmark_id,
                "method": self.method,
                "user_id": self.user_id,
                "started_at": started_at.isoformat(),
                "collection_interval_seconds": self.collection_interval_seconds,
                "tick_holdback_seconds": self.tick_holdback_seconds,
                "tick_boundary_source": "plan.created",
                "evidence_collection": self.collect_evidence,
                "causal_state_collection": self.collect_causal_state,
            }
        )
        self._next_collection_at = started_at + timedelta(
            seconds=self.collection_interval_seconds
        )

    def bind_job(
        self,
        *,
        authored_job_id: str,
        store_job_id: str,
        job_kind: str,
        benchmark_job_path: Path,
    ) -> None:
        """Register the immutable identity and artifact location for one job."""
        if authored_job_id in self._bindings:
            raise ValueError(f"authored job {authored_job_id!r} is already bound")
        self._bindings[authored_job_id] = JobBinding(
            authored_job_id=authored_job_id,
            store_job_id=store_job_id,
            job_kind=job_kind,
            benchmark_job_path=benchmark_job_path,
            aiperf_profile_path=(
                self.run_dir
                / authored_job_id
                / "artifacts"
                / "profile_export.jsonl"
            ),
        )

    def job_binding(self, authored_job_id: str) -> JobBinding:
        """Return the binding needed by later Store and artifact readers."""
        return self._bindings[authored_job_id]

    def collect(self, observed_at: datetime) -> bool:
        """Return whether a minute-level collection pass is due at this time.

        Store and AIPerf readers are added in later phases. Keeping the cadence
        gate here ensures all readers share one stable schedule.
        """
        assert self._next_collection_at is not None
        if observed_at < self._next_collection_at:
            return False
        interval = timedelta(seconds=self.collection_interval_seconds)
        while self._next_collection_at <= observed_at:
            self._next_collection_at += interval
        return True

    def collect_store_events(
        self, event_log: PostgresEventLog, *, limit: int = 100
    ) -> int:
        """Project one durable EventLog batch and acknowledge each consumed event.

        The caller controls when this method runs through ``collect``. Events
        retain Store's ``created_at`` timestamp, so minute-level polling delays
        visibility without changing trace timing. Ignored events are still
        acknowledged because the Store cursor is shared across all event types.
        """
        consumer_name = f"real-trace:{self.run_id}"
        events = event_log.read_for_consumer(consumer_name, limit=limit)
        for event in events:
            trace_event = self._trace_event(event)
            if trace_event is not None:
                self._append(trace_event)
            event_log.ack(consumer_name, event.event_id)
        return len(events)

    def collect_store_snapshots(
        self,
        *,
        jobs: JobStore,
        resources: ResourceMapStore,
        observed_at: datetime,
        evidence: EvidenceStore | None = None,
        causal: CausalGraphStore | None = None,
    ) -> None:
        """Capture Store state needed to explain a real benchmark tick.

        Job rows are emitted each collection because their status is a useful
        reconciliation point. Rank, ResourceMap, and Evidence records are only
        emitted when their Store content changes. Evidence and causal state are
        opt-in because non-Koi baselines must not query Koi-owned state.
        """
        self._snapshot_jobs(jobs, observed_at)
        self._snapshot_resource_map(resources, observed_at)
        if self.collect_evidence:
            if evidence is None:
                raise ValueError(
                    "evidence store is required when collection is enabled"
                )
            self._snapshot_evidence(evidence, observed_at)
        if self.collect_causal_state:
            if causal is None:
                raise ValueError("causal store is required when collection is enabled")
            self._snapshot_causal_state(causal, observed_at)

    def close_open_tick(self, ended_at: datetime) -> None:
        """Close the final plan window when the benchmark run ends."""
        if self._open_tick is None:
            return
        self._closed_ticks.append(
            TraceTick(
                tick=self._open_tick.tick,
                plan_id=self._open_tick.plan_id,
                started_at=self._open_tick.started_at,
                ended_at=ended_at,
                finalizes_at=ended_at + timedelta(seconds=self.tick_holdback_seconds),
            )
        )
        self._open_tick = None

    def tick_for(self, timestamp: datetime) -> int | None:
        """Return an unfinalized tick accepting this source timestamp."""
        for tick in self._closed_ticks:
            if tick.contains(timestamp):
                return tick.tick
        if self._open_tick is not None and self._open_tick.contains(timestamp):
            return self._open_tick.tick
        return None

    def finalize_closed_ticks(self, observed_at: datetime) -> tuple[TraceTick, ...]:
        """Permanently release windows whose fixed holdback has elapsed."""
        finalized = tuple(
            tick
            for tick in self._closed_ticks
            if tick.finalizes_at is not None and tick.finalizes_at <= observed_at
        )
        self._closed_ticks = [
            tick for tick in self._closed_ticks if tick not in finalized
        ]
        return finalized

    def read_aiperf_records(self, authored_job_id: str) -> tuple[AIPerfRequest, ...]:
        """Read newly flushed complete AIPerf JSONL records for one bound job.

        AIPerf writes records asynchronously, so an incomplete final line stays
        buffered until a later read. The byte offset advances to end-of-file;
        the retained fragment is prepended to the next append. Request IDs make
        repeated exported records harmless.
        """
        path = self.job_binding(authored_job_id).aiperf_profile_path
        if not path.exists():
            return ()
        offset = self._aiperf_offsets.get(authored_job_id, 0)
        with path.open("rb") as artifact:
            artifact.seek(offset)
            appended = artifact.read()
        self._aiperf_offsets[authored_job_id] = offset + len(appended)
        lines = (self._aiperf_tails.get(authored_job_id, b"") + appended).split(b"\n")
        self._aiperf_tails[authored_job_id] = lines.pop()
        seen_request_ids = self._aiperf_request_ids.setdefault(authored_job_id, set())
        records = []
        for line in lines:
            record = self._aiperf_request(line)
            if record is None or record.request_id in seen_request_ids:
                continue
            seen_request_ids.add(record.request_id)
            records.append(record)
        return tuple(records)

    def collect_aiperf_records(self, authored_job_id: str) -> int:
        """Assign newly read AIPerf terminal records to their open tick."""
        assigned = 0
        for record in self.read_aiperf_records(authored_job_id):
            timestamp = record.ended_at or record.cancelled_at
            tick = self.tick_for(timestamp) if timestamp is not None else None
            if tick is None:
                continue
            self._aiperf_records_by_window.setdefault(
                (tick, authored_job_id), []
            ).append(record)
            assigned += 1
        return assigned

    def emit_online_job_window(
        self,
        tick: TraceTick,
        authored_job_id: str,
        *,
        ttft_target_ms: float | None,
        tpot_target_ms: float | None,
    ) -> AIPerfSummary:
        """Write one finalized client-outcome summary for an online job tick."""
        binding = self.job_binding(authored_job_id)
        if binding.job_kind != "online":
            raise ValueError("AIPerf windows require an online job")
        records = tuple(
            self._aiperf_records_by_window.pop((tick.tick, authored_job_id), [])
        )
        summary = summarize_aiperf(
            records,
            ttft_target_ms=ttft_target_ms,
            tpot_target_ms=tpot_target_ms,
        )
        self._append(
            {
                "kind": "online_job_window",
                "tick": tick.tick,
                "job_id": binding.store_job_id,
                "authored_job_id": authored_job_id,
                "window_start": tick.started_at.isoformat(),
                "window_end": tick.ended_at.isoformat() if tick.ended_at else None,
                **asdict(summary),
            }
        )
        return summary

    def emit_gpu_samples(self, tick: TraceTick, metrics: GpuMetricStore) -> int:
        """Write raw Store GPU samples that fall inside one finalized tick."""
        if tick.ended_at is None:
            raise ValueError("GPU samples require a closed tick")
        emitted = 0
        for binding in self._bindings.values():
            rows = metrics.rows_for_job_window(
                self.user_id,
                binding.store_job_id,
                tick.started_at,
                tick.ended_at,
            )
            for metric in rows:
                if (
                    metric.metric_id in self._gpu_metric_ids
                    or metric.job_id != binding.store_job_id
                    or not tick.contains(metric.ts)
                ):
                    continue
                self._gpu_metric_ids.add(metric.metric_id)
                self._append(self._gpu_sample(tick, metric))
                emitted += 1
        return emitted

    def emit_rank_windows(self, tick: TraceTick, metrics: GpuMetricStore) -> int:
        """Aggregate finalized Store GPU rows into ten-second rank trajectories."""
        if tick.ended_at is None:
            raise ValueError("rank windows require a closed tick")
        rows = [
            metric
            for binding in self._bindings.values()
            for metric in metrics.rows_for_job_window(
                self.user_id,
                binding.store_job_id,
                tick.started_at,
                tick.ended_at,
            )
            if metric.job_id == binding.store_job_id and tick.contains(metric.ts)
        ]
        emitted = 0
        for (job_id, rank_id), trajectory in rank_trajectories(rows).items():
            self._append(
                {
                    "kind": "rank_window",
                    "tick": tick.tick,
                    "job_id": job_id,
                    "rank_id": rank_id,
                    "chain_indices": trajectory["chain_indices"],
                    "window_start": tick.started_at.isoformat(),
                    "window_end": tick.ended_at.isoformat(),
                    "sample_bucket_starts": trajectory["sample_bucket_starts"],
                    "metrics": trajectory["metrics"],
                    "telemetry_scope": "real_observed_window",
                }
            )
            emitted += 1
        return emitted

    def close_run(self, finished_at: datetime, *, status: str) -> None:
        """Write the terminal run record after scenario-owned shutdown completes."""
        self._append(
            {
                "kind": "run_finished",
                "run_id": self.run_id,
                "finished_at": finished_at.isoformat(),
                "status": status,
            }
        )

    def _append(self, event: dict[str, object]) -> None:
        with (self.run_dir / "trace.jsonl").open("a", encoding="utf-8") as trace:
            trace.write(self._json(event) + "\n")

    def _trace_event(self, event: Event) -> dict[str, object] | None:
        if event.user_id != self.user_id:
            return None
        if event.type == "plan.applied":
            return None
        if event.type == "plan.created":
            return self._plan_created_event(event)
        if (
            event.type not in LIFECYCLE_EVENT_TYPES
            or not self._is_bound_job(event.job_id)
        ):
            return None
        return {
            "kind": event.type,
            "tick": self._tick_number(),
            "source_event_id": event.event_id,
            "created_at": event.created_at.isoformat(),
            "job_id": event.job_id,
            "rank_id": event.rank_id,
            "payload": event.payload_json,
        }

    def _plan_created_event(self, event: Event) -> dict[str, object]:
        self.close_open_tick(event.created_at)
        self._open_tick = TraceTick(
            tick=self._next_tick,
            plan_id=str(event.payload_json["plan_id"]),
            started_at=event.created_at,
        )
        self._next_tick += 1
        return {
            "kind": "plan_created",
            "tick": self._open_tick.tick,
            "source_event_id": event.event_id,
            "plan_id": event.payload_json["plan_id"],
            "created_at": event.created_at.isoformat(),
        }

    def _is_bound_job(self, job_id: str | None) -> bool:
        return job_id in {binding.store_job_id for binding in self._bindings.values()}

    def _tick_number(self) -> int | None:
        return self._open_tick.tick if self._open_tick is not None else None

    def _aiperf_request(self, line: bytes) -> AIPerfRequest | None:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            return None
        metadata = raw.get("metadata")
        metrics = raw.get("metrics")
        if not isinstance(metadata, dict) or not isinstance(metrics, dict):
            return None
        request_id = metadata.get("x_request_id")
        if not isinstance(request_id, str) or not request_id:
            return None
        cancelled = metadata.get("was_cancelled") is True
        return AIPerfRequest(
            request_id=request_id,
            started_at=_timestamp(metadata.get("request_start_ns")),
            ended_at=_timestamp(metadata.get("request_end_ns")),
            cancelled_at=_timestamp(metadata.get("cancellation_time_ns")),
            failed=bool(raw.get("error")),
            cancelled=cancelled,
            ttft_ms=_milliseconds(metrics, "time_to_first_token"),
            tpot_ms=_milliseconds(metrics, "inter_token_latency"),
            latency_ms=_milliseconds(metrics, "request_latency"),
            input_length=_whole_number(metrics, "input_sequence_length"),
            output_tokens=_whole_number(metrics, "output_sequence_length"),
        )

    def _gpu_sample(self, tick: TraceTick, metric: GpuMetric) -> dict[str, object]:
        values = gpu_metric_to_metrics(metric)
        if values["sm_utilization"] is not None:
            values["sm_utilization"] = float(values["sm_utilization"]) * 100
        return {
            "kind": "gpu_sample",
            "tick": tick.tick,
            "metric_id": metric.metric_id,
            "ts": metric.ts.isoformat(),
            "job_id": metric.job_id,
            "gpu_uuid": metric.gpu_uuid,
            "rank_id": metric.rank_id,
            "chain_index": metric.chain_index,
            "local_rank": metric.local_rank,
            "role": metric.role,
            "node_name": metric.node_name,
            "instance_type": metric.instance_type,
            "model_name": metric.model_name,
            "metrics": values,
        }

    def _snapshot_jobs(self, jobs: JobStore, observed_at: datetime) -> None:
        for job in jobs.list_jobs(self.user_id):
            if not self._is_bound_job(job.job_id):
                continue
            self._append(
                {
                    "kind": "job_snapshot",
                    "tick": self._tick_number(),
                    "observed_at": observed_at.isoformat(),
                    "job": job.model_dump(mode="json"),
                }
            )
            for rank in jobs.ranks(job.job_id):
                self._snapshot_rank(rank, observed_at)

    def _snapshot_rank(self, rank: Rank, observed_at: datetime) -> None:
        if self._rank_updated_at.get(rank.rank_id) == rank.updated_at:
            return
        self._rank_updated_at[rank.rank_id] = rank.updated_at
        self._append(
            {
                "kind": "rank_snapshot",
                "tick": self._tick_number(),
                "observed_at": observed_at.isoformat(),
                "rank": rank.model_dump(mode="json"),
            }
        )

    def _snapshot_resource_map(
        self, resources: ResourceMapStore, observed_at: datetime
    ) -> None:
        resource_map = resources.get()
        if self._resource_version == resource_map.version:
            return
        self._resource_version = resource_map.version
        self._append(
            {
                "kind": "resource_snapshot",
                "tick": self._tick_number(),
                "observed_at": observed_at.isoformat(),
                "resource_map": resource_map.model_dump(mode="json"),
            }
        )

    def _snapshot_evidence(
        self, evidence: EvidenceStore, observed_at: datetime
    ) -> None:
        for row in evidence.latest(self.user_id):
            if not self._is_bound_job(row.job_id):
                continue
            body = self._json_value(asdict(row))
            digest = self._json(body)
            if self._evidence_digests.get(row.row_id) == digest:
                continue
            self._evidence_digests[row.row_id] = digest
            self._append(
                {
                    "kind": "evidence_snapshot",
                    "tick": self._tick_number(),
                    "observed_at": observed_at.isoformat(),
                    "evidence": body,
                }
            )

    def _snapshot_causal_state(
        self, causal: CausalGraphStore, observed_at: datetime
    ) -> None:
        edges, edge_metadata = causal.load_edges()
        mechanisms, mechanism_metadata = causal.load_mechanisms()
        self._append(
            {
                "kind": "causal_state_snapshot",
                "tick": self._tick_number(),
                "observed_at": observed_at.isoformat(),
                "nodes": self._json_value(
                    {
                        node_id: asdict(node)
                        for node_id, node in causal.load_nodes().items()
                    }
                ),
                "edges": self._json_value(
                    {edge_id: asdict(edge) for edge_id, edge in edges.items()}
                ),
                "edge_metadata": self._json_value(
                    {
                        edge_id: asdict(metadata)
                        for edge_id, metadata in edge_metadata.items()
                    }
                ),
                "mechanisms": self._json_value(
                    {
                        mechanism_id: asdict(mechanism)
                        for mechanism_id, mechanism in mechanisms.items()
                    }
                ),
                "mechanism_metadata": self._json_value(
                    {
                        mechanism_id: asdict(metadata)
                        for mechanism_id, metadata in mechanism_metadata.items()
                    }
                ),
            }
        )

    def _json(self, value: object) -> str:
        return json.dumps(value, default=self._json_default, sort_keys=True)

    def _json_value(self, value: object) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self._json(value)))

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, set):
            return sorted(value)
        return str(value)


def _mean(values: tuple[float, ...]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: tuple[float, ...], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _timestamp(value: object) -> datetime | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        return None
    return datetime.fromtimestamp(value / 1_000_000_000, UTC)


def _milliseconds(metrics: dict[str, object], name: str) -> float | None:
    raw = metrics.get(name)
    if not isinstance(raw, dict):
        return None
    value = raw.get("value")
    unit = raw.get("unit")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    factors = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1000.0, "sec": 1000.0}
    if not isinstance(unit, str) or unit not in factors:
        return None
    return float(value) * factors[unit]


def _whole_number(metrics: dict[str, object], name: str) -> int | None:
    raw = metrics.get(name)
    if not isinstance(raw, dict):
        return None
    value = raw.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)
