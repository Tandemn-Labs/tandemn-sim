"""Run benchmark AIPerf workloads when their real-hardware routers are ready."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Protocol, TextIO
from uuid import NAMESPACE_URL, uuid5

import yaml
from tandemn_system_data.clients import (
    CausalGraphStore,
    EvidenceStore,
    GpuMetricStore,
    JobStore,
    PlanStore,
    PostgresClient,
    PostgresEventLog,
    ResourceMapStore,
)
from tandemn_system_data.models import JobStatus

from data_generation.controller import (
    BENCHMARKS_ROOT,
    ROOT,
    _load_yaml,
    aiperf_command,
    build_aiperf_config,
)
from data_generation.traces import RealTraceCollector, TraceTick

LOGGER = logging.getLogger(__name__)
DEFAULT_ROUTER_CONFIG_DIR = "~/.tandemn/router-configs"
GOODPUT_WINDOW_SECONDS = 12_600
_CROCKFORD_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def stable_store_id(benchmark_id: str, job_name: str) -> str:
    raw = (
        b"\x00" * 6
        + uuid5(NAMESPACE_URL, f"tandemn-sim:{benchmark_id}:{job_name}").bytes[:10]
    )
    value = int.from_bytes(raw, "big")
    return "".join(
        _CROCKFORD_BASE32[(value >> shift) & 31] for shift in range(125, -1, -5)
    )


def resolve_endpoint(job_id: str, config_dir: Path) -> str:
    """Read one real deployment's router endpoint from its config file."""
    path = config_dir / f"{job_id}.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(
            f"no router config for {job_id!r} in {config_dir} - is the job placed?"
        ) from None
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"unreadable router config {path}: {error}") from None
    port = config.get("listen_port") or _router_listen_port(job_id)
    if not isinstance(port, int) or not 0 < port < 65536:
        raise SystemExit(f"router config {path} has invalid listen_port {port!r}")
    return f"http://127.0.0.1:{port}"


def check_ready(endpoint: str, timeout: float = 2.0) -> bool:
    """Return whether one router responds to its readiness endpoint."""
    try:
        with urllib.request.urlopen(f"{endpoint}/readyz", timeout=timeout) as response:
            return bool(response.status == 200)
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _router_listen_port(job_id: str) -> int:
    digest = hashlib.sha256(job_id.encode()).digest()
    return 28_000 + int.from_bytes(digest[:4], "big") % 1_000


class State(StrEnum):
    """Lifecycle of one manifest workload."""

    PENDING = "PENDING"
    WAIT_READY = "WAIT_READY"
    LOADING = "LOADING"
    OBSERVING = "OBSERVING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


TERMINAL_STATES = {State.DONE, State.FAILED, State.SKIPPED}


@dataclass(frozen=True)
class ManifestJob:
    """One workload declared by a benchmark manifest."""

    job_id: str
    user_id: str
    path: Path
    skip_reason: str | None = None
    kind: str = "online"
    driver: str = "aiperf"


@dataclass(frozen=True)
class Manifest:
    """Real-hardware jobs and polling settings for one benchmark."""

    benchmark_id: str
    poll_interval: float
    jobs: tuple[ManifestJob, ...]


@dataclass(frozen=True)
class StoreJob:
    """Store fields needed to match and start one workload."""

    job_id: str
    status: JobStatus
    job_name: str | None


class StoreView(Protocol):
    """Read the Store jobs relevant to a benchmark."""

    def list_jobs(self, user_ids: Sequence[str]) -> Sequence[StoreJob]: ...


class PostgresStoreView:
    """Read canonical jobs from tandemn-store."""

    def __init__(self, client: PostgresClient | None = None) -> None:
        self._jobs = JobStore(client or PostgresClient())

    def list_jobs(self, user_ids: Sequence[str]) -> tuple[StoreJob, ...]:
        jobs = []
        for user_id in user_ids:
            for job in self._jobs.list_jobs(user_id):
                name = job.input_source.get("job_name")
                jobs.append(
                    StoreJob(
                        job_id=job.job_id,
                        status=job.status,
                        job_name=name if isinstance(name, str) else None,
                    )
                )
        return tuple(jobs)


@dataclass
class TraceRuntime:
    """Connect one trace collector to the Store clients used during a real run.

    The runtime reads Store and artifact data only. It never receives an
    AIPerf process handle, so process lifecycle remains entirely owned by
    ``real_scenario``.
    """

    collector: RealTraceCollector
    events: PostgresEventLog
    jobs: JobStore
    plans: PlanStore
    resources: ResourceMapStore
    metrics: GpuMetricStore
    evidence: EvidenceStore | None = None
    causal: CausalGraphStore | None = None

    def start(self, started_at: datetime) -> None:
        """Open the run trace before workload observation begins."""
        self.collector.start_run(started_at)

    def bind_job(self, manifest: ManifestJob, store_job_id: str) -> None:
        """Associate one benchmark artifact path with its canonical Store job."""
        self.collector.bind_job(
            authored_job_id=manifest.job_id,
            store_job_id=store_job_id,
            job_kind=manifest.kind,
            benchmark_job_path=manifest.path,
        )

    def collect(self, observed_at: datetime) -> None:
        """Collect one due minute and emit every tick whose holdback elapsed."""
        if not self.collector.collect(observed_at):
            return
        self._read(observed_at)
        self._emit(self.collector.finalize_closed_ticks(observed_at))

    def collect_final_aiperf_metrics(self, authored_job_id: str) -> None:
        """Read AIPerf's final flushed records after the scenario sees exit."""
        self.collector.collect_aiperf_records(authored_job_id)

    def finish(self, finished_at: datetime, *, status: str) -> None:
        """Close the final tick, emit summaries, and terminate the trace file."""
        self._read(finished_at)
        self.collector.close_open_tick(finished_at)
        final_at = finished_at + timedelta(
            seconds=self.collector.tick_holdback_seconds
        )
        self._emit(self.collector.finalize_closed_ticks(final_at))
        self.collector.emit_run_summary(duration_seconds=GOODPUT_WINDOW_SECONDS)
        self.collector.close_run(finished_at, status=status)

    def _read(self, observed_at: datetime) -> None:
        self.collector.collect_store_events(self.events, plans=self.plans)
        self.collector.collect_store_snapshots(
            jobs=self.jobs,
            resources=self.resources,
            evidence=self.evidence,
            causal=self.causal,
            observed_at=observed_at,
        )
        for binding in self.collector.job_bindings():
            if binding.job_kind == "online":
                self.collector.collect_aiperf_records(binding.authored_job_id)

    def _emit(self, ticks: Sequence[TraceTick]) -> None:
        for tick in ticks:
            for binding in self.collector.job_bindings():
                if binding.job_kind != "online":
                    continue
                ttft, tpot = _online_targets(binding.benchmark_job_path)
                self.collector.emit_online_job_window(
                    tick,
                    binding.authored_job_id,
                    ttft_target_ms=ttft,
                    tpot_target_ms=tpot,
                )
            self.collector.emit_gpu_samples(tick, self.metrics)
            self.collector.emit_rank_windows(tick, self.metrics)
            self.collector.emit_batch_job_window(tick, self.metrics)


def _create_trace_runtime(
    client: PostgresClient,
    *,
    run_dir: Path,
    run_id: str,
    benchmark_id: str,
    method: str,
    user_id: str,
    collect_evidence: bool,
    collect_causal_state: bool,
) -> TraceRuntime:
    collector = RealTraceCollector(
        run_dir=run_dir,
        run_id=run_id,
        benchmark_id=benchmark_id,
        method=method,
        user_id=user_id,
        collect_evidence=collect_evidence,
        collect_causal_state=collect_causal_state,
    )
    return TraceRuntime(
        collector=collector,
        events=PostgresEventLog(client),
        jobs=JobStore(client),
        plans=PlanStore(client),
        resources=ResourceMapStore(client, user_id=user_id),
        metrics=GpuMetricStore(client),
        evidence=EvidenceStore(client) if collect_evidence else None,
        causal=(
            CausalGraphStore(client, user_id=user_id)
            if collect_causal_state
            else None
        ),
    )


def _online_targets(path: Path) -> tuple[float | None, float | None]:
    features = _load_yaml(path).get("spec", {}).get("features", {})

    def target(name: str) -> float | None:
        value = features.get(name)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        return None

    return target("target_p99_ttft_ms"), target("target_p99_tpot_ms")


@dataclass
class TrackedJob:
    """Mutable daemon state for one manifest workload."""

    manifest: ManifestJob
    state: State = State.PENDING
    running_seen_at: float | None = None
    store_job_id: str | None = None
    process: subprocess.Popen[bytes] | None = None
    log_file: BinaryIO | None = None
    stop_requested: bool = False


def load_manifest(benchmark_id: str) -> Manifest:
    """Load job metadata and polling settings from one benchmark manifest."""
    manifest_path = BENCHMARKS_ROOT / benchmark_id / "benchmark.yaml"
    document = _load_yaml(manifest_path)
    benchmark = document.get("benchmark")
    if not isinstance(benchmark, dict) or benchmark.get("id") != benchmark_id:
        raise ValueError(f"Invalid benchmark manifest: {manifest_path}")
    jobs_name = benchmark.get("jobs")
    if not isinstance(jobs_name, str):
        raise ValueError(f"Benchmark {benchmark_id} does not define a jobs directory")

    jobs = []
    for path in sorted((manifest_path.parent / jobs_name).glob("*.yaml")):
        job = _load_yaml(path)
        schedule = job.get("schedule")
        spec = job.get("spec")
        if not isinstance(schedule, dict) or not isinstance(spec, dict):
            raise ValueError(f"Invalid benchmark job: {path}")
        job_id = job.get("job_id")
        user_id = spec.get("user_id")
        if not isinstance(job_id, str) or not isinstance(user_id, str):
            raise ValueError(f"Invalid benchmark job identity: {path}")
        kind = schedule.get("kind")
        driver = schedule.get("driver")
        if kind not in {"online", "batch"} or not isinstance(driver, str):
            raise ValueError(f"Invalid benchmark job schedule: {path}")
        reason = None
        if schedule.get("enabled") is not True:
            reason = "job is disabled"
        elif kind != "batch" and driver != "aiperf":
            reason = f"unsupported driver {driver!r}"
        jobs.append(ManifestJob(job_id, user_id, path, reason, kind, driver))

    poll_interval = float(benchmark.get("poll_interval_seconds", 10))
    if poll_interval <= 0:
        raise ValueError("poll interval must be positive")
    return Manifest(benchmark_id, poll_interval, tuple(jobs))


def run_scenario(
    manifest: Manifest,
    store: StoreView,
    *,
    poll_interval: float,
    ready_timeout: float,
    output_dir: Path,
    router_config_dir: Path,
    source_cloud: str,
    source_region: str,
    source_zone: str,
    user_ids: Sequence[str] | None = None,
    job_ids: Sequence[str] | None = None,
    trace_client: PostgresClient | None = None,
    trace_method: str = "real",
    collect_evidence: bool = False,
    collect_causal_state: bool = False,
) -> int:
    """Poll Store and run each selected AIPerf workload once."""
    if poll_interval <= 0 or ready_timeout <= 0:
        raise ValueError("poll interval and ready timeout must be positive")

    selected = set(job_ids or ())
    known = {job.job_id for job in manifest.jobs}
    unknown = selected - known
    if unknown:
        raise ValueError(f"Unknown manifest job IDs: {', '.join(sorted(unknown))}")
    jobs = [
        TrackedJob(job)
        for job in manifest.jobs
        if not selected or job.job_id in selected
    ]
    query_users = tuple(
        dict.fromkeys(user_ids or [job.manifest.user_id for job in jobs])
    )

    started_at = datetime.now(UTC)
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"{timestamp}-{manifest.benchmark_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    trace = None
    if trace_client is not None:
        if len(query_users) != 1:
            raise ValueError("real traces require exactly one Store user")
        trace = _create_trace_runtime(
            trace_client,
            run_dir=run_dir,
            run_id=run_dir.name,
            benchmark_id=manifest.benchmark_id,
            method=trace_method,
            user_id=query_users[0],
            collect_evidence=collect_evidence,
            collect_causal_state=collect_causal_state,
        )
        trace.start(started_at)

    with events_path.open("a", encoding="utf-8") as events:
        for job in jobs:
            if job.manifest.skip_reason:
                _transition(job, State.SKIPPED, job.manifest.skip_reason, events)
            else:
                _event(job, "waiting for Store job", events)

        while any(job.state not in TERMINAL_STATES for job in jobs):
            snapshot = store.list_jobs(query_users)
            by_id = {job.job_id: job for job in snapshot}
            by_name: dict[str, StoreJob] = {}
            for store_job in snapshot:
                if store_job.job_name:
                    by_name.setdefault(store_job.job_name, store_job)
            now = time.monotonic()

            for job in jobs:
                if job.state in TERMINAL_STATES:
                    continue
                found = (
                    by_id.get(job.manifest.job_id)
                    or by_id.get(
                        stable_store_id(manifest.benchmark_id, job.manifest.job_id)
                    )
                    or by_name.get(job.manifest.job_id)
                )
                if job.state == State.LOADING:
                    _poll_process(job, events, trace)
                    if (
                        job.state == State.LOADING
                        and found is not None
                        and found.status == JobStatus.FINISHED
                        and not job.stop_requested
                    ):
                        if job.process is None:
                            raise RuntimeError("LOADING job has no AIPerf process")
                        job.process.terminate()
                        job.stop_requested = True
                        _event(job, "Store job finished; stopping AIPerf", events)
                    continue
                if (
                    job.state == State.WAIT_READY
                    and job.running_seen_at is not None
                    and now - job.running_seen_at >= ready_timeout
                ):
                    _transition(job, State.FAILED, "router readiness timed out", events)
                    continue
                if found is None:
                    continue
                if job.store_job_id is None:
                    job.store_job_id = found.job_id
                    if trace is not None:
                        trace.bind_job(job.manifest, found.job_id)

                if job.manifest.kind == "batch":
                    if found.status == JobStatus.FINISHED:
                        _transition(job, State.DONE, "Store batch job finished", events)
                    elif (
                        job.state == State.PENDING
                        and found.status == JobStatus.RUNNING
                    ):
                        _transition(
                            job,
                            State.OBSERVING,
                            "Store batch job is running",
                            events,
                        )
                    continue

                if found.status == JobStatus.FINISHED:
                    _transition(
                        job,
                        State.SKIPPED,
                        "job finished before router was ready",
                        events,
                    )
                    continue

                if job.state == State.PENDING and found.status == JobStatus.RUNNING:
                    job.running_seen_at = now
                    _transition(
                        job, State.WAIT_READY, "Store status is running", events
                    )

                if job.state != State.WAIT_READY:
                    continue
                if found.status != JobStatus.RUNNING:
                    continue
                try:
                    endpoint = resolve_endpoint(found.job_id, router_config_dir)
                except SystemExit:
                    continue
                if not check_ready(endpoint):
                    continue
                _launch(
                    job,
                    endpoint,
                    run_dir,
                    source_cloud,
                    source_region,
                    source_zone,
                    events,
                )

            if trace is not None:
                trace.collect(datetime.now(UTC))
            if any(job.state not in TERMINAL_STATES for job in jobs):
                time.sleep(poll_interval)

    exit_code = int(any(job.state == State.FAILED for job in jobs))
    if trace is not None:
        trace.finish(
            datetime.now(UTC),
            status="failed" if exit_code else "completed",
        )
    return exit_code


def _launch(
    job: TrackedJob,
    endpoint: str,
    run_dir: Path,
    source_cloud: str,
    source_region: str,
    source_zone: str,
    events: TextIO,
) -> None:
    """Render and start one AIPerf process."""
    job_dir = run_dir / job.manifest.job_id
    artifact_dir = job_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_path = job_dir / "aiperf.yaml"
    config = build_aiperf_config(
        job.manifest.path,
        endpoint_url=endpoint,
        session_header="X-Tandemn-Session-ID",
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    headers = {
        "X-Tandemn-Source-Cloud": source_cloud,
        "X-Tandemn-Source-Region": source_region,
        "X-Tandemn-Source-Zone": source_zone,
    }
    log_path = job_dir / "aiperf.log"
    job.log_file = log_path.open("wb")
    try:
        job.process = subprocess.Popen(
            aiperf_command(
                config_path,
                headers=headers,
                artifact_dir=artifact_dir,
            ),
            stdout=job.log_file,
            stderr=subprocess.STDOUT,
        )
    except OSError as error:
        job.log_file.close()
        job.log_file = None
        _transition(job, State.FAILED, f"failed to start AIPerf: {error}", events)
        return
    _transition(job, State.LOADING, f"AIPerf started against {endpoint}", events)


def _poll_process(
    job: TrackedJob,
    events: TextIO,
    trace: TraceRuntime | None = None,
) -> None:
    """Record a completed AIPerf process."""
    if job.process is None:
        raise RuntimeError("LOADING job has no AIPerf process")
    exit_code = job.process.poll()
    if exit_code is None:
        return
    if job.log_file is not None:
        job.log_file.close()
        job.log_file = None
    if trace is not None:
        trace.collect_final_aiperf_metrics(job.manifest.job_id)
    state = State.DONE if exit_code == 0 or job.stop_requested else State.FAILED
    _transition(job, state, f"AIPerf exited with code {exit_code}", events, exit_code)


def _transition(
    job: TrackedJob,
    state: State,
    detail: str,
    events: TextIO,
    exit_code: int | None = None,
) -> None:
    job.state = state
    _event(job, detail, events, exit_code)


def _event(
    job: TrackedJob,
    detail: str,
    events: TextIO,
    exit_code: int | None = None,
) -> None:
    event: dict[str, object] = {
        "ts": datetime.now(UTC).isoformat(),
        "job_id": job.manifest.job_id,
        "state": job.state,
        "detail": detail,
    }
    if exit_code is not None:
        event["exit_code"] = exit_code
    events.write(json.dumps(event) + "\n")
    events.flush()
    LOGGER.info("%s %s: %s", job.manifest.job_id, job.state, detail)


def main() -> None:
    """Run real-hardware benchmark workloads as their routers become ready."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_id")
    parser.add_argument("--poll-interval", type=float)
    parser.add_argument("--ready-timeout", type=float, default=1200)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--router-config-dir",
        type=Path,
        default=Path(
            os.environ.get("TANDEMN_ROUTER_CONFIG_DIR", DEFAULT_ROUTER_CONFIG_DIR)
        ),
    )
    parser.add_argument("--source-cloud", default="aws")
    parser.add_argument("--source-region", default="us-east-1")
    parser.add_argument("--source-zone", default="us-east-1a")
    parser.add_argument("--user-id", action="append")
    parser.add_argument("--job-id", action="append")
    parser.add_argument("--method", default=os.environ.get("BASELINE", "real"))
    parser.add_argument("--collect-evidence", action="store_true")
    parser.add_argument("--collect-causal-state", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        manifest = load_manifest(args.benchmark_id)
        client = PostgresClient()
        exit_code = run_scenario(
            manifest,
            PostgresStoreView(client),
            poll_interval=(
                manifest.poll_interval
                if args.poll_interval is None
                else args.poll_interval
            ),
            ready_timeout=args.ready_timeout,
            output_dir=args.output_dir.expanduser(),
            router_config_dir=args.router_config_dir.expanduser(),
            source_cloud=args.source_cloud,
            source_region=args.source_region,
            source_zone=args.source_zone,
            user_ids=args.user_id,
            job_ids=args.job_id,
            trace_client=client,
            trace_method=args.method,
            collect_evidence=args.collect_evidence,
            collect_causal_state=args.collect_causal_state,
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
