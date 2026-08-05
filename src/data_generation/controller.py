"""Run one benchmark and orchestrate all jobs in its overview."""

import argparse
import json
import logging
import os
import signal
import subprocess
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import FrameType
from typing import BinaryIO, cast

import yaml
from dotenv import load_dotenv

from data_generation.batch_dataset_generator import generate_batch_dataset
from data_generation.scheduling import (
    DecisionStatus,
    OptimizationGoal,
    SchedulerJob,
    SchedulerWorkload,
    SchedulingDecision,
    SchedulingMethod,
)
from data_generation.submission import (
    JobSubmitter,
    SubmissionError,
    SubmissionRequest,
    TandemnStoreSubmitter,
)

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_ROOT = ROOT / "benchmarks"
DEFAULT_BENCHMARK = "simulation_3h"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts"
BENCHMARK_STATE_FILENAME = "benchmark_state.json"
PROMPTS_FILENAME = "prompts.jsonl"
RESPONSES_FILENAME = "responses.jsonl"
RUNTIME_AIPERF_FILENAME = "runtime-aiperf.yaml"
LOGGER = logging.getLogger(__name__)


class JobKind(StrEnum):
    """Benchmark job lifecycle types."""

    ONLINE = "online"
    BATCH = "batch"


class JobState(StrEnum):
    """Controller states for one benchmark job."""

    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    MISSED = "missed"
    DISABLED = "disabled"
    PAUSED = "paused"
    REJECTED = "rejected"


TERMINAL_STATES = {
    JobState.COMPLETED,
    JobState.FAILED,
    JobState.MISSED,
    JobState.DISABLED,
    JobState.PAUSED,
    JobState.REJECTED,
}


@dataclass(frozen=True)
class Job:
    """One job declared in a benchmark overview."""

    job_id: str
    enabled: bool
    output_to_terminal: bool
    kind: JobKind
    driver: str
    config: Path
    command: tuple[str, ...] | None
    start_minute: float
    stop_minute: float | None
    deadline_minute: float | None
    hard_stop_minute: float
    parent_job_id: str | None
    submission: SubmissionRequest | None
    optimization_goal: OptimizationGoal | None = None
    workload: SchedulerWorkload | None = None


@dataclass(frozen=True)
class Benchmark:
    """One benchmark overview and its complete job schedule."""

    benchmark_id: str
    poll_interval_seconds: float
    shutdown_grace_seconds: float
    jobs: tuple[Job, ...]


@dataclass
class RunningJob:
    """The live process and state for one scheduled job."""

    job: Job
    state: JobState
    process: subprocess.Popen[bytes] | None = None
    log_file: BinaryIO | None = None
    stop_requested_at: float | None = None
    deadline_missed: bool = False
    pause_requested: bool = False
    launch_count: int = 0
    first_started_at: str | None = None
    last_started_at: str | None = None
    paused_at: str | None = None
    resumed_at: str | None = None
    completed_at: str | None = None
    accumulated_runtime_seconds: float = 0.0
    process_started_at: float | None = None
    exit_code: int | None = None
    tandemn_job_id: str | None = None
    endpoint_url: str | None = None
    raw_record_offsets: dict[Path, int] = field(default_factory=dict)
    recorded_prompt_ids: set[str] = field(default_factory=set)
    recorded_response_ids: set[str] = field(default_factory=set)
    artifact_ids_loaded: bool = False
    scheduling_decision: SchedulingDecision | None = None
    allocation_released: bool = False


@dataclass(frozen=True)
class RunSummary:
    """Final benchmark run result."""

    run_id: str
    states: Mapping[str, JobState]
    deadline_misses: frozenset[str]


@dataclass(frozen=True)
class PlacementSummary:
    """Final result of a placement-only benchmark simulation."""

    run_id: str
    decisions: Mapping[str, SchedulingDecision]


def main() -> None:
    """Run or validate one benchmark."""
    load_dotenv(ROOT / ".env")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark",
        nargs="?",
        default=DEFAULT_BENCHMARK,
        help="benchmark name or path to its overview.yaml",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument(
        "--aiperf-export-level",
        choices=("summary", "records", "raw"),
        default="raw",
        help="AI-Perf output detail; summary excludes per-request payloads",
    )
    parser.add_argument(
        "--discard-request-data",
        action="store_true",
        help="do not create controller request files and discard AI-Perf inputs",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="resume a paused benchmark run without advancing paused time",
    )
    parser.add_argument(
        "--generate-batch",
        type=Path,
        metavar="CONFIG",
        help="generate batch JSONL from a batch-dataset/v1 YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSONL output path; otherwise records print to the terminal",
    )
    parser.add_argument(
        "--records",
        type=int,
        help="optional generation limit used with --generate-batch",
    )
    parser.add_argument(
        "--method",
        choices=("dynamo-aic",),
        help="scheduler method used to place managed jobs",
    )
    parser.add_argument(
        "--placement-only",
        action="store_true",
        help="simulate placements without waiting or starting data senders",
    )
    args = parser.parse_args()

    if args.generate_batch is not None:
        try:
            generate_batch_dataset(
                args.generate_batch,
                args.output,
                record_limit=args.records,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return

    if args.output is not None or args.records is not None:
        parser.error("--output and --records require --generate-batch")
    if args.placement_only and args.method is None:
        parser.error("--placement-only requires --method")
    if args.placement_only and args.resume is not None:
        parser.error("--placement-only cannot be resumed")
    if args.discard_request_data and args.aiperf_export_level != "summary":
        parser.error("--discard-request-data requires --aiperf-export-level summary")

    try:
        overview_path = resolve_benchmark_overview(args.benchmark)
        benchmark = load_benchmark(overview_path)
    except ValueError as error:
        parser.error(str(error))

    scheduling_method: SchedulingMethod | None = None
    if args.method == "dynamo-aic":
        from data_generation.methods.dynamo_aic import load_dynamo_aic_method

        try:
            scheduling_method = load_dynamo_aic_method(overview_path.parent)
        except (OSError, ValueError, yaml.YAMLError) as error:
            parser.error(str(error))

    if not args.placement_only:
        log_benchmark(benchmark)
    if not args.validate_only:
        if args.placement_only:
            if scheduling_method is None:
                parser.error("no scheduler method is configured")
            try:
                run_placement_only(
                    benchmark,
                    args.artifact_root,
                    scheduling_method,
                )
            except ValueError as error:
                parser.error(str(error))
            return
        previous_sigterm = signal.signal(signal.SIGTERM, interrupt_controller)
        try:
            run_benchmark(
                benchmark,
                args.artifact_root,
                resume_run_id=args.resume,
                aiperf_export_level=args.aiperf_export_level,
                discard_request_data=args.discard_request_data,
                scheduling_method=scheduling_method,
            )
        except ValueError as error:
            parser.error(str(error))
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)


def run_benchmark(
    benchmark: Benchmark,
    artifact_root: Path,
    *,
    run_id: str | None = None,
    resume_run_id: str | None = None,
    aiperf_export_level: str = "raw",
    discard_request_data: bool = False,
    job_submitter: JobSubmitter | None = None,
    scheduling_method: SchedulingMethod | None = None,
) -> RunSummary:
    """Run or resume every enabled job in one benchmark."""
    validate_runnable_jobs(benchmark)
    if run_id is not None and resume_run_id is not None:
        raise ValueError("run_id and resume_run_id cannot be used together")
    if resume_run_id is not None and scheduling_method is not None:
        raise ValueError("scheduler method runs cannot currently be resumed")

    active_run_id = (
        resume_run_id or run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    benchmark_directory = artifact_root / "benchmarks" / benchmark.benchmark_id
    run_directory = benchmark_directory / active_run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    run_index = benchmark_directory / "runs.json"
    jobs = {
        job.job_id: RunningJob(
            job=job,
            state=JobState.PENDING if job.enabled else JobState.DISABLED,
        )
        for job in benchmark.jobs
    }
    active_submitter = job_submitter
    if active_submitter is None and any(
        item.job.submission is not None for item in jobs.values()
    ):
        active_submitter = TandemnStoreSubmitter()
    for item in jobs.values():
        ensure_job_artifacts(
            run_directory,
            item.job.job_id,
            create_record_files=not discard_request_data,
        )
    elapsed_before_start = 0.0
    created_at = utc_timestamp()
    milestones: list[dict[str, object]] = []

    if resume_run_id is not None:
        elapsed_before_start, created_at, milestones = restore_benchmark_state(
            run_directory,
            benchmark,
            jobs,
        )
        add_milestone(
            milestones,
            "benchmark_resumed",
            elapsed_before_start,
        )

    active_period_started_at = time.time()
    status = "running"

    LOGGER.info(
        "[controller] benchmark=%s run=%s elapsed=%.1fs%s",
        benchmark.benchmark_id,
        active_run_id,
        elapsed_before_start,
        " resumed=true" if resume_run_id is not None else "",
    )
    if resume_run_id is None:
        add_milestone(milestones, "benchmark_started", 0.0)

    save_benchmark_state(
        run_directory,
        benchmark,
        active_run_id,
        status,
        created_at,
        elapsed_before_start,
        jobs,
        milestones,
        run_index,
    )

    try:
        while not all(item.state in TERMINAL_STATES for item in jobs.values()):
            now = time.time()
            elapsed_seconds = elapsed_before_start + max(
                0.0,
                now - active_period_started_at,
            )
            elapsed_minutes = elapsed_seconds / 60

            for item in jobs.values():
                previous_state = item.state
                previous_launch_count = item.launch_count
                reconcile_job(
                    item,
                    elapsed_minutes,
                    now,
                    run_directory,
                    benchmark.shutdown_grace_seconds,
                    aiperf_export_level,
                    discard_request_data,
                    active_submitter,
                    scheduling_method,
                )
                if not discard_request_data:
                    sync_job_records(item, run_directory)
                record_job_milestones(
                    milestones,
                    item,
                    previous_state,
                    previous_launch_count,
                    elapsed_seconds,
                )

            save_benchmark_state(
                run_directory,
                benchmark,
                active_run_id,
                status,
                created_at,
                elapsed_seconds,
                jobs,
                milestones,
                run_index,
            )

            if not all(item.state in TERMINAL_STATES for item in jobs.values()):
                time.sleep(benchmark.poll_interval_seconds)
    except KeyboardInterrupt:
        elapsed_seconds = elapsed_before_start + max(
            0.0,
            time.time() - active_period_started_at,
        )
        status = "paused"
        LOGGER.info("[controller] paused elapsed=%.1fs", elapsed_seconds)
        states_before_shutdown = {job_id: item.state for job_id, item in jobs.items()}
        stop_all(
            jobs.values(),
            benchmark.shutdown_grace_seconds,
            paused=True,
        )
        release_all_allocations(jobs.values(), scheduling_method, run_directory)
        if not discard_request_data:
            sync_all_job_records(jobs.values(), run_directory)
        record_shutdown_milestones(
            milestones,
            jobs,
            states_before_shutdown,
            elapsed_seconds,
        )
        add_milestone(milestones, "benchmark_paused", elapsed_seconds)
        save_benchmark_state(
            run_directory,
            benchmark,
            active_run_id,
            status,
            created_at,
            elapsed_seconds,
            jobs,
            milestones,
            run_index,
        )
    except Exception:
        elapsed_seconds = elapsed_before_start + max(
            0.0,
            time.time() - active_period_started_at,
        )
        status = "failed"
        LOGGER.exception("[controller] fatal error")
        states_before_shutdown = {job_id: item.state for job_id, item in jobs.items()}
        stop_all(
            jobs.values(),
            benchmark.shutdown_grace_seconds,
            paused=True,
        )
        release_all_allocations(jobs.values(), scheduling_method, run_directory)
        if not discard_request_data:
            sync_all_job_records(jobs.values(), run_directory)
        record_shutdown_milestones(
            milestones,
            jobs,
            states_before_shutdown,
            elapsed_seconds,
        )
        add_milestone(milestones, "benchmark_failed", elapsed_seconds)
        save_benchmark_state(
            run_directory,
            benchmark,
            active_run_id,
            status,
            created_at,
            elapsed_seconds,
            jobs,
            milestones,
            run_index,
        )
        raise
    else:
        elapsed_seconds = elapsed_before_start + max(
            0.0,
            time.time() - active_period_started_at,
        )
        status = "completed"
        if not discard_request_data:
            sync_all_job_records(jobs.values(), run_directory)
        add_milestone(milestones, "benchmark_completed", elapsed_seconds)
        save_benchmark_state(
            run_directory,
            benchmark,
            active_run_id,
            status,
            created_at,
            elapsed_seconds,
            jobs,
            milestones,
            run_index,
        )

    summary = RunSummary(
        run_id=active_run_id,
        states={job_id: item.state for job_id, item in jobs.items()},
        deadline_misses=frozenset(
            item.job.job_id for item in jobs.values() if item.deadline_missed
        ),
    )
    log_summary(summary)
    return summary


def run_placement_only(
    benchmark: Benchmark,
    artifact_root: Path,
    scheduling_method: SchedulingMethod,
    *,
    run_id: str | None = None,
) -> PlacementSummary:
    """Simulate benchmark arrivals and placements without starting senders."""
    active_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_directory = (
        artifact_root / "benchmarks" / benchmark.benchmark_id / active_run_id
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    decisions: dict[str, SchedulingDecision] = {}
    releases: list[tuple[float, str]] = []
    resource_events: list[dict[str, object]] = []
    created_at = utc_timestamp()
    save_placement_run_artifact(
        run_directory,
        benchmark,
        active_run_id,
        scheduling_method,
        decisions,
        resource_events,
        created_at,
    )
    save_resource_artifact(run_directory, scheduling_method)

    indexed_jobs = enumerate(benchmark.jobs)
    ordered_jobs = sorted(
        indexed_jobs,
        key=lambda value: (value[1].start_minute, value[0]),
    )
    for _, job in ordered_jobs:
        if not job.enabled or job.submission is None:
            continue
        pending_releases = sorted(releases)
        releases = []
        for release_minute, release_job_id in pending_releases:
            if release_minute > job.start_minute:
                releases.append((release_minute, release_job_id))
                continue
            scheduling_method.release(release_job_id)
            resource_events.append(
                {
                    "offset_minutes": release_minute,
                    "event": "released",
                    "job_id": release_job_id,
                    "resource_snapshot": scheduling_method.resource_snapshot(),
                }
            )

        decision = scheduling_method.schedule(scheduler_job(job))
        decisions[job.job_id] = decision
        save_placement_artifacts(
            run_directory,
            decision,
            scheduling_method,
        )
        resource_events.append(
            {
                "offset_minutes": job.start_minute,
                "event": decision.status.value,
                "job_id": job.job_id,
                "resource_snapshot": decision.resource_snapshot_after,
            }
        )
        log_placement_decision(job.start_minute, decision)
        save_placement_run_artifact(
            run_directory,
            benchmark,
            active_run_id,
            scheduling_method,
            decisions,
            resource_events,
            created_at,
        )
        if decision.status == DecisionStatus.ACCEPTED:
            releases.append((job_end_minute(job), job.job_id))

    for release_minute, release_job_id in sorted(releases):
        scheduling_method.release(release_job_id)
        resource_events.append(
            {
                "offset_minutes": release_minute,
                "event": "released",
                "job_id": release_job_id,
                "resource_snapshot": scheduling_method.resource_snapshot(),
            }
        )
    save_placement_run_artifact(
        run_directory,
        benchmark,
        active_run_id,
        scheduling_method,
        decisions,
        resource_events,
        created_at,
    )
    save_resource_artifact(run_directory, scheduling_method)
    LOGGER.debug(
        "[placement-only] benchmark=%s run=%s accepted=%d rejected=%d",
        benchmark.benchmark_id,
        active_run_id,
        sum(
            decision.status == DecisionStatus.ACCEPTED
            for decision in decisions.values()
        ),
        sum(
            decision.status == DecisionStatus.REJECTED
            for decision in decisions.values()
        ),
    )
    return PlacementSummary(active_run_id, decisions)


def log_placement_decision(
    release_minute: float,
    decision: SchedulingDecision,
) -> None:
    """Log one placement result at its benchmark release offset."""
    if decision.status == DecisionStatus.REJECTED:
        category = (
            "unknown"
            if decision.failure_category is None
            else decision.failure_category.value
        )
        LOGGER.info(
            "[placement] T+%gm job=%s rejected category=%s reason=%s",
            release_minute,
            decision.job_id,
            category,
            decision.reason,
        )
        return

    configuration = decision.configuration or {}
    LOGGER.info(
        "[placement] T+%gm job=%s hardware=%s devices=%s replicas=%s parallel=%s",
        release_minute,
        decision.job_id,
        decision.accelerator,
        decision.devices,
        configuration.get("replicas_needed", 1),
        configuration.get("parallel", "unknown"),
    )


def save_placement_run_artifact(
    run_directory: Path,
    benchmark: Benchmark,
    run_id: str,
    scheduling_method: SchedulingMethod,
    decisions: Mapping[str, SchedulingDecision],
    resource_events: list[dict[str, object]],
    created_at: str,
) -> None:
    """Checkpoint all placement decisions and resource events so far."""
    write_json(
        run_directory / "placements.json",
        {
            "schema_version": "placement-run/v1",
            "benchmark_id": benchmark.benchmark_id,
            "run_id": run_id,
            "method": scheduling_method.method_name,
            "created_at": created_at,
            "updated_at": utc_timestamp(),
            "decisions": [decision.to_dict() for decision in decisions.values()],
            "resource_events": resource_events,
            "current_resource_snapshot": scheduling_method.resource_snapshot(),
            "final_resource_snapshot": scheduling_method.resource_snapshot(),
        },
    )


def job_end_minute(job: Job) -> float:
    """Return the allocation release offset for one benchmark job."""
    if job.kind == JobKind.ONLINE and job.stop_minute is not None:
        return job.stop_minute
    if job.kind == JobKind.BATCH and job.deadline_minute is not None:
        return job.deadline_minute
    return job.hard_stop_minute


def scheduler_job(job: Job) -> SchedulerJob:
    """Expose only canonical job inputs to a scheduler method."""
    if job.submission is None:
        raise ValueError(f"Job {job.job_id} has no scheduler-visible submission")
    if job.workload is None:
        raise ValueError(f"Job {job.job_id} has no scheduler-visible workload")
    return SchedulerJob(
        job_id=job.job_id,
        kind=job.kind.value,
        spec=job.submission.spec_json,
        workload=job.workload,
        optimization_goal=job.optimization_goal,
    )


def save_placement_artifacts(
    run_directory: Path,
    decision: SchedulingDecision,
    scheduling_method: SchedulingMethod,
) -> None:
    """Persist one decision and the resulting resource-map snapshot."""
    job_directory = ensure_job_artifacts(
        run_directory,
        decision.job_id,
        create_record_files=False,
    )
    write_json(job_directory / "placement.json", decision.to_dict())
    save_resource_artifact(run_directory, scheduling_method)


def save_resource_artifact(
    run_directory: Path,
    scheduling_method: SchedulingMethod,
) -> None:
    """Persist the scheduler's latest mutable resource map."""
    write_json(
        run_directory / "resource_map.json",
        {
            "schema_version": "resource-state/v1",
            "method": scheduling_method.method_name,
            "updated_at": utc_timestamp(),
            "resource_snapshot": scheduling_method.resource_snapshot(),
        },
    )


def write_json(path: Path, document: Mapping[str, object]) -> None:
    """Atomically write one JSON artifact."""
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def reconcile_job(
    item: RunningJob,
    elapsed_minutes: float,
    now: float,
    run_directory: Path,
    shutdown_grace_seconds: float,
    aiperf_export_level: str = "raw",
    discard_request_data: bool = False,
    job_submitter: JobSubmitter | None = None,
    scheduling_method: SchedulingMethod | None = None,
) -> None:
    """Apply the current benchmark time to one job."""
    record_process_exit(item)
    release_terminal_allocation(item, scheduling_method, run_directory)
    job = item.job

    if item.state == JobState.PENDING:
        window_closed = (
            job.kind == JobKind.ONLINE
            and job.stop_minute is not None
            and elapsed_minutes >= job.stop_minute
        ) or (job.kind == JobKind.BATCH and elapsed_minutes >= job.hard_stop_minute)
        if window_closed:
            item.state = JobState.MISSED
            LOGGER.error("[missed] %s", job.job_id)
        elif elapsed_minutes >= job.start_minute:
            start_job(
                item,
                run_directory,
                aiperf_export_level=aiperf_export_level,
                discard_request_data=discard_request_data,
                job_submitter=job_submitter,
                scheduling_method=scheduling_method,
            )

    if item.state == JobState.RUNNING:
        if (
            job.kind == JobKind.BATCH
            and job.deadline_minute is not None
            and elapsed_minutes >= job.deadline_minute
            and not item.deadline_missed
        ):
            item.deadline_missed = True
            LOGGER.warning("[deadline-missed] %s", job.job_id)

        should_stop = (
            job.kind == JobKind.ONLINE
            and job.stop_minute is not None
            and elapsed_minutes >= job.stop_minute
        ) or (job.kind == JobKind.BATCH and elapsed_minutes >= job.hard_stop_minute)
        if should_stop:
            request_stop(item, now)

    if (
        item.state == JobState.STOPPING
        and item.stop_requested_at is not None
        and now - item.stop_requested_at >= shutdown_grace_seconds
        and item.process is not None
        and item.process.poll() is None
    ):
        signal_process(item.process, signal.SIGKILL)
        LOGGER.warning("[force-stop] %s", job.job_id)

    record_process_exit(item)
    release_terminal_allocation(item, scheduling_method, run_directory)


def start_job(
    item: RunningJob,
    run_directory: Path,
    *,
    aiperf_export_level: str = "raw",
    discard_request_data: bool = False,
    job_submitter: JobSubmitter | None = None,
    scheduling_method: SchedulingMethod | None = None,
) -> None:
    """Start one independent AIPerf generator or batch sender."""
    job_directory = ensure_job_artifacts(
        run_directory,
        item.job.job_id,
        create_record_files=not discard_request_data,
    )
    if discard_request_data and item.job.driver == "aiperf":
        discard_aiperf_inputs(job_directory)
    try:
        if (
            scheduling_method is not None
            and item.job.submission is not None
            and item.scheduling_decision is None
        ):
            item.scheduling_decision = scheduling_method.schedule(
                scheduler_job(item.job)
            )
            save_placement_artifacts(
                run_directory,
                item.scheduling_decision,
                scheduling_method,
            )
            if item.scheduling_decision.status == DecisionStatus.REJECTED:
                item.state = JobState.REJECTED
                category = (
                    "unknown"
                    if item.scheduling_decision.failure_category is None
                    else item.scheduling_decision.failure_category.value
                )
                LOGGER.warning(
                    "[rejected] %s category=%s reason=%s",
                    item.job.job_id,
                    category,
                    item.scheduling_decision.reason,
                )
                return
        runtime_config = item.job.config
        if item.job.submission is not None:
            if item.tandemn_job_id is None or item.endpoint_url is None:
                if job_submitter is None:
                    raise SubmissionError("no Tandemn job submitter is configured")
                submission_result = job_submitter.submit(item.job.submission)
                item.tandemn_job_id = submission_result.tandemn_job_id
                item.endpoint_url = submission_result.endpoint_url
            runtime_config = materialize_runtime_config(
                item.job,
                job_directory,
                item.endpoint_url,
            )
        command = job_command(
            item.job,
            job_directory,
            config=runtime_config,
            endpoint_url=item.endpoint_url,
            aiperf_export_level=aiperf_export_level,
        )
    except (OSError, SubmissionError, ValueError, yaml.YAMLError) as error:
        item.state = JobState.FAILED
        release_terminal_allocation(item, scheduling_method, run_directory)
        LOGGER.error("[failed] %s setup error: %s", item.job.job_id, error)
        return
    log_file = None
    if not item.job.output_to_terminal:
        log_file = (job_directory / "process.log").open("ab")

    try:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=None if item.job.output_to_terminal else subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as error:
        if log_file is not None:
            log_file.close()
        item.state = JobState.FAILED
        release_terminal_allocation(item, scheduling_method, run_directory)
        LOGGER.error("[failed] %s launch error: %s", item.job.job_id, error)
        return

    item.process = process
    item.log_file = log_file
    item.state = JobState.RUNNING
    item.launch_count += 1
    started_at = utc_timestamp()
    if item.first_started_at is None:
        item.first_started_at = started_at
    else:
        item.resumed_at = started_at
    item.last_started_at = started_at
    item.process_started_at = time.time()
    item.pause_requested = False
    item.exit_code = None
    LOGGER.info(
        "[start] %s driver=%s pid=%d",
        item.job.job_id,
        item.job.driver,
        process.pid,
    )


def release_terminal_allocation(
    item: RunningJob,
    scheduling_method: SchedulingMethod | None,
    run_directory: Path,
) -> None:
    """Release resources once a scheduled job reaches a terminal state."""
    if (
        scheduling_method is None
        or item.scheduling_decision is None
        or item.scheduling_decision.status != DecisionStatus.ACCEPTED
        or item.allocation_released
        or item.state not in TERMINAL_STATES
        or item.state == JobState.PAUSED
    ):
        return
    scheduling_method.release(item.job.job_id)
    item.allocation_released = True
    save_resource_artifact(run_directory, scheduling_method)
    LOGGER.info("[release] %s", item.job.job_id)


def release_all_allocations(
    jobs: Iterable[RunningJob],
    scheduling_method: SchedulingMethod | None,
    run_directory: Path,
) -> None:
    """Release every live allocation when a method run cannot continue."""
    if scheduling_method is None:
        return
    for item in jobs:
        if (
            item.scheduling_decision is None
            or item.scheduling_decision.status != DecisionStatus.ACCEPTED
            or item.allocation_released
        ):
            continue
        scheduling_method.release(item.job.job_id)
        item.allocation_released = True
    save_resource_artifact(run_directory, scheduling_method)


def job_artifact_directory(run_directory: Path, job_id: str) -> Path:
    """Return the artifact directory for one job in a benchmark run."""
    return run_directory / "jobs" / job_id


def ensure_job_artifacts(
    run_directory: Path,
    job_id: str,
    *,
    create_record_files: bool = True,
) -> Path:
    """Create the stable prompt and response files for one job."""
    job_directory = job_artifact_directory(run_directory, job_id)
    job_directory.mkdir(parents=True, exist_ok=True)
    if create_record_files:
        (job_directory / PROMPTS_FILENAME).touch(exist_ok=True)
        (job_directory / RESPONSES_FILENAME).touch(exist_ok=True)
    return job_directory


def discard_aiperf_inputs(job_directory: Path) -> None:
    """Redirect AI-Perf's generated input export to the null device."""
    input_paths = (
        job_directory / "inputs.json",
        job_directory / "inputs.tmp",
    )
    for inputs_path in input_paths:
        if inputs_path.exists() or inputs_path.is_symlink():
            inputs_path.unlink()
        try:
            inputs_path.symlink_to(os.devnull)
        except OSError as error:
            raise RuntimeError(
                f"Cannot safely discard AI-Perf inputs at {inputs_path}"
            ) from error


def sync_all_job_records(
    jobs: Iterable[RunningJob],
    run_directory: Path,
) -> None:
    """Capture all request and response records currently available."""
    for item in jobs:
        sync_job_records(item, run_directory)


def sync_job_records(item: RunningJob, run_directory: Path) -> None:
    """Append new AIPerf raw records to the job's lean JSONL files."""
    if item.job.driver != "aiperf":
        return

    job_directory = job_artifact_directory(run_directory, item.job.job_id)
    if not job_directory.is_dir():
        return

    prompts_path = job_directory / PROMPTS_FILENAME
    responses_path = job_directory / RESPONSES_FILENAME
    if not item.artifact_ids_loaded:
        item.recorded_prompt_ids = load_recorded_request_ids(prompts_path)
        item.recorded_response_ids = load_recorded_request_ids(responses_path)
        item.artifact_ids_loaded = True

    final_raw_path = job_directory / f"{item.job.job_id}_raw.jsonl"
    raw_record_paths = sorted(
        (job_directory / "raw_records").glob("raw_records_*.jsonl")
    )
    for raw_path in [final_raw_path, *raw_record_paths]:
        sync_raw_record_file(
            raw_path,
            prompts_path,
            responses_path,
            item,
        )


def load_recorded_request_ids(path: Path) -> set[str]:
    """Load request IDs already written before a resumed launch."""
    request_ids: set[str] = set()
    if not path.is_file():
        return request_ids

    with path.open(encoding="utf-8") as records:
        for line_number, line in enumerate(records, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning(
                    "[artifacts] ignored invalid JSON in %s line=%d",
                    path,
                    line_number,
                )
                continue
            if isinstance(record, dict):
                request_id = record.get("request_id")
                if isinstance(request_id, str) and request_id:
                    request_ids.add(request_id)
    return request_ids


def sync_raw_record_file(
    raw_path: Path,
    prompts_path: Path,
    responses_path: Path,
    item: RunningJob,
) -> None:
    """Read complete, unseen lines from one AIPerf raw JSONL file."""
    try:
        raw_size = raw_path.stat().st_size
    except FileNotFoundError:
        return

    offset = item.raw_record_offsets.get(raw_path, 0)
    if raw_size < offset:
        offset = 0

    try:
        with raw_path.open("rb") as raw_records:
            raw_records.seek(offset)
            while True:
                line_start = raw_records.tell()
                line = raw_records.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    raw_records.seek(line_start)
                    break

                item.raw_record_offsets[raw_path] = raw_records.tell()
                try:
                    raw_record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    LOGGER.warning(
                        "[artifacts] ignored invalid raw record in %s",
                        raw_path,
                    )
                    continue
                append_job_record(
                    raw_record,
                    prompts_path,
                    responses_path,
                    item,
                )
    except FileNotFoundError:
        return


def append_job_record(
    value: object,
    prompts_path: Path,
    responses_path: Path,
    item: RunningJob,
) -> None:
    """Append one paired prompt and response for an AIPerf job."""
    if not isinstance(value, dict):
        return
    raw_record = cast(dict[str, object], value)
    metadata_value = raw_record.get("metadata")
    if not isinstance(metadata_value, dict):
        return
    metadata = cast(dict[str, object], metadata_value)
    request_id = metadata.get("x_request_id")
    if not isinstance(request_id, str) or not request_id:
        LOGGER.warning("[artifacts] raw record has no request ID")
        return

    if request_id not in item.recorded_prompt_ids:
        prompt_record = {
            "request_id": request_id,
            "sent_at": timestamp_from_nanoseconds(metadata.get("request_start_ns")),
            "request": raw_record.get("payload"),
        }
        append_jsonl(prompts_path, prompt_record)
        item.recorded_prompt_ids.add(request_id)

    if request_id not in item.recorded_response_ids:
        response_record = {
            "request_id": request_id,
            "received_at": timestamp_from_nanoseconds(metadata.get("request_end_ns")),
            "status_code": raw_record.get("status"),
            "response": raw_record.get("responses"),
            "error": raw_record.get("error"),
        }
        append_jsonl(responses_path, response_record)
        item.recorded_response_ids.add(request_id)


def append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    """Append one compact JSON object to a JSONL artifact."""
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, separators=(",", ":")))
        output.write("\n")


def timestamp_from_nanoseconds(value: object) -> str | None:
    """Convert a wall-clock nanosecond timestamp to UTC."""
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return datetime.fromtimestamp(value / 1_000_000_000, UTC).isoformat(
        timespec="microseconds"
    )


def request_stop(item: RunningJob, now: float) -> None:
    """Ask one running process group to stop cleanly."""
    if item.process is None or item.process.poll() is not None:
        record_process_exit(item)
        return

    signal_process(item.process, signal.SIGINT)
    item.state = JobState.STOPPING
    item.stop_requested_at = now
    LOGGER.info("[stop] %s", item.job.job_id)


def record_process_exit(item: RunningJob) -> None:
    """Record an exited process as paused, completed, or failed."""
    if item.process is None or item.state in TERMINAL_STATES:
        return

    return_code = item.process.poll()
    if return_code is None:
        return

    if item.log_file is not None:
        item.log_file.close()
        item.log_file = None

    expected_stop = item.state == JobState.STOPPING
    stopped_at = utc_timestamp()
    if item.process_started_at is not None:
        item.accumulated_runtime_seconds += max(
            0.0,
            time.time() - item.process_started_at,
        )
        item.process_started_at = None
    item.process = None
    item.exit_code = return_code
    if item.pause_requested:
        item.state = JobState.PAUSED
        item.paused_at = stopped_at
    elif return_code == 0 or expected_stop:
        item.state = JobState.COMPLETED
        item.completed_at = stopped_at
    else:
        item.state = JobState.FAILED
    LOGGER.info(
        "[exit] %s state=%s code=%d",
        item.job.job_id,
        item.state,
        return_code,
    )


def stop_all(
    jobs: Iterable[RunningJob],
    shutdown_grace_seconds: float,
    *,
    paused: bool = False,
) -> None:
    """Stop all live child process groups when the controller exits."""
    running_jobs = tuple(jobs)
    previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    previous_sigterm = signal.signal(signal.SIGTERM, signal.SIG_IGN)

    try:
        now = time.time()
        for item in running_jobs:
            if item.state == JobState.RUNNING:
                item.pause_requested = paused
                request_stop(item, now)

        deadline = time.time() + shutdown_grace_seconds
        while time.time() < deadline:
            for item in running_jobs:
                record_process_exit(item)
            if all(item.state != JobState.STOPPING for item in running_jobs):
                break
            time.sleep(0.1)

        for item in running_jobs:
            if (
                item.state == JobState.STOPPING
                and item.process is not None
                and item.process.poll() is None
            ):
                signal_process(item.process, signal.SIGKILL)
                LOGGER.warning("[force-stop] %s", item.job.job_id)

        for item in running_jobs:
            if item.process is not None:
                item.process.wait()
                record_process_exit(item)
            if item.log_file is not None:
                item.log_file.close()
                item.log_file = None
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def current_runtime_seconds(item: RunningJob) -> float:
    """Return accumulated runtime including the current live process."""
    runtime_seconds = item.accumulated_runtime_seconds
    if item.process_started_at is not None:
        runtime_seconds += max(0.0, time.time() - item.process_started_at)
    return runtime_seconds


def save_benchmark_state(
    run_directory: Path,
    benchmark: Benchmark,
    run_id: str,
    status: str,
    created_at: str,
    elapsed_seconds: float,
    jobs: Mapping[str, RunningJob],
    milestones: list[dict[str, object]],
    run_index: Path,
) -> None:
    """Persist the standalone benchmark and per-job resume state."""
    state = {
        "schema_version": "benchmark-state/v1",
        "benchmark_id": benchmark.benchmark_id,
        "run_id": run_id,
        "status": status,
        "created_at": created_at,
        "updated_at": utc_timestamp(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "jobs": [
            {
                "job_id": job_id,
                "enabled": item.job.enabled,
                "kind": item.job.kind.value,
                "driver": item.job.driver,
                "config": str(item.job.config),
                "start_at_offset_minutes": item.job.start_minute,
                "stop_at_offset_minutes": item.job.stop_minute,
                "deadline_at_offset_minutes": item.job.deadline_minute,
                "hard_stop_at_offset_minutes": item.job.hard_stop_minute,
                "submission": submission_state(item.job.submission),
                "optimization_goal": (
                    None
                    if item.job.optimization_goal is None
                    else item.job.optimization_goal.value
                ),
                "state": item.state.value,
                "pid": None if item.process is None else item.process.pid,
                "launch_count": item.launch_count,
                "first_started_at": item.first_started_at,
                "last_started_at": item.last_started_at,
                "paused_at": item.paused_at,
                "resumed_at": item.resumed_at,
                "completed_at": item.completed_at,
                "runtime_seconds": round(current_runtime_seconds(item), 3),
                "exit_code": item.exit_code,
                "deadline_missed": item.deadline_missed,
                "tandemn_job_id": item.tandemn_job_id,
                "endpoint_url": item.endpoint_url,
            }
            for job_id, item in jobs.items()
        ],
        "milestones": milestones,
    }
    state_path = run_directory / BENCHMARK_STATE_FILENAME
    temporary_path = state_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(state_path)
    save_run_index(
        run_index,
        benchmark.benchmark_id,
        run_id,
        status,
        created_at,
        elapsed_seconds,
        state_path,
        jobs,
        milestones,
    )


def save_run_index(
    run_index: Path,
    benchmark_id: str,
    run_id: str,
    status: str,
    created_at: str,
    elapsed_seconds: float,
    state_path: Path,
    jobs: Mapping[str, RunningJob],
    milestones: list[dict[str, object]],
) -> None:
    """Update the lightweight index of runs for one benchmark."""
    run_index.parent.mkdir(parents=True, exist_ok=True)
    if run_index.is_file():
        document = dict(
            mapping(
                json.loads(run_index.read_text(encoding="utf-8")),
                "run index",
            )
        )
        if document.get("schema_version") != "benchmark-runs/v1":
            raise ValueError(f"Unsupported run index: {run_index}")
        if document.get("benchmark_id") != benchmark_id:
            raise ValueError(f"Run index belongs to another benchmark: {run_index}")
        runs = dict(mapping(document.get("runs"), "run index runs"))
    else:
        document = {
            "schema_version": "benchmark-runs/v1",
            "benchmark_id": benchmark_id,
        }
        runs = {}

    runs[run_id] = {
        "status": status,
        "created_at": created_at,
        "updated_at": utc_timestamp(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "state_file": str(state_path),
        "jobs": {job_id: item.state.value for job_id, item in jobs.items()},
        "latest_milestone": milestones[-1] if milestones else None,
    }
    document["runs"] = runs

    temporary_path = run_index.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(document, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(run_index)


def restore_benchmark_state(
    run_directory: Path,
    benchmark: Benchmark,
    jobs: Mapping[str, RunningJob],
) -> tuple[float, str, list[dict[str, object]]]:
    """Restore completed jobs and requeue jobs paused mid-window."""
    state_path = run_directory / BENCHMARK_STATE_FILENAME
    if not state_path.is_file():
        raise ValueError(f"Benchmark state not found: {state_path}")

    document = mapping(
        json.loads(state_path.read_text(encoding="utf-8")),
        "benchmark state",
    )
    if document.get("schema_version") != "benchmark-state/v1":
        raise ValueError("Unsupported benchmark-state schema_version")
    if document.get("benchmark_id") != benchmark.benchmark_id:
        raise ValueError("State belongs to a different benchmark")
    if document.get("status") == "completed":
        raise ValueError(f"Run {run_directory.name} is already completed")

    saved_jobs_value = document.get("jobs")
    if not isinstance(saved_jobs_value, list):
        raise ValueError("benchmark state jobs must be an array")
    saved_jobs: dict[str, Mapping[str, object]] = {}
    for value in saved_jobs_value:
        definition = mapping(value, "benchmark state job")
        job_id = text(definition.get("job_id"), "benchmark state job_id")
        if job_id in saved_jobs:
            raise ValueError(f"Duplicate job in benchmark state: {job_id}")
        saved_jobs[job_id] = definition
    if set(saved_jobs) != set(jobs):
        raise ValueError("State jobs do not match the benchmark overview")

    for job_id, item in jobs.items():
        definition = saved_jobs[job_id]
        validate_saved_job(item.job, definition)
        saved_state = JobState(
            text(definition.get("state"), f"state.jobs.{job_id}.state")
        )
        if saved_state in {
            JobState.RUNNING,
            JobState.STOPPING,
            JobState.PAUSED,
        }:
            item.state = JobState.PENDING
        else:
            item.state = saved_state

        item.launch_count = whole_number(
            definition.get("launch_count", 0),
            f"state.jobs.{job_id}.launch_count",
        )
        item.first_started_at = optional_text(
            definition.get("first_started_at"),
            f"state.jobs.{job_id}.first_started_at",
        )
        item.last_started_at = optional_text(
            definition.get("last_started_at"),
            f"state.jobs.{job_id}.last_started_at",
        )
        item.paused_at = optional_text(
            definition.get("paused_at"),
            f"state.jobs.{job_id}.paused_at",
        )
        item.resumed_at = optional_text(
            definition.get("resumed_at"),
            f"state.jobs.{job_id}.resumed_at",
        )
        item.completed_at = optional_text(
            definition.get("completed_at"),
            f"state.jobs.{job_id}.completed_at",
        )
        item.accumulated_runtime_seconds = nonnegative_number(
            definition.get("runtime_seconds", 0),
            f"state.jobs.{job_id}.runtime_seconds",
        )
        item.exit_code = optional_integer(
            definition.get("exit_code"),
            f"state.jobs.{job_id}.exit_code",
        )
        item.deadline_missed = boolean(
            definition.get("deadline_missed", False),
            f"state.jobs.{job_id}.deadline_missed",
        )
        item.tandemn_job_id = optional_text(
            definition.get("tandemn_job_id"),
            f"state.jobs.{job_id}.tandemn_job_id",
        )
        item.endpoint_url = optional_text(
            definition.get("endpoint_url"),
            f"state.jobs.{job_id}.endpoint_url",
        )
        if (item.tandemn_job_id is None) != (item.endpoint_url is None):
            raise ValueError(
                f"State submission is incomplete for benchmark job {job_id}"
            )

    milestone_value = document.get("milestones", [])
    if not isinstance(milestone_value, list):
        raise ValueError("benchmark state milestones must be a list")
    milestones = [
        dict(mapping(value, "benchmark milestone")) for value in milestone_value
    ]
    elapsed_seconds = nonnegative_number(
        document.get("elapsed_seconds"),
        "benchmark state elapsed_seconds",
    )
    created_at = text(document.get("created_at"), "benchmark state created_at")
    return elapsed_seconds, created_at, milestones


def validate_saved_job(
    job: Job,
    definition: Mapping[str, object],
) -> None:
    """Prevent resuming state against a changed benchmark overview."""
    expected = {
        "enabled": job.enabled,
        "kind": job.kind.value,
        "driver": job.driver,
        "config": str(job.config),
        "start_at_offset_minutes": job.start_minute,
        "stop_at_offset_minutes": job.stop_minute,
        "deadline_at_offset_minutes": job.deadline_minute,
        "hard_stop_at_offset_minutes": job.hard_stop_minute,
        "submission": submission_state(job.submission),
        "optimization_goal": (
            None if job.optimization_goal is None else job.optimization_goal.value
        ),
    }
    for field_name, expected_value in expected.items():
        if definition.get(field_name) != expected_value:
            raise ValueError(f"Benchmark changed for {job.job_id}: {field_name}")


def submission_state(submission: SubmissionRequest | None) -> dict[str, object] | None:
    """Return the source-controlled submission fields saved for resume checks."""
    if submission is None:
        return None
    return {
        "user_id": submission.user_id,
        "spec_json": dict(submission.spec_json),
    }


def record_job_milestones(
    milestones: list[dict[str, object]],
    item: RunningJob,
    previous_state: JobState,
    previous_launch_count: int,
    elapsed_seconds: float,
) -> None:
    """Record durable job transitions for inspection after a run."""
    if item.launch_count > previous_launch_count:
        add_milestone(
            milestones,
            "job_started",
            elapsed_seconds,
            item.job.job_id,
        )
    if item.state == previous_state or item.state == JobState.RUNNING:
        return

    add_milestone(
        milestones,
        f"job_{item.state.value}",
        elapsed_seconds,
        item.job.job_id,
    )


def record_shutdown_milestones(
    milestones: list[dict[str, object]],
    jobs: Mapping[str, RunningJob],
    previous_states: Mapping[str, JobState],
    elapsed_seconds: float,
) -> None:
    """Record jobs whose processes were stopped with the controller."""
    for job_id, item in jobs.items():
        if item.state != previous_states[job_id]:
            add_milestone(
                milestones,
                f"job_{item.state.value}",
                elapsed_seconds,
                job_id,
            )


def add_milestone(
    milestones: list[dict[str, object]],
    event: str,
    elapsed_seconds: float,
    job_id: str | None = None,
) -> None:
    """Append one timestamped benchmark or job event."""
    milestone: dict[str, object] = {
        "timestamp": utc_timestamp(),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "event": event,
    }
    if job_id is not None:
        milestone["job_id"] = job_id
    milestones.append(milestone)


def utc_timestamp() -> str:
    """Return one readable UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def interrupt_controller(signum: int, frame: FrameType | None) -> None:
    """Route SIGTERM through the same cleanup path as Ctrl+C."""
    del signum, frame
    raise KeyboardInterrupt


def materialize_runtime_config(
    job: Job,
    artifact_directory: Path,
    endpoint_url: str | None,
) -> Path:
    """Create the generator config that owns one job's runtime endpoint."""
    if endpoint_url is None:
        raise SubmissionError("Tandemn submission returned no endpoint URL")
    if job.driver != "aiperf":
        return job.config

    document = dict(
        mapping(
            yaml.safe_load(job.config.read_text(encoding="utf-8")),
            f"AIPerf config for {job.job_id}",
        )
    )
    document.pop("job", None)
    benchmark_document = cast(
        dict[str, object],
        mapping(document.get("benchmark"), f"AIPerf benchmark for {job.job_id}"),
    )
    endpoint_document = cast(
        dict[str, object],
        mapping(
            benchmark_document.get("endpoint"),
            f"AIPerf endpoint for {job.job_id}",
        ),
    )
    endpoint_document["url"] = endpoint_url
    endpoint_document.pop("path", None)

    runtime_config = artifact_directory / RUNTIME_AIPERF_FILENAME
    runtime_config.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return runtime_config


def job_command(
    job: Job,
    artifact_directory: Path,
    *,
    config: Path | None = None,
    endpoint_url: str | None = None,
    aiperf_export_level: str = "raw",
) -> tuple[str, ...]:
    """Build the process command declared by a job's driver."""
    active_config = job.config if config is None else config
    if job.driver == "aiperf":
        if aiperf_export_level not in {"summary", "records", "raw"}:
            raise ValueError(f"Unsupported AIPerf export level: {aiperf_export_level}")
        return (
            "aiperf",
            "profile",
            "--config",
            str(active_config),
            "--export-level",
            aiperf_export_level,
            "--output-artifact-dir",
            str(artifact_directory),
            "--profile-export-prefix",
            job.job_id,
        )

    if job.command is None:
        raise ValueError(f"{job.job_id} requires a batch sender command")

    replacements = {
        "{artifact_dir}": str(artifact_directory),
        "{config}": str(active_config),
        "{job_id}": job.job_id,
    }
    if endpoint_url is not None:
        replacements["{endpoint_url}"] = endpoint_url
    command = []
    for argument in job.command:
        for marker, replacement in replacements.items():
            argument = argument.replace(marker, replacement)
        command.append(argument)
    return tuple(command)


def signal_process(
    process: subprocess.Popen[bytes],
    requested_signal: signal.Signals,
) -> None:
    """Signal the whole job process group."""
    try:
        if os.name == "posix":
            os.killpg(process.pid, requested_signal)
        else:
            process.send_signal(requested_signal)
    except ProcessLookupError:
        return


def resolve_benchmark_overview(value: str) -> Path:
    """Resolve a benchmark name or an explicit overview YAML path."""
    supplied_path = Path(value)
    if supplied_path.is_file():
        return supplied_path.resolve()
    return (BENCHMARKS_ROOT / value / "overview.yaml").resolve()


def load_benchmark(path: Path) -> Benchmark:
    """Read one benchmark overview and resolve every job YAML."""
    overview_path = path.resolve()
    if not overview_path.is_file():
        raise ValueError(f"Benchmark overview not found: {path}")

    document = mapping(
        yaml.safe_load(overview_path.read_text(encoding="utf-8")),
        "benchmark overview",
    )
    if document.get("schema_version") != "benchmark/v1":
        raise ValueError("Unsupported benchmark schema_version")

    benchmark_document = mapping(document.get("benchmark"), "benchmark")
    benchmark_id = text(benchmark_document.get("id"), "benchmark.id")
    poll_interval_seconds = positive_number(
        benchmark_document.get("poll_interval_seconds", 60),
        "benchmark.poll_interval_seconds",
    )
    shutdown_grace_seconds = positive_number(
        benchmark_document.get("shutdown_grace_seconds", 60),
        "benchmark.shutdown_grace_seconds",
    )
    hard_stop_minute = positive_number(
        benchmark_document.get("hard_stop_at_offset_minutes"),
        "benchmark.hard_stop_at_offset_minutes",
    )

    job_documents = mapping(document.get("jobs"), "jobs")
    jobs = tuple(
        load_job(
            job_id,
            value,
            overview_path.parent,
            hard_stop_minute,
        )
        for job_id, value in job_documents.items()
    )
    validate_jobs(jobs)
    return Benchmark(
        benchmark_id=benchmark_id,
        poll_interval_seconds=poll_interval_seconds,
        shutdown_grace_seconds=shutdown_grace_seconds,
        jobs=jobs,
    )


def load_job(
    job_id: str,
    value: object,
    benchmark_directory: Path,
    default_hard_stop_minute: float,
) -> Job:
    """Read one job entry and resolve its configuration file."""
    definition = mapping(value, f"jobs.{job_id}")
    enabled = boolean(definition.get("enabled"), f"jobs.{job_id}.enabled")
    output_to_terminal = boolean(
        definition.get("output_to_terminal", False),
        f"jobs.{job_id}.output_to_terminal",
    )
    kind = JobKind(text(definition.get("kind"), f"jobs.{job_id}.kind"))
    driver = text(definition.get("driver"), f"jobs.{job_id}.driver")
    if driver not in {"aiperf", "external_mock_data"}:
        raise ValueError(f"Unsupported driver for {job_id}: {driver}")

    config = (
        benchmark_directory / text(definition.get("config"), f"jobs.{job_id}.config")
    ).resolve()
    if not config.is_file():
        raise ValueError(f"Config not found for {job_id}: {config}")
    config_document = load_job_config_document(config, job_id)

    command_value = definition.get("command")
    command = None if command_value is None else string_list(command_value)
    start_minute = nonnegative_number(
        definition.get("start_at_offset_minutes"),
        f"jobs.{job_id}.start_at_offset_minutes",
    )
    stop_minute = optional_number(
        definition.get("stop_at_offset_minutes"),
        f"jobs.{job_id}.stop_at_offset_minutes",
    )
    deadline_minute = optional_number(
        definition.get("deadline_at_offset_minutes"),
        f"jobs.{job_id}.deadline_at_offset_minutes",
    )
    hard_stop_minute = positive_number(
        definition.get(
            "hard_stop_at_offset_minutes",
            default_hard_stop_minute,
        ),
        f"jobs.{job_id}.hard_stop_at_offset_minutes",
    )
    parent_value = definition.get("parent_job_id")
    parent_job_id = (
        None
        if parent_value is None
        else text(parent_value, f"jobs.{job_id}.parent_job_id")
    )
    workload, sender_model = load_scheduler_workload(
        config_document,
        driver,
        kind,
        job_id,
    )
    submission, optimization_goal = load_job_submission(
        config_document,
        sender_model,
        job_id,
        kind,
    )

    if kind == JobKind.ONLINE and (stop_minute is None or stop_minute <= start_minute):
        raise ValueError(f"{job_id} requires a stop after its start")
    if kind == JobKind.BATCH and (
        deadline_minute is None or deadline_minute < start_minute
    ):
        raise ValueError(f"{job_id} requires a deadline")
    if hard_stop_minute <= start_minute:
        raise ValueError(f"{job_id} hard stop must be after its start")
    return Job(
        job_id=job_id,
        enabled=enabled,
        output_to_terminal=output_to_terminal,
        kind=kind,
        driver=driver,
        config=config,
        command=command,
        start_minute=start_minute,
        stop_minute=stop_minute,
        deadline_minute=deadline_minute,
        hard_stop_minute=hard_stop_minute,
        parent_job_id=parent_job_id,
        submission=submission,
        optimization_goal=optimization_goal,
        workload=workload,
    )


def load_scheduler_workload(
    config_document: Mapping[str, object] | None,
    driver: str,
    kind: JobKind,
    job_id: str,
) -> tuple[SchedulerWorkload | None, str | None]:
    """Read the actual sender rate and token shape used for scheduling."""
    if config_document is None:
        return None, None
    if kind == JobKind.BATCH:
        dataset = mapping(
            config_document.get("dataset"),
            f"batch dataset for {job_id}",
        )
        io = mapping(dataset.get("io"), f"batch io for {job_id}")
        input_tokens = token_mean(
            io.get("input_tokens"),
            f"batch input tokens for {job_id}",
        )
        output_tokens = token_mean(
            io.get("output_tokens"),
            f"batch output tokens for {job_id}",
        )
        record_count = positive_number(
            dataset.get("record_count"),
            f"batch record count for {job_id}",
        )
        job_definition = mapping(
            config_document.get("job"),
            f"job config for {job_id}",
        )
        features = mapping(
            job_definition.get("features"),
            f"job features for {job_id}",
        )
        deadline_hours = positive_number(
            features.get("deadline_hrs"),
            f"batch deadline for {job_id}",
        )
        return (
            SchedulerWorkload(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_rate=record_count / (deadline_hours * 60 * 60),
            ),
            text(dataset.get("model"), f"batch model for {job_id}"),
        )
    if driver != "aiperf":
        return None, None
    benchmark_document = mapping(
        config_document.get("benchmark"),
        f"AIPerf benchmark for {job_id}",
    )
    sender_model = text(
        benchmark_document.get("model"),
        f"AIPerf benchmark model for {job_id}",
    )
    dataset = mapping(
        benchmark_document.get("dataset"),
        f"AIPerf dataset for {job_id}",
    )
    input_tokens, output_tokens = aiperf_token_means(dataset, job_id)
    phases_value = benchmark_document.get("phases")
    if not isinstance(phases_value, list) or not phases_value:
        raise ValueError(f"AIPerf phases for {job_id} must be a non-empty array")
    phases = [mapping(value, f"AIPerf phase for {job_id}") for value in phases_value]
    request_rate = max(
        positive_number(
            phase.get("rate"),
            f"AIPerf phase rate for {job_id}",
        )
        for phase in phases
    )
    concurrency_values = [
        whole_number(
            phase.get("concurrency"),
            f"AIPerf phase concurrency for {job_id}",
        )
        for phase in phases
        if phase.get("concurrency") is not None
    ]
    max_concurrency = max(concurrency_values) if concurrency_values else None
    return (
        SchedulerWorkload(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_rate=request_rate,
            max_concurrency=max_concurrency,
        ),
        sender_model,
    )


def aiperf_token_means(
    dataset: Mapping[str, object],
    job_id: str,
) -> tuple[float, float]:
    """Return mean input and output tokens from an AIPerf dataset."""
    prompts_value = dataset.get("prompts")
    if isinstance(prompts_value, Mapping):
        prompts = cast(Mapping[str, object], prompts_value)
        return (
            token_mean(prompts.get("isl"), f"AIPerf ISL for {job_id}"),
            token_mean(prompts.get("osl"), f"AIPerf OSL for {job_id}"),
        )

    rankings_value = dataset.get("rankings")
    if isinstance(rankings_value, Mapping):
        rankings = cast(Mapping[str, object], rankings_value)
        passages = token_mean(
            rankings.get("passages"),
            f"AIPerf passages for {job_id}",
        )
        passage_tokens = token_mean(
            rankings.get("passageTokens"),
            f"AIPerf passage tokens for {job_id}",
        )
        query_tokens = token_mean(
            rankings.get("queryTokens"),
            f"AIPerf query tokens for {job_id}",
        )
        return query_tokens + passages * passage_tokens, 1.0
    raise ValueError(f"AIPerf dataset for {job_id} has no supported token shape")


def token_mean(value: object, name: str) -> float:
    """Read a scalar or distribution mean token count."""
    if isinstance(value, Mapping):
        value = value.get("mean", value.get("average"))
    return nonnegative_number(value, name)


def load_job_config_document(
    config: Path,
    job_id: str,
) -> Mapping[str, object] | None:
    """Load a YAML job file while allowing non-YAML support processes."""
    if config.suffix != ".yaml":
        return None
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    if document is None:
        return None
    return mapping(
        document,
        f"job config for {job_id}",
    )


def load_job_submission(
    config_document: Mapping[str, object] | None,
    sender_model: str | None,
    job_id: str,
    kind: JobKind,
) -> tuple[SubmissionRequest | None, OptimizationGoal | None]:
    """Load scheduler policy and submission data from one job YAML."""
    if config_document is None or config_document.get("job") is None:
        return None, None
    definition = mapping(config_document.get("job"), f"job config for {job_id}")
    user_id = text(
        definition.get("user_id"),
        f"job config for {job_id}.job.user_id",
    )
    model_id = sender_model or text(
        definition.get("model_id"),
        f"job config for {job_id}.job.model_id",
    )
    features = dict(
        mapping(
            definition.get("features"),
            f"job config for {job_id}.job.features",
        )
    )
    optimization_goal = OptimizationGoal(
        text(
            definition.get("optimization_goal"),
            f"job config for {job_id}.job.optimization_goal",
        )
    )
    return (
        SubmissionRequest(
            user_id=user_id,
            kind=kind.value,
            spec_json={"model_id": model_id, "job_features": features},
        ),
        optimization_goal,
    )


def validate_jobs(jobs: tuple[Job, ...]) -> None:
    """Check additive jobs against their parent job windows."""
    jobs_by_id = {job.job_id: job for job in jobs}
    for job in jobs:
        if job.parent_job_id is None:
            continue
        parent = jobs_by_id.get(job.parent_job_id)
        if parent is None:
            raise ValueError(f"Parent not found for {job.job_id}")
        if parent.start_minute > job.start_minute:
            raise ValueError(f"{job.job_id} starts before its parent")
        if (
            parent.stop_minute is not None
            and job.stop_minute is not None
            and parent.stop_minute < job.stop_minute
        ):
            raise ValueError(f"{job.job_id} stops after its parent")


def validate_runnable_jobs(benchmark: Benchmark) -> None:
    """Require a sender command for every enabled external job."""
    for job in benchmark.jobs:
        if job.enabled and job.driver == "external_mock_data" and job.command is None:
            raise ValueError(f"{job.job_id} requires a batch sender command")


def log_benchmark(benchmark: Benchmark) -> None:
    """Print the benchmark schedule in job-start order."""
    LOGGER.info(
        "[benchmark] id=%s poll=%.0fs",
        benchmark.benchmark_id,
        benchmark.poll_interval_seconds,
    )
    for job in sorted(
        benchmark.jobs,
        key=lambda item: (item.start_minute, item.job_id),
    ):
        end_minute = (
            job.stop_minute if job.kind == JobKind.ONLINE else job.deadline_minute
        )
        LOGGER.info(
            "[job] %s enabled=%s kind=%s start=%.0fm end=%.0fm driver=%s",
            job.job_id,
            job.enabled,
            job.kind,
            job.start_minute,
            0.0 if end_minute is None else end_minute,
            job.driver,
        )


def log_summary(summary: RunSummary) -> None:
    """Print the final job counts for one run."""
    LOGGER.info(
        "[summary] completed=%d failed=%d missed=%d paused=%d disabled=%d late=%d",
        sum(state == JobState.COMPLETED for state in summary.states.values()),
        sum(state == JobState.FAILED for state in summary.states.values()),
        sum(state == JobState.MISSED for state in summary.states.values()),
        sum(state == JobState.PAUSED for state in summary.states.values()),
        sum(state == JobState.DISABLED for state in summary.states.values()),
        len(summary.deadline_misses),
    )


def mapping(value: object, name: str) -> Mapping[str, object]:
    """Require a YAML mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, object], value)


def text(value: object, name: str) -> str:
    """Require a non-empty string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def boolean(value: object, name: str) -> bool:
    """Require a boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def optional_text(value: object, name: str) -> str | None:
    """Read an optional non-empty string."""
    return None if value is None else text(value, name)


def optional_integer(value: object, name: str) -> int | None:
    """Read an optional integer."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def whole_number(value: object, name: str) -> int:
    """Require a nonnegative integer."""
    number = optional_integer(value, name)
    if number is None or number < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return number


def positive_number(value: object, name: str) -> float:
    """Require a positive number."""
    number = number_value(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return number


def nonnegative_number(value: object, name: str) -> float:
    """Require a nonnegative number."""
    number = number_value(value, name)
    if number < 0:
        raise ValueError(f"{name} must be zero or greater")
    return number


def optional_number(value: object, name: str) -> float | None:
    """Read an optional number."""
    return None if value is None else number_value(value, name)


def number_value(value: object, name: str) -> float:
    """Require a number that is not a boolean."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    return float(value)


def string_list(value: object) -> tuple[str, ...]:
    """Require a command argument list."""
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError("command must be a list of non-empty strings")
    return tuple(cast(list[str], value))


if __name__ == "__main__":
    main()
