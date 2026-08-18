"""Run cluster-simulation control ticks in logical time."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from math import ceil
from pathlib import Path
from time import perf_counter, time

from koi_trace import compile_koi_events
from models import BatchJob, BatchProgress, OnlineJob, PlannerResult, Request, Scenario
from planning import PlannerDriver
from routing import StickyRoundRobinRouter
from scenario import build_batch_probe, materialize_online_requests, requests_in_tick
from serving import (
    BatchProbeMeasurement,
    RankRun,
    SERVING_WINDOW_SECONDS,
    batch_probe_measurements,
    build_rank_deployments,
    commit_tick_metrics,
    completed_request_summary,
    gpu_metrics_from_results,
    merge_hardware_catalogs,
    profile_coverage_error,
    prepare_rank_run,
    resolve_rank_hardware,
    run_ranks,
)
from telemetry import (
    OUTCOME_METRICS,
    aggregate_job_metrics,
    cluster_accounting,
    collect_rank_telemetry,
    pool_telemetry,
)
from trace import (
    append_event,
    cluster_window,
    job_admitted,
    job_finished,
    job_state_changed,
    job_submitted,
    job_window,
    gpu_sample,
    planner_finished,
    pool_window,
    placement_observed,
    rank_failed,
    rank_window,
    resource_snapshot,
    run_finished,
    run_started,
    tick_started,
    trace_path,
)


@dataclass(frozen=True)
class _PreparedJob:
    running: object
    rank_runs: tuple[RankRun, ...]
    batch_progress: BatchProgress | None = None
    online_requests: tuple[Request, ...] | None = None
    source_exhausted: bool = False


@dataclass(frozen=True)
class _RankFailure:
    rank_id: str
    reason_code: str
    error_detail: str


class LogicalClock:
    """Track the simulation's UTC time without sleeping."""

    def __init__(self, start: datetime) -> None:
        """Initialize the clock at the supplied UTC start time."""
        if start.tzinfo is None:
            raise ValueError("logical clock start must include a timezone")
        self.start = start
        self._now = start

    def now(self) -> datetime:
        """Return the current logical UTC time."""
        return self._now

    def advance(self, duration: timedelta) -> None:
        """Move the logical clock forward by one positive duration."""
        if duration <= timedelta():
            raise ValueError("logical clock duration must be positive")
        self._now += duration


class ClusterSimulator:
    """Coordinate lifecycle events, planning, rollouts, and telemetry per tick."""

    def __init__(
        self,
        scenario: Scenario,
        user_id: str,
        postgres: object,
        planner: PlannerDriver,
        clock: LogicalClock,
        llmservingsim_root: Path | None = None,
        container: str = "servingsim_docker",
        run_name: str = "cluster-sim",
        parallel_simulators: int = 5,
        rank_timeout_seconds: float = 330,
        trace_root: Path | None = None,
        trace_run_id: str | None = None,
    ) -> None:
        """Create the simulator after its external dependencies are configured."""
        from tandemn_system_data.clients import JobStore

        verify_store_user(user_id, postgres)
        self.scenario = scenario
        self.user_id = user_id
        self.postgres = postgres
        self.planner = planner
        self.clock = clock
        self.jobs = JobStore(postgres)
        self.llmservingsim_root = llmservingsim_root
        self.container = container
        self.run_name = run_name
        self.parallel_simulators = parallel_simulators
        self.rank_timeout_seconds = rank_timeout_seconds
        self.trace_file: Path | None = None
        if trace_root is not None and trace_run_id is not None:
            self.trace_file = trace_path(
                trace_root,
                "simulation",
                scenario.benchmark_name,
                planner.method_name,
                trace_run_id,
            )
            append_event(
                self.trace_file,
                run_started(
                    trace_run_id,
                    "simulation",
                    scenario.benchmark_name,
                    planner.method_name,
                    clock.now().isoformat(),
                    scenario.tick_duration_ns / 1_000_000_000,
                ),
            )
        self._submitted_jobs: set[str] = set()
        self._trace_job_states: dict[str, str] = {}
        self._admitted_jobs: set[str] = set()
        self._admission_latencies_ms: list[float] = []
        self._admissions_in_tick = 0
        self._traffic_starts: dict[str, int] = {}
        self._requests = {
            job.store_job_id: materialize_online_requests(job) for job in scenario.online_jobs
        }
        self._online_jobs = {job.store_job_id: job for job in scenario.online_jobs}
        self._online_backlog = {job.store_job_id: () for job in scenario.online_jobs}
        self._batch_jobs = {job.store_job_id: job for job in scenario.batch_jobs}
        self._batch_progress = {
            job.store_job_id: BatchProgress(total_requests=int(job.workload["record_count"]))
            for job in scenario.batch_jobs
        }
        self._router = StickyRoundRobinRouter()

    def run(self, max_ticks: int | None = None) -> tuple[PlannerResult, ...]:
        """Run logical ticks through the requested limit or drain boundary."""
        if max_ticks is not None and max_ticks < 1:
            raise ValueError("max ticks must be positive")
        results = []
        tick_number = 0
        while (
            self._offset_ns(self.clock.now()) < self.scenario.drain_end_ns
            and (max_ticks is None or tick_number < max_ticks)
        ):
            results.append(self.run_tick(tick_number))
            tick_number += 1
        reached_drain = self._offset_ns(self.clock.now()) >= self.scenario.drain_end_ns
        termination_reason = (
            "SIMULATION_HARD_STOP" if reached_drain else "SIMULATION_TICK_LIMIT"
        )
        self._hard_stop(termination_reason)
        trace_file = getattr(self, "trace_file", None)
        if trace_file is not None:
            append_event(
                trace_file,
                run_finished(
                    self.clock.now().isoformat(),
                    termination_reason,
                    termination_reason == "SIMULATION_TICK_LIMIT",
                ),
            )
        return tuple(results)

    def run_tick(self, tick_number: int) -> PlannerResult:
        """Submit due jobs, apply one plan, and advance one logical tick."""
        if tick_number < 0:
            raise ValueError("tick number must be nonnegative")
        print(f"[tick {tick_number}] Planning placements", flush=True)
        start = self.clock.now()
        start_offset_ns = self._offset_ns(start)
        end_offset_ns = start_offset_ns + self.scenario.tick_duration_ns
        trace_file = getattr(self, "trace_file", None)
        if trace_file is not None:
            append_event(trace_file, tick_started(tick_number, start.isoformat()))
            from tandemn_system_data.clients import ResourceMapStore

            resource_map = ResourceMapStore(self.postgres, user_id=self.user_id).get()
            pools = []
            for env_key, summary in resource_map.scheduling_summary().items():
                for pool in summary["pools"]:
                    pools.append(
                        {
                            "pool_id": f"{env_key}|{pool['fabric_id']}|{pool['instance_type']}",
                            "instance_type": pool["instance_type"],
                            "instance_count": pool["total_instances"],
                            "gpus_per_instance": pool["gpus_per_instance"],
                            "gpu_count": pool["total"],
                            "price_per_instance_hour": pool["price_per_instance_hour"],
                        }
                    )
            append_event(trace_file, resource_snapshot(tick_number, pools))

        self._submit_jobs_before_or_at(start_offset_ns)
        self._admissions_in_tick = 0
        decision_started = perf_counter()
        result = self.planner.create_plans(tick_number, start)
        decision_time_ms = (perf_counter() - decision_started) * 1_000
        if trace_file is not None:
            debug_logger = getattr(self.planner, "debug_logger", None)
            events_path = getattr(debug_logger, "events_path", None)
            if events_path is not None:
                compiled_koi_events = compile_koi_events(Path(events_path))
                event_count = getattr(self, "_koi_event_count", 0)
                for event in compiled_koi_events[event_count:]:
                    append_event(trace_file, event)
                self._koi_event_count = len(compiled_koi_events)
            from tandemn_system_data.clients import PlanStore

            plans = []
            for created in result.plans:
                plan = PlanStore(self.postgres).get(created.plan_id)
                if plan is None:
                    continue
                plans.append(
                    {
                        "plan_id": plan.plan_id,
                        "created_at": plan.created_at.isoformat(),
                        "logical_created_at": start.isoformat(),
                        "status": plan.status,
                        "actions": [
                            action.model_dump(mode="json") for action in plan.actions
                        ],
                    }
                )
            planned_job_ids = tuple(
                action["job_id"] for plan in plans for action in plan["actions"]
            )
            append_event(
                trace_file,
                planner_finished(
                    tick_number,
                    start.isoformat(),
                    start.isoformat(),
                    decision_time_ms,
                    plans,
                    planned_job_ids,
                    None,
                    result.error,
                ),
            )
            for plan in plans:
                for action in plan["actions"]:
                    job_id = action["job_id"]
                    if action["type"] != "place" or job_id in self._admitted_jobs:
                        continue
                    job = self.jobs.get(job_id)
                    if job is None:
                        continue
                    append_event(
                        trace_file,
                        job_admitted(
                            job_id,
                            plan["plan_id"],
                            job.created_at.isoformat(),
                            start.isoformat(),
                            "logical_place_plan_created",
                        ),
                    )
                    self._admitted_jobs.add(job_id)
                    self._admission_latencies_ms.append(
                        (start - job.created_at).total_seconds() * 1_000
                    )
                    self._admissions_in_tick += 1
        apply_pending_plans(self.user_id, self.postgres)
        if trace_file is not None:
            for running in self.jobs.running_jobs(self.user_id):
                self._trace_job_state(
                    running.job.job_id, "RUNNING", start, "ranks_materialized")
                for rank in running.ranks:
                    append_event(
                        trace_file,
                        placement_observed(
                            tick_number,
                            rank.plan_id,
                            running.job.job_id,
                            rank.rank_id,
                            rank.n_replicas,
                            rank.shape_json,
                        ),
                    )
        print(f"[tick {tick_number}] Preparing ServingSim inputs", flush=True)
        self._simulate_tick(tick_number, start_offset_ns, end_offset_ns)
        self._submit_jobs_before(end_offset_ns)
        self.clock.advance(timedelta(microseconds=self.scenario.tick_duration_ns // 1_000))
        return result

    def _submit_jobs_before_or_at(self, offset_ns: int) -> None:
        """Submit jobs released exactly at a control boundary before planning."""
        for job in self._jobs_released_before(offset_ns + 1):
            self._submit(job)

    def _submit_jobs_before(self, offset_ns: int) -> None:
        """Submit jobs released inside the current interval for the next tick."""
        for job in self._jobs_released_before(offset_ns):
            self._submit(job)

    def _jobs_released_before(self, offset_ns: int) -> tuple[OnlineJob | BatchJob, ...]:
        """Return unsubmitted jobs with benchmark releases before one offset."""
        jobs = (*self.scenario.online_jobs, *self.scenario.batch_jobs)
        return tuple(
            job
            for job in jobs
            if job.name not in self._submitted_jobs and self._release_ns(job) < offset_ns
        )

    def _submit(self, job: OnlineJob | BatchJob) -> None:
        """Persist one waiting job with its exact logical benchmark release time."""
        from tandemn_system_data.clients import ModelCatalogStore

        submitted_at = self.clock.start + timedelta(microseconds=self._release_ns(job) // 1_000)
        catalog = ModelCatalogStore(self.postgres).get(job.model_name)
        submit_waiting_job(
            job,
            self.user_id,
            self.jobs,
            submitted_at,
            prefix_cache_enabled=bool(catalog and catalog.prefix_cache_enabled),
        )
        trace_file = getattr(self, "trace_file", None)
        if trace_file is not None:
            features = job.workload["features"]
            deadline_at = None
            if isinstance(job, BatchJob):
                deadline_at = (
                    self.clock.start
                    + timedelta(microseconds=job.deadline_at_ns // 1_000)
                ).isoformat()
            append_event(
                trace_file,
                job_submitted(
                    job.store_job_id,
                    "online" if isinstance(job, OnlineJob) else "batch",
                    submitted_at.isoformat(),
                    features.get("tenant_id") or self.user_id,
                    features.get("target_p99_ttft_ms"),
                    features.get("target_p99_tpot_ms"),
                    deadline_at,
                ),
            )
        self._trace_job_state(job.store_job_id, "WAITING", submitted_at, "submitted")
        self._submitted_jobs.add(job.name)

    def _offset_ns(self, timestamp: datetime) -> int:
        """Return a logical timestamp's elapsed nanoseconds from the epoch."""
        return round((timestamp - self.clock.start).total_seconds() * 1_000_000_000)

    @staticmethod
    def _release_ns(job: OnlineJob | BatchJob) -> int:
        """Read the benchmark release offset for either job kind."""
        return job.starts_at_ns if isinstance(job, OnlineJob) else job.releases_at_ns

    def _simulate_tick(
        self,
        tick_number: int,
        start_offset_ns: int,
        end_offset_ns: int,
    ) -> None:
        """Run every active rank in one pool and commit the successful tick together."""
        if self.llmservingsim_root is None:
            return

        from tandemn_system_data.clients import (
            GpuMetricStore,
            HardwareCatalogStore,
            ResourceMapStore,
        )

        catalog_store = HardwareCatalogStore(self.postgres)
        all_catalogs = getattr(catalog_store, "all", lambda: None)()
        catalogs = all_catalogs if isinstance(all_catalogs, list) else [catalog_store.get()]
        catalog = merge_hardware_catalogs([item for item in catalogs if item is not None])
        if not catalog["regions"]:
            raise ValueError("hardware catalog is required for rank simulation")
        resource_map = ResourceMapStore(self.postgres, user_id=self.user_id).get()
        prices = self._prices(resource_map)
        tick_start = self.clock.now()
        prepared_jobs: list[_PreparedJob] = []
        finished_online = []
        failed_rank_ids = set()

        running_jobs = self.jobs.running_jobs(self.user_id)
        for running in running_jobs:
            if not running.ranks:
                continue
            job = self._online_jobs.get(running.job.job_id)
            if job is None:
                continue
            traffic_start = self._traffic_starts.setdefault(running.job.job_id, start_offset_ns)
            duration = job.stops_at_ns - job.starts_at_ns
            relative_start = start_offset_ns - traffic_start
            backlog = tuple(
                replace(request, arrives_at_ns=start_offset_ns)
                for request in self._online_backlog[job.store_job_id]
            )
            if relative_start >= duration:
                if not backlog:
                    finished_online.append((running, tick_start))
                    continue
                fresh_requests = ()
                source_exhausted = True
            else:
                relative_end = min(end_offset_ns - traffic_start, duration)
                source_start = job.starts_at_ns + relative_start
                source_end = job.starts_at_ns + relative_end
                fresh_requests = tuple(
                    replace(
                        request,
                        arrives_at_ns=(
                            start_offset_ns + request.arrives_at_ns - source_start
                        ),
                    )
                    for request in requests_in_tick(
                        self._requests[job.store_job_id], source_start, source_end
                    )
                )
                source_exhausted = relative_end >= duration
            requests = (*backlog, *fresh_requests)
            rank_runs, failures = self._prepare_job_runs(
                running,
                job.name,
                requests,
                tick_number,
                start_offset_ns,
                catalog,
            )
            failed_rank_ids.update(failure.rank_id for failure in failures)
            self._fail_ranks(running, job.name, failures)
            prepared_jobs.append(
                _PreparedJob(
                    running,
                    rank_runs,
                    online_requests=requests,
                    source_exhausted=source_exhausted,
                )
            )

        for running in running_jobs:
            if not running.ranks:
                continue
            job = self._batch_jobs.get(running.job.job_id)
            if job is None:
                continue
            progress = self._batch_progress[job.store_job_id]
            probe = tuple(
                replace(request, arrives_at_ns=start_offset_ns)
                for request in build_batch_probe(job, progress)
            )
            rank_runs, failures = self._prepare_job_runs(
                running,
                job.name,
                probe,
                tick_number,
                start_offset_ns,
                catalog,
            )
            failed_rank_ids.update(failure.rank_id for failure in failures)
            self._fail_ranks(running, job.name, failures)
            prepared_jobs.append(
                _PreparedJob(
                    running,
                    rank_runs,
                    batch_progress=progress,
                )
            )

        rank_runs = tuple(
            rank_run
            for prepared in prepared_jobs
            for rank_run in prepared.rank_runs
        )
        deadline_epoch_seconds = time() + SERVING_WINDOW_SECONDS
        if rank_runs:
            print(f"[tick {tick_number}] ServingSim ranks: 0/{len(rank_runs)} complete", flush=True)
        results = (
            run_ranks(
                rank_runs,
                self.parallel_simulators,
                deadline_epoch_seconds,
                progress=lambda completed, total, rank_run: print(
                    f"[tick {tick_number}] ServingSim ranks: {completed}/{total} complete "
                    f"({rank_run.deployment.rank_name})",
                    flush=True,
                ),
            )
            if rank_runs
            else ()
        )
        metrics = []
        result_offset = 0
        backlog_updates = {}
        batch_progress_updates = {}
        finished_batch = []
        request_summaries = {}
        for prepared in prepared_jobs:
            job_results = results[
                result_offset : result_offset + len(prepared.rank_runs)
            ]
            result_offset += len(prepared.rank_runs)
            job_id = prepared.running.job.job_id
            job = self._online_jobs.get(job_id) or self._batch_jobs.get(job_id)
            result_pairs = tuple(zip(prepared.rank_runs, job_results, strict=True))
            if any(
                getattr(result, "status", "completed") == "oom"
                for _, result in result_pairs
            ):
                failed_rank_ids.update(
                    result.deployment.rank_name
                    for _, result in result_pairs
                    if getattr(result, "status", "completed") == "oom"
                )
                self._fail_oom(prepared.running, job.name, job_results)
            active_pairs = tuple(
                (rank_run, result)
                for rank_run, result in result_pairs
                if getattr(result, "status", "completed") != "oom"
            )
            active_rank_runs = tuple(rank_run for rank_run, _ in active_pairs)
            active_results = tuple(result for _, result in active_pairs)
            features = job.workload.get("features", {})
            attempted_request_count = sum(
                rank_run.request_count for rank_run in prepared.rank_runs
            )
            request_summary = completed_request_summary(
                active_results,
                features.get("target_p99_ttft_ms"),
                features.get("target_p99_tpot_ms"),
                job.workload.get("task", "generate"),
            )
            completed_request_count = int(
                request_summary["completed_request_count"]
            )
            backlog_names = {
                request.request_name for request in self._online_backlog.get(job_id, ())
            }
            request_summaries[job_id] = {
                **request_summary,
                "slo_count_scope": "completed_requests",
                "request_count_scope": "tick_requests",
                "batch_latency_diagnostic": False,
                "offered_request_count": (
                    len(prepared.online_requests) - len(self._online_backlog[job_id])
                    if prepared.online_requests is not None else None
                ),
                "admitted_request_count": (
                    sum(
                        request.request_name not in backlog_names
                        for result in active_results
                        for request in getattr(result, "admitted_requests", ())
                    ) if prepared.online_requests is not None else None
                ),
                "attempted_request_count": attempted_request_count,
                "unfinished_request_count": max(
                    0, attempted_request_count - completed_request_count
                ),
                "completion_ratio": (
                    completed_request_count / attempted_request_count
                    if attempted_request_count
                    else None
                ),
            }
            if prepared.online_requests is not None:
                metrics.extend(
                    gpu_metrics_from_results(
                        prepared.running.job,
                        tuple(rank_run.deployment for rank_run in active_rank_runs),
                        active_results,
                        tick_start,
                        self.scenario.tick_duration_ns,
                        prices,
                    )
                )
                completed_names = {
                    request.request_name
                    for result in active_results
                    for request in result.completed_requests
                }
                unfinished = tuple(
                    request
                    for request in prepared.online_requests
                    if request.request_name not in completed_names
                )
                backlog_updates[job_id] = unfinished
                if prepared.source_exhausted and not unfinished:
                    finished_online.append(
                        (
                            prepared.running,
                            tick_start
                            + timedelta(
                                microseconds=(
                                    self.scenario.tick_duration_ns // 1_000
                                )
                            ),
                        )
                    )
                continue
            progress = prepared.batch_progress
            if progress is None:
                continue
            staged_progress = replace(progress)
            measurements = batch_probe_measurements(active_results)
            projected_completed = advance_batch_progress(
                staged_progress,
                measurements,
                self.scenario.tick_duration_ns,
                start_offset_ns,
                attempted_requests=sum(
                    rank_run.request_count for rank_run in active_rank_runs
                ),
            )
            measurements_by_rank = {
                measurement.deployment.rank_name: measurement
                for measurement in measurements
            }
            active_ns = self.scenario.tick_duration_ns
            if staged_progress.completed_at_ns is not None:
                active_ns = max(0, staged_progress.completed_at_ns - start_offset_ns)
            metrics.extend(
                gpu_metrics_from_results(
                    prepared.running.job,
                    tuple(rank_run.deployment for rank_run in active_rank_runs),
                    active_results,
                    tick_start,
                    self.scenario.tick_duration_ns,
                    prices,
                    batch_measurements=measurements_by_rank,
                    batch_active_ns=active_ns,
                )
            )
            request_rate = _batch_request_rate_per_second(measurements)
            request_summaries[job_id].update(
                request_count_scope="batch_probe",
                batch_latency_diagnostic=True,
                slo_count_scope="batch_probe_diagnostic",
                probe_attempted_request_count=attempted_request_count,
                probe_completed_request_count=completed_request_count,
                probe_unfinished_request_count=max(
                    0, attempted_request_count - completed_request_count
                ),
                campaign_total_request_count=staged_progress.total_requests,
                campaign_completed_in_window=projected_completed,
                campaign_completed_request_count=staged_progress.completed_requests,
                campaign_remaining_request_count=(
                    staged_progress.total_requests - staged_progress.completed_requests
                ),
                campaign_request_per_sec=request_rate,
            )
            batch_progress_updates[job_id] = staged_progress
            if staged_progress.completed_requests == staged_progress.total_requests:
                finished_batch.append(
                    (
                        prepared.running,
                        self.clock.start
                        + timedelta(
                            microseconds=staged_progress.completed_at_ns // 1_000
                        ),
                    )
                )

        metric_store = None
        if metrics:
            metric_store = GpuMetricStore(self.postgres)
            commit_tick_metrics(metric_store, metrics)

        self._online_backlog.update(backlog_updates)
        self._batch_progress.update(batch_progress_updates)
        for running, finished_at in finished_online:
            self._finish(running, finished_at)
        for running, finished_at in finished_batch:
            self._finish(running, finished_at)

        if metrics:
            trace_file = getattr(self, "trace_file", None)
            if trace_file is not None:
                from tandemn_system_data.models.gpu_metric import gpu_metric_to_metrics

                window_end = tick_start + timedelta(
                    microseconds=self.scenario.tick_duration_ns // 1_000
                )
                rows, trajectories = collect_rank_telemetry(
                    metric_store,
                    self.user_id,
                    (metric.job_id for metric in metrics if metric.job_id is not None),
                    tick_start,
                    window_end,
                )
                for metric in rows:
                    append_event(
                        trace_file,
                        gpu_sample(
                            tick_number,
                            metric.ts.isoformat(),
                            metric.gpu_uuid,
                            metric.job_id,
                            metric.rank_id,
                            metric.chain_index,
                            metric.node_name,
                            metric.instance_type,
                            gpu_metric_to_metrics(metric),
                        ),
                    )
                chain_indices_by_rank = {}
                for metric in rows:
                    if (
                        metric.job_id is None
                        or metric.rank_id is None
                        or metric.chain_index is None
                    ):
                        continue
                    rank = (metric.job_id, metric.rank_id)
                    chain_indices_by_rank.setdefault(rank, set()).add(metric.chain_index)
                for rank, actual in trajectories.items():
                    v_actual = {
                        name: values
                        for name, values in actual.items()
                        if name not in OUTCOME_METRICS
                    }
                    y_actual = {
                        name: values
                        for name, values in actual.items()
                        if name in OUTCOME_METRICS
                    }
                    append_event(
                        trace_file,
                        rank_window(
                            tick_number,
                            rank[0],
                            rank[1],
                            tuple(sorted(chain_indices_by_rank[rank])),
                            tick_start.isoformat(),
                            window_end.isoformat(),
                            v_actual,
                            y_actual,
                            telemetry_scope=(
                                "projected_batch_probe"
                                if rank[0] in self._batch_jobs
                                else "observed_tick"
                            ),
                            projection_method=(
                                "stationary_after_probe"
                                if rank[0] in self._batch_jobs
                                else None
                            ),
                        ),
                    )
                jobs_by_id = {**self._online_jobs, **self._batch_jobs}
                actual_by_job = aggregate_job_metrics(trajectories)
                for job_id, request_summary in request_summaries.items():
                    job = jobs_by_id.get(job_id)
                    if job is None:
                        continue
                    actual = actual_by_job.get(job_id, {})
                    features = job.workload["features"]
                    target_ttft = features.get("target_p99_ttft_ms")
                    target_tpot = features.get("target_p99_tpot_ms")
                    p99_ttft = request_summary["p99_ttft_ms"]
                    p99_tpot = request_summary["p99_tpot_ms"]
                    is_batch = not isinstance(job, OnlineJob)
                    append_event(
                        trace_file,
                        job_window(
                            tick_number,
                            job_id,
                            "online" if isinstance(job, OnlineJob) else "batch",
                            tick_start.isoformat(),
                            window_end.isoformat(),
                            p99_ttft,
                            p99_tpot,
                            actual.get("throughput_token_per_sec"),
                            actual.get("cost_per_token"),
                            target_ttft,
                            target_tpot,
                            p99_ttft <= target_ttft
                            if not is_batch
                            and p99_ttft is not None
                            and target_ttft is not None
                            else None,
                            p99_tpot <= target_tpot
                            if not is_batch
                            and p99_tpot is not None
                            and target_tpot is not None
                            else None,
                            int(request_summary["attempted_request_count"]),
                            int(request_summary["completed_request_count"]),
                            int(request_summary["unfinished_request_count"]),
                            request_summary["completion_ratio"],
                            (
                                "completed_batch_probe_requests_diagnostic"
                                if is_batch
                                else "encoder_latency_as_ttft"
                                if job.workload.get("task") == "embeddings"
                                else "completed_requests"
                            ),
                            request_summary["slo_pass_count"],
                            request_summary["slo_fail_count"],
                            request_summary["slo_count_scope"],
                            request_summary["ttft_pass_count"],
                            request_summary["ttft_fail_count"],
                            request_summary["tpot_pass_count"],
                            request_summary["tpot_fail_count"],
                            request_summary["offered_request_count"],
                            request_summary["admitted_request_count"],
                            request_summary["qualified_output_tokens"],
                            request_count_scope=request_summary[
                                "request_count_scope"
                            ],
                            batch_latency_diagnostic=request_summary[
                                "batch_latency_diagnostic"
                            ],
                            probe_attempted_request_count=request_summary.get(
                                "probe_attempted_request_count"
                            ),
                            probe_completed_request_count=request_summary.get(
                                "probe_completed_request_count"
                            ),
                            probe_unfinished_request_count=request_summary.get(
                                "probe_unfinished_request_count"
                            ),
                            campaign_total_request_count=request_summary.get(
                                "campaign_total_request_count"
                            ),
                            campaign_completed_in_window=request_summary.get(
                                "campaign_completed_in_window"
                            ),
                            campaign_completed_request_count=request_summary.get(
                                "campaign_completed_request_count"
                            ),
                            campaign_remaining_request_count=request_summary.get(
                                "campaign_remaining_request_count"
                            ),
                            campaign_request_per_sec=request_summary.get(
                                "campaign_request_per_sec"
                            ),
                        ),
                    )
                allocated_by_pool = {}
                pool_by_rank = {}
                for running in running_jobs:
                    for rank in running.ranks:
                        if rank.rank_id in failed_rank_ids:
                            continue
                        shape = rank.shape_json
                        env = shape.get("env")
                        if not isinstance(env, list):
                            continue
                        key = ("|".join(env), shape.get("instance_type"))
                        pool_by_rank[rank.rank_id] = key
                        # Store count is the total rank footprint; replicas are chains.
                        allocated_by_pool[key] = allocated_by_pool.get(key, 0) + (
                            int(shape["count"])
                        )
                pool_facts = []
                for env_key, summary in resource_map.scheduling_summary().items():
                    for pool in summary["pools"]:
                        instance_type = pool["instance_type"]
                        observation = pool_telemetry(
                            int(pool["total"]),
                            allocated_by_pool.get((env_key, instance_type), 0),
                            (
                                row
                                for row in rows
                                if pool_by_rank.get(row.rank_id) == (env_key, instance_type)
                            ),
                        )
                        append_event(
                            trace_file,
                            pool_window(
                                tick_number,
                                f"{env_key}|{pool['fabric_id']}|{instance_type}",
                                observation["expected_gpu_count"],
                                observation["reported_gpu_count"],
                                observation["allocated_gpu_count"],
                                observation["physical_sm_utilization_percent"],
                                observation["telemetry_coverage_percent"],
                            ),
                        )
                        pool_facts.append(
                            {**pool, "allocated_gpu_count": observation["allocated_gpu_count"]}
                        )
                accounting = cluster_accounting(
                    pool_facts,
                    self.scenario.tick_duration_ns / 3_600_000_000_000,
                )
                append_event(
                    trace_file,
                    cluster_window(
                        tick_number,
                        tick_start.isoformat(),
                        window_end.isoformat(),
                        accounting["capacity_gpu_hours"],
                        accounting["allocated_gpu_hours"],
                        accounting["capacity_cost"],
                        accounting["allocated_cost"],
                        getattr(self, "_admissions_in_tick", 0),
                        len(getattr(self, "_admission_latencies_ms", [])),
                        _mean_or_none(getattr(self, "_admission_latencies_ms", [])),
                    ),
                )

    def _prepare_job_runs(
        self,
        running,
        job_name: str,
        requests,
        tick_number: int,
        tick_start_ns: int,
        catalog,
    ) -> tuple[tuple[RankRun, ...], tuple[_RankFailure, ...]]:
        """Prepare one job's rank simulations for the tick-wide worker pool."""
        from tandemn_system_data.clients import ModelCatalogStore
        from tandemn_system_data.models.enums import ReasonCode

        deployments = build_rank_deployments(running.job, running.ranks)
        model_catalog = ModelCatalogStore(self.postgres).get(running.job.spec_json["model_id"])
        if model_catalog is None:
            raise ValueError(f"model catalog missing {running.job.spec_json['model_id']!r}")
        benchmark_job = self._online_jobs.get(running.job.job_id) or self._batch_jobs[running.job.job_id]
        for deployment in deployments:
            deployment.shape["weight_dtype"] = model_catalog.weight_dtype
            deployment.shape["weight_quantization_method"] = (
                model_catalog.weight_quantization_method
            )
            deployment.shape["task"] = benchmark_job.workload.get("task", "generate")
        failures = []
        supported = []
        for deployment in deployments:
            try:
                error_detail = profile_coverage_error(
                    deployment,
                    resolve_rank_hardware(deployment, catalog),
                    self.llmservingsim_root,
                )
            except (FileNotFoundError, KeyError, ValueError) as error:
                error_detail = f"PROFILE_UNAVAILABLE: {error}"
            if error_detail is None:
                supported.append(deployment)
            else:
                failures.append(
                    _RankFailure(
                        deployment.rank_name,
                        ReasonCode.SOFTWARE_STACK_RANK_FAILURE,
                        error_detail,
                    )
                )
        deployments = tuple(supported)
        if not deployments:
            return (), tuple(failures)
        if len(deployments) == 1:
            rank_shares = {deployments[0].rank_name: 1.0}
        else:
            rank_shares = {
                deployment.rank_name: float(deployment.shape["rank_traffic_share"])
                for deployment in deployments
            }
        routed = self._router.route(
            job_name,
            requests,
            rank_shares,
        )
        rank_runs = []
        for deployment in deployments:
            rank_requests = routed[deployment.rank_name]
            if not rank_requests:
                continue
            run_name = f"{self.run_name}-t{tick_number}-{job_name}-{deployment.rank_name}"
            rank_runs.append(
                prepare_rank_run(
                    deployment,
                    rank_requests,
                    resolve_rank_hardware(deployment, catalog),
                    self.llmservingsim_root,
                    self.llmservingsim_root
                    / "outputs"
                    / "tandemn-sim"
                    / self.run_name
                    / f"tick-{tick_number}"
                    / job_name
                    / deployment.rank_name,
                    self.container,
                    run_name,
                    tick_start_ns,
                    self.rank_timeout_seconds,
                )
            )
        if not rank_runs:
            return (), tuple(failures)
        return tuple(rank_runs), tuple(failures)

    @staticmethod
    def _prices(resource_map) -> dict[str, float]:
        """Index seeded ResourceMap instance prices for cost-per-token telemetry."""
        prices = {}
        for summary in resource_map.scheduling_summary().values():
            for pool in summary["pools"]:
                if pool["price_per_instance_hour"] is not None:
                    prices[pool["instance_type"]] = pool["price_per_instance_hour"]
        return prices

    def _finish(self, running, finished_at: datetime | None = None) -> None:
        """Stop active chains and finish a job after simulated work is complete."""
        from tandemn_system_data.models.enums import JobStatus, RankStatus

        for rank in running.ranks:
            self.jobs.set_rank_status(
                rank.rank_id,
                RankStatus.STOPPED,
                [RankStatus.LAUNCHING, RankStatus.RUNNING],
            )
        self.jobs.transition(running.job.job_id, JobStatus.FINISHED, [JobStatus.RUNNING])
        trace_file = getattr(self, "trace_file", None)
        if trace_file is not None:
            finished_at = finished_at or self.clock.now()
            self._trace_job_state(running.job.job_id, "FINISHED", finished_at, "completed")
            batch_job = self._batch_jobs.get(running.job.job_id)
            deadline_slo_met = None
            if batch_job is not None:
                deadline_at = self.clock.start + timedelta(
                    microseconds=batch_job.deadline_at_ns // 1_000
                )
                deadline_slo_met = finished_at <= deadline_at
            append_event(
                trace_file,
                job_finished(
                    running.job.job_id,
                    finished_at.isoformat(),
                    True,
                    None,
                    deadline_slo_met,
                ),
            )

    def _fail_ranks(self, running, job_name, failures: tuple[_RankFailure, ...]) -> None:
        """Mark rank-local failures while siblings continue serving the job."""
        from tandemn_system_data.models.enums import RankStatus

        failures_by_rank = {failure.rank_id: failure for failure in failures}
        trace_file = getattr(self, "trace_file", None)
        for rank in running.ranks:
            failure = failures_by_rank.get(rank.rank_id)
            if failure is None:
                continue
            self.jobs.set_rank_status(
                rank.rank_id,
                RankStatus.FAILED,
                [RankStatus.LAUNCHING, RankStatus.RUNNING],
                reason_code=failure.reason_code,
            )
            self._router.remove_rank(job_name, rank.rank_id)
            if trace_file is not None:
                shape = getattr(rank, "shape_json", {})
                gpu_count = shape.get("gpu_count", shape.get("count")) if isinstance(shape, dict) else None
                append_event(
                    trace_file,
                    rank_failed(
                        running.job.job_id,
                        rank.rank_id,
                        self.clock.now().isoformat(),
                        failure.reason_code,
                        gpu_count if isinstance(gpu_count, int) and not isinstance(gpu_count, bool) else None,
                        failure.error_detail[-1_000:],
                    ),
                )

    def _fail_oom(self, running, job_name, results) -> None:
        """Mark only ranks that exceed simulated device memory as failed."""
        from tandemn_system_data.models.enums import ReasonCode

        oom_results = tuple(result for result in results if result.status == "oom")
        self._fail_ranks(
            running,
            job_name,
            tuple(
                _RankFailure(
                    result.deployment.rank_name,
                    ReasonCode.OOM,
                    result.stderr or result.stdout or "LLMServingSim OOM",
                )
                for result in oom_results
            ),
        )

    def _hard_stop(self, reason: str = "SIMULATION_HARD_STOP") -> None:
        """Cancel every job still running when the simulation terminates."""
        from tandemn_system_data.models.enums import JobStatus, RankStatus

        tick_limited = reason == "SIMULATION_TICK_LIMIT"
        trace_file = getattr(self, "trace_file", None)
        for running in self.jobs.running_jobs(self.user_id):
            job_id = running.job.job_id
            if job_id in self._online_jobs:
                unserved_request_count = len(self._online_backlog.get(job_id, ()))
            elif job_id in self._batch_jobs:
                progress = self._batch_progress[job_id]
                unserved_request_count = (
                    progress.total_requests - progress.completed_requests
                )
            else:
                unserved_request_count = 0
            for rank in running.ranks:
                self.jobs.set_rank_status(
                    rank.rank_id,
                    RankStatus.STOPPED,
                    [RankStatus.LAUNCHING, RankStatus.RUNNING],
                )
            self.jobs.transition(
                running.job.job_id,
                JobStatus.FINISHED,
                [JobStatus.RUNNING],
                finish_reason=reason,
            )
            if trace_file is not None:
                is_batch = job_id in self._batch_jobs
                self._trace_job_state(
                    job_id, "FINISHED", self.clock.now(), reason)
                append_event(
                    trace_file,
                    job_finished(
                        running.job.job_id,
                        self.clock.now().isoformat(),
                        False,
                        reason,
                        False if is_batch and not tick_limited else None,
                        unserved_request_count,
                    ),
                )
        for waiting in getattr(self.jobs, "waiting_jobs", lambda _: ()) (self.user_id):
            job_id = waiting.job_id
            if job_id in self._online_jobs:
                unserved_request_count = sum(
                    request.arrives_at_ns < self._offset_ns(self.clock.now())
                    for request in self._requests[job_id]
                )
            elif job_id in self._batch_jobs:
                unserved_request_count = self._batch_progress[job_id].total_requests
            else:
                unserved_request_count = 0
            self.jobs.transition(
                job_id,
                JobStatus.FINISHED,
                [JobStatus.WAITING],
                finish_reason=reason,
            )
            if trace_file is not None:
                self._trace_job_state(
                    job_id, "FINISHED", self.clock.now(), reason)
                append_event(
                    trace_file,
                    job_finished(
                        job_id,
                        self.clock.now().isoformat(),
                        False,
                        reason,
                        False if job_id in self._batch_jobs and not tick_limited else None,
                        unserved_request_count,
                    ),
                )

    def _trace_job_state(self, job_id, to_status, at, reason) -> None:
        """Append one deduplicated logical lifecycle transition."""
        trace_file = getattr(self, "trace_file", None)
        if trace_file is None:
            return
        states = getattr(self, "_trace_job_states", None)
        if states is None:
            states = self._trace_job_states = {}
        from_status = states.get(job_id)
        if from_status == to_status:
            return
        append_event(trace_file, job_state_changed(
            job_id, from_status, to_status, at.isoformat(), reason))
        states[job_id] = to_status


def verify_store_user(user_id: str, postgres: object) -> None:
    """Require the externally seeded simulation Store user to exist."""
    from tandemn_system_data.db.orm import UserRow

    with postgres.session() as session:
        if session.get(UserRow, user_id) is None:
            raise ValueError(f"simulation Store user does not exist: {user_id}")


def submit_waiting_job(
    job: OnlineJob | BatchJob,
    user_id: str,
    job_store: object,
    submitted_at: datetime,
    *,
    prefix_cache_enabled: bool = False,
) -> None:
    """Submit one released benchmark job with its logical submission time."""
    from tandemn_orca.scripts.submit_job import normalize_job_spec
    from tandemn_system_data.models import Job, JobKind

    kind = JobKind.ONLINE if isinstance(job, OnlineJob) else JobKind.BATCH
    spec_json = normalize_job_spec(
        {
            "model_id": job.model_name,
            "prefix_cache_enabled": prefix_cache_enabled,
            "job_features": dict(job.workload["features"]),
        },
        kind,
    )
    job_store.submit(
        Job(
            job_id=job.store_job_id,
            user_id=user_id,
            kind=kind,
            spec_json=spec_json,
            input_source={"type": "cluster_sim", "job_name": job.name},
            created_at=submitted_at,
        )
    )


def apply_pending_plans(user_id: str, postgres: object) -> int:
    """Apply one user's created plans without Kubernetes or real GPU launches."""
    from tandemn_orca.launcher import NoopLauncher
    from tandemn_orca.orca import Orca

    return Orca(postgres, launcher=NoopLauncher()).apply_pending(user_id)


def active_ranks(store_job_id: str, job_store: object) -> list[object]:
    """Return active canonical ranks available for rank simulation."""
    return job_store.active_ranks(store_job_id)


def advance_batch_progress(
    progress: BatchProgress,
    measurements: tuple[BatchProbeMeasurement, ...],
    tick_duration_ns: int,
    tick_start_ns: int = 0,
    *,
    attempted_requests: int | None = None,
) -> int:
    """Advance a full batch backlog from one tick's representative probe."""
    if attempted_requests is None:
        attempted_requests = sum(
            measurement.completed_request_count for measurement in measurements
        )
    progress.next_request_number += attempted_requests

    request_rate = _batch_request_rate_per_second(measurements)
    projected_capacity = int(request_rate * tick_duration_ns / 1_000_000_000)
    if projected_capacity == 0:
        return 0

    remaining = progress.total_requests - progress.completed_requests
    completed = min(remaining, projected_capacity)
    progress.completed_requests += completed
    if progress.completed_requests == progress.total_requests:
        progress.completed_at_ns = tick_start_ns + ceil(
            remaining * 1_000_000_000 / request_rate
        )
    return completed


def _batch_request_rate_per_second(
    measurements: tuple[BatchProbeMeasurement, ...],
) -> float:
    """Return the combined completed-request rate across batch rank probes."""
    return sum(
        measurement.completed_request_count
        * 1_000_000_000
        / measurement.horizon_ns
        for measurement in measurements
    )


def _mean_or_none(values: list[float]) -> float | None:
    """Return a running arithmetic mean without inventing an empty value."""
    return sum(values) / len(values) if values else None


def parse_args() -> object:
    """Parse command-line arguments for one cluster simulation run."""
    pass


def main() -> None:
    """Run the cluster simulator command-line entry point."""
    pass


if __name__ == "__main__":
    main()
