"""Run cluster-simulation control ticks in logical time."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from models import BatchJob, BatchProgress, OnlineJob, PlannerResult, Scenario
from planning import PlannerDriver
from routing import StickyRoundRobinRouter
from scenario import build_batch_probe, materialize_online_requests, requests_in_tick
from serving import (
    RankResult,
    build_rank_deployments,
    commit_tick_metrics,
    gpu_metrics_from_results,
    prepare_rank_run,
    resolve_rank_hardware,
    run_ranks,
)


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
        parallel_ranks_per_job: int = 1,
        rank_timeout_seconds: float = 300,
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
        self.parallel_ranks_per_job = parallel_ranks_per_job
        self.rank_timeout_seconds = rank_timeout_seconds
        self._submitted_jobs: set[str] = set()
        self._traffic_starts: dict[str, int] = {}
        self._requests = {
            job.store_job_id: materialize_online_requests(job) for job in scenario.online_jobs
        }
        self._online_jobs = {job.store_job_id: job for job in scenario.online_jobs}
        self._batch_jobs = {job.store_job_id: job for job in scenario.batch_jobs}
        self._batch_progress = {
            job.store_job_id: BatchProgress(total_requests=int(job.workload["record_count"]))
            for job in scenario.batch_jobs
        }
        self._router = StickyRoundRobinRouter()

    def run(self) -> tuple[PlannerResult, ...]:
        """Run every logical tick through the configured drain boundary."""
        results = []
        tick_number = 0
        while self._offset_ns(self.clock.now()) < self.scenario.drain_end_ns:
            results.append(self.run_tick(tick_number))
            tick_number += 1
        self._hard_stop()
        return tuple(results)

    def run_tick(self, tick_number: int) -> PlannerResult:
        """Submit due jobs, apply one plan, and advance one logical tick."""
        if tick_number < 0:
            raise ValueError("tick number must be nonnegative")
        start = self.clock.now()
        start_offset_ns = self._offset_ns(start)
        end_offset_ns = start_offset_ns + self.scenario.tick_duration_ns

        self._submit_jobs_before_or_at(start_offset_ns)
        result = self.planner.create_plans(tick_number, start)
        apply_pending_plans(self.user_id, self.postgres)
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
        submitted_at = self.clock.start + timedelta(microseconds=self._release_ns(job) // 1_000)
        submit_waiting_job(job, self.user_id, self.jobs, submitted_at)
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
        """Run active jobs serially and commit every successful tick metric together."""
        if self.llmservingsim_root is None:
            return

        from tandemn_system_data.clients import (
            GpuMetricStore,
            HardwareCatalogStore,
            ResourceMapStore,
        )

        catalog = HardwareCatalogStore(self.postgres).get()
        if catalog is None:
            raise ValueError("hardware catalog is required for rank simulation")
        prices = self._prices(ResourceMapStore(self.postgres, user_id=self.user_id).get())
        tick_start = self.clock.now()
        metrics = []

        running_jobs = self.jobs.running_jobs(self.user_id)
        for running in running_jobs:
            job = self._online_jobs.get(running.job.job_id)
            if job is None:
                continue
            traffic_start = self._traffic_starts.setdefault(running.job.job_id, start_offset_ns)
            duration = job.stops_at_ns - job.starts_at_ns
            relative_start = start_offset_ns - traffic_start
            if relative_start >= duration:
                self._finish(running)
                continue
            relative_end = min(end_offset_ns - traffic_start, duration)
            source_start = job.starts_at_ns + relative_start
            source_end = job.starts_at_ns + relative_end
            requests = tuple(
                replace(
                    request,
                    arrives_at_ns=start_offset_ns + request.arrives_at_ns - source_start,
                )
                for request in requests_in_tick(self._requests[job.store_job_id], source_start, source_end)
            )
            job_metrics, _ = self._simulate_job(
                running,
                job.name,
                requests,
                tick_number,
                start_offset_ns,
                catalog.catalog,
                prices,
                tick_start,
            )
            metrics.extend(job_metrics)

        for running in running_jobs:
            job = self._batch_jobs.get(running.job.job_id)
            if job is None:
                continue
            progress = self._batch_progress[job.store_job_id]
            probe = tuple(
                replace(request, arrives_at_ns=start_offset_ns)
                for request in build_batch_probe(job, progress)
            )
            job_metrics, results = self._simulate_job(
                running,
                job.name,
                probe,
                tick_number,
                start_offset_ns,
                catalog.catalog,
                prices,
                tick_start,
            )
            metrics.extend(job_metrics)
            advance_batch_progress(progress, results, self.scenario.tick_duration_ns)
            if progress.completed_requests == progress.total_requests:
                self._finish(running)

        if metrics:
            commit_tick_metrics(GpuMetricStore(self.postgres), metrics)

    def _simulate_job(
        self,
        running,
        job_name: str,
        requests,
        tick_number: int,
        tick_start_ns: int,
        catalog,
        prices: dict[str, float],
        tick_start: datetime,
    ) -> tuple[list[object], tuple[RankResult, ...]]:
        """Run one job's ranks concurrently and return its staged telemetry."""
        deployments = build_rank_deployments(running.job, running.chains)
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
            return [], ()
        results = run_ranks(rank_runs, self.parallel_ranks_per_job)
        return (
            gpu_metrics_from_results(
                running.job,
                tuple(rank_run.deployment for rank_run in rank_runs),
                results,
                tick_start,
                self.scenario.tick_duration_ns,
                prices,
            ),
            results,
        )

    @staticmethod
    def _prices(resource_map) -> dict[str, float]:
        """Index seeded ResourceMap instance prices for cost-per-token telemetry."""
        prices = {}
        for summary in resource_map.scheduling_summary().values():
            for pool in summary["pools"]:
                if pool["price_per_instance_hour"] is not None:
                    prices[pool["instance_type"]] = pool["price_per_instance_hour"]
        return prices

    def _finish(self, running) -> None:
        """Stop active chains and finish a job after simulated work is complete."""
        from tandemn_system_data.models.enums import ChainStatus, JobStatus

        for chain in running.chains:
            self.jobs.set_chain_status(
                chain.chain_id,
                ChainStatus.STOPPED,
                [ChainStatus.LAUNCHING, ChainStatus.RUNNING],
            )
        self.jobs.transition(running.job.job_id, JobStatus.FINISHED, [JobStatus.RUNNING])

    def _hard_stop(self) -> None:
        """Cancel every job still running at the benchmark's logical hard stop."""
        from tandemn_system_data.models.enums import ChainStatus, JobStatus

        for running in self.jobs.running_jobs(self.user_id):
            for chain in running.chains:
                self.jobs.set_chain_status(
                    chain.chain_id,
                    ChainStatus.STOPPED,
                    [ChainStatus.LAUNCHING, ChainStatus.RUNNING],
                )
            self.jobs.transition(
                running.job.job_id,
                JobStatus.FINISHED,
                [JobStatus.RUNNING],
                finish_reason="SIMULATION_HARD_STOP",
            )


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
) -> None:
    """Submit one released benchmark job with its logical submission time."""
    from tandemn_orca.scripts.submit_job import normalize_job_spec
    from tandemn_system_data.models import Job, JobKind

    kind = JobKind.ONLINE if isinstance(job, OnlineJob) else JobKind.BATCH
    spec_json = normalize_job_spec(
        {
            "model_id": job.model_name,
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


def active_chains(store_job_id: str, job_store: object) -> list[object]:
    """Return the active canonical chains available for rank simulation."""
    return job_store.active_chains(store_job_id)


def advance_batch_progress(
    progress: BatchProgress,
    results: tuple[RankResult, ...],
    tick_duration_ns: int,
) -> int:
    """Advance a full batch backlog from one tick's representative probe."""
    horizon_ns = max(
        (row["end_time"] for result in results for row in result.request_rows),
        default=0,
    )
    probe_requests = sum(len(result.request_rows) for result in results)
    if horizon_ns <= 0 or probe_requests == 0:
        return 0

    completed = min(
        progress.total_requests - progress.completed_requests,
        probe_requests * tick_duration_ns // horizon_ns,
    )
    progress.completed_requests += completed
    progress.next_request_number = min(
        progress.total_requests,
        progress.next_request_number + probe_requests,
    )
    return completed


def parse_args() -> object:
    """Parse command-line arguments for one cluster simulation run."""
    pass


def main() -> None:
    """Run the cluster simulator command-line entry point."""
    pass


if __name__ == "__main__":
    main()
