"""Run cluster-simulation control ticks in logical time."""

from __future__ import annotations

from datetime import datetime, timedelta

from models import BatchJob, OnlineJob, PlannerResult, Scenario
from planning import PlannerDriver


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
        self._submitted_jobs: set[str] = set()

    def run(self) -> tuple[PlannerResult, ...]:
        """Run every logical tick through the configured drain boundary."""
        results = []
        tick_number = 0
        while self._offset_ns(self.clock.now()) < self.scenario.drain_end_ns:
            results.append(self.run_tick(tick_number))
            tick_number += 1
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


def parse_args() -> object:
    """Parse command-line arguments for one cluster simulation run."""
    pass


def main() -> None:
    """Run the cluster simulator command-line entry point."""
    pass


if __name__ == "__main__":
    main()
