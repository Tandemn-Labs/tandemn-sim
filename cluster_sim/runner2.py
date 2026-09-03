"""Generate and apply benchmark plans without simulating workload telemetry."""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic, sleep

from models import BatchJob, OnlineJob, PlannerResult, Scenario
from planning import PlannerDriver, build_planner
from scenario import generate_job_ulid, load_scenario

ROOT = Path(__file__).resolve().parents[1]
PLANNERS = (
    "dynamo_aic",
    "bestfit",
    "vram",
    "edf",
    "prism",
    "cp_sat",
    "gpt_only",
    "openevolve",
    "contextual_bandit",
    "evox",
)
LOGGER = logging.getLogger(__name__)


class LogicalClock:
    """Track benchmark time independently from wall-clock polling."""

    def __init__(self, start: datetime) -> None:
        """Initialize the clock at a timezone-aware timestamp."""
        if start.tzinfo is None:
            raise ValueError("logical clock start must include a timezone")
        self.start = start
        self._now = start

    def now(self) -> datetime:
        """Return the current logical time."""
        return self._now

    def advance(self, duration: timedelta) -> None:
        """Advance the logical clock by one positive duration."""
        if duration <= timedelta():
            raise ValueError("logical clock duration must be positive")
        self._now += duration


class PlanRunner:
    """Submit due jobs and apply planner output at each benchmark tick."""

    def __init__(
        self,
        scenario: Scenario,
        user_id: str,
        postgres: object,
        planner: PlannerDriver,
        clock: LogicalClock,
    ) -> None:
        """Create a runner for an existing, seeded Store user."""
        from tandemn_system_data.clients import JobStore

        _verify_store_user(user_id, postgres)
        self.scenario = scenario
        self.user_id = user_id
        self.postgres = postgres
        self.planner = planner
        self.clock = clock
        self.jobs = JobStore(postgres)
        self._submitted_jobs: set[str] = set()

    def run(self, max_ticks: int | None = None) -> tuple[PlannerResult, ...]:
        """Create plans on the benchmark's wall-clock cadence."""
        if max_ticks is not None and max_ticks < 1:
            raise ValueError("max ticks must be positive")

        results = []
        tick_number = 0
        next_tick_at = monotonic()
        while self._offset_ns(self.clock.now()) < self.scenario.drain_end_ns and (
            max_ticks is None or tick_number < max_ticks
        ):
            results.append(self.run_tick(tick_number))
            tick_number += 1
            self.clock.advance(
                timedelta(microseconds=self.scenario.tick_duration_ns // 1_000)
            )
            if self._offset_ns(self.clock.now()) >= self.scenario.drain_end_ns or (
                max_ticks is not None and tick_number >= max_ticks
            ):
                break

            next_tick_at += self.scenario.tick_duration_ns / 1_000_000_000
            delay = max(0.0, next_tick_at - monotonic())
            LOGGER.info("next planning tick in %.1f seconds", delay)
            sleep(delay)

        reached_drain = self._offset_ns(self.clock.now()) >= self.scenario.drain_end_ns
        self._finish_submitted_jobs(
            "SIMULATION_HARD_STOP" if reached_drain else "SIMULATION_TICK_LIMIT"
        )
        return tuple(results)

    def run_tick(self, tick_number: int) -> PlannerResult:
        """Submit released jobs, create plans, and apply them in Store."""
        if tick_number < 0:
            raise ValueError("tick number must be nonnegative")

        start = self.clock.now()
        start_offset_ns = self._offset_ns(start)
        end_offset_ns = start_offset_ns + self.scenario.tick_duration_ns
        self._submit_jobs_before(start_offset_ns + 1)

        LOGGER.info("tick %d: planning placements", tick_number)
        result = self.planner.create_plans(tick_number, start)
        LOGGER.info("tick %d: created %d plan(s)", tick_number, len(result.plans))
        if result.error is not None:
            LOGGER.warning("tick %d planner error: %s", tick_number, result.error)

        self._submit_jobs_before(end_offset_ns)
        return result

    def _submit_jobs_before(self, offset_ns: int) -> None:
        """Submit unsubmitted benchmark jobs released before one offset."""
        jobs = (*self.scenario.online_jobs, *self.scenario.batch_jobs)
        for job in jobs:
            release_ns = (
                job.starts_at_ns if isinstance(job, OnlineJob) else job.releases_at_ns
            )
            if job.name in self._submitted_jobs or release_ns >= offset_ns:
                continue
            self._submit(job, release_ns)

    def _submit(self, job: OnlineJob | BatchJob, release_ns: int) -> None:
        """Persist one job with its logical benchmark release time."""
        from tandemn_system_data.clients import ModelCatalogStore

        catalog = ModelCatalogStore(self.postgres).get(job.model_name)
        _submit_waiting_job(
            job,
            self.user_id,
            self.jobs,
            self.clock.start + timedelta(microseconds=release_ns // 1_000),
            prefix_cache_enabled=bool(catalog and catalog.prefix_cache_enabled),
        )
        self._submitted_jobs.add(job.name)

    def _offset_ns(self, timestamp: datetime) -> int:
        """Return elapsed logical nanoseconds from the benchmark epoch."""
        return round((timestamp - self.clock.start).total_seconds() * 1_000_000_000)

    def _finish_submitted_jobs(self, reason: str) -> None:
        """Leave Store jobs and ranks in terminal states when the run ends."""
        from tandemn_system_data.models.enums import JobStatus, RankStatus

        for running in self.jobs.running_jobs(self.user_id):
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

        for getter, status in (
            (self.jobs.waiting_jobs, JobStatus.WAITING),
            (self.jobs.paused_jobs, JobStatus.PAUSED),
        ):
            for job in getter(self.user_id):
                self.jobs.transition(
                    job.job_id,
                    JobStatus.FINISHED,
                    [status],
                    finish_reason=reason,
                )


def _verify_store_user(user_id: str, postgres: object) -> None:
    """Require the externally seeded Store user to exist."""
    from tandemn_system_data.db.orm import UserRow

    with postgres.session() as session:
        if session.get(UserRow, user_id) is None:
            raise ValueError(f"simulation Store user does not exist: {user_id}")


def _submit_waiting_job(
    job: OnlineJob | BatchJob,
    user_id: str,
    job_store: object,
    submitted_at: datetime,
    *,
    prefix_cache_enabled: bool = False,
) -> None:
    """Submit one released benchmark job to Store."""
    from simulation_orca import normalize_job_spec
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


def _benchmark_path(value: str) -> Path:
    """Resolve a benchmark name or an explicit benchmark YAML path."""
    path = Path(value)
    if path.is_file():
        return path
    return ROOT / "benchmarks" / value / "benchmark.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one plan-only benchmark run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="simulation_3h")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--postgres-url")
    parser.add_argument("--planner", required=True, choices=PLANNERS)
    parser.add_argument("--logical-epoch", default="2026-01-01T00:00:00+00:00")
    parser.add_argument("--ticks", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load one benchmark and generate plans until its drain boundary."""
    args = parse_args(argv)
    epoch = datetime.fromisoformat(args.logical_epoch)
    if epoch.tzinfo is None:
        raise ValueError("--logical-epoch must include a timezone")
    if args.ticks is not None and args.ticks < 1:
        raise ValueError("--ticks must be positive")

    from tandemn_system_data.clients import PostgresClient

    scenario = generate_job_ulid(load_scenario(_benchmark_path(args.benchmark)))
    clock = LogicalClock(epoch.astimezone(UTC))
    postgres = PostgresClient(args.postgres_url)
    planner = build_planner(
        args.planner,
        argparse.Namespace(
            user_id=args.user_id,
            postgres=postgres,
            scenario=scenario,
        ),
    )
    PlanRunner(scenario, args.user_id, postgres, planner, clock).run(args.ticks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
