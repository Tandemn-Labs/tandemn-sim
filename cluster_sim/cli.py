"""Run one standalone Tandemn cluster simulation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from models import Scenario
from planning import PlannerContext, build_planner
from runner import ClusterSimulator, LogicalClock
from scenario import load_scenario

ROOT = Path(__file__).resolve().parents[1]
TANDEMN_ROOT = ROOT.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one simulation run and planner configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="simulation_3h")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--planner", default="koi")
    parser.add_argument("--logical-epoch", default="2026-01-01T00:00:00+00:00")
    parser.add_argument("--run-name", default="cluster-sim")
    parser.add_argument(
        "--llmservingsim-root",
        type=Path,
        default=TANDEMN_ROOT / "tandemn-LLMServingSim",
    )
    parser.add_argument("--container", default="servingsim_docker")
    parser.add_argument("--parallel-ranks-per-job", type=int, default=1)
    parser.add_argument("--rank-timeout-seconds", type=float, default=300)
    parser.add_argument("--koi-root", type=Path)
    parser.add_argument(
        "--koi-arg",
        action="append",
        default=[],
        help="one Koi runner argument; use --koi-arg=--api-key",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load one scenario, build one planner, and run all logical ticks."""
    args = parse_args(argv)
    epoch = datetime.fromisoformat(args.logical_epoch)
    if epoch.tzinfo is None:
        raise ValueError("--logical-epoch must include a timezone")
    if args.parallel_ranks_per_job < 1:
        raise ValueError("--parallel-ranks-per-job must be positive")

    from tandemn_system_data.clients import PostgresClient

    scenario = _fresh_store_job_ids(load_scenario(_benchmark_path(args.benchmark)))
    clock = LogicalClock(epoch.astimezone(UTC))
    postgres = PostgresClient()
    planner = build_planner(
        args.planner,
        PlannerContext(
            user_id=args.user_id,
            postgres=postgres,
            logical_clock=clock,
            koi_root=args.koi_root,
            koi_args=list(args.koi_arg),
        ),
    )
    simulator = ClusterSimulator(
        scenario,
        args.user_id,
        postgres,
        planner,
        clock,
        args.llmservingsim_root,
        args.container,
        args.run_name,
        args.parallel_ranks_per_job,
        args.rank_timeout_seconds,
    )
    simulator.run()


def _benchmark_path(value: str) -> Path:
    """Resolve a benchmark name or explicit overview path from this repository."""
    path = Path(value)
    if path.is_file():
        return path
    return ROOT / "benchmarks" / value / "benchmark.yaml"


def _fresh_store_job_ids(scenario: Scenario) -> Scenario:
    """Assign fresh Store IDs so repeated runs never collide with old jobs."""
    return replace(
        scenario,
        online_jobs=tuple(
            replace(job, store_job_id=f"job-{uuid4()}") for job in scenario.online_jobs
        ),
        batch_jobs=tuple(
            replace(job, store_job_id=f"job-{uuid4()}") for job in scenario.batch_jobs
        ),
    )


if __name__ == "__main__":
    main()
