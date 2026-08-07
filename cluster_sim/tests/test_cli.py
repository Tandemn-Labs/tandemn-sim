"""Test standalone cluster-simulator CLI setup without running a simulation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from cli import _benchmark_path, _fresh_store_job_ids, parse_args
from scenario import load_scenario


class CliTest(unittest.TestCase):
    """Verify CLI configuration stays independent from any planner implementation."""

    def test_parse_planner_and_koi_arguments(self) -> None:
        """Accept a planner name and pass-through Koi arguments."""
        args = parse_args(
            [
                "--user-id",
                "usr_sim",
                "--planner",
                "first-fit",
                "--parallel-ranks-per-job",
                "6",
                "--koi-arg=--api-key",
                "--koi-arg=test-key",
            ]
        )

        self.assertEqual(args.user_id, "usr_sim")
        self.assertEqual(args.planner, "first-fit")
        self.assertEqual(args.parallel_ranks_per_job, 6)
        self.assertEqual(args.koi_arg, ["--api-key", "test-key"])

    def test_benchmark_name_resolves_to_existing_overview(self) -> None:
        """Resolve the existing benchmark name without planner-specific logic."""
        self.assertEqual(
            _benchmark_path("simulation_3h"),
            ROOT / "benchmarks" / "simulation_3h" / "benchmark.yaml",
        )

    def test_each_run_receives_fresh_store_job_ids(self) -> None:
        """Avoid Store primary-key collisions when repeating one benchmark run."""
        scenario = load_scenario(ROOT / "benchmarks" / "simulation_3h" / "benchmark.yaml")
        fresh = _fresh_store_job_ids(scenario)

        original = (*scenario.online_jobs, *scenario.batch_jobs)
        updated = (*fresh.online_jobs, *fresh.batch_jobs)
        self.assertEqual([job.name for job in original], [job.name for job in updated])
        self.assertTrue(all(left.store_job_id != right.store_job_id for left, right in zip(original, updated)))
        self.assertEqual(len({job.store_job_id for job in updated}), len(updated))


if __name__ == "__main__":
    unittest.main()
