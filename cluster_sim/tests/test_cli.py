"""Test standalone cluster-simulator CLI setup without running a simulation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from cli import _benchmark_path, _fresh_store_job_ids, main, parse_args
from models import Scenario
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
                "--trace-root",
                "traces",
                "--trace-run-id",
                "run-1",
                "--koi-arg=--api-key",
                "--koi-arg=test-key",
            ]
        )

        self.assertEqual(args.user_id, "usr_sim")
        self.assertEqual(args.planner, "first-fit")
        self.assertEqual(args.parallel_ranks_per_job, 6)
        self.assertEqual(args.trace_root, Path("traces"))
        self.assertEqual(args.trace_run_id, "run-1")
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

    def test_trace_options_must_be_supplied_together(self) -> None:
        """Reject an output directory without an explicit trace identity."""
        args = parse_args(["--user-id", "user", "--trace-root", "x"])
        with patch("cli.parse_args", return_value=args):
            with self.assertRaisesRegex(ValueError, "must be used together"):
                main()

    def test_main_passes_trace_configuration_to_simulator(self) -> None:
        """Pass the explicit trace output identity to one simulation run."""
        scenario = Scenario("trace-smoke", 300_000_000_000, 300_000_000_000, (), ())
        simulator = Mock()

        with (
            patch("tandemn_system_data.clients.PostgresClient", return_value=object()),
            patch("cli.load_scenario", return_value=scenario),
            patch("cli.build_planner", return_value=SimpleNamespace(method_name="first-fit")),
            patch("cli.ClusterSimulator", return_value=simulator) as simulator_class,
        ):
            main(
                [
                    "--user-id",
                    "user",
                    "--trace-root",
                    "traces",
                    "--trace-run-id",
                    "run-1",
                ]
            )

        self.assertEqual(simulator_class.call_args.args[-2:], (Path("traces"), "run-1"))
        simulator.run.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
