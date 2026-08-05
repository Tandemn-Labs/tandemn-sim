"""Repository-level benchmark, method, metric, and experiment catalog checks."""

import unittest

from data_generation.benchmarks.registry import list_benchmarks
from data_generation.cli import resolve_run_selections
from data_generation.experiments.registry import load_experiment
from data_generation.methods.registry import list_methods


class CatalogTests(unittest.TestCase):
    """Keep the paper-aligned execution matrix internally consistent."""

    def test_catalog_declares_all_six_benchmarks(self) -> None:
        benchmarks = list_benchmarks()

        self.assertEqual(len(benchmarks), 6)
        self.assertEqual(
            {definition.environment for definition in benchmarks.values()},
            {"simulation", "real_hardware"},
        )
        self.assertEqual(
            {definition.duration_hours for definition in benchmarks.values()},
            {3, 12, 24},
        )
        self.assertEqual(benchmarks["simulation_3h"].status, "executable")
        self.assertEqual(
            sum(item.status == "definition_only" for item in benchmarks.values()),
            5,
        )

    def test_catalog_declares_paper_methods_and_implementation_status(self) -> None:
        methods = list_methods()

        self.assertEqual(len(methods), 10)
        implemented = {
            method_id
            for method_id, definition in methods.items()
            if definition.status == "implemented"
        }
        self.assertEqual(implemented, {"dynamo_aic", "gpt_only"})

    def test_main_simulation_resolves_its_complete_matrix(self) -> None:
        experiment = load_experiment("main_simulation")
        selections = resolve_run_selections(
            experiment_id="main_simulation",
            benchmark_filter=None,
            method_filter=None,
            validate_only=True,
        )

        self.assertEqual(len(experiment.benchmarks), 3)
        self.assertEqual(len(experiment.methods), 9)
        self.assertEqual(len(selections), 27)
        self.assertIn("goodput", experiment.metrics)
        self.assertIn("invalid_plan_rate", experiment.metrics)

    def test_experiment_can_select_one_benchmark_method_pair(self) -> None:
        selections = resolve_run_selections(
            experiment_id="agentic_ablations",
            benchmark_filter="simulation_3h",
            method_filter="gpt_only",
            validate_only=True,
        )

        self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0].experiment_id, "agentic_ablations")
        self.assertEqual(selections[0].benchmark.benchmark_id, "simulation_3h")
        self.assertEqual(selections[0].method_id, "gpt_only")

    def test_direct_selection_requires_an_implemented_method_for_execution(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "not implemented"):
            resolve_run_selections(
                experiment_id=None,
                benchmark_filter="simulation_3h",
                method_filter="fcfs_first_fit",
                validate_only=False,
            )


if __name__ == "__main__":
    unittest.main()
