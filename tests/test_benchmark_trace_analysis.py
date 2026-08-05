import unittest
from pathlib import Path

from data_generation.benchmark_trace_analysis import (
    WindowSelection,
    _pattern_score,
    _windows_overlap,
    load_manifest,
)

MANIFEST_PATH = (
    Path(__file__).parents[1] / "datasets" / "simulation_3h" / "trace_mapping.yaml"
)


class BenchmarkTraceAnalysisTests(unittest.TestCase):
    def test_defines_all_online_jobs_once(self) -> None:
        manifest = load_manifest(MANIFEST_PATH)
        slugs = [spec.slug for spec in manifest.jobs]

        self.assertEqual(len(slugs), 18)
        self.assertEqual(len(set(slugs)), 18)
        self.assertEqual(
            manifest.output_directory,
            MANIFEST_PATH.parent / "traces",
        )

    def test_pattern_scores_prefer_expected_shapes(self) -> None:
        constant = [10] * 60
        bursty = [0] * 55 + [20, 40, 80, 40, 20]

        self.assertGreater(
            _pattern_score(bursty, "Bursty"),
            _pattern_score(constant, "Bursty"),
        )
        self.assertGreater(
            _pattern_score(constant, "Constant"),
            _pattern_score(bursty, "Constant"),
        )

    def test_window_overlap_uses_half_open_intervals(self) -> None:
        excluded = [WindowSelection(10, 20, 0, 1)]

        self.assertTrue(_windows_overlap(15, 25, excluded))
        self.assertFalse(_windows_overlap(20, 30, excluded))


if __name__ == "__main__":
    unittest.main()
