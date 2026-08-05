import unittest

from data_generation.traffic_trace_analysis import _normalize_row, analyze_trace


class TrafficTraceAnalysisTests(unittest.TestCase):
    def test_normalizes_camel_case_columns_without_elapsed(self) -> None:
        row = _normalize_row(
            {
                "TIMESTAMP": "2024-05-10T00:00:00Z",
                "ContextTokens": "100",
                "GeneratedTokens": "20",
            },
            row_number=1,
        )

        self.assertIsNone(row["elapsed_seconds"])
        self.assertEqual(row["input_tokens"], "100")
        self.assertEqual(row["output_tokens"], "20")

    def test_derives_elapsed_from_real_start_and_end_times(self) -> None:
        row = _normalize_row(
            {
                "start_time": "10.25",
                "end_time": "11.75",
                "input_tokens": "100",
                "output_tokens": "20",
            },
            row_number=1,
        )

        self.assertEqual(row["elapsed_seconds"], 1.5)

    def test_analysis_does_not_require_elapsed_time(self) -> None:
        rows = [
            {
                "timestamp_seconds": 0,
                "elapsed_seconds": None,
                "input_tokens": 10,
                "output_tokens": 5,
            },
            {
                "timestamp_seconds": 1,
                "elapsed_seconds": None,
                "input_tokens": 20,
                "output_tokens": 10,
            },
        ]

        result = analyze_trace(rows)

        self.assertEqual(result["request_rate"]["mean_rps"], 1.0)
        self.assertEqual(result["tokens"]["input"]["mean_tokens"], 15.0)

    def test_analysis_includes_empty_window_buckets(self) -> None:
        rows = [
            {
                "timestamp_seconds": 1,
                "elapsed_seconds": None,
                "input_tokens": 10,
                "output_tokens": 5,
            },
            {
                "timestamp_seconds": 2,
                "elapsed_seconds": None,
                "input_tokens": 20,
                "output_tokens": 10,
            },
        ]

        result = analyze_trace(rows, window_start=0, window_end=4)

        self.assertEqual(result["request_rate"]["count"], 4)
        self.assertEqual(result["request_rate"]["mean_rps"], 0.5)


if __name__ == "__main__":
    unittest.main()
