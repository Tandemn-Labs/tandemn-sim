"""Tandemn submission contract checks."""

import unittest

from data_generation.submission import SubmissionError, normalize_job_spec


class SubmissionTests(unittest.TestCase):
    """Check traffic intent normalization before JobStore submission."""

    def test_concurrency_remains_the_authoritative_online_input(self) -> None:
        spec = normalize_job_spec(
            {
                "model_id": "test-model",
                "job_features": {
                    "max_concurrent_streaming": 20,
                    "target_p99_ttft_ms": 500,
                    "target_p99_tpot_ms": 10,
                },
            },
            "online",
        )
        features = spec["job_features"]

        self.assertEqual(features["_traffic_mode"], "concurrency")
        self.assertEqual(features["max_concurrent_streaming"], 20)
        self.assertGreater(features["request_arrival_rate"], 0)

    def test_online_submission_rejects_two_traffic_modes(self) -> None:
        with self.assertRaisesRegex(SubmissionError, "not both"):
            normalize_job_spec(
                {
                    "model_id": "test-model",
                    "requestRate": 2,
                    "concurrency": 10,
                    "target_p99_ttft_ms": 500,
                    "target_p99_tpot_ms": 10,
                },
                "online",
            )


if __name__ == "__main__":
    unittest.main()
