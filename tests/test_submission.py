"""Tandemn submission contract checks."""

import unittest
from types import SimpleNamespace
from unittest import mock

from data_generation.submission import (
    SubmissionError,
    SubmissionRequest,
    TandemnStoreSubmitter,
    normalize_job_spec,
)


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

    def test_store_submission_preserves_canonical_benchmark_job_id(self) -> None:
        submitted: list[object] = []

        class Job:
            def __init__(self, **values: object) -> None:
                self.__dict__.update(values)

        class JobStore:
            def __init__(self, _client: object) -> None:
                pass

            def submit(self, job: object) -> object:
                submitted.append(job)
                return job

        clients = SimpleNamespace(
            PostgresClient=lambda: object(),
            JobStore=JobStore,
        )
        models = SimpleNamespace(
            JobKind=SimpleNamespace(ONLINE="online", BATCH="batch"),
            Job=Job,
        )
        request = SubmissionRequest(
            user_id="tenant-a",
            kind="online",
            spec_json={
                "model_id": "test-model",
                "job_features": {
                    "target_p99_ttft_ms": 500,
                    "target_p99_tpot_ms": 10,
                },
            },
            job_id="benchmark-job-1",
        )

        with mock.patch(
            "data_generation.submission.import_module",
            side_effect=(clients, models),
        ):
            result = TandemnStoreSubmitter().submit(request)

        self.assertEqual(result.tandemn_job_id, "benchmark-job-1")
        self.assertEqual(submitted[0].job_id, "benchmark-job-1")


if __name__ == "__main__":
    unittest.main()
