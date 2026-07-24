"""Benchmark controller lifecycle checks."""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import yaml

from data_generation.controller import (
    Job,
    JobKind,
    JobState,
    RunningJob,
    load_benchmark,
    run_benchmark,
    sync_job_records,
)

ROOT = Path(__file__).resolve().parents[1]


class ControllerTests(unittest.TestCase):
    """Check benchmark loading, job lifecycles, pause, and resume."""

    def test_three_hour_benchmark_loads_all_jobs(self) -> None:
        benchmark = load_benchmark(ROOT / "benchmarks/simulation_3h/overview.yaml")

        self.assertEqual(benchmark.benchmark_id, "simulation_3h")
        self.assertEqual(len(benchmark.jobs), 26)
        self.assertEqual(sum(job.enabled for job in benchmark.jobs), 18)
        self.assertEqual(benchmark.poll_interval_seconds, 60)

    def test_online_job_stops_at_its_end_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="online",
                    end_field="stop_at_offset_minutes",
                    end_minute=0.001,
                    process_seconds=5,
                )
            )

            summary = run_benchmark(
                benchmark,
                root / "artifacts",
                run_id="online-test",
            )

            self.assertEqual(summary.states["test-job"], JobState.COMPLETED)

    def test_batch_job_can_finish_after_its_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="batch",
                    end_field="deadline_at_offset_minutes",
                    end_minute=0.001,
                    process_seconds=0.1,
                )
            )

            summary = run_benchmark(
                benchmark,
                root / "artifacts",
                run_id="batch-test",
            )

            self.assertEqual(summary.states["test-job"], JobState.COMPLETED)
            self.assertEqual(summary.deadline_misses, frozenset({"test-job"}))

    def test_paused_job_restarts_when_benchmark_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            benchmark = load_benchmark(
                write_benchmark(
                    root,
                    kind="online",
                    end_field="stop_at_offset_minutes",
                    end_minute=0.002,
                    process_seconds=5,
                )
            )
            real_sleep = time.sleep
            sleep_calls = 0

            def interrupt_first_poll(seconds: float) -> None:
                nonlocal sleep_calls
                sleep_calls += 1
                if sleep_calls == 1:
                    raise KeyboardInterrupt
                real_sleep(seconds)

            with mock.patch(
                "data_generation.controller.time.sleep",
                side_effect=interrupt_first_poll,
            ):
                paused = run_benchmark(
                    benchmark,
                    root / "artifacts",
                    run_id="resume-test",
                )

            self.assertEqual(
                paused.states["test-job"],
                JobState.PAUSED,
            )

            resumed = run_benchmark(
                benchmark,
                root / "artifacts",
                resume_run_id="resume-test",
            )
            checkpoint_path = (
                root
                / "artifacts"
                / "benchmarks"
                / "test-benchmark"
                / "resume-test"
                / "benchmark_state.json"
            )
            job_path = checkpoint_path.parent / "jobs" / "test-job"
            state = json.loads(checkpoint_path.read_text())
            saved_job = state["jobs"][0]

            self.assertEqual(resumed.states["test-job"], JobState.COMPLETED)
            self.assertEqual(state["status"], "completed")
            self.assertEqual(saved_job["job_id"], "test-job")
            self.assertEqual(saved_job["launch_count"], 2)
            self.assertIsNotNone(saved_job["paused_at"])
            self.assertIsNotNone(saved_job["resumed_at"])
            self.assertTrue((job_path / "prompts.jsonl").is_file())
            self.assertTrue((job_path / "responses.jsonl").is_file())

    def test_aiperf_raw_records_are_split_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            job_path = run_directory / "jobs" / "online-job"
            raw_path = job_path / "raw_records" / "raw_records_worker.jsonl"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_text(
                json.dumps(
                    {
                        "metadata": {
                            "x_request_id": "request-1",
                            "request_start_ns": 1_750_000_000_000_000_000,
                            "request_end_ns": 1_750_000_001_000_000_000,
                        },
                        "payload": {"messages": [{"content": "hello"}]},
                        "status": 200,
                        "responses": [{"content": "world"}],
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            item = RunningJob(
                job=Job(
                    job_id="online-job",
                    enabled=True,
                    output_to_terminal=False,
                    kind=JobKind.ONLINE,
                    driver="aiperf",
                    config=run_directory / "job.yaml",
                    command=None,
                    start_minute=0,
                    stop_minute=1,
                    deadline_minute=None,
                    hard_stop_minute=1,
                    parent_job_id=None,
                ),
                state=JobState.RUNNING,
            )

            sync_job_records(item, run_directory)
            sync_job_records(item, run_directory)
            resumed_item = RunningJob(job=item.job, state=JobState.RUNNING)
            sync_job_records(resumed_item, run_directory)

            prompts = read_jsonl(job_path / "prompts.jsonl")
            responses = read_jsonl(job_path / "responses.jsonl")
            self.assertEqual(len(prompts), 1)
            self.assertEqual(len(responses), 1)
            self.assertEqual(prompts[0]["request_id"], "request-1")
            self.assertEqual(responses[0]["request_id"], "request-1")
            self.assertEqual(prompts[0]["request"]["messages"][0]["content"], "hello")
            self.assertEqual(responses[0]["response"][0]["content"], "world")


def write_benchmark(
    root: Path,
    *,
    kind: str,
    end_field: str,
    end_minute: float,
    process_seconds: float,
) -> Path:
    """Write one small command-backed benchmark."""
    config = root / "job.yaml"
    config.write_text("# test job\n")
    overview_path = root / "overview.yaml"
    overview_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "benchmark/v1",
                "benchmark": {
                    "id": "test-benchmark",
                    "poll_interval_seconds": 0.01,
                    "shutdown_grace_seconds": 0.05,
                    "hard_stop_at_offset_minutes": 2,
                },
                "jobs": {
                    "test-job": {
                        "enabled": True,
                        "kind": kind,
                        "driver": "external_mock_data",
                        "config": "job.yaml",
                        "command": [
                            sys.executable,
                            "-c",
                            f"import time; time.sleep({process_seconds})",
                        ],
                        "start_at_offset_minutes": 0,
                        end_field: end_minute,
                    }
                },
            },
            sort_keys=False,
        )
    )
    return overview_path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL artifact into records."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


if __name__ == "__main__":
    unittest.main()
