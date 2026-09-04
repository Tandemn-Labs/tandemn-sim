"""Tests for the real-hardware AIPerf scenario daemon."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import yaml
from tandemn_system_data.models import JobStatus

from data_generation.controller import ROOT
from data_generation.real_scenario import (
    Manifest,
    ManifestJob,
    State,
    StoreJob,
    load_manifest,
    run_scenario,
    stable_store_id,
)

JOB_PATH = ROOT / "benchmarks" / "15min_real" / "jobs" / "code-chat.yaml"


class FakeStore:
    """Return one scripted Store snapshot per daemon tick."""

    def __init__(self, snapshots: Sequence[Sequence[StoreJob]]) -> None:
        self.snapshots = list(snapshots)
        self.calls: list[tuple[str, ...]] = []

    def list_jobs(self, user_ids: Sequence[str]) -> Sequence[StoreJob]:
        self.calls.append(tuple(user_ids))
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]


def process(*poll_results: int | None) -> mock.Mock:
    result = mock.Mock()
    result.poll.side_effect = poll_results
    return result


class RealScenarioTests(unittest.TestCase):
    """Verify readiness gating, launch behavior, and terminal states."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.directory.name)
        self.job = ManifestJob("authored-job", "tenant", JOB_PATH)
        self.manifest = Manifest("test", 0.01, (self.job,))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def run_daemon(self, store: FakeStore, **overrides: object) -> int:
        arguments: dict[str, object] = {
            "poll_interval": 0.01,
            "ready_timeout": 1200.0,
            "output_dir": self.output_dir,
            "router_config_dir": Path("router-configs"),
            "source_cloud": "aws",
            "source_region": "us-east-1",
            "source_zone": "us-east-1a",
        }
        arguments.update(overrides)
        with mock.patch("data_generation.real_scenario.time.sleep"):
            return run_scenario(self.manifest, store, **arguments)  # type: ignore[arg-type]

    @mock.patch("data_generation.real_scenario.subprocess.Popen")
    @mock.patch("data_generation.real_scenario.check_ready", side_effect=[False, True])
    @mock.patch(
        "data_generation.real_scenario.resolve_endpoint",
        side_effect=[
            SystemExit("missing"),
            "http://127.0.0.1:28001",
            "http://127.0.0.1:28001",
        ],
    )
    def test_waits_for_running_and_ready_then_uses_actual_store_job(
        self,
        resolve: mock.Mock,
        ready: mock.Mock,
        popen: mock.Mock,
    ) -> None:
        popen.return_value = process(0)
        store = FakeStore(
            [
                (StoreJob("store-ulid", JobStatus.WAITING, "authored-job"),),
                (StoreJob("store-ulid", JobStatus.RUNNING, "authored-job"),),
                (StoreJob("store-ulid", JobStatus.RUNNING, "authored-job"),),
                (StoreJob("store-ulid", JobStatus.RUNNING, "authored-job"),),
                (StoreJob("store-ulid", JobStatus.RUNNING, "authored-job"),),
            ]
        )

        self.assertEqual(self.run_daemon(store), 0)

        self.assertEqual(resolve.call_count, 3)
        ready.assert_has_calls(
            [mock.call("http://127.0.0.1:28001"), mock.call("http://127.0.0.1:28001")]
        )
        command = popen.call_args.args[0]
        self.assertFalse(
            any(value.startswith("X-Tandemn-Session-ID:") for value in command)
        )
        self.assertIn("X-Tandemn-Source-Cloud:aws", command)
        self.assertIn("X-Tandemn-Source-Region:us-east-1", command)
        self.assertIn("X-Tandemn-Source-Zone:us-east-1a", command)
        config_path = Path(command[command.index("--config") + 1])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            config["benchmark"]["endpoint"]["url"], "http://127.0.0.1:28001"
        )
        self.assertEqual(
            config["benchmark"]["endpoint"]["sessionHeader"],
            "X-Tandemn-Session-ID",
        )

    @mock.patch("data_generation.real_scenario.subprocess.Popen")
    @mock.patch("data_generation.real_scenario.check_ready", return_value=True)
    @mock.patch(
        "data_generation.real_scenario.resolve_endpoint",
        return_value="http://127.0.0.1:28001",
    )
    def test_finished_store_job_stops_running_aiperf(
        self,
        _resolve: mock.Mock,
        _ready: mock.Mock,
        popen: mock.Mock,
    ) -> None:
        popen.return_value = process(None, -15)
        store = FakeStore(
            [
                (StoreJob("authored-job", JobStatus.RUNNING, None),),
                (StoreJob("authored-job", JobStatus.FINISHED, None),),
                (StoreJob("authored-job", JobStatus.FINISHED, None),),
            ]
        )

        self.assertEqual(self.run_daemon(store), 0)
        events = self.read_events()
        self.assertEqual(events[-1]["state"], State.DONE)
        self.assertEqual(events[-1]["exit_code"], -15)
        popen.return_value.terminate.assert_called_once_with()

    @mock.patch("data_generation.real_scenario.subprocess.Popen")
    @mock.patch("data_generation.real_scenario.check_ready", return_value=True)
    @mock.patch(
        "data_generation.real_scenario.resolve_endpoint",
        return_value="http://127.0.0.1:28001",
    )
    def test_matches_runner_deterministic_store_id(
        self,
        _resolve: mock.Mock,
        _ready: mock.Mock,
        popen: mock.Mock,
    ) -> None:
        popen.return_value = process(0)
        store_id = stable_store_id("test", "authored-job")
        store = FakeStore([(StoreJob(store_id, JobStatus.RUNNING, None),)])

        self.assertEqual(self.run_daemon(store), 0)

    @mock.patch("data_generation.real_scenario.check_ready", return_value=False)
    @mock.patch(
        "data_generation.real_scenario.resolve_endpoint",
        return_value="http://127.0.0.1:28001",
    )
    @mock.patch("data_generation.real_scenario.time.monotonic", side_effect=[1.0, 3.0])
    def test_ready_timeout_fails_without_launch(
        self,
        _clock: mock.Mock,
        _resolve: mock.Mock,
        _ready: mock.Mock,
    ) -> None:
        store = FakeStore([(StoreJob("authored-job", JobStatus.RUNNING, None),)])

        self.assertEqual(self.run_daemon(store, ready_timeout=2.0), 1)
        self.assertEqual(self.read_events()[-1]["detail"], "router readiness timed out")

    @mock.patch("data_generation.real_scenario.subprocess.Popen")
    @mock.patch("data_generation.real_scenario.check_ready", return_value=True)
    @mock.patch(
        "data_generation.real_scenario.resolve_endpoint",
        return_value="http://127.0.0.1:28001",
    )
    def test_launches_ready_jobs_concurrently(
        self,
        _resolve: mock.Mock,
        _ready: mock.Mock,
        popen: mock.Mock,
    ) -> None:
        second = ManifestJob("second-job", "other-tenant", JOB_PATH)
        self.manifest = Manifest("test", 0.01, (self.job, second))
        popen.side_effect = [process(0), process(0)]
        store = FakeStore(
            [
                (
                    StoreJob("authored-job", JobStatus.RUNNING, None),
                    StoreJob("second-job", JobStatus.RUNNING, None),
                )
            ]
        )

        self.assertEqual(self.run_daemon(store), 0)
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(store.calls[0], ("tenant", "other-tenant"))

    @mock.patch("data_generation.real_scenario.subprocess.Popen")
    @mock.patch("data_generation.real_scenario.check_ready", return_value=True)
    @mock.patch(
        "data_generation.real_scenario.resolve_endpoint",
        return_value="http://127.0.0.1:28001",
    )
    def test_records_aiperf_failure_exit_code(
        self,
        _resolve: mock.Mock,
        _ready: mock.Mock,
        popen: mock.Mock,
    ) -> None:
        popen.return_value = process(17)
        store = FakeStore([(StoreJob("authored-job", JobStatus.RUNNING, None),)])

        self.assertEqual(self.run_daemon(store), 1)
        self.assertEqual(self.read_events()[-1]["state"], State.FAILED)
        self.assertEqual(self.read_events()[-1]["exit_code"], 17)

    def test_skips_disabled_and_non_aiperf_jobs(self) -> None:
        benchmark_root = self.output_dir / "benchmarks" / "sample"
        jobs_dir = benchmark_root / "jobs"
        jobs_dir.mkdir(parents=True)
        (benchmark_root / "benchmark.yaml").write_text(
            yaml.safe_dump(
                {
                    "benchmark": {
                        "id": "sample",
                        "jobs": "jobs",
                        "poll_interval_seconds": 3,
                    }
                }
            ),
            encoding="utf-8",
        )
        for name, enabled, driver in (
            ("disabled", False, "aiperf"),
            ("batch", True, "external_mock_data"),
        ):
            (jobs_dir / f"{name}.yaml").write_text(
                yaml.safe_dump(
                    {
                        "job_id": name,
                        "schedule": {"enabled": enabled, "driver": driver},
                        "spec": {"user_id": "tenant"},
                    }
                ),
                encoding="utf-8",
            )

        with mock.patch(
            "data_generation.real_scenario.BENCHMARKS_ROOT",
            self.output_dir / "benchmarks",
        ):
            manifest = load_manifest("sample")
        self.manifest = manifest

        self.assertEqual(self.run_daemon(FakeStore([()])), 0)
        self.assertEqual(
            [event["state"] for event in self.read_events()],
            [State.SKIPPED, State.SKIPPED],
        )

    def read_events(self) -> list[dict[str, object]]:
        events_path = next(self.output_dir.glob("*-test/events.jsonl"), None)
        if events_path is None:
            events_path = next(self.output_dir.glob("*-sample/events.jsonl"))
        return [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
        ]


if __name__ == "__main__":
    unittest.main()
