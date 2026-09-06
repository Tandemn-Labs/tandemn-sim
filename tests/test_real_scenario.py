"""Tests for the real-hardware AIPerf scenario daemon."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
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
    TraceRuntime,
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

    @mock.patch("data_generation.real_scenario._create_trace_runtime")
    @mock.patch("data_generation.real_scenario.subprocess.Popen")
    @mock.patch("data_generation.real_scenario.check_ready", return_value=True)
    @mock.patch(
        "data_generation.real_scenario.resolve_endpoint",
        return_value="http://127.0.0.1:28001",
    )
    def test_integrates_trace_runtime_without_passing_aiperf_process(
        self,
        _resolve: mock.Mock,
        _ready: mock.Mock,
        popen: mock.Mock,
        create_trace: mock.Mock,
    ) -> None:
        popen.return_value = process(0)
        trace = create_trace.return_value
        store = FakeStore(
            [(StoreJob("store-ulid", JobStatus.RUNNING, "authored-job"),)]
        )

        self.assertEqual(self.run_daemon(store, trace_client=object()), 0)

        trace.start.assert_called_once()
        trace.bind_job.assert_called_once_with(self.job, "store-ulid")
        trace.collect.assert_called()
        trace.collect_final_aiperf_metrics.assert_called_once_with("authored-job")
        trace.finish.assert_called_once()
        for call in trace.mock_calls:
            self.assertNotIn(popen.return_value, call.args)

    def test_observes_batch_job_without_launching_aiperf(self) -> None:
        batch = ManifestJob(
            "batch-job",
            "tenant",
            JOB_PATH,
            kind="batch",
            driver="external_mock_data",
        )
        self.manifest = Manifest("test", 0.01, (batch,))
        store = FakeStore(
            [
                (StoreJob("batch-store", JobStatus.RUNNING, "batch-job"),),
                (StoreJob("batch-store", JobStatus.FINISHED, "batch-job"),),
            ]
        )

        self.assertEqual(self.run_daemon(store), 0)

        self.assertEqual(
            [event["state"] for event in self.read_events()],
            [State.PENDING, State.OBSERVING, State.DONE],
        )
        self.assertEqual(store.finished, [])

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
                        "schedule": {
                            "enabled": enabled,
                            "kind": "online",
                            "driver": driver,
                        },
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


class TraceRuntimeTests(unittest.TestCase):
    """Verify the trace runtime avoids pre-run EventLog history."""

    def test_start_initializes_the_event_consumer_at_the_user_tail(self) -> None:
        collector = mock.Mock()
        collector.run_id = "run-1"
        collector.user_id = "tenant"
        events = mock.Mock()
        runtime = TraceRuntime(
            collector=collector,
            events=events,
            jobs=mock.Mock(),
            plans=mock.Mock(),
            resources=mock.Mock(),
            metrics=mock.Mock(),
        )

        runtime.start(mock.sentinel.started_at)

        collector.start_run.assert_called_once_with(mock.sentinel.started_at)
        events.initialize_consumer_at_user_tail.assert_called_once_with(
            "real-trace:run-1", "tenant"
        )

    def test_emit_uses_one_tick_telemetry_call(self) -> None:
        collector = mock.Mock()
        collector.job_bindings.return_value = ()
        metrics = mock.Mock()
        runtime = TraceRuntime(
            collector=collector,
            events=mock.Mock(),
            jobs=mock.Mock(),
            plans=mock.Mock(),
            resources=mock.Mock(),
            metrics=metrics,
        )

        runtime._emit((mock.sentinel.tick,))

        collector.emit_tick_telemetry.assert_called_once_with(
            mock.sentinel.tick, metrics
        )
        collector.emit_gpu_samples.assert_not_called()
        collector.emit_rank_windows.assert_not_called()
        collector.emit_batch_job_window.assert_not_called()

    @mock.patch("data_generation.real_scenario.time.sleep")
    @mock.patch("data_generation.real_scenario.datetime")
    def test_finish_waits_only_for_the_remaining_holdback(
        self, mocked_datetime: mock.Mock, sleep: mock.Mock
    ) -> None:
        collector = mock.Mock()
        collector.tick_holdback_seconds = 60
        finalized_ticks = (mock.sentinel.tick,)
        collector.finalize_closed_ticks.return_value = finalized_ticks
        runtime = TraceRuntime(
            collector=collector,
            events=mock.Mock(),
            jobs=mock.Mock(),
            plans=mock.Mock(),
            resources=mock.Mock(),
            metrics=mock.Mock(),
        )
        runtime._read = mock.Mock()
        runtime._emit = mock.Mock()
        finished_at = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)
        after_initial_read = finished_at + timedelta(seconds=15)
        observed_at = finished_at + timedelta(seconds=60)
        mocked_datetime.now.side_effect = [after_initial_read, observed_at]

        runtime.finish(finished_at, status="completed")

        sleep.assert_called_once_with(45)
        runtime._read.assert_has_calls([mock.call(finished_at), mock.call(observed_at)])
        collector.close_open_tick.assert_called_once_with(finished_at)
        collector.finalize_closed_ticks.assert_called_once_with(observed_at)
        runtime._emit.assert_called_once_with(finalized_ticks)
        collector.close_run.assert_called_once_with(observed_at, status="completed")


if __name__ == "__main__":
    unittest.main()
