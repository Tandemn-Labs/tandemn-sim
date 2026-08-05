"""Tests for the local internal benchmark dashboard."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml
from batch_sender import send_batch
from dashboard import (
    DashboardServer,
    DashboardState,
    parse_job_route,
    parse_traffic_route,
)
from run import InternalTestRunner, materialize_benchmark

ROOT = Path(__file__).resolve().parents[1]


class MaterializeBenchmarkTests(unittest.TestCase):
    """Verify generation preserves the full schedule and enables every job."""

    def test_materializes_all_jobs_without_scaling_workload(self) -> None:
        """Redirect online endpoints and attach senders to all batch jobs."""
        source = ROOT / "benchmarks" / "simulation_3h" / "overview.yaml"
        with tempfile.TemporaryDirectory() as temporary_directory:
            overview_path, plan = materialize_benchmark(
                source,
                Path(temporary_directory),
                host="127.0.0.1",
                port=9876,
            )
            overview = yaml.safe_load(overview_path.read_text(encoding="utf-8"))
            online_id = "job-benchmark_timing_3hr-0-1"
            online_job = overview["jobs"][online_id]
            online_config = yaml.safe_load(
                (overview_path.parent / online_job["config"]).read_text(
                    encoding="utf-8"
                )
            )
            batch_job = overview["jobs"]["job-benchmark_timing_3hr-15-1"]

            self.assertEqual(
                overview["benchmark"]["id"],
                "internal_test_simulation_3h",
            )
            self.assertEqual(len(overview["jobs"]), 26)
            self.assertEqual(len(plan["jobs"]), 26)
            self.assertEqual(
                plan["jobs"][0]["display_name"],
                "Chat Free Core",
            )
            self.assertEqual(
                plan["jobs"][0]["workload_type"],
                "Chat Free",
            )
            self.assertEqual(online_job["stop_at_offset_minutes"], 180)
            self.assertEqual(
                online_config["benchmark"]["phases"][0]["rate"],
                74.162,
            )
            self.assertEqual(
                online_config["benchmark"]["dataset"]["entries"],
                32,
            )
            self.assertIn(
                f"/jobs/{online_id}/",
                online_config["benchmark"]["endpoint"]["url"],
            )
            self.assertTrue(batch_job["enabled"])
            self.assertIn("batch_sender.py", " ".join(batch_job["command"]))
            self.assertEqual(
                sum(job["kind"] == "batch" for job in plan["jobs"]),
                8,
            )


class DashboardServerTests(unittest.TestCase):
    """Verify controls, attribution, and compatible local responses."""

    def setUp(self) -> None:
        """Start one ephemeral receiver."""
        self.start_calls = 0
        self.state = DashboardState(test_plan())
        self.server = DashboardServer(
            ("127.0.0.1", 0),
            self.state,
            start_callback=self.start_run,
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.thread.start()

    def tearDown(self) -> None:
        """Stop the ephemeral receiver."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def start_run(self) -> bool:
        """Simulate the runner accepting exactly one start request."""
        self.start_calls += 1
        if self.start_calls > 1:
            return False
        self.state.controller_starting()
        self.state.controller_started(123)
        return True

    def test_start_endpoint_launches_run_once(self) -> None:
        """Connect the dashboard button API to the runner callback."""
        first_response = self.post("/api/run/start", {})
        snapshot = self.state.snapshot()

        self.assertEqual(first_response["status"], "starting")
        self.assertEqual(self.start_calls, 1)
        self.assertEqual(snapshot["controller"]["status"], "running")

        with self.assertRaises(HTTPError):
            self.post("/api/run/start", {})

    def test_records_chat_request_by_job(self) -> None:
        """Count a request without retaining any payload metadata."""
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
        response = self.post("/jobs/job-one/v1/chat/completions", payload)
        snapshot = self.state.snapshot()

        self.assertEqual(response["choices"][0]["message"]["content"], "ok")
        self.assertEqual(snapshot["jobs"]["job-one"]["total_requests"], 1)
        self.assertEqual(
            set(snapshot["jobs"]["job-one"]),
            {"total_requests"},
        )

        host, port = self.server.server_address
        with urlopen(f"http://{host}:{port}/api/jobs/job-one/traffic") as traffic:
            series = json.loads(traffic.read())
        self.assertEqual(series["second_buckets"], [1])
        self.assertEqual(series["total_requests"], 1)

    def test_serves_dashboard_application(self) -> None:
        """Serve the light, directly scrubbable requests-per-second chart."""
        host, port = self.server.server_address
        with urlopen(f"http://{host}:{port}/") as response:
            html = response.read().decode()
        with urlopen(f"http://{host}:{port}/app.js") as response:
            javascript = response.read().decode()

        self.assertIn("Start 3-hour run", html)
        self.assertIn("Requests per second", html)
        self.assertNotIn("Planned req/s", html)
        self.assertNotIn("Inspect elapsed second", html)
        self.assertNotIn("Total requests", html)
        self.assertIn('fetch("/api/run/start"', javascript)
        self.assertIn("second_buckets", javascript)
        self.assertIn("`${minute} min`", javascript)
        self.assertNotIn("cumulative", javascript)
        self.assertIn("function drawChart()", javascript)

    def test_returns_embedding_response(self) -> None:
        """Return one mock embedding per input item."""
        response = self.post(
            "/jobs/job-one/v1/embeddings",
            {"model": "test-model", "input": ["one", "two"]},
        )

        self.assertEqual(len(response["data"]), 2)
        self.assertEqual(response["data"][1]["index"], 1)

    def test_batch_sender_streams_complete_limited_dataset(self) -> None:
        """Send generated batch bodies through the attributed local route."""
        config = (
            ROOT
            / "benchmarks"
            / "simulation_3h"
            / "jobs"
            / "batch-mini-nightly-eval.yaml"
        )
        host, port = self.server.server_address

        send_batch(
            config,
            "job-one",
            f"http://{host}:{port}",
            record_limit=2,
        )

        self.assertEqual(
            self.state.snapshot()["jobs"]["job-one"]["total_requests"],
            2,
        )

    def post(self, path: str, body: object) -> object:
        """Send one JSON request to the ephemeral server."""
        host, port = self.server.server_address
        request = Request(
            f"http://{host}:{port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            return json.loads(response.read())


class StateAndRouteTests(unittest.TestCase):
    """Verify run-clock and receiver-route behavior."""

    def test_dashboard_is_idle_until_controller_starts(self) -> None:
        """Keep the benchmark clock at zero before the button is clicked."""
        state = DashboardState(test_plan())

        snapshot = state.snapshot()

        self.assertEqual(snapshot["virtual_minute"], 0)
        self.assertEqual(snapshot["controller"]["status"], "idle")

    def test_controller_completion_freezes_benchmark_clock(self) -> None:
        """Keep a completed dashboard on its final benchmark minute."""
        state = DashboardState(test_plan())
        state.controller_starting()
        state.controller_finished(0)

        first_minute = state.snapshot()["virtual_minute"]
        second_minute = state.snapshot()["virtual_minute"]

        self.assertEqual(first_minute, second_minute)

    def test_parses_job_route(self) -> None:
        """Keep the original endpoint suffix."""
        self.assertEqual(
            parse_job_route("/jobs/job-1/v1/chat/completions"),
            ("job-1", "/v1/chat/completions"),
        )

    def test_records_traffic_in_elapsed_second(self) -> None:
        """Expose one count per second and its cumulative request total."""
        state = DashboardState(test_plan())
        state.started_monotonic = 100.0

        with patch("dashboard.time.monotonic", return_value=102.9):
            state.record_request("job-one")
            traffic = state.traffic_snapshot("job-one")

        self.assertIsNotNone(traffic)
        assert traffic is not None
        self.assertEqual(traffic["second_buckets"], [0, 0, 1])
        self.assertEqual(traffic["total_requests"], 1)

    def test_parses_job_traffic_route(self) -> None:
        """Resolve a selected-job traffic-series endpoint."""
        self.assertEqual(
            parse_traffic_route("/api/jobs/job%20one/traffic"),
            "job one",
        )

    @patch("run.os.killpg")
    def test_runner_stops_controller_process_group(self, killpg: MagicMock) -> None:
        """Send SIGTERM to the controller group during dashboard shutdown."""
        state = DashboardState(test_plan())
        runner = InternalTestRunner(Path("overview.yaml"), state)
        artifact_root = runner.artifact_root
        controller = MagicMock()
        controller.pid = 123
        controller.poll.return_value = None
        runner.controller = controller

        runner.stop()

        killpg.assert_called_once()
        controller.wait.assert_called_once_with(timeout=70)
        self.assertFalse(artifact_root.exists())


def test_plan() -> dict[str, object]:
    """Return the minimal plan required by dashboard state."""
    return {
        "benchmark_id": "test",
        "horizon_minutes": 180,
        "timeline_minutes": 210,
        "jobs": [
            {
                "job_id": "job-one",
                "start_minute": 0,
                "stop_minute": 180,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
