"""Local request receiver and dashboard server for internal benchmark tests."""

import json
import logging
import threading
import time
from array import array
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

STATIC_ROOT = Path(__file__).resolve().parent / "static"
LOGGER = logging.getLogger(__name__)


class DashboardState:
    """Store aggregate request traffic for one local benchmark run."""

    def __init__(self, plan: dict[str, Any]) -> None:
        """Initialize job buckets from a materialized dashboard plan."""
        self.plan = plan
        self.started_monotonic: float | None = None
        self.started_at: str | None = None
        self.controller_status = "idle"
        self.controller_pid: int | None = None
        self.controller_exit_code: int | None = None
        self.finished_virtual_minute: float | None = None
        self.launch_error: str | None = None
        self.lock = threading.Lock()
        bucket_count = int(float(plan["timeline_minutes"]) * 60) + 1
        self.jobs = {
            job["job_id"]: {
                "second_buckets": array("I", [0]) * bucket_count,
                "total_requests": 0,
            }
            for job in plan["jobs"]
        }

    def controller_starting(self) -> None:
        """Start the benchmark clock when the user requests a run."""
        with self.lock:
            self.started_monotonic = time.monotonic()
            self.started_at = datetime.now(UTC).isoformat(timespec="seconds")
            self.controller_status = "starting"
            self.launch_error = None

    def controller_started(self, pid: int) -> None:
        """Record that the benchmark controller is running."""
        with self.lock:
            self.controller_status = "running"
            self.controller_pid = pid

    def controller_launch_failed(self, error: str) -> None:
        """Expose a controller launch failure."""
        with self.lock:
            self.finished_virtual_minute = self.virtual_minute()
            self.controller_status = "failed"
            self.launch_error = error

    def controller_finished(self, return_code: int) -> None:
        """Record the controller's terminal result."""
        with self.lock:
            self.finished_virtual_minute = min(
                float(self.plan["timeline_minutes"]),
                self.virtual_minute(),
            )
            self.controller_status = "completed" if return_code == 0 else "failed"
            self.controller_exit_code = return_code

    def record_request(
        self,
        job_id: str,
    ) -> None:
        """Attribute one received request to its job and elapsed second."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            bucket_index = min(
                len(job["second_buckets"]) - 1,
                max(0, int(self.virtual_minute() * 60)),
            )
            job["second_buckets"][bucket_index] += 1
            job["total_requests"] += 1

    def virtual_minute(self) -> float:
        """Return elapsed time on the benchmark's original virtual clock."""
        if self.finished_virtual_minute is not None:
            return self.finished_virtual_minute
        if self.started_monotonic is None:
            return 0.0
        elapsed_seconds = max(0.0, time.monotonic() - self.started_monotonic)
        return elapsed_seconds / 60

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable dashboard snapshot."""
        with self.lock:
            jobs = {
                job_id: {
                    "total_requests": data["total_requests"],
                }
                for job_id, data in self.jobs.items()
            }
            return {
                "plan": self.plan,
                "started_at": self.started_at,
                "virtual_minute": round(self.virtual_minute(), 2),
                "controller": {
                    "status": self.controller_status,
                    "pid": self.controller_pid,
                    "exit_code": self.controller_exit_code,
                    "launch_error": self.launch_error,
                },
                "jobs": jobs,
            }

    def traffic_snapshot(self, job_id: str) -> dict[str, Any] | None:
        """Return the per-second request series for one selected job."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            elapsed_second = min(
                len(job["second_buckets"]) - 1,
                max(0, int(self.virtual_minute() * 60)),
            )
            return {
                "job_id": job_id,
                "second_buckets": list(job["second_buckets"][: elapsed_second + 1]),
                "total_requests": job["total_requests"],
            }


class DashboardServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying one dashboard state."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        state: DashboardState,
        *,
        start_callback: Callable[[], bool] | None = None,
    ) -> None:
        """Bind the request handler and shared state."""
        super().__init__(address, DashboardRequestHandler)
        self.state = state
        self.start_callback = start_callback


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Serve the dashboard and OpenAI-compatible mock endpoints."""

    server: DashboardServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        """Serve dashboard assets and live snapshots."""
        path = urlsplit(self.path).path
        if path == "/api/health":
            self.send_json({"status": "ready"})
            return
        if path == "/api/snapshot":
            self.send_json(self.server.state.snapshot())
            return
        job_id = parse_traffic_route(path)
        if job_id is not None:
            traffic = self.server.state.traffic_snapshot(job_id)
            if traffic is None:
                self.send_json({"error": "unknown job"}, status=404)
            else:
                self.send_json(traffic)
            return
        if path == "/":
            self.send_static("index.html", "text/html; charset=utf-8")
            return
        static_files = {
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        }
        asset = static_files.get(path)
        if asset is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_static(*asset)

    def do_POST(self) -> None:
        """Record one AIPerf request and return a compatible mock response."""
        path = urlsplit(self.path).path
        if path == "/api/run/start":
            content_length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(content_length)
            if self.server.start_callback is not None and self.server.start_callback():
                self.send_json({"status": "starting"}, status=202)
            else:
                self.send_json(
                    {"error": "benchmark has already been started"},
                    status=409,
                )
            return
        route = parse_job_route(path)
        if route is None:
            self.send_json({"error": "unknown receiver route"}, status=404)
            return
        job_id, endpoint_path = route
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_json({"error": "request body must be JSON"}, status=400)
            return
        if not isinstance(payload, dict):
            self.send_json({"error": "request body must be an object"}, status=400)
            return

        self.server.state.record_request(job_id)
        if endpoint_path.endswith("/embeddings"):
            inputs = payload.get("input", [])
            input_count = len(inputs) if isinstance(inputs, list) else 1
            self.send_json(
                {
                    "object": "list",
                    "model": payload.get("model"),
                    "data": [
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": [0.0],
                        }
                        for index in range(input_count)
                    ],
                }
            )
        elif endpoint_path.endswith("/rerank"):
            texts = payload.get("texts", [])
            text_count = len(texts) if isinstance(texts, list) else 0
            self.send_json(
                [
                    {"index": index, "score": 1.0 / (index + 1)}
                    for index in range(text_count)
                ]
            )
        elif payload.get("stream"):
            self.send_streaming_chat(payload.get("model"))
        else:
            self.send_json(chat_response(payload.get("model")))

    def send_static(self, filename: str, content_type: str) -> None:
        """Send one known dashboard asset."""
        body = (STATIC_ROOT / filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, body: object, status: int = 200) -> None:
        """Send a JSON response."""
        encoded = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def send_streaming_chat(self, model: object) -> None:
        """Send a minimal OpenAI-compatible streaming chat response."""
        response = {
            "id": "internal-test",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }
        encoded = f"data: {json.dumps(response)}\n\ndata: [DONE]\n\n".encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress the standard library access log."""


def parse_job_route(path: str) -> tuple[str, str] | None:
    """Parse `/jobs/<job-id>/<endpoint>` into attribution fields."""
    parts = path.split("/", maxsplit=3)
    if len(parts) != 4 or parts[1] != "jobs" or not parts[2]:
        return None
    return unquote(parts[2]), f"/{parts[3]}"


def parse_traffic_route(path: str) -> str | None:
    """Parse `/api/jobs/<job-id>/traffic` into a job identifier."""
    prefix = "/api/jobs/"
    suffix = "/traffic"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    encoded_job_id = path[len(prefix) : -len(suffix)]
    if not encoded_job_id or "/" in encoded_job_id:
        return None
    return unquote(encoded_job_id)


def chat_response(model: object) -> dict[str, Any]:
    """Return a minimal non-streaming chat completion."""
    return {
        "id": "internal-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }
