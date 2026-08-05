"""Serve an idle dashboard that can start one full local benchmark run."""

import argparse
import copy
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import yaml
from dashboard import DashboardServer, DashboardState

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "simulation_3h" / "overview.yaml"
RUNTIME_ROOT = Path(__file__).resolve().parent / ".runtime"
BATCH_SENDER = Path(__file__).resolve().parent / "batch_sender.py"
LOGGER = logging.getLogger(__name__)


class InternalTestRunner:
    """Own the single controller process started from the dashboard."""

    def __init__(
        self,
        overview_path: Path,
        state: DashboardState,
    ) -> None:
        """Store the prepared run inputs without starting the controller."""
        self.overview_path = overview_path
        self.state = state
        self.controller: subprocess.Popen[bytes] | None = None
        self.lock = threading.Lock()
        self.cleanup_lock = threading.Lock()
        self.temporary_artifacts = tempfile.TemporaryDirectory(
            prefix="tei-internal-test-"
        )
        self.artifact_root = Path(self.temporary_artifacts.name)
        self.artifacts_cleaned = False

    def start(self) -> bool:
        """Start the controller once and return whether this call launched it."""
        with self.lock:
            if self.controller is not None:
                return False
            self.state.controller_starting()
            command = (
                sys.executable,
                "-m",
                "data_generation.controller",
                str(self.overview_path),
                "--artifact-root",
                str(self.artifact_root),
                "--aiperf-export-level",
                "summary",
                "--discard-request-data",
            )
            try:
                controller = subprocess.Popen(command, start_new_session=True)
            except OSError as error:
                self.state.controller_launch_failed(str(error))
                LOGGER.error("[internal-test] controller launch failed: %s", error)
                return False
            self.controller = controller
            self.state.controller_started(controller.pid)
            monitor = threading.Thread(
                target=self._monitor,
                args=(controller,),
                daemon=True,
            )
            monitor.start()
            LOGGER.info("[internal-test] full benchmark started pid=%d", controller.pid)
            return True

    def stop(self) -> None:
        """Pause the controller when the dashboard process is stopped."""
        with self.lock:
            controller = self.controller
        if controller is None or controller.poll() is not None:
            return
        if sys.platform == "win32":
            controller.terminate()
        else:
            try:
                os.killpg(controller.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        try:
            controller.wait(timeout=70)
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                controller.kill()
            else:
                try:
                    os.killpg(controller.pid, signal.SIGKILL)
                except ProcessLookupError:
                    return
            controller.wait()
        finally:
            self._cleanup_artifacts()

    def _monitor(self, controller: subprocess.Popen[bytes]) -> None:
        """Publish completion while leaving the dashboard available."""
        return_code = controller.wait()
        self.state.controller_finished(return_code)
        self._cleanup_artifacts()
        LOGGER.info("[internal-test] controller exited code=%d", return_code)
        LOGGER.info("[internal-test] dashboard remains available; press Ctrl+C to stop")

    def _cleanup_artifacts(self) -> None:
        """Delete the temporary AI-Perf summary directory exactly once."""
        with self.cleanup_lock:
            if self.artifacts_cleaned:
                return
            self.temporary_artifacts.cleanup()
            self.artifacts_cleaned = True


def main() -> None:
    """Prepare the full local benchmark and serve its idle dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_name = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = RUNTIME_ROOT / run_name
    overview_path, plan = materialize_benchmark(
        args.benchmark.resolve(),
        run_root / "benchmark",
        host=args.host,
        port=args.port,
    )
    state = DashboardState(plan)
    runner = InternalTestRunner(overview_path, state)
    server = DashboardServer(
        (args.host, args.port),
        state,
        start_callback=runner.start,
    )

    LOGGER.info("[internal-test] dashboard http://%s:%d", args.host, args.port)
    LOGGER.info(
        "[internal-test] ready benchmark=%s jobs=%d duration=%.0fm drain=%.0fm",
        plan["benchmark_id"],
        len(plan["jobs"]),
        float(plan["horizon_minutes"]),
        float(plan["timeline_minutes"]) - float(plan["horizon_minutes"]),
    )
    LOGGER.info("[internal-test] open the dashboard and click Start 3-hour run")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("[internal-test] stopping")
    finally:
        server.shutdown()
        server.server_close()
        runner.stop()


def parse_args() -> argparse.Namespace:
    """Parse dashboard settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="path to an executable benchmark overview",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.benchmark.is_file():
        parser.error(f"benchmark overview not found: {args.benchmark}")
    return args


def materialize_benchmark(
    source_overview: Path,
    destination: Path,
    *,
    host: str,
    port: int,
) -> tuple[Path, dict[str, Any]]:
    """Create a full-speed local copy of every job in one benchmark."""
    document = load_yaml_mapping(source_overview)
    benchmark = mapping(document.get("benchmark"), "benchmark")
    jobs = mapping(document.get("jobs"), "jobs")
    horizon_minutes = number(benchmark.get("horizon_minutes", 180), "horizon")
    hard_stop = number(
        benchmark.get("hard_stop_at_offset_minutes"),
        "hard stop",
    )

    destination.mkdir(parents=True, exist_ok=True)
    jobs_directory = destination / "jobs"
    jobs_directory.mkdir(parents=True, exist_ok=True)
    generated_jobs: dict[str, Any] = {}
    dashboard_jobs: list[dict[str, Any]] = []

    for job_id, raw_job in jobs.items():
        job = mapping(raw_job, f"job {job_id}")
        config_path = (
            source_overview.parent / string(job.get("config"), "job config")
        ).resolve()
        config = load_yaml_mapping(config_path)
        generated_config = jobs_directory / f"{job_id}.yaml"

        if job.get("driver") == "aiperf":
            local_config, job_plan = localize_online_job(
                config,
                job_id,
                host=host,
                port=port,
            )
            generated_job = copy.deepcopy(job)
        else:
            local_config = copy.deepcopy(config)
            job_plan = batch_job_plan(config)
            generated_job = local_batch_job(
                job,
                host=host,
                port=port,
            )

        generated_job.pop("submission", None)
        generated_config.write_text(
            yaml.safe_dump(local_config, sort_keys=False),
            encoding="utf-8",
        )
        generated_job["config"] = f"jobs/{job_id}.yaml"
        generated_jobs[job_id] = generated_job
        job_plan.update(schedule_plan(job_id, job, config_path))
        dashboard_jobs.append(job_plan)

    generated_overview = {
        "schema_version": "benchmark/v1",
        "benchmark": {
            "id": f"internal_test_{benchmark.get('id', 'benchmark')}",
            "poll_interval_seconds": benchmark.get("poll_interval_seconds", 60),
            "shutdown_grace_seconds": benchmark.get(
                "shutdown_grace_seconds",
                60,
            ),
            "hard_stop_at_offset_minutes": hard_stop,
        },
        "jobs": generated_jobs,
    }
    overview_path = destination / "overview.yaml"
    overview_path.write_text(
        yaml.safe_dump(generated_overview, sort_keys=False),
        encoding="utf-8",
    )
    plan = {
        "benchmark_id": benchmark.get("id", "benchmark"),
        "horizon_minutes": horizon_minutes,
        "timeline_minutes": hard_stop,
        "jobs": dashboard_jobs,
    }
    (destination / "dashboard_plan.json").write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8",
    )
    return overview_path, plan


def localize_online_job(
    source: dict[str, Any],
    job_id: str,
    *,
    host: str,
    port: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Redirect one unchanged online workload to its local job endpoint."""
    config = copy.deepcopy(source)
    benchmark = mapping(config.get("benchmark"), "AIPerf benchmark")
    endpoint = mapping(benchmark.get("endpoint"), "AIPerf endpoint")
    source_url = endpoint.get("url")
    if source_url is None:
        original_path = string(endpoint.get("path"), "endpoint path")
    else:
        original_path = urlsplit(string(source_url, "endpoint URL")).path or "/"
    receiver_host = receiver_hostname(host)
    endpoint["url"] = (
        f"http://{receiver_host}:{port}/jobs/{quote(job_id, safe='')}{original_path}"
    )
    endpoint.pop("path", None)
    tokenizer = benchmark.get("tokenizer")
    if isinstance(tokenizer, dict):
        tokenizer["name"] = "builtin"
    dataset = mapping(benchmark.get("dataset"), "AIPerf dataset")
    entries = dataset.get("entries")
    if isinstance(entries, int):
        dataset["entries"] = min(entries, 32)

    phases = benchmark.get("phases")
    if not isinstance(phases, list) or not phases or not isinstance(phases[0], dict):
        raise ValueError(f"{job_id} must define at least one AIPerf phase")
    first_phase = phases[0]
    phase_documents = [phase for phase in phases if isinstance(phase, dict)]
    if len(phase_documents) != len(phases):
        raise ValueError(f"{job_id} has an invalid AIPerf phase")
    production_rate = max(
        number(phase.get("rate", 0), "phase rate") for phase in phase_documents
    )
    rate_ramp = first_phase.get("rateRamp")
    plan = {
        "kind": "online",
        "model": benchmark.get("model", "unknown"),
        "endpoint_type": endpoint.get("type", "unknown"),
        "endpoint_path": original_path,
        "traffic_pattern": first_phase.get("type", "unknown"),
        "smoothness": first_phase.get("smoothness"),
        "production_rate": production_rate,
        "rate_ramp_minutes": (
            None if rate_ramp is None else number(rate_ramp, "rate ramp") / 60
        ),
        "record_count": None,
    }
    return config, plan


def local_batch_job(
    source: dict[str, Any],
    *,
    host: str,
    port: int,
) -> dict[str, Any]:
    """Enable one batch job with the internal streaming sender command."""
    job = copy.deepcopy(source)
    job["enabled"] = True
    job["driver"] = "external_mock_data"
    job["command"] = [
        sys.executable,
        str(BATCH_SENDER),
        "--config",
        "{config}",
        "--job-id",
        "{job_id}",
        "--receiver",
        f"http://{receiver_hostname(host)}:{port}",
    ]
    return job


def batch_job_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Return dashboard metadata for one finite batch dataset."""
    dataset = mapping(config.get("dataset"), "batch dataset")
    return {
        "kind": "batch",
        "model": dataset.get("model", "unknown"),
        "endpoint_type": dataset.get("request_format", "unknown"),
        "endpoint_path": dataset.get("endpoint", "unknown"),
        "traffic_pattern": "batch",
        "smoothness": None,
        "production_rate": None,
        "rate_ramp_minutes": None,
        "record_count": number(dataset.get("record_count"), "record count"),
    }


def schedule_plan(
    job_id: str,
    job: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    """Normalize online and batch schedule fields for the dashboard."""
    kind = string(job.get("kind"), "job kind")
    stop_value = (
        job.get("stop_at_offset_minutes")
        if kind == "online"
        else job.get("deadline_at_offset_minutes")
    )
    return {
        "job_id": job_id,
        "display_name": config_path.stem.replace("-", " ").title(),
        "workload_type": string(job.get("tenant"), "job tenant")
        .removeprefix("tenant_")
        .replace("_", " ")
        .title(),
        "operation": string(job.get("operation", "finite batch"), "job operation")
        .replace("_", " ")
        .title(),
        "start_minute": number(
            job.get("start_at_offset_minutes"),
            "job start",
        ),
        "stop_minute": number(stop_value, "job stop or deadline"),
        "deadline_minute": (
            None
            if job.get("deadline_at_offset_minutes") is None
            else number(job["deadline_at_offset_minutes"], "job deadline")
        ),
        "hard_stop_minute": (
            None
            if job.get("hard_stop_at_offset_minutes") is None
            else number(job["hard_stop_at_offset_minutes"], "job hard stop")
        ),
    }


def receiver_hostname(host: str) -> str:
    """Return a connectable host when the server binds all interfaces."""
    return "127.0.0.1" if host == "0.0.0.0" else host


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load one YAML document with a mapping root."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return mapping(value, str(path))


def mapping(value: object, name: str) -> dict[str, Any]:
    """Return a string-keyed mapping or reject the input."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping")
    return value


def string(value: object, name: str) -> str:
    """Return a non-empty string or reject the input."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def number(value: object, name: str) -> float:
    """Return a numeric value or reject the input."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    return float(value)


if __name__ == "__main__":
    main()
