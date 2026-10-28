"""Opt-in AIC-to-ServingSim smoke tests without evaluator source changes."""

# ruff: noqa: E402

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from aic import (
    AicRequest,
    HardwareCandidate,
    NvidiaAicClient,
    ResourceTarget,
    _build_aic_request,
    _load_settings,
    _normalize_result,
    _place_action,
)
from models import BatchProgress, Request
from policy import load_config_policy
from runner import advance_batch_progress
from scenario import build_batch_probe, load_scenario
from serving import (
    RankRun,
    batch_probe_measurements,
    build_rank_cluster,
    build_rank_command,
    build_rank_deployments,
    parse_rank_result,
    profile_coverage_error,
    write_rank_workload,
)

from benchmarks.seed_benchmark_store import load_catalogs

LOGGER = logging.getLogger(__name__)
CONTAINER = "servingsim_docker"
EVALUATOR_ROOT = ROOT.parent / "tandemn-LLMServingSim"
HARDWARE = {
    "profile": "H100",
    "memory_gb": 80,
    "memory_bandwidth_gb_s": 3350,
    "link_bandwidth_gb_s": 900,
}


@unittest.skipUnless(
    os.getenv("AIC_SERVINGSIM_SMOKE") == "1",
    "set AIC_SERVINGSIM_SMOKE=1 with servingsim_docker running",
)
class AicServingSimSmokeTest(unittest.TestCase):
    """Run real AIC selections against native target models in ServingSim."""

    def test_native_moe(self) -> None:
        """Execute whole-expert Mixtral placement selected by native AIC."""
        self._run_case("mixtral", "mistralai/Mixtral-8x7B-Instruct-v0.1", "native")

    def test_proxy_next(self) -> None:
        """Execute original hybrid-attention Next using a proxy-backed selection."""
        self._run_case("next", "Qwen/Qwen3-Next-80B-A3B-Thinking", "proxy")

    def test_dense(self) -> None:
        """Verify the dense workload path still executes after the MoE changes."""
        self._run_case("dense", "meta-llama/Llama-3.2-1B-Instruct", None)

    def test_one_qwen3_next_job_through_its_benchmark_ticks(self) -> None:
        """Run AIC's placement and runner-style probes through the batch deadline."""
        scenario = load_scenario(
            ROOT / "benchmarks" / "simulation_3h" / "benchmark.yaml"
        )
        job = next(item for item in scenario.batch_jobs if item.name.endswith("-160-1"))
        features = job.workload["features"]
        model_catalog = next(
            catalog
            for catalog in load_catalogs(
                ROOT / "benchmarks" / "model_catalogs.csv",
                ROOT / "benchmarks" / "model_catalog_gpu_profiles.csv",
            )
            if catalog.model_id == job.model_name
        )
        catalog_values = model_catalog.model_dump(mode="json")
        self.assertTrue(catalog_values["chunked_prefill_enable"])
        self.assertEqual(
            next(
                value["value"]
                for value in catalog_values["max_num_batched_tokens"]
                if value["gpu_type"] == "H100"
            ),
            8192,
        )
        request_rate = features["total_token_budget"] / (
            (features["isl_token_avg"] + features["osl_token_avg"])
            * features["deadline_hrs"]
            * 3600
        )
        rule = load_config_policy().rule_for("H100", job.model_name)
        self.assertIsNotNone(rule)
        hardware_target = ResourceTarget(
            "reserved|aws|us-east-1|us-east-1a|H100",
            "H100",
            None,
            {"gpu_type": "H100", "total": 16, "free": 16, "pools": []},
            None,
        )
        settings = _load_settings(
            {
                "backend": "vllm",
                "model_configs_root": str(EVALUATOR_ROOT / "configs" / "model"),
            }
        )
        known = {**features, "kind": "batch", "model_id": job.model_name}
        known["_model_catalog"] = catalog_values
        aic_request = _build_aic_request(
            known,
            HardwareCandidate("h100_sxm", (hardware_target,), 16),
            settings,
            "throughput",
        )
        prediction = NvidiaAicClient().evaluate(aic_request)
        self.assertEqual(prediction.configuration["aic_prediction"]["mode"], "proxy")

        progress = BatchProgress(total_requests=int(job.workload["record_count"]))
        tick_offsets = range(
            job.releases_at_ns, job.deadline_at_ns, scenario.tick_duration_ns
        )
        tick_results = []
        for tick_number, tick_start_ns in enumerate(tick_offsets):
            probe = tuple(
                replace(request, arrives_at_ns=tick_start_ns)
                for request in build_batch_probe(job, progress)
            )
            if not probe:
                break
            result = self._run_case(
                f"simulation-3h-{job.name}-tick-{tick_number}",
                job.model_name,
                "proxy",
                input_tokens=int(features["isl_token_avg"]),
                output_tokens=int(features["osl_token_avg"]),
                target_ttft_ms=float(features["target_p99_ttft_ms"]),
                target_tpot_ms=float(features["target_p99_tpot_ms"]),
                request_rate=request_rate,
                requests=probe,
                tick_start_ns=tick_start_ns,
                max_wall_time_seconds=330,
                prediction=prediction,
                aic_request=aic_request,
                model_catalog=catalog_values,
                allow_partial=True,
            )
            measurements = batch_probe_measurements((result,))
            projected = advance_batch_progress(
                progress,
                measurements,
                scenario.tick_duration_ns,
                tick_start_ns,
                attempted_requests=(len(probe) if result.status != "oom" else 0),
            )
            tick_results.append(
                {
                    "tick": tick_number,
                    "logical_start_minute": tick_start_ns // 60_000_000_000,
                    "probe_requests": len(probe),
                    "probe_status": result.status,
                    "probe_completed": len(result.request_rows),
                    "probe_simulated_seconds": (result.sim_time_ns / 1_000_000_000),
                    "projected_campaign_completions": projected,
                    "campaign_completed": progress.completed_requests,
                    "campaign_remaining": (
                        progress.total_requests - progress.completed_requests
                    ),
                }
            )
            if (
                result.status == "oom"
                or progress.completed_requests == progress.total_requests
            ):
                break
        summary = {
            "job": job.name,
            "model": job.model_name,
            "total_requests": progress.total_requests,
            "release_minute": job.releases_at_ns // 60_000_000_000,
            "deadline_minute": job.deadline_at_ns // 60_000_000_000,
            "tick_minutes": scenario.tick_duration_ns // 60_000_000_000,
            "placement_parallel": prediction.configuration["parallel"],
            "placement_replicas": prediction.replicas,
            "prediction_model": prediction.configuration["aic_prediction"][
                "prediction_model_id"
            ],
            "ticks_run": len(tick_results),
            "campaign_completed": progress.completed_requests,
            "campaign_remaining": progress.total_requests - progress.completed_requests,
            "tick_results": tick_results,
        }
        run_dir = ROOT / "artifacts" / "aic-servingsim-smoke" / f"{job.name}-campaign"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        LOGGER.warning("Benchmark batch campaign result: %s", json.dumps(summary))
        self.assertEqual(len(tick_results), 4)
        self.assertTrue(any(item["probe_completed"] > 0 for item in tick_results))

    def _run_case(
        self,
        label: str,
        model: str,
        expected_mode: str | None,
        *,
        input_tokens: int = 128,
        output_tokens: int = 32,
        target_ttft_ms: float = 10000,
        target_tpot_ms: float = 1000,
        request_rate: float = 1,
        requests: tuple[Request, ...] | None = None,
        max_wall_time_seconds: int = 90,
        tick_start_ns: int = 0,
        prediction=None,
        aic_request: AicRequest | None = None,
        model_catalog: dict[str, object] | None = None,
        allow_partial: bool = False,
    ):
        run_id = f"aic-smoke-{label}-{uuid4().hex}"
        run_dir = ROOT / "artifacts" / "aic-servingsim-smoke" / run_id
        run_dir.mkdir(parents=True)
        rule = load_config_policy().rule_for("H100", model)
        assert rule is not None
        pairs = (
            ((1, 1), (2, 1), (4, 1), (8, 1))
            if label == "dense"
            else ((4, 1), (4, 2), (8, 1))
        )
        request = aic_request or AicRequest(
            model=model,
            system="h100_sxm",
            available_devices=16,
            backend="vllm",
            backend_version=None,
            database_mode="HYBRID",
            serving_mode="agg",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            target_ttft_ms=target_ttft_ms,
            target_tpot_ms=target_tpot_ms,
            optimization_goal="throughput",
            target_request_rate=request_rate,
            target_concurrency=None,
            parallelism_candidates=pairs,
            model_configs_root=EVALUATOR_ROOT / "configs" / "model",
            precision=rule.precision,
        )
        prediction = prediction or NvidiaAicClient().evaluate(request)
        (run_dir / "prediction.json").write_text(
            json.dumps(asdict(prediction), indent=2)
        )
        if expected_mode is not None:
            self.assertEqual(
                prediction.configuration["aic_prediction"]["mode"], expected_mode
            )
        target = ResourceTarget(
            "reserved|aws|us-east-1|us-east-1a|H100",
            "H100",
            "sim.h100",
            {"free": 16},
            None,
        )
        placement = _normalize_result(
            prediction, request, HardwareCandidate("h100_sxm", (target,), 16)
        )
        job = SimpleNamespace(job_id=run_id, spec_json={"model_id": model})
        job_payload: dict[str, object] = {"kind": "batch", "model_id": model}
        if model_catalog is not None:
            job_payload["_model_catalog"] = model_catalog
        action = _place_action(run_id, job_payload, placement, "vllm")
        ladder = action["ladder"][0]
        rank = SimpleNamespace(
            rank_id=ladder["rank_id"],
            n_replicas=ladder["n_replicas"],
            shape_json=ladder["config"],
        )
        deployment = build_rank_deployments(job, (rank,))[0]
        self.assertIsNone(profile_coverage_error(deployment, HARDWARE, EVALUATOR_ROOT))
        cluster = build_rank_cluster(deployment, HARDWARE)
        (run_dir / "cluster.json").write_text(json.dumps(cluster, indent=2))
        if requests is None:
            requests = tuple(
                Request(f"request-{index}", f"session-{index}", 0, 0, isl, osl, None)
                for index, (isl, osl) in enumerate(
                    ((96, 24), (128, 32), (160, 40), (128, 32))
                )
            )
        write_rank_workload(requests, run_dir / "workload.jsonl", tick_start_ns)

        scratch = f"/tmp/opencode/{run_id}"
        _docker("exec", CONTAINER, "mkdir", "-p", scratch)
        _docker("cp", f"{run_dir}/.", f"{CONTAINER}:{scratch}")
        # Private working directories avoid evaluator-source writes from ASTRA/Chakra.
        _docker(
            "exec",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            CONTAINER,
            "python3",
            "-c",
            """
import shutil
import sys
from pathlib import Path
source = Path('/app/LLMServingSim')
runtime = Path(sys.argv[1]) / 'runtime'
runtime.mkdir()
for name in ('serving', 'configs', 'profiler'):
    (runtime / name).symlink_to(source / name, target_is_directory=True)
astra = runtime / 'astra-sim'
astra.mkdir()
(astra / 'build').symlink_to(source / 'astra-sim/build', target_is_directory=True)
chakra = astra / 'extern/graph_frontend/chakra'
chakra.mkdir(parents=True)
(chakra / 'chakra').symlink_to(
    source / 'astra-sim/extern/graph_frontend/chakra/chakra', target_is_directory=True)
template = astra / 'inputs/system/system.json'
template.parent.mkdir(parents=True)
shutil.copyfile(source / 'astra-sim/inputs/system/system.json', template)
""",
            scratch,
        )
        command = list(
            build_rank_command(
                CONTAINER,
                run_dir,
                run_dir / "cluster.json",
                run_dir / "workload.jsonl",
                run_dir / "requests.csv",
                run_dir / "metrics.jsonl",
                run_id,
                len(requests),
                max_wall_time_seconds + 15,
            )
        )
        command[command.index("-w") + 1] = f"{scratch}/runtime"
        command[2:2] = ["-e", "PYTHONDONTWRITEBYTECODE=1"]
        for flag in ("--cluster-config", "--dataset", "--output", "--metrics-output"):
            index = command.index(flag) + 1
            prefix = ".." if flag in {"--cluster-config", "--dataset"} else scratch
            command[index] = f"{prefix}/{command[index]}"
        command[command.index("--max-wall-time-seconds") + 1] = str(
            max_wall_time_seconds
        )
        command.extend(["--inputs-root", f"{scratch}/inputs"])
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max_wall_time_seconds + 30,
        )
        (run_dir / "stdout.log").write_text(completed.stdout)
        (run_dir / "stderr.log").write_text(completed.stderr)
        for filename in ("requests.csv", "metrics.jsonl"):
            subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{CONTAINER}:{scratch}/{filename}",
                    str(run_dir / filename),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        rank_run = RankRun(
            deployment,
            requests,
            tuple(command),
            run_dir / "requests.csv",
            run_dir / "metrics.jsonl",
            len(requests),
            max_wall_time_seconds + 15,
        )
        result = parse_rank_result(rank_run, completed)
        summary = {
            "model": model,
            "mode": prediction.configuration["aic_prediction"]["mode"],
            "parallel": prediction.configuration["parallel"],
            "replicas": prediction.replicas,
            "status": result.status,
            "completed": len(result.request_rows),
            "offered": len(requests),
            "metric_samples": len(result.metric_samples),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        LOGGER.warning("AIC smoke result %s; artifacts: %s", summary, run_dir)
        if allow_partial:
            self.assertIn(result.status, {"completed", "wall_time_limit", "oom"})
        else:
            self.assertEqual(result.status, "completed", f"See {run_dir}")
            self.assertEqual(len(result.request_rows), len(requests), f"See {run_dir}")
        if result.status != "oom":
            self.assertTrue(result.metric_samples, f"See {run_dir}")
        return result


def _docker(*args: str) -> None:
    """Run a bounded Docker setup step and report its captured error on failure."""
    subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=True, timeout=30
    )


if __name__ == "__main__":
    unittest.main()
