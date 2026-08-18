"""Build per-rank LLMServingSim inputs and later convert results to telemetry."""

from __future__ import annotations

import hashlib
import json
import csv
import math
import shutil
import subprocess
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Mapping, Sequence

from models import ChainBinding, RankDeployment, RankMetricSample, Request


CSV_HEADER = [
    "instance id",
    "request id",
    "model",
    "input",
    "output",
    "arrival",
    "end_time",
    "latency",
    "queuing_delay",
    "TTFT",
    "TPOT",
    "ITL",
    "npu_prefix_hit_tokens",
    "prefix_cache_requested_tokens",
    "prefix_cache_hit_ratio",
]
SECOND = 1_000_000_000
SAMPLE_BUCKET_NS = 10 * SECOND
SERVING_WINDOW_SECONDS = 300
PROFILE_ALIASES = {
    "H100": "H100",
    "H100-SXM-80GB": "H100",
    "A100_80GB": "A100-80GB",
    "A100-SXM-80GB": "A100-80GB",
    "A100_40GB": "A100-40GB",
    "A100-PCIe-40GB": "A100-40GB",
    "A10G": "A10G",
    "A10G-24GB": "A10G",
    "L40S": "L40S",
    "L40S-48GB": "L40S",
    "MI300": "MI300X",
    "MI300X-192GB": "MI300X",
}


@dataclass(frozen=True)
class RankRun:
    """One rank-local LLMServingSim command and its expected output."""

    deployment: RankDeployment
    requests: tuple[Request, ...]
    command: tuple[str, ...]
    output_path: Path
    metrics_path: Path
    request_count: int
    timeout_seconds: float
    cleanup_dir: Path | None = None


@dataclass(frozen=True)
class RankResult:
    """Validated request results and periodic runtime metrics for one rank."""

    deployment: RankDeployment
    request_rows: tuple[dict[str, float], ...]
    metric_samples: tuple[RankMetricSample, ...]
    stdout: str
    stderr: str
    status: str = "completed"
    completed_requests: tuple[Request, ...] = ()
    sim_time_ns: int = 0
    admitted_requests: tuple[Request, ...] = ()


@dataclass(frozen=True)
class BatchProbeMeasurement:
    """Elapsed simulation time and completed work from one batch rank probe."""

    deployment: RankDeployment
    horizon_ns: int
    completed_request_count: int
    completed_output_tokens: float


def batch_probe_measurements(
    results: Sequence[RankResult],
) -> tuple[BatchProbeMeasurement, ...]:
    """Return completed work and elapsed simulation time for each batch probe."""
    measurements = []
    for result in results:
        if result.status == "wall_time_limit":
            horizon_ns = result.sim_time_ns
        elif result.request_rows:
            horizon_ns = max(row["end_time"] for row in result.request_rows)
        else:
            continue
        if horizon_ns <= 0:
            continue
        measurements.append(
            BatchProbeMeasurement(
                result.deployment,
                int(horizon_ns),
                len(result.request_rows),
                sum(row["output"] for row in result.request_rows),
            )
        )
    return tuple(measurements)


def build_rank_deployments(job: object, active_ranks: Sequence[object]) -> tuple[RankDeployment, ...]:
    """Build one simulation deployment per canonical Store rank."""
    deployments = []
    for rank in sorted(active_ranks, key=lambda item: item.rank_id):
        rank_id = str(rank.rank_id)
        shape = dict(rank.shape_json)
        shape["rank_id"] = rank_id
        # Orca stores the total DP footprint in count/gpu_count. Each
        # LLMServingSim chain represents one DP replica, so it needs TP * PP.
        count = int(shape.get("tp", 1)) * int(shape.get("pp", 1))
        shape["count"] = count
        bindings = tuple(
            ChainBinding(rank_id, chain_index, count)
            for chain_index in range(int(rank.n_replicas))
        )
        deployments.append(
            RankDeployment(
                store_job_id=job.job_id,
                rank_name=rank_id,
                model_name=job.spec_json["model_id"],
                shape=shape,
                chains=bindings,
            )
        )
    return tuple(deployments)


def build_rank_cluster(
    deployment: RankDeployment,
    hardware: Mapping[str, object],
) -> dict[str, object]:
    """Build one homogeneous LLMServingSim cluster JSON document for a rank."""
    shape = deployment.shape
    count = int(shape["count"])
    tp = int(shape.get("tp", 1))
    pp = int(shape.get("pp", 1))
    if count != tp * pp:
        raise ValueError(f"{deployment.rank_name} count must equal tp * pp")

    instance = {
        "model_name": deployment.model_name,
        "hardware": hardware["profile"],
        "fp": _serving_precision(shape),
        "dtype": _serving_dtype(shape),
        "npu_mem": {
            "mem_size": hardware["memory_gb"],
            "mem_bw": hardware["memory_bandwidth_gb_s"],
            "mem_latency": 0,
        },
        "num_npus": count,
        "tp_size": tp,
        "pp_size": pp,
        "ep_size": int(shape.get("ep", 1)),
        "pd_type": None,
        "max_num_seqs": int(shape.get("max_num_seq", 128)),
        "max_num_batched_tokens": int(shape.get("max_num_batched_tokens", 2048)),
        "block_size": int(shape.get("block_size", 16)),
        "enable_prefix_caching": bool(shape.get("prefix_cache_enabled", False)),
        "enable_chunked_prefill": bool(shape.get("chunked_prefill_enable", False)),
    }
    return {
        "num_nodes": 1,
        "link_bw": hardware["link_bandwidth_gb_s"],
        "link_latency": hardware.get("link_latency_ns", 0),
        "nodes": [
            {
                "num_instances": len(deployment.chains),
                "cpu_mem": {"mem_size": 512, "mem_bw": 256, "mem_latency": 0},
                "instances": [dict(instance) for _ in deployment.chains],
            }
        ],
    }


def _serving_precision(shape: Mapping[str, object]) -> str:
    """Translate Store model precision into LLMServingSim's profile directory name."""
    dtype = str(shape.get("weight_dtype") or "bfloat16").lower()
    if shape.get("weight_quantization_method") == "fp8" and dtype in {"bf16", "bfloat16"}:
        return "bf16-fp8"
    if dtype in {"fp16", "float16"}:
        return "fp16"
    return "bf16"


def _serving_dtype(shape: Mapping[str, object]) -> str:
    """Translate Store weight dtype to the LLMServingSim runtime dtype option."""
    dtype = str(shape.get("weight_dtype") or "bfloat16").lower()
    return "float16" if dtype in {"fp16", "float16"} else "bfloat16"


def profile_coverage_error(
    deployment: RankDeployment,
    hardware: Mapping[str, object],
    llmservingsim_root: Path,
) -> str | None:
    """Return the runtime profile lookup failure before launching ServingSim."""
    model_path = Path(deployment.model_name)
    if model_path.is_absolute() or ".." in model_path.parts:
        return f"PROFILE_UNAVAILABLE: invalid model path={deployment.model_name!r}"
    profile = str(hardware["profile"])
    variant = _serving_precision(deployment.shape)
    root = llmservingsim_root / "profiler" / "perf" / profile / model_path
    variant_root = root / variant
    if not variant_root.is_dir():
        variants = sorted(path.name for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
        return (
            f"PROFILE_UNAVAILABLE: hardware={profile} model={deployment.model_name} "
            f"variant={variant} available_variants={variants}"
        )
    if not (variant_root / "meta.yaml").is_file():
        return f"PROFILE_UNAVAILABLE: meta.yaml missing at {variant_root}"
    try:
        tp = int(deployment.shape.get("tp", 1))
    except (TypeError, ValueError):
        return f"PROFILE_UNAVAILABLE: invalid tp={deployment.shape.get('tp')!r}"
    if not (variant_root / f"tp{tp}").is_dir():
        available_tp = sorted(
            int(path.name[2:])
            for path in variant_root.iterdir()
            if path.is_dir() and path.name.startswith("tp") and path.name[2:].isdigit()
        )
        return (
            f"PROFILE_UNAVAILABLE: hardware={profile} model={deployment.model_name} "
            f"variant={variant} tp={tp} available_tp={available_tp}"
        )
    return None


def resolve_rank_hardware(
    deployment: RankDeployment,
    catalog: Mapping[str, object],
) -> dict[str, object]:
    """Resolve one rank's Store catalog entry into LLMServingSim hardware facts."""
    shape = deployment.shape
    _, cloud, region, _, gpu_type = shape["env"]
    instance_type = shape["instance_type"]
    regions = list(catalog["regions"])
    for region_catalog in sorted(
        regions,
        key=lambda item: item["cloud"] != cloud or item["region"] != region,
    ):
        if region_catalog["cloud"] != cloud:
            continue
        for instance in region_catalog["instance_types"]:
            if instance["instance_type"] != instance_type:
                continue
            for accelerator in instance["accelerators"]:
                if _gpu_name(gpu_type) not in {
                    _gpu_name(accelerator.get("name")),
                    _gpu_name(accelerator.get("canonical_gpu_name")),
                }:
                    continue
                profile = PROFILE_ALIASES.get(gpu_type)
                if profile is None:
                    raise ValueError(f"no LLMServingSim profile alias for {gpu_type}")
                link = (
                    accelerator.get("infinity_fabric_bandwidth_gbps")
                    or accelerator.get("nvlink_bandwidth_gbps")
                    or accelerator.get("pcie_bandwidth_gbps")
                )
                if not link:
                    link = max(card["peak_bandwidth_gbps"] for card in instance["network"]["network_cards"])
                return {
                    "profile": profile,
                    "memory_gb": accelerator["memory_mib_each"] / 1024,
                    "memory_bandwidth_gb_s": accelerator["gpu_bandwidth_gbps"],
                    "link_bandwidth_gb_s": link,
                    "link_latency_ns": 0,
                }
    raise ValueError(f"hardware catalog has no {cloud}/{region}/{instance_type}/{gpu_type} entry")


def _gpu_name(value: object) -> str:
    return str(value or "").upper().replace("_", "-")


def merge_hardware_catalogs(catalogs: Sequence[object]) -> dict[str, object]:
    """Normalize every seeded cloud catalog into resolver regions."""
    regions = []
    for hardware in catalogs:
        catalog = hardware.catalog
        source_regions = catalog.get("regions", [])
        if source_regions and isinstance(source_regions[0], dict):
            regions.extend(source_regions)
        elif catalog.get("cloud") == "azure":
            for region in source_regions:
                regions.append({
                    "cloud": "azure",
                    "region": region,
                    "instance_types": [
                        instance for instance in catalog["instance_types"]
                        if any(offering["region"] == region for offering in instance["offerings"])
                    ],
                })
    return {"regions": regions}


def write_rank_workload(
    requests: Sequence[Request],
    output_path: Path,
    tick_start_ns: int,
) -> None:
    """Write one rank's routed requests as LLMServingSim flat-request JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for request in requests:
            row = {
                "input_toks": request.input_tokens,
                "output_toks": request.output_tokens,
                "arrival_time_ns": request.arrives_at_ns - tick_start_ns,
            }
            if request.prefix_key is not None:
                row["input_tok_ids"] = _prefix_tokens(request.prefix_key, request.input_tokens)
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _prefix_tokens(prefix_key: str, input_tokens: int) -> list[int]:
    """Create stable synthetic token IDs so equal prefix keys hit the same cache."""
    length = min(int(prefix_key.rsplit("-", 1)[1]), input_tokens)
    seed = int.from_bytes(hashlib.sha256(prefix_key.encode()).digest()[:4], "big")
    return [seed + index for index in range(length)]


def build_rank_command(
    container: str,
    llmservingsim_root: Path,
    cluster_path: Path,
    workload_path: Path,
    output_path: Path,
    metrics_path: Path,
    run_name: str,
    request_count: int,
) -> tuple[str, ...]:
    """Build one Docker command using paths visible inside LLMServingSim."""
    root = llmservingsim_root.resolve()

    def relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(root))
        except ValueError as error:
            raise ValueError(f"{path} must be inside {root}") from error

    return (
        "docker",
        "exec",
        "-i",
        "-w",
        "/app/LLMServingSim",
        container,
        "python3",
        "-m",
        "serving",
        "--cluster-config",
        relative(cluster_path),
        "--dataset",
        relative(workload_path),
        "--output",
        relative(output_path),
        "--metrics-output",
        relative(metrics_path),
        "--num-reqs",
        str(request_count),
        "--max-wall-time-seconds",
        str(SERVING_WINDOW_SECONDS),
        "--request-routing-policy",
        "RR",
        "--run-id",
        run_name,
        "--log-level",
        "WARNING",
        "--cleanup-inputs",
    )


def prepare_rank_run(
    deployment: RankDeployment,
    requests: Sequence[Request],
    hardware: Mapping[str, object],
    llmservingsim_root: Path,
    run_dir: Path,
    container: str,
    run_name: str,
    tick_start_ns: int,
    timeout_seconds: float,
) -> RankRun:
    """Write one rank's files and return its ready-to-run Docker command."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cluster_path = run_dir / "cluster.json"
    workload_path = run_dir / "workload.jsonl"
    output_path = run_dir / "requests.csv"
    metrics_path = run_dir / "metrics.jsonl"
    cluster_path.write_text(json.dumps(build_rank_cluster(deployment, hardware)))
    write_rank_workload(requests, workload_path, tick_start_ns)
    return RankRun(
        deployment,
        tuple(requests),
        build_rank_command(
            container,
            llmservingsim_root,
            cluster_path,
            workload_path,
            output_path,
            metrics_path,
            run_name,
            len(requests),
        ),
        output_path,
        metrics_path,
        len(requests),
        timeout_seconds,
        run_dir,
    )


def run_ranks(
    rank_runs: Sequence[RankRun],
    max_parallel: int,
    deadline_epoch_seconds: float | None = None,
    progress=None,
) -> tuple[RankResult, ...]:
    """Run independent rank simulations concurrently and fail as one batch."""
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(_run_rank, rank_run, deadline_epoch_seconds): index
            for index, rank_run in enumerate(rank_runs)
        }
        results = [None] * len(rank_runs)
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            results[index] = future.result()
            if progress is not None:
                progress(completed, len(rank_runs), rank_runs[index])
        return tuple(results)


def _run_rank(
    rank_run: RankRun, deadline_epoch_seconds: float | None = None
) -> RankResult:
    """Run and parse one rank simulation command."""
    if deadline_epoch_seconds is not None and time() >= deadline_epoch_seconds:
        if rank_run.cleanup_dir is not None:
            shutil.rmtree(rank_run.cleanup_dir)
        return RankResult(
            rank_run.deployment,
            (),
            (),
            "",
            "",
            "wall_time_limit",
            (),
        )
    command = rank_run.command
    if deadline_epoch_seconds is not None:
        command = (
            *command,
            "--wall-deadline-epoch-seconds",
            str(deadline_epoch_seconds),
        )
    try:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=rank_run.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"{rank_run.deployment.rank_name} simulation timed out") from error
        return parse_rank_result(rank_run, completed)
    finally:
        if rank_run.cleanup_dir is not None and rank_run.cleanup_dir.exists():
            shutil.rmtree(rank_run.cleanup_dir)


def parse_rank_result(
    rank_run: RankRun,
    completed: subprocess.CompletedProcess[str],
) -> RankResult:
    """Parse one rank's request CSV and periodic metrics artifact."""
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode:
        lowered = output.lower()
        if (
            "memoryerror" in lowered
            or "out of memory" in lowered
            or (
                "[memorymodel]" in lowered
                and "exceeds total npu memory" in lowered
            )
        ):
            return RankResult(
                rank_run.deployment,
                (),
                (),
                completed.stdout,
                completed.stderr,
                "oom",
            )
        raise RuntimeError(f"{rank_run.deployment.rank_name} failed: {output[-1000:]}")
    summary = _run_summary(output)
    if not rank_run.output_path.is_file():
        raise RuntimeError(f"{rank_run.deployment.rank_name} wrote no request CSV")

    with rank_run.output_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_HEADER:
            raise ValueError(f"unexpected LLMServingSim CSV header: {reader.fieldnames!r}")
        rows = tuple(_row(row) for row in reader)
    if summary["completed_requests"] != len(rows):
        raise ValueError("LLMServingSim summary does not match completed request rows")
    if summary["loaded_requests"] != rank_run.request_count:
        raise ValueError("LLMServingSim summary does not match submitted requests")
    if summary["status"] == "completed" and len(rows) != rank_run.request_count:
        raise ValueError(f"expected {rank_run.request_count} request rows, got {len(rows)}")
    if len(rows) > rank_run.request_count:
        raise ValueError(f"expected at most {rank_run.request_count} request rows, got {len(rows)}")
    completed_ids = []
    completed_requests = []
    for row in rows:
        request_id = row["request id"]
        if not math.isfinite(request_id) or request_id != int(request_id):
            raise ValueError("LLMServingSim returned an invalid request ID")
        request_id = int(request_id)
        if request_id < 0 or request_id >= len(rank_run.requests):
            raise ValueError("LLMServingSim returned an unknown request ID")
        completed_ids.append(request_id)
        completed_requests.append(rank_run.requests[request_id])
    if len(set(completed_ids)) != len(completed_ids):
        raise ValueError("LLMServingSim returned duplicate request IDs")

    expected_instances = set(range(len(rank_run.deployment.chains)))
    for row in rows:
        instance_id = _whole_number(row["instance id"], "instance ID")
        if instance_id not in expected_instances:
            raise ValueError("LLMServingSim returned an unknown instance ID")
    metric_samples = _read_metric_samples(rank_run.metrics_path, expected_instances)

    return RankResult(
        rank_run.deployment,
        rows,
        metric_samples,
        completed.stdout,
        completed.stderr,
        summary["status"],
        tuple(completed_requests),
        summary["sim_time_ns"],
        tuple(rank_run.requests[request_id] for request_id in summary["admitted_request_ids"]),
    )


def _run_summary(output: str) -> dict[str, object]:
    """Return and validate the terminal LLMServingSim status."""
    for line in reversed(output.splitlines()):
        if not line.startswith("SIM_RUN_SUMMARY "):
            continue
        try:
            summary = json.loads(line.removeprefix("SIM_RUN_SUMMARY "))
            status = summary["status"]
            loaded = summary["loaded_requests"]
            completed = summary["completed_requests"]
            unfinished = summary["unfinished_requests"]
            sim_time_ns = summary["sim_time_ns"]
            admitted = summary.get("admitted_request_ids", [])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError("invalid LLMServingSim run summary") from error
        if (
            status not in {"completed", "wall_time_limit"}
            or any(
                type(value) is not int or value < 0
                for value in (loaded, completed, unfinished, sim_time_ns)
            )
            or loaded != completed + unfinished
            or not isinstance(admitted, list)
            or any(type(value) is not int or value < 0 or value >= loaded for value in admitted)
            or len(set(admitted)) != len(admitted)
        ):
            raise ValueError("invalid LLMServingSim run summary")
        return {
            "status": status,
            "loaded_requests": loaded,
            "completed_requests": completed,
            "unfinished_requests": unfinished,
            "sim_time_ns": sim_time_ns,
            "admitted_request_ids": admitted,
        }
    raise ValueError("LLMServingSim wrote no run summary")


def _row(row: Mapping[str, str]) -> dict[str, float]:
    """Convert one validated LLMServingSim CSV row to numeric values."""
    try:
        return {
            key: float(value)
            for key, value in row.items()
            if key not in {"model", "ITL"}
        }
    except ValueError as error:
        raise ValueError("LLMServingSim CSV contains a nonnumeric value") from error


def _whole_number(value: object, name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}") from error
    if not math.isfinite(numeric) or numeric != int(numeric):
        raise ValueError(f"invalid {name}")
    return int(numeric)


def _read_metric_samples(path: Path, expected_instances: set[int]) -> tuple[RankMetricSample, ...]:
    """Read every periodic LLMServingSim JSONL metric sample for one rank."""
    if not path.is_file():
        raise ValueError(f"LLMServingSim wrote no metrics JSONL: {path}")

    samples = []
    by_time: dict[int, set[int]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid metrics JSONL at line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"metrics JSONL line {line_number} must be an object")
        if row.get("sample_kind") not in {"periodic", "final"}:
            raise ValueError(f"metrics JSONL line {line_number} has invalid sample_kind")
        try:
            sim_time_ns = _whole_number(row["sim_time_ns"], "metric sim_time_ns")
            instance_id = _whole_number(row["instance_id"], "metric instance_id")
            sample_interval_ns = (
                _whole_number(row["sample_interval_ns"], "metric sample_interval_ns")
                if row["sample_kind"] == "periodic"
                else sim_time_ns
            )
        except (KeyError, ValueError) as error:
            raise ValueError(f"metrics JSONL line {line_number} has invalid identity") from error
        if sim_time_ns < 0 or sample_interval_ns <= 0:
            raise ValueError(f"metrics JSONL line {line_number} has invalid interval")
        if instance_id not in expected_instances:
            raise ValueError(f"metrics JSONL line {line_number} has unknown instance")

        values = {}
        for name, value in row.items():
            if name in {"sim_time_ns", "sample_interval_ns", "sample_kind", "instance_id"}:
                continue
            if value is None:
                values[name] = None
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"metrics JSONL line {line_number} has invalid {name}") from error
            if not math.isfinite(numeric):
                raise ValueError(f"metrics JSONL line {line_number} has non-finite {name}")
            values[name] = numeric
        samples.append(
            RankMetricSample(
                sim_time_ns,
                sample_interval_ns,
                instance_id,
                values,
                row["sample_kind"],
            )
        )
        by_time.setdefault(sim_time_ns, set()).add(instance_id)

    for sim_time_ns, instances in by_time.items():
        if instances != expected_instances:
            raise ValueError(f"metrics sample at {sim_time_ns} is missing rank instances")
    return tuple(samples)


def gpu_metrics_from_results(
    job: object,
    deployments: Sequence[RankDeployment],
    results: Sequence[RankResult],
    tick_start: datetime,
    tick_duration_ns: int,
    prices_by_pool: Mapping[tuple[str, str], tuple[float, int]],
    *,
    batch_measurements: Mapping[str, BatchProbeMeasurement] | None = None,
    batch_active_ns: int | None = None,
) -> list[object]:
    """Derive per-chain and per-GPU Store telemetry from rank results."""
    from tandemn_system_data.models import GpuMetric

    results_by_rank = {result.deployment.rank_name: result for result in results}
    metrics = []
    for deployment in deployments:
        result = results_by_rank[deployment.rank_name]
        if batch_measurements is not None:
            measurement = batch_measurements.get(deployment.rank_name)
            if measurement is not None:
                metrics.extend(
                    _project_batch_probe_metrics(
                        job,
                        deployment,
                        result,
                        measurement,
                        tick_start,
                        tick_duration_ns,
                        prices_by_pool,
                        batch_active_ns,
                    )
                )
            continue
        shape = deployment.shape
        target_ttft = float(shape.get("target_p99_ttft_ms", 0))
        prefix_cache_enabled = bool(shape.get("prefix_cache_enabled", False))
        samples = _latest_samples_per_bucket(result.metric_samples, tick_duration_ns)

        for instance_id, binding in enumerate(deployment.chains):
            rows = [row for row in result.request_rows if int(row["instance id"]) == instance_id]
            for sample in samples.get(instance_id, ()):
                direct = {
                    name: value
                    for name, value in sample.metrics.items()
                    if name in GpuMetric.model_fields and name != "kvcache_hit_rate"
                }
                interval_rows = _completed_in_sample(result.request_rows, sample)
                observed = _observed_y(
                    _completed_in_sample(rows, sample),
                    sample.sample_interval_ns,
                    target_ttft,
                    _chain_price_per_hour(deployment, binding, prices_by_pool),
                    shape.get("task"),
                )
                values = {
                    **direct,
                    **observed,
                    "kvcache_hit_rate": (
                        _kvcache_hit_rate(interval_rows)
                        if prefix_cache_enabled
                        else None
                    ),
                }
                for local_rank in range(binding.gpu_count):
                    metrics.append(
                        GpuMetric(
                            ts=_logical_sample_time(tick_start, sample.sim_time_ns),
                            job_id=job.job_id,
                            gpu_uuid=(
                                f"sim:{binding.rank_id}:{binding.chain_index}:{local_rank}"
                            ),
                            rank_id=binding.rank_id,
                            chain_index=binding.chain_index,
                            local_rank=str(local_rank),
                            role="aggregate",
                            instance_type=str(shape["instance_type"]),
                            model_name=deployment.model_name,
                            **values,
                        )
                    )
    return metrics


def _project_batch_probe_metrics(
    job: object,
    deployment: RankDeployment,
    result: RankResult,
    measurement: BatchProbeMeasurement,
    tick_start: datetime,
    tick_duration_ns: int,
    prices_by_pool: Mapping[tuple[str, str], tuple[float, int]],
    active_ns: int | None,
) -> list[object]:
    """Keep observed probe buckets, then project until the batch stops."""
    from tandemn_system_data.models import GpuMetric

    instance_type = str(deployment.shape["instance_type"])
    observed = _batch_probe_bucket_metrics(result, prices_by_pool)
    chain_metrics = batch_probe_chain_metrics(
        result,
        measurement,
        prices_by_pool,
    )
    active_ns = tick_duration_ns if active_ns is None else min(tick_duration_ns, active_ns)
    metrics = []
    for bucket_start_ns in range(0, active_ns, SAMPLE_BUCKET_NS):
        timestamp = _logical_sample_time(
            tick_start,
            bucket_start_ns + SAMPLE_BUCKET_NS // 2,
        )
        for instance_id, binding in enumerate(deployment.chains):
            values = observed.get(instance_id, {}).get(bucket_start_ns)
            if values is None:
                if bucket_start_ns < measurement.horizon_ns:
                    continue
                values = chain_metrics[instance_id]
            for local_rank in range(binding.gpu_count):
                metrics.append(
                    GpuMetric(
                        ts=timestamp,
                        job_id=job.job_id,
                        gpu_uuid=(
                            f"sim:{binding.rank_id}:{binding.chain_index}:{local_rank}"
                        ),
                        rank_id=binding.rank_id,
                        chain_index=binding.chain_index,
                        local_rank=str(local_rank),
                        role="aggregate",
                        instance_type=instance_type,
                        model_name=deployment.model_name,
                        **values,
                    )
                )
    return metrics


def _batch_probe_bucket_metrics(
    result: RankResult,
    prices_by_pool: Mapping[tuple[str, str], tuple[float, int]],
) -> dict[int, dict[int, dict[str, float | None]]]:
    """Return measured chain metrics in canonical probe-time buckets."""
    from tandemn_system_data.models import GpuMetric

    selected = {}
    periodic_instances = {
        sample.instance_id
        for sample in result.metric_samples
        if sample.sample_kind == "periodic"
    }
    for sample in result.metric_samples:
        if sample.sample_kind == "final" and sample.instance_id in periodic_instances:
            continue
        bucket_start_ns = max(
            0,
            ((sample.sim_time_ns - 1) // SAMPLE_BUCKET_NS) * SAMPLE_BUCKET_NS,
        )
        key = (sample.instance_id, bucket_start_ns)
        previous = selected.get(key)
        if previous is None or _probe_sample_priority(sample) > _probe_sample_priority(
            previous
        ):
            selected[key] = sample

    target_ttft = float(result.deployment.shape.get("target_p99_ttft_ms", 0))
    prefix_cache_enabled = bool(
        result.deployment.shape.get("prefix_cache_enabled", False)
    )
    metrics: dict[int, dict[int, dict[str, float | None]]] = {}
    for (instance_id, bucket_start_ns), sample in selected.items():
        rows = _completed_in_sample(
            tuple(
                row
                for row in result.request_rows
                if int(row["instance id"]) == instance_id
            ),
            sample,
        )
        direct = {
            name: value
            for name, value in sample.metrics.items()
            if name in GpuMetric.model_fields and name != "kvcache_hit_rate"
        }
        metrics.setdefault(instance_id, {})[bucket_start_ns] = {
            **direct,
            **_observed_y(
                rows,
                sample.sample_interval_ns,
                target_ttft,
                _chain_price_per_hour(
                    result.deployment,
                    result.deployment.chains[instance_id],
                    prices_by_pool,
                ),
                result.deployment.shape.get("task"),
            ),
            "kvcache_hit_rate": (
                _kvcache_hit_rate(rows) if prefix_cache_enabled else None
            ),
        }
    return metrics


def _probe_sample_priority(sample: RankMetricSample) -> tuple[bool, int]:
    """Prefer interval telemetry over a terminal snapshot in the same bucket."""
    return sample.sample_kind == "periodic", sample.sim_time_ns


def batch_probe_chain_metrics(
    result: RankResult,
    measurement: BatchProbeMeasurement,
    prices_by_pool: Mapping[tuple[str, str], tuple[float, int]],
) -> dict[int, dict[str, float | None]]:
    """Build one stationary metric set for each chain in a batch probe."""
    from tandemn_system_data.models import GpuMetric

    shape = result.deployment.shape
    target_ttft = float(shape.get("target_p99_ttft_ms", 0))
    prefix_cache_enabled = bool(shape.get("prefix_cache_enabled", False))
    metrics = {}
    for instance_id in range(len(result.deployment.chains)):
        rows = tuple(
            row
            for row in result.request_rows
            if int(row["instance id"]) == instance_id
        )
        sample = _stationary_probe_sample(result.metric_samples, instance_id)
        direct = {
            name: value
            for name, value in (sample.metrics if sample is not None else {}).items()
            if name in GpuMetric.model_fields and name != "kvcache_hit_rate"
        }
        metrics[instance_id] = {
            **direct,
            **_observed_y(
                rows,
                measurement.horizon_ns,
                target_ttft,
                _chain_price_per_hour(
                    result.deployment,
                    result.deployment.chains[instance_id],
                    prices_by_pool,
                ),
                shape.get("task"),
            ),
            "kvcache_hit_rate": _kvcache_hit_rate(rows) if prefix_cache_enabled else None,
        }
    return metrics


def _chain_price_per_hour(
    deployment: RankDeployment,
    binding: ChainBinding,
    prices_by_pool: Mapping[tuple[str, str], tuple[float, int]],
) -> float | None:
    """Return the full instance cost of one chain's allocated nodes."""
    environment = deployment.shape.get("env")
    if not isinstance(environment, list):
        return None
    pool = prices_by_pool.get(
        ("|".join(str(value) for value in environment), deployment.shape["instance_type"])
    )
    if pool is None:
        return None
    price_per_instance, gpus_per_instance = pool
    return price_per_instance * math.ceil(binding.gpu_count / gpus_per_instance)


def _stationary_probe_sample(
    samples: Sequence[RankMetricSample],
    instance_id: int,
) -> RankMetricSample | None:
    """Prefer steady-state samples and use the final sample only as fallback."""
    matching = tuple(sample for sample in samples if sample.instance_id == instance_id)
    periodic = tuple(sample for sample in matching if sample.sample_kind == "periodic")
    return max(periodic or matching, key=lambda sample: sample.sim_time_ns, default=None)


def _latest_samples_per_bucket(
    samples: Sequence[RankMetricSample], tick_duration_ns: int
) -> dict[int, tuple[RankMetricSample, ...]]:
    """Keep the latest periodic sample per instance in each Koi-sized bucket."""
    selected: dict[tuple[int, int], RankMetricSample] = {}
    for sample in samples:
        if sample.sim_time_ns >= tick_duration_ns:
            continue
        key = (sample.instance_id, sample.sim_time_ns // SAMPLE_BUCKET_NS)
        previous = selected.get(key)
        if previous is None or sample.sim_time_ns > previous.sim_time_ns:
            selected[key] = sample
    by_instance: dict[int, list[RankMetricSample]] = {}
    for sample in selected.values():
        by_instance.setdefault(sample.instance_id, []).append(sample)
    return {
        instance_id: tuple(sorted(instance_samples, key=lambda sample: sample.sim_time_ns))
        for instance_id, instance_samples in by_instance.items()
    }


def _completed_in_sample(
    rows: Sequence[dict[str, float]], sample: RankMetricSample
) -> tuple[dict[str, float], ...]:
    """Return requests completed in one half-open simulator sample interval."""
    start_ns = max(0, sample.sim_time_ns - sample.sample_interval_ns)
    return tuple(
        row for row in rows if start_ns < row["end_time"] <= sample.sim_time_ns
    )


def _logical_sample_time(tick_start: datetime, sim_time_ns: int) -> datetime:
    """Map a simulator-local sample to its logical cluster timestamp."""
    return tick_start + timedelta(microseconds=max(0, sim_time_ns) // 1_000)


def _kvcache_hit_rate(rows: Sequence[dict[str, float]]) -> float | None:
    """Return the token-weighted NPU prefix-cache hit rate for request rows."""
    requested = sum(row.get("prefix_cache_requested_tokens", 0.0) for row in rows)
    hits = sum(row.get("npu_prefix_hit_tokens", 0.0) for row in rows)
    if requested <= 0:
        return None
    if hits < 0 or hits > requested:
        raise ValueError("invalid NPU prefix-cache hit totals")
    return hits / requested


def _observed_y(
    rows: Sequence[dict[str, float]],
    horizon_ns: float,
    target_ttft: float,
    price_per_hour: float | None,
    task: object = None,
) -> dict[str, float | None]:
    """Derive chain outcomes from its completed LLMServingSim request rows."""
    if not rows:
        return {
            "throughput_token_per_sec": 0.0,
            "p99_ttft_ms": None,
            "p99_tpot_ms": None,
            "cost_per_token": None,
            "slo_margin": None,
            "input_length_observed": None,
            "output_length_observed": None,
        }

    throughput = sum(row["output"] for row in rows) * SECOND / horizon_ns
    encoder = task == "embeddings"
    p99_ttft = _percentile(
        [row["latency"] if encoder else row["TTFT"] for row in rows], 0.99
    ) / 1_000_000
    p99_tpot = None if encoder else _percentile(
        [row["TPOT"] for row in rows], 0.99
    ) / 1_000_000
    return {
        "throughput_token_per_sec": throughput,
        "p99_ttft_ms": p99_ttft,
        "p99_tpot_ms": p99_tpot,
        "cost_per_token": price_per_hour / (throughput * 3600) if price_per_hour and throughput else None,
        "slo_margin": (target_ttft - p99_ttft) / target_ttft if target_ttft else None,
        "input_length_observed": sum(row["input"] for row in rows) / len(rows),
        "output_length_observed": sum(row["output"] for row in rows) / len(rows),
    }


def completed_request_summary(
    results: Sequence[RankResult],
    target_ttft_ms: float | None = None,
    target_tpot_ms: float | None = None,
    task: object = None,
) -> dict[str, int | float | None]:
    """Summarize completed request latency across every rank of one job."""
    rows = [row for result in results for row in result.request_rows]
    if not rows:
        return {
            "completed_request_count": 0,
            "p99_ttft_ms": None,
            "p99_tpot_ms": None,
            "ttft_pass_count": 0 if target_ttft_ms is not None else None,
            "ttft_fail_count": 0 if target_ttft_ms is not None else None,
            "tpot_pass_count": None if task == "embeddings" else 0 if target_tpot_ms is not None else None,
            "tpot_fail_count": None if task == "embeddings" else 0 if target_tpot_ms is not None else None,
            "slo_pass_count": 0,
            "slo_fail_count": 0,
            "qualified_output_tokens": 0 if target_ttft_ms is not None and target_tpot_ms is not None else None,
        }
    encoder = task == "embeddings"
    ttft_passed = sum(
        (row["latency"] if encoder else row["TTFT"]) <= target_ttft_ms * 1_000_000
        for row in rows
    ) if target_ttft_ms is not None else None
    tpot_passed = None if encoder else sum(
        row["TPOT"] <= target_tpot_ms * 1_000_000 for row in rows
    ) if target_tpot_ms is not None else None
    passed = sum(
        row["TTFT"] <= target_ttft_ms * 1_000_000
        and row["TPOT"] <= target_tpot_ms * 1_000_000
        for row in rows
    ) if not encoder and ttft_passed is not None and tpot_passed is not None else ttft_passed
    return {
        "completed_request_count": len(rows),
        "p99_ttft_ms": _percentile(
            [row["latency"] if encoder else row["TTFT"] for row in rows], 0.99
        ) / 1_000_000,
        "p99_tpot_ms": None if encoder else _percentile([row["TPOT"] for row in rows], 0.99) / 1_000_000,
        "ttft_pass_count": ttft_passed,
        "ttft_fail_count": len(rows) - ttft_passed if ttft_passed is not None else None,
        "tpot_pass_count": tpot_passed,
        "tpot_fail_count": len(rows) - tpot_passed if tpot_passed is not None else None,
        "slo_pass_count": passed,
        "slo_fail_count": len(rows) - passed if passed is not None else None,
        "qualified_output_tokens": sum(
            row["output"] for row in rows
            if (row["latency"] if encoder else row["TTFT"]) <= target_ttft_ms * 1_000_000
            and (encoder or row["TPOT"] <= target_tpot_ms * 1_000_000)
        ) if passed is not None else None,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return one linearly interpolated percentile without another dependency."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def commit_tick_metrics(store: object, metrics: Sequence[object]) -> None:
    """Atomically persist all fully validated telemetry for one logical tick."""
    store.put_many(metrics)
