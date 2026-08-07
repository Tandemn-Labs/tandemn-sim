"""Build per-rank LLMServingSim inputs and later convert results to telemetry."""

from __future__ import annotations

import hashlib
import json
import csv
import re
import shutil
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from models import ChainBinding, RankDeployment, Request


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
]
METRICS_LINE = re.compile(r"^SIM_METRICS instance=(\d+) (.*)$", re.MULTILINE)
SECOND = 1_000_000_000
PROFILE_ALIASES = {
    "H100-SXM-80GB": "H100",
    "A100-SXM-80GB": "A100-80GB",
    "A100-PCIe-40GB": "A100-40GB",
    "A10G-24GB": "A10G",
    "L40S-48GB": "L40S",
    # "MI300X-192GB": "MI300X",  # Enable after the MI300X profile is added.
}


@dataclass(frozen=True)
class RankRun:
    """One rank-local LLMServingSim command and its expected output."""

    deployment: RankDeployment
    command: tuple[str, ...]
    output_path: Path
    request_count: int
    timeout_seconds: float
    cleanup_dir: Path | None = None


@dataclass(frozen=True)
class RankResult:
    """Validated request results and final runtime metrics for one rank."""

    deployment: RankDeployment
    request_rows: tuple[dict[str, float], ...]
    metrics_by_instance: dict[int, dict[str, float | None]]
    stdout: str
    stderr: str


def build_rank_deployments(job: object, active_chains: Sequence[object]) -> tuple[RankDeployment, ...]:
    """Group active Store chains into independent homogeneous rank deployments."""
    groups: dict[str, list[object]] = {}
    for chain in active_chains:
        rank_name = chain.shape_json["rank_id"]
        groups.setdefault(rank_name, []).append(chain)

    deployments = []
    for rank_name, chains in sorted(groups.items()):
        shape = dict(chains[0].shape_json)
        if any(chain.shape_json != shape for chain in chains):
            raise ValueError(f"{rank_name} chains must have one shared shape")
        count = int(shape["count"])
        bindings = tuple(
            ChainBinding(rank_name, chain.chain_id, number, count)
            for number, chain in enumerate(sorted(chains, key=lambda item: item.chain_id))
        )
        deployments.append(
            RankDeployment(
                store_job_id=job.job_id,
                rank_name=rank_name,
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


def resolve_rank_hardware(
    deployment: RankDeployment,
    catalog: Mapping[str, object],
) -> dict[str, object]:
    """Resolve one rank's Store catalog entry into LLMServingSim hardware facts."""
    shape = deployment.shape
    _, cloud, region, _, gpu_type = shape["env"]
    instance_type = shape["instance_type"]
    for region_catalog in catalog["regions"]:
        if region_catalog["cloud"] != cloud or region_catalog["region"] != region:
            continue
        for instance in region_catalog["instance_types"]:
            if instance["instance_type"] != instance_type:
                continue
            for accelerator in instance["accelerators"]:
                if gpu_type not in {
                    accelerator.get("name"),
                    accelerator.get("canonical_gpu_name"),
                }:
                    continue
                profile = PROFILE_ALIASES.get(gpu_type)
                if profile is None:
                    raise ValueError(f"no LLMServingSim profile alias for {gpu_type}")
                link = accelerator.get("nvlink_bandwidth_gbps") or accelerator.get(
                    "pcie_bandwidth_gbps"
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
        "--num-reqs",
        str(request_count),
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
    cluster_path.write_text(json.dumps(build_rank_cluster(deployment, hardware)))
    write_rank_workload(requests, workload_path, tick_start_ns)
    return RankRun(
        deployment,
        build_rank_command(
            container,
            llmservingsim_root,
            cluster_path,
            workload_path,
            output_path,
            run_name,
            len(requests),
        ),
        output_path,
        len(requests),
        timeout_seconds,
        run_dir,
    )


def run_ranks(rank_runs: Sequence[RankRun], max_parallel: int) -> tuple[RankResult, ...]:
    """Run independent rank simulations concurrently and fail as one batch."""
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [executor.submit(_run_rank, rank_run) for rank_run in rank_runs]
        return tuple(future.result() for future in futures)


def _run_rank(rank_run: RankRun) -> RankResult:
    """Run and parse one rank simulation command."""
    try:
        completed = subprocess.run(
            rank_run.command,
            capture_output=True,
            text=True,
            timeout=rank_run.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{rank_run.deployment.rank_name} simulation timed out") from error
    result = parse_rank_result(rank_run, completed)
    if rank_run.cleanup_dir is not None:
        shutil.rmtree(rank_run.cleanup_dir)
    return result


def parse_rank_result(
    rank_run: RankRun,
    completed: subprocess.CompletedProcess[str],
) -> RankResult:
    """Parse one rank's request CSV and final SIM_METRICS output."""
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode:
        raise RuntimeError(f"{rank_run.deployment.rank_name} failed: {output[-1000:]}")
    if "ERROR: Some Requests Remain" in output or "All Request Has Been Exited" not in output:
        raise RuntimeError(f"{rank_run.deployment.rank_name} did not finish all requests")
    if not rank_run.output_path.is_file():
        raise RuntimeError(f"{rank_run.deployment.rank_name} wrote no request CSV")

    with rank_run.output_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_HEADER:
            raise ValueError(f"unexpected LLMServingSim CSV header: {reader.fieldnames!r}")
        rows = tuple(_row(row) for row in reader)
    if len(rows) != rank_run.request_count:
        raise ValueError(f"expected {rank_run.request_count} request rows, got {len(rows)}")
    if len({row["request id"] for row in rows}) != len(rows):
        raise ValueError("LLMServingSim returned duplicate request IDs")

    metrics = {
        instance: values
        for instance, values in (_metrics(match) for match in METRICS_LINE.finditer(output))
    }
    expected_instances = set(range(len(rank_run.deployment.chains)))
    if set(metrics) != expected_instances:
        raise ValueError(f"missing final metrics for rank instances: {expected_instances - set(metrics)}")
    if any(int(row["instance id"]) not in expected_instances for row in rows):
        raise ValueError("LLMServingSim returned an unknown instance ID")

    return RankResult(rank_run.deployment, rows, metrics, completed.stdout, completed.stderr)


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


def _metrics(match: re.Match[str]) -> tuple[int, dict[str, float | None]]:
    """Parse one final runtime-metrics line for one simulator instance."""
    values: dict[str, float | None] = {}
    for pair in match.group(2).split():
        key, value = pair.split("=", 1)
        values[key] = None if value == "N/A" else float(value)
    return int(match.group(1)), values


def gpu_metrics_from_results(
    job: object,
    deployments: Sequence[RankDeployment],
    results: Sequence[RankResult],
    sample_time: datetime,
    price_by_instance_type: Mapping[str, float],
) -> list[object]:
    """Derive per-chain and per-GPU Store telemetry from rank results."""
    from tandemn_system_data.models import GpuMetric

    horizon_ns = max(
        (row["end_time"] for result in results for row in result.request_rows),
        default=0,
    )
    if horizon_ns <= 0:
        raise ValueError("rank results require a positive simulation horizon")

    results_by_rank = {result.deployment.rank_name: result for result in results}
    metrics = []
    for deployment in deployments:
        result = results_by_rank[deployment.rank_name]
        shape = deployment.shape
        instance_type = str(shape["instance_type"])
        target_ttft = float(shape.get("target_p99_ttft_ms", 0))
        price_per_hour = price_by_instance_type.get(instance_type)

        for instance_id, binding in enumerate(deployment.chains):
            rows = [row for row in result.request_rows if int(row["instance id"]) == instance_id]
            direct = {
                name: value
                for name, value in result.metrics_by_instance[instance_id].items()
                if name in GpuMetric.model_fields
            }
            observed = _observed_y(rows, horizon_ns, target_ttft, price_per_hour)
            values = {**direct, **observed}
            for local_rank in range(binding.gpu_count):
                metrics.append(
                    GpuMetric(
                        ts=sample_time,
                        job_id=job.job_id,
                        gpu_uuid=f"sim:{binding.chain_id}:{local_rank}",
                        rank_id=deployment.rank_name,
                        chain_id=binding.chain_id,
                        local_rank=str(local_rank),
                        role="aggregate",
                        instance_type=instance_type,
                        model_name=deployment.model_name,
                        **values,
                    )
                )
    return metrics


def _observed_y(
    rows: Sequence[dict[str, float]],
    horizon_ns: float,
    target_ttft: float,
    price_per_hour: float | None,
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
    p99_ttft = _percentile([row["TTFT"] for row in rows], 0.99) / 1_000_000
    p99_tpot = _percentile([row["TPOT"] for row in rows], 0.99) / 1_000_000
    return {
        "throughput_token_per_sec": throughput,
        "p99_ttft_ms": p99_ttft,
        "p99_tpot_ms": p99_tpot,
        "cost_per_token": price_per_hour / (throughput * 3600) if price_per_hour and throughput else None,
        "slo_margin": (target_ttft - p99_ttft) / target_ttft if target_ttft else None,
        "input_length_observed": sum(row["input"] for row in rows) / len(rows),
        "output_length_observed": sum(row["output"] for row in rows) / len(rows),
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
