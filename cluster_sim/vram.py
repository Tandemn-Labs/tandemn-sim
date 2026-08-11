"""Calculate and schedule benchmark workloads by maximum VRAM demand."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

from models import CreatedPlan, PlannerResult, Scenario
from policy import ConfigPolicy, ConfigPolicyRule, load_config_policy

VRAM_REQUIREMENTS_GIB: dict[str, dict[str, float]] = {
    "real_life_12h": {
        "job-benchmark_timing_real_12hr_192gpu-0-1": 320.905,
        "job-benchmark_timing_real_12hr_192gpu-0-2": 179.304,
        "job-benchmark_timing_real_12hr_192gpu-0-3": 500.411,
        "job-benchmark_timing_real_12hr_192gpu-0-4": 1497.795,
        "job-benchmark_timing_real_12hr_192gpu-0-5": 356.794,
        "job-benchmark_timing_real_12hr_192gpu-0-6": 1.053,
        "job-benchmark_timing_real_12hr_192gpu-10-1": 741.768,
        "job-benchmark_timing_real_12hr_192gpu-10-2": 204.726,
        "job-benchmark_timing_real_12hr_192gpu-10-3": 1.053,
        "job-benchmark_timing_real_12hr_192gpu-45-1": 5328.781,
        "job-benchmark_timing_real_12hr_192gpu-45-2": 1.077,
        "job-benchmark_timing_real_12hr_192gpu-45-3": 1026.932,
        "job-benchmark_timing_real_12hr_192gpu-90-1": 869.698,
        "job-benchmark_timing_real_12hr_192gpu-90-2": 613.243,
        "job-benchmark_timing_real_12hr_192gpu-150-1": 7511.741,
        "job-benchmark_timing_real_12hr_192gpu-150-2": 794.06,
        "job-benchmark_timing_real_12hr_192gpu-210-1": 4660.637,
        "job-benchmark_timing_real_12hr_192gpu-210-2": 775.948,
        "job-benchmark_timing_real_12hr_192gpu-270-1": 1667.625,
        "job-benchmark_timing_real_12hr_192gpu-270-2": 647.627,
        "job-benchmark_timing_real_12hr_192gpu-360-1": 3218.094,
        "job-benchmark_timing_real_12hr_192gpu-360-2": 840.193,
        "job-benchmark_timing_real_12hr_192gpu-420-1": 181.67,
        "job-benchmark_timing_real_12hr_192gpu-420-2": 772.286,
        "job-benchmark_timing_real_12hr_192gpu-480-1": 2027.827,
        "job-benchmark_timing_real_12hr_192gpu-480-2": 521.621,
        "job-benchmark_timing_real_12hr_192gpu-480-3": 1432.109,
        "job-benchmark_timing_real_12hr_192gpu-570-1": 457.339,
        "job-benchmark_timing_real_12hr_192gpu-570-2": 274.252,
        "job-benchmark_timing_real_12hr_192gpu-630-1": 313.143,
        "job-benchmark_timing_real_12hr_192gpu-630-2": 904.862,
        "job-benchmark_timing_real_12hr_192gpu-690-1": 1078.202,
        "job-benchmark_timing_real_12hr_192gpu-690-2": 251.363,
    },
    "real_life_24h": {
        "job-benchmark_timing_real_24hr_192gpu-0-1": 356.306,
        "job-benchmark_timing_real_24hr_192gpu-0-2": 197.004,
        "job-benchmark_timing_real_24hr_192gpu-0-3": 170.454,
        "job-benchmark_timing_real_24hr_192gpu-0-4": 533.956,
        "job-benchmark_timing_real_24hr_192gpu-0-5": 1628.41,
        "job-benchmark_timing_real_24hr_192gpu-0-6": 389.021,
        "job-benchmark_timing_real_24hr_192gpu-0-7": 1.053,
        "job-benchmark_timing_real_24hr_192gpu-10-1": 741.768,
        "job-benchmark_timing_real_24hr_192gpu-10-2": 204.726,
        "job-benchmark_timing_real_24hr_192gpu-10-3": 1647.942,
        "job-benchmark_timing_real_24hr_192gpu-30-1": 1667.625,
        "job-benchmark_timing_real_24hr_192gpu-30-2": 613.243,
        "job-benchmark_timing_real_24hr_192gpu-60-1": 869.698,
        "job-benchmark_timing_real_24hr_192gpu-60-2": 1.053,
        "job-benchmark_timing_real_24hr_192gpu-120-1": 5640.632,
        "job-benchmark_timing_real_24hr_192gpu-120-2": 1270.897,
        "job-benchmark_timing_real_24hr_192gpu-180-1": 802.803,
        "job-benchmark_timing_real_24hr_192gpu-180-2": 1647.942,
        "job-benchmark_timing_real_24hr_192gpu-300-1": 8759.148,
        "job-benchmark_timing_real_24hr_192gpu-300-2": 857.956,
        "job-benchmark_timing_real_24hr_192gpu-420-1": 5120.598,
        "job-benchmark_timing_real_24hr_192gpu-420-2": 856.514,
        "job-benchmark_timing_real_24hr_192gpu-540-1": 197.966,
        "job-benchmark_timing_real_24hr_192gpu-540-2": 521.431,
        "job-benchmark_timing_real_24hr_192gpu-600-1": 1667.625,
        "job-benchmark_timing_real_24hr_192gpu-690-1": 737.836,
        "job-benchmark_timing_real_24hr_192gpu-720-1": 1270.897,
        "job-benchmark_timing_real_24hr_192gpu-720-2": 274.252,
        "job-benchmark_timing_real_24hr_192gpu-840-1": 3565.995,
        "job-benchmark_timing_real_24hr_192gpu-840-2": 969.099,
        "job-benchmark_timing_real_24hr_192gpu-930-1": 1432.109,
        "job-benchmark_timing_real_24hr_192gpu-930-2": 1338.244,
        "job-benchmark_timing_real_24hr_192gpu-960-1": 772.286,
        "job-benchmark_timing_real_24hr_192gpu-960-2": 851.28,
        "job-benchmark_timing_real_24hr_192gpu-1080-1": 2059.429,
        "job-benchmark_timing_real_24hr_192gpu-1080-2": 600.585,
        "job-benchmark_timing_real_24hr_192gpu-1200-1": 2211.014,
        "job-benchmark_timing_real_24hr_192gpu-1200-2": 7511.741,
        "job-benchmark_timing_real_24hr_192gpu-1320-1": 313.143,
        "job-benchmark_timing_real_24hr_192gpu-1320-2": 904.862,
        "job-benchmark_timing_real_24hr_192gpu-1410-1": 251.363,
        "job-benchmark_timing_real_24hr_192gpu-1410-2": 1667.625,
    },
    "real_life_3h": {
        "job-benchmark_timing_real_3hr_192gpu-0-1": 250.105,
        "job-benchmark_timing_real_3hr_192gpu-0-2": 433.321,
        "job-benchmark_timing_real_3hr_192gpu-0-3": 1236.565,
        "job-benchmark_timing_real_3hr_192gpu-0-4": 292.341,
        "job-benchmark_timing_real_3hr_192gpu-0-5": 1.053,
        "job-benchmark_timing_real_3hr_192gpu-12-1": 4393.226,
        "job-benchmark_timing_real_3hr_192gpu-12-2": 1.077,
        "job-benchmark_timing_real_3hr_192gpu-12-3": 856.034,
        "job-benchmark_timing_real_3hr_192gpu-25-1": 741.768,
        "job-benchmark_timing_real_3hr_192gpu-25-2": 1.053,
        "job-benchmark_timing_real_3hr_192gpu-25-3": 613.243,
        "job-benchmark_timing_real_3hr_192gpu-45-1": 2870.194,
        "job-benchmark_timing_real_3hr_192gpu-45-2": 646.833,
        "job-benchmark_timing_real_3hr_192gpu-70-1": 1996.224,
        "job-benchmark_timing_real_3hr_192gpu-70-2": 442.657,
        "job-benchmark_timing_real_3hr_192gpu-70-3": 203.419,
        "job-benchmark_timing_real_3hr_192gpu-95-1": 334.995,
        "job-benchmark_timing_real_3hr_192gpu-95-2": 313.143,
        "job-benchmark_timing_real_3hr_192gpu-120-1": 772.286,
        "job-benchmark_timing_real_3hr_192gpu-145-1": 1432.109,
    },
    "simulation_12h": {
        "job-benchmark_timing_12hr-0-1": 1418.318,
        "job-benchmark_timing_12hr-0-2": 887.312,
        "job-benchmark_timing_12hr-0-3": 2144.112,
        "job-benchmark_timing_12hr-0-4": 7897.942,
        "job-benchmark_timing_12hr-0-5": 1130.232,
        "job-benchmark_timing_12hr-0-6": 1.053,
        "job-benchmark_timing_12hr-10-1": 741.768,
        "job-benchmark_timing_12hr-10-2": 204.726,
        "job-benchmark_timing_12hr-10-3": 656.349,
        "job-benchmark_timing_12hr-25-1": 24975.426,
        "job-benchmark_timing_12hr-25-2": 1.077,
        "job-benchmark_timing_12hr-25-3": 8610.374,
        "job-benchmark_timing_12hr-25-4": 5213.944,
        "job-benchmark_timing_12hr-45-1": 1.053,
        "job-benchmark_timing_12hr-45-2": 1270.897,
        "job-benchmark_timing_12hr-90-1": 869.698,
        "job-benchmark_timing_12hr-90-2": 613.243,
        "job-benchmark_timing_12hr-135-1": 43686.516,
        "job-benchmark_timing_12hr-135-2": 2838.737,
        "job-benchmark_timing_12hr-180-1": 36857.902,
        "job-benchmark_timing_12hr-180-2": 3354.073,
        "job-benchmark_timing_12hr-240-1": 362.783,
        "job-benchmark_timing_12hr-240-2": 1518.135,
        "job-benchmark_timing_12hr-300-1": 1667.625,
        "job-benchmark_timing_12hr-330-1": 3308.785,
        "job-benchmark_timing_12hr-375-1": 2206.612,
        "job-benchmark_timing_12hr-420-1": 18177.811,
        "job-benchmark_timing_12hr-420-2": 3224.958,
        "job-benchmark_timing_12hr-465-1": 335.976,
        "job-benchmark_timing_12hr-480-1": 772.286,
        "job-benchmark_timing_12hr-510-1": 2354.386,
        "job-benchmark_timing_12hr-510-2": 1469.192,
        "job-benchmark_timing_12hr-570-1": 274.252,
        "job-benchmark_timing_12hr-570-2": 457.339,
        "job-benchmark_timing_12hr-630-1": 313.143,
        "job-benchmark_timing_12hr-630-2": 904.862,
        "job-benchmark_timing_12hr-690-1": 4335.038,
        "job-benchmark_timing_12hr-690-2": 1381.009,
    },
    "simulation_24h": {
        "job-benchmark_timing_24hr-0-1": 1506.818,
        "job-benchmark_timing_24hr-0-2": 887.312,
        "job-benchmark_timing_24hr-0-3": 798.811,
        "job-benchmark_timing_24hr-0-4": 2311.836,
        "job-benchmark_timing_24hr-0-5": 7897.942,
        "job-benchmark_timing_24hr-0-6": 1162.458,
        "job-benchmark_timing_24hr-0-7": 1.053,
        "job-benchmark_timing_24hr-0-8": 1.077,
        "job-benchmark_timing_24hr-10-1": 741.768,
        "job-benchmark_timing_24hr-10-2": 204.726,
        "job-benchmark_timing_24hr-10-3": 1647.942,
        "job-benchmark_timing_24hr-10-4": 274.252,
        "job-benchmark_timing_24hr-30-1": 1667.625,
        "job-benchmark_timing_24hr-30-2": 613.243,
        "job-benchmark_timing_24hr-60-1": 869.698,
        "job-benchmark_timing_24hr-60-2": 1.053,
        "job-benchmark_timing_24hr-120-1": 1270.897,
        "job-benchmark_timing_24hr-120-2": 782.303,
        "job-benchmark_timing_24hr-180-1": 802.803,
        "job-benchmark_timing_24hr-180-2": 1647.942,
        "job-benchmark_timing_24hr-240-1": 904.862,
        "job-benchmark_timing_24hr-300-1": 49923.546,
        "job-benchmark_timing_24hr-300-2": 3094.322,
        "job-benchmark_timing_24hr-420-1": 39924.309,
        "job-benchmark_timing_24hr-420-2": 3756.905,
        "job-benchmark_timing_24hr-540-1": 405.044,
        "job-benchmark_timing_24hr-540-2": 16716.606,
        "job-benchmark_timing_24hr-540-3": 1691.475,
        "job-benchmark_timing_24hr-600-1": 1667.625,
        "job-benchmark_timing_24hr-690-1": 3759.829,
        "job-benchmark_timing_24hr-720-1": 1270.897,
        "job-benchmark_timing_24hr-720-2": 274.252,
        "job-benchmark_timing_24hr-840-1": 19569.413,
        "job-benchmark_timing_24hr-840-2": 3869.489,
        "job-benchmark_timing_24hr-930-1": 1432.109,
        "job-benchmark_timing_24hr-930-2": 1464.654,
        "job-benchmark_timing_24hr-960-1": 772.286,
        "job-benchmark_timing_24hr-960-2": 2636.558,
        "job-benchmark_timing_24hr-1020-1": 313.143,
        "job-benchmark_timing_24hr-1020-2": 556.716,
        "job-benchmark_timing_24hr-1080-1": 2480.796,
        "job-benchmark_timing_24hr-1080-2": 1679.763,
        "job-benchmark_timing_24hr-1140-1": 892.167,
        "job-benchmark_timing_24hr-1200-1": 9291.092,
        "job-benchmark_timing_24hr-1200-2": 32459.862,
        "job-benchmark_timing_24hr-1320-1": 313.143,
        "job-benchmark_timing_24hr-1320-2": 904.862,
        "job-benchmark_timing_24hr-1410-1": 251.363,
        "job-benchmark_timing_24hr-1410-2": 1667.625,
    },
    "simulation_3h": {
        "job-benchmark_timing_3hr-0-1": 1064.314,
        "job-benchmark_timing_3hr-0-2": 1640.938,
        "job-benchmark_timing_3hr-0-3": 6591.789,
        "job-benchmark_timing_3hr-0-4": 807.966,
        "job-benchmark_timing_3hr-0-5": 1.053,
        "job-benchmark_timing_3hr-7-1": 16243.584,
        "job-benchmark_timing_3hr-7-2": 1.077,
        "job-benchmark_timing_3hr-7-3": 3932.206,
        "job-benchmark_timing_3hr-15-1": 741.768,
        "job-benchmark_timing_3hr-15-2": 1.053,
        "job-benchmark_timing_3hr-15-3": 613.243,
        "job-benchmark_timing_3hr-32-1": 19992.668,
        "job-benchmark_timing_3hr-32-2": 2071.983,
        "job-benchmark_timing_3hr-49-1": 11219.803,
        "job-benchmark_timing_3hr-49-2": 1935.896,
        "job-benchmark_timing_3hr-67-1": 2206.908,
        "job-benchmark_timing_3hr-67-2": 995.407,
        "job-benchmark_timing_3hr-67-3": 331.271,
        "job-benchmark_timing_3hr-85-1": 570.518,
        "job-benchmark_timing_3hr-100-1": 772.286,
        "job-benchmark_timing_3hr-100-2": 1743.919,
        "job-benchmark_timing_3hr-125-1": 702.026,
        "job-benchmark_timing_3hr-125-2": 313.143,
        "job-benchmark_timing_3hr-145-1": 1171.456,
        "job-benchmark_timing_3hr-145-2": 1432.109,
        "job-benchmark_timing_3hr-160-1": 251.363,
    },
}

_HARDWARE_ALIASES = {
    "h100": "h100-sxm-80gb",
    "h100-sxm": "h100-sxm-80gb",
    "h100-sxm-80gb": "h100-sxm-80gb",
    "a100": "a100-sxm-80gb",
    "a100-sxm": "a100-sxm-80gb",
    "a100-sxm-80gb": "a100-sxm-80gb",
    "a100-pcie": "a100-pcie-40gb",
    "a100-pcie-40gb": "a100-pcie-40gb",
    "a10g": "a10g-24gb",
    "a10g-24gb": "a10g-24gb",
    "l40s": "l40s-48gb",
    "l40s-48gb": "l40s-48gb",
    "mi300x": "mi300x-192gb",
    "mi300x-192gb": "mi300x-192gb",
}

_HARDWARE_MEMORY_GIB = {
    "h100-sxm-80gb": 80,
    "a100-sxm-80gb": 80,
    "a100-pcie-40gb": 40,
    "a10g-24gb": 24,
    "l40s-48gb": 48,
    "mi300x-192gb": 192,
}

_PRECISION_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
}


@dataclass(frozen=True)
class ModelArchitecture:
    """Architecture fields that determine inference-time VRAM."""

    parameter_count: int
    hidden_size: int
    intermediate_size: int
    attention_layers: int = 0
    kv_heads: int = 0
    head_dim: int = 0
    mla_cache_elements: int | None = None
    linear_attention_layers: int = 0
    linear_cache_elements: int = 0
    causal: bool = True


MODEL_ARCHITECTURES: dict[str, ModelArchitecture] = {
    "meta-llama/Llama-3.2-1B-Instruct": ModelArchitecture(
        parameter_count=1_235_814_400,
        hidden_size=2_048,
        intermediate_size=8_192,
        attention_layers=16,
        kv_heads=8,
        head_dim=64,
    ),
    "meta-llama/Llama-3.3-70B-Instruct": ModelArchitecture(
        parameter_count=70_553_706_496,
        hidden_size=8_192,
        intermediate_size=28_672,
        attention_layers=80,
        kv_heads=8,
        head_dim=128,
    ),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelArchitecture(
        parameter_count=46_702_792_704,
        hidden_size=4_096,
        intermediate_size=14_336,
        attention_layers=32,
        kv_heads=8,
        head_dim=128,
    ),
    "microsoft/phi-4": ModelArchitecture(
        parameter_count=14_659_507_200,
        hidden_size=5_120,
        intermediate_size=17_920,
        attention_layers=40,
        kv_heads=10,
        head_dim=128,
    ),
    "Qwen/Qwen2.5-Coder-32B-Instruct": ModelArchitecture(
        parameter_count=32_763_876_352,
        hidden_size=5_120,
        intermediate_size=27_648,
        attention_layers=64,
        kv_heads=8,
        head_dim=128,
    ),
    "Qwen/Qwen3-Next-80B-A3B-Thinking": ModelArchitecture(
        parameter_count=81_324_862_720,
        hidden_size=2_048,
        intermediate_size=5_120,
        attention_layers=12,
        kv_heads=2,
        head_dim=256,
        linear_attention_layers=36,
        linear_cache_elements=557_056,
    ),
    "zai-org/GLM-4.5-Air": ModelArchitecture(
        parameter_count=110_468_824_832,
        hidden_size=4_096,
        intermediate_size=10_944,
        attention_layers=46,
        kv_heads=8,
        head_dim=128,
    ),
    "moonshotai/Kimi-K2-Instruct": ModelArchitecture(
        parameter_count=1_026_470_731_056,
        hidden_size=7_168,
        intermediate_size=18_432,
        attention_layers=61,
        mla_cache_elements=576,
    ),
    "deepseek-ai/DeepSeek-V3": ModelArchitecture(
        parameter_count=684_531_386_000,
        hidden_size=7_168,
        intermediate_size=18_432,
        attention_layers=61,
        mla_cache_elements=576,
    ),
    "intfloat/multilingual-e5-large": ModelArchitecture(
        parameter_count=559_890_946,
        hidden_size=1_024,
        intermediate_size=4_096,
        causal=False,
    ),
    "BAAI/bge-reranker-v2-m3": ModelArchitecture(
        parameter_count=567_755_777,
        hidden_size=1_024,
        intermediate_size=4_096,
        causal=False,
    ),
}


def calculate_max_vram_gib(
    architecture: ModelArchitecture,
    job: Mapping[str, object],
) -> float:
    """Calculate peak inference VRAM for one workload.

    Causal models retain the maximum input-plus-output KV cache for every
    online stream or request in a batch shard. Encoder workloads have no
    persistent KV cache, so their peak adds one maximum-sized layer working
    set to the model weights.
    """
    spec = _mapping(job.get("spec_json"), "job spec_json")
    features = _mapping(spec.get("job_features"), "job features")
    precision = str(spec.get("precision") or "fp16").lower()
    try:
        bytes_per_element = _PRECISION_BYTES[precision]
    except KeyError as error:
        raise ValueError(f"unsupported model precision {precision!r}") from error

    input_tokens = _positive_int(features.get("isl_token_max"), "isl_token_max")
    output_tokens = _positive_int(features.get("osl_token_max"), "osl_token_max")
    sequence_tokens = input_tokens + output_tokens
    total_bytes = architecture.parameter_count * bytes_per_element

    if architecture.causal:
        active_sequences = _active_sequences(job, features)
        cache_elements = _cache_elements_per_token(architecture)
        total_bytes += active_sequences * (
            sequence_tokens * cache_elements * bytes_per_element
            + architecture.linear_attention_layers
            * architecture.linear_cache_elements
            * bytes_per_element
        )
    else:
        total_bytes += (
            sequence_tokens
            * (architecture.hidden_size + architecture.intermediate_size)
            * bytes_per_element
        )
    return round(total_bytes / 2**30, 3)


def generate_vram_requirements(
    benchmarks_root: Path,
) -> dict[str, dict[str, float]]:
    """Calculate the checked-in VRAM map from all benchmark definitions."""
    requirements: dict[str, dict[str, float]] = {}
    for path in sorted(benchmarks_root.glob("*/benchmark.md")):
        jobs = _benchmark_jobs(path)
        requirements[path.parent.name] = {}
        for job in jobs:
            spec = _mapping(job.get("spec_json"), "job spec_json")
            model_id = _text(spec.get("model_id"), "model_id")
            try:
                architecture = MODEL_ARCHITECTURES[model_id]
            except KeyError as error:
                raise ValueError(f"missing architecture for {model_id}") from error
            job_id = _text(job.get("job_id"), "job_id")
            requirements[path.parent.name][job_id] = calculate_max_vram_gib(
                architecture,
                job,
            )
    return requirements


def _active_sequences(
    job: Mapping[str, object],
    features: Mapping[str, object],
) -> int:
    """Return the maximum resident sequence count for one workload."""
    kind = _text(job.get("kind"), "job kind")
    if kind == "online":
        return _positive_int(
            features.get("max_concurrent_streaming"),
            "max_concurrent_streaming",
        )
    if kind == "batch":
        input_source = _mapping(job.get("input_source"), "batch input_source")
        return _positive_int(
            input_source.get("requests_per_shard"),
            "requests_per_shard",
        )
    raise ValueError(f"unsupported job kind {kind!r}")


def _cache_elements_per_token(architecture: ModelArchitecture) -> int:
    """Return cached elements per token across all attention layers."""
    if architecture.mla_cache_elements is not None:
        elements_per_layer = architecture.mla_cache_elements
    else:
        elements_per_layer = 2 * architecture.kv_heads * architecture.head_dim
    return architecture.attention_layers * elements_per_layer


def _benchmark_jobs(path: Path) -> tuple[dict[str, Any], ...]:
    """Read canonical jobs from the JSON blocks in one benchmark document."""
    jobs = []
    content = path.read_text(encoding="utf-8")
    for document in re.findall(r"```json\s*(.*?)```", content, re.DOTALL):
        payload = json.loads(document)
        for value in payload.get("jobs", []):
            jobs.append(_mapping(value, f"job in {path}"))
    return tuple(jobs)


@dataclass(frozen=True)
class StoreSnapshot:
    """One frozen Store view used throughout a VRAM control tick."""

    resources: dict[str, dict[str, Any]]
    running_jobs: tuple[dict[str, Any], ...]
    waiting_jobs: tuple[dict[str, Any], ...]
    paused_jobs: tuple[dict[str, Any], ...]
    finished_jobs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ResourceTarget:
    """One concrete Store accelerator target."""

    environment_key: str
    environment: dict[str, Any]
    pool: dict[str, Any] | None
    gpu_type: str
    instance_type: str | None


class JobStoreClient(Protocol):
    """Read canonical lifecycle groups from Tandemn Store."""

    def running_jobs(self, user_id: str) -> Sequence[object]: ...

    def waiting_jobs(self, user_id: str) -> Sequence[object]: ...

    def paused_jobs(self, user_id: str) -> Sequence[object]: ...


class ResourceMap(Protocol):
    """Expose the Store resource-map fields used by this adapter."""

    market: object

    def scheduling_summary(self) -> Mapping[str, Mapping[str, Any]]: ...


class ResourceMapStoreClient(Protocol):
    """Read one current Store resource map."""

    def get(self) -> ResourceMap: ...


class PlanStoreClient(Protocol):
    """Persist one canonical Store plan."""

    def create(self, plan: object) -> object: ...


class VramPlannerDriver:
    """Create Store plans from benchmark VRAM requirements and live capacity."""

    method_name = "vram"

    def __init__(
        self,
        user_id: str,
        scenario: Scenario,
        job_store: JobStoreClient,
        resource_map_store: ResourceMapStoreClient,
        plan_store: PlanStoreClient,
        backend: str,
        config_policy: ConfigPolicy,
    ) -> None:
        """Keep Store clients and benchmark-to-Store job identity."""
        self.user_id = user_id
        self.benchmark_name = scenario.benchmark_name
        jobs = (*scenario.online_jobs, *scenario.batch_jobs)
        self.benchmark_job_ids = {job.store_job_id: job.name for job in jobs}
        self.job_store = job_store
        self.resource_map_store = resource_map_store
        self.plan_store = plan_store
        self.backend = backend
        self.config_policy = config_policy

    @classmethod
    def build(
        cls,
        user_id: str,
        scenario: Scenario,
        settings: Mapping[str, object],
        postgres_client: object | None = None,
    ) -> VramPlannerDriver:
        """Build VRAM placement with Tandemn Store public clients."""
        from tandemn_system_data.clients import (
            JobStore,
            PlanStore,
            PostgresClient,
            ResourceMapStore,
        )

        postgres = postgres_client or PostgresClient()
        return cls(
            user_id=user_id,
            scenario=scenario,
            job_store=JobStore(postgres),
            resource_map_store=ResourceMapStore(postgres, user_id=user_id),
            plan_store=PlanStore(postgres),
            backend=_text(settings.get("backend"), "backend"),
            config_policy=load_config_policy(),
        )

    def create_plans(
        self,
        tick_number: int,
        logical_time: datetime,
    ) -> PlannerResult:
        """Read one Store snapshot, persist one plan, and return its identity."""
        del logical_time
        try:
            snapshot = self._read_store_snapshot()
            actions = self._build_actions(snapshot)
            if not actions:
                return PlannerResult(tick_number=tick_number, plans=())

            from tandemn_system_data.models import Plan

            plan = Plan(
                user_id=self.user_id,
                koi_version=self.method_name,
                tick_rationale=_tick_rationale(tick_number, actions),
                actions=actions,
            )
            self.plan_store.create(plan)
            return PlannerResult(
                tick_number=tick_number,
                plans=(CreatedPlan(plan_id=plan.plan_id),),
            )
        except Exception as error:
            return PlannerResult(
                tick_number=tick_number,
                plans=(),
                degraded=True,
                error=repr(error),
            )

    def _read_store_snapshot(self) -> StoreSnapshot:
        """Freeze resources and canonical lifecycle groups for one tick."""
        resource_map = self.resource_map_store.get()
        running = tuple(self.job_store.running_jobs(self.user_id))
        waiting = tuple(self.job_store.waiting_jobs(self.user_id))
        paused = tuple(self.job_store.paused_jobs(self.user_id))
        finished = tuple(self._finished_jobs())
        return StoreSnapshot(
            resources=_resources_with_free_capacity(resource_map, running),
            running_jobs=tuple(_running_job_payload(job) for job in running),
            waiting_jobs=tuple(_model_payload(job) for job in waiting),
            paused_jobs=tuple(_model_payload(job) for job in paused),
            finished_jobs=tuple(_model_payload(job) for job in finished),
        )

    def _finished_jobs(self) -> Sequence[object]:
        """Read terminal jobs through the Store's available lifecycle API."""
        reader = getattr(self.job_store, "finished_jobs", None)
        if callable(reader):
            return cast(Sequence[object], reader(self.user_id))
        status_reader = getattr(self.job_store, "_jobs_with_status", None)
        if not callable(status_reader):
            raise RuntimeError("JobStore does not expose finished jobs")

        from tandemn_system_data.models import JobStatus

        return cast(
            Sequence[object],
            status_reader(self.user_id, JobStatus.FINISHED),
        )

    def _build_actions(self, snapshot: StoreSnapshot) -> list[dict[str, object]]:
        """Place waiting jobs greedily and include every lifecycle action."""
        benchmark = VRAM_REQUIREMENTS_GIB.get(self.benchmark_name, {})
        resources = deepcopy(snapshot.resources)
        actions: list[dict[str, object]] = []
        for running in snapshot.running_jobs:
            actions.append({"job_id": running["job_id"], "type": "keep"})
        for waiting in snapshot.waiting_jobs:
            job_id = _text(waiting.get("job_id"), "waiting Store job_id")
            known = _known_job_fields(waiting)
            model_id = _text(known.get("model_id"), "model_id")
            benchmark_job_id = self.benchmark_job_ids.get(job_id, job_id)
            required_vram_gib = benchmark.get(benchmark_job_id)
            placement = (
                None
                if required_vram_gib is None
                else select_vram_placement(
                    required_vram_gib,
                    resources,
                    model_id=model_id,
                    config_policy=self.config_policy,
                )
            )
            if placement is None:
                actions.append({"job_id": job_id, "type": "defer"})
                continue
            target, tp, pp, dp = placement
            reserve_vram_placement(target, tp * pp * dp)
            actions.append(
                _place_action(
                    job_id,
                    waiting,
                    target,
                    tp,
                    pp,
                    dp,
                    self.backend,
                    self.config_policy,
                )
            )
        for paused in snapshot.paused_jobs:
            actions.append({"job_id": paused["job_id"], "type": "defer"})
        for finished in snapshot.finished_jobs:
            actions.append({"job_id": finished["job_id"], "type": "preempt"})
        _validate_complete_job_actions(snapshot, actions)
        return actions


def select_vram_placement(
    required_vram_gib: float,
    resources: dict[str, dict[str, Any]],
    *,
    model_id: str | None = None,
    config_policy: ConfigPolicy | None = None,
) -> tuple[ResourceTarget, int, int, int] | None:
    """Choose the feasible hardware with the lowest provisioned VRAM."""
    if (model_id is None) != (config_policy is None):
        raise ValueError("model_id and config_policy must be supplied together")

    feasible = []
    for target in _resource_targets(resources):
        hardware = _canonical_hardware(target.gpu_type)
        memory_gib = _HARDWARE_MEMORY_GIB.get(hardware)
        if memory_gib is None:
            continue
        minimum_gpus = math.ceil(required_vram_gib / memory_gib)
        desired_tp = min(minimum_gpus, 8)
        if config_policy is None or model_id is None:
            tp = desired_tp
        else:
            rule = config_policy.rule_for(target.gpu_type, model_id)
            if rule is None:
                continue
            tp = _policy_tp(desired_tp, rule)
        pp = math.ceil(minimum_gpus / tp)
        dp = 1
        gpu_count = tp * pp * dp
        footprint = _target_footprint(target, gpu_count)
        if footprint is None or footprint > _target_available(target):
            continue
        feasible.append(
            (
                gpu_count * memory_gib,
                gpu_count,
                _target_available(target) - footprint,
                target.environment_key,
                target.instance_type or "",
                target,
                tp,
                pp,
                dp,
            )
        )
    if not feasible:
        return None
    selected = min(feasible, key=lambda value: value[:5])
    return selected[5], selected[6], selected[7], selected[8]


def _resource_targets(
    resources: dict[str, dict[str, Any]],
) -> tuple[ResourceTarget, ...]:
    """Return deterministic concrete targets from current Store resources."""
    targets = []
    for environment_key, environment in sorted(resources.items()):
        gpu_type = str(environment.get("gpu_type") or environment_key.split("|")[-1])
        pools = environment.get("pools", [])
        if not isinstance(pools, list):
            raise ValueError("resource pools must be a list")
        for pool in pools or [None]:
            if pool is not None and not isinstance(pool, dict):
                raise ValueError("resource pool must be an object")
            targets.append(
                ResourceTarget(
                    environment_key=environment_key,
                    environment=environment,
                    pool=pool,
                    gpu_type=gpu_type,
                    instance_type=(
                        None
                        if pool is None
                        else _optional_text(pool.get("instance_type"))
                    ),
                )
            )
    return tuple(targets)


def _place_action(
    job_id: str,
    store_job: Mapping[str, object],
    target: ResourceTarget,
    tp: int,
    pp: int,
    dp: int,
    backend: str,
    config_policy: ConfigPolicy | None = None,
) -> dict[str, object]:
    """Combine deterministic Store fields with one mapped placement."""
    known = _known_job_fields(store_job)
    model_id = _text(known.get("model_id"), "model_id")
    precision = None
    if config_policy is not None:
        rule = config_policy.rule_for(target.gpu_type, model_id)
        if rule is None or tp not in rule.allowed_tp:
            raise RuntimeError("selected VRAM placement violates config policy")
        precision = rule.precision
    gpu_count = tp * pp * dp
    return {
        "job_id": job_id,
        "type": "place",
        "tenant_id": known.get("tenant_id") or store_job.get("user_id"),
        "target_tps": _target_tps(known),
        "target_p99_ttft_ms": _number(known.get("target_p99_ttft_ms")),
        "target_p99_tpot_ms": _number(known.get("target_p99_tpot_ms")),
        "mechanism_id": known.get("mechanism_id"),
        "budget_ref": known.get("budget_ref"),
        "rationale": known.get("rationale"),
        "ladder": [
            {
                "role": known.get("role") or "aggregate",
                "rank_id": known.get("rank_id") or "rank_0",
                "env": target.environment_key.split("|"),
                "config": {
                    "model_id": model_id,
                    "precision": precision,
                    "engine_name": known.get("engine_name") or backend,
                    "instance_type": target.instance_type,
                    "gpu_type": target.gpu_type,
                    "gpu_count": gpu_count,
                    "count": known.get("count"),
                    "tp": tp,
                    "pp": pp,
                    "dp": dp,
                    "ep": known.get("ep"),
                    "max_num_seq": known.get("max_num_seq"),
                    "max_num_batched_tokens": known.get("max_num_batched_tokens"),
                    "gpu_mem_util": known.get("gpu_mem_util"),
                    "prefix_cache_enabled": known.get("prefix_cache_enabled"),
                    "rank_traffic_share": 1.0,
                },
                "n_replicas": 1,
                "mechanism_id": known.get("mechanism_id"),
            }
        ],
    }


def _target_available(target: ResourceTarget) -> int:
    """Return current capacity for one concrete target."""
    available = int(target.environment.get("free", 0))
    if target.pool is not None:
        available = min(available, int(target.pool.get("free", 0)))
    return max(0, available)


def _policy_tp(desired_tp: int, rule: ConfigPolicyRule) -> int:
    """Choose the closest allowed TP while preserving the VRAM preference."""
    larger = [tp for tp in rule.allowed_tp if tp >= desired_tp]
    if larger:
        return min(larger)
    return max(rule.allowed_tp)


def _target_footprint(target: ResourceTarget, gpu_count: int) -> int | None:
    """Translate mapped GPUs into the target's allocation unit."""
    if target.pool is None:
        return gpu_count
    allocation_kind = str(
        target.pool.get("allocation_kind")
        or target.pool.get("allocation_unit")
        or "instance"
    ).lower()
    if allocation_kind == "gpu":
        return gpu_count
    gpus_per_instance = _positive_int(
        target.pool.get("gpus_per_instance"),
        "gpus_per_instance",
    )
    instances = math.ceil(gpu_count / gpus_per_instance)
    return instances * gpus_per_instance


def reserve_vram_placement(target: ResourceTarget, gpu_count: int) -> None:
    """Reserve a selected VRAM placement for the current control tick."""
    footprint = _target_footprint(target, gpu_count)
    if footprint is None or footprint > _target_available(target):
        raise RuntimeError("selected VRAM placement no longer fits")
    target.environment["free"] = int(target.environment["free"]) - footprint
    if target.pool is not None:
        target.pool["free"] = int(target.pool["free"]) - footprint


def _resources_with_free_capacity(
    resource_map: ResourceMap,
    running_jobs: Sequence[object],
) -> dict[str, dict[str, Any]]:
    """Normalize Store totals and subtract active canonical rank allocations."""
    raw = resource_map.scheduling_summary()
    default_market = _default_market(resource_map)
    resources = {
        _normalize_env_key(str(key), default_market): dict(value)
        for key, value in raw.items()
    }
    used_by_env: dict[str, int] = {}
    used_by_pool: dict[tuple[str, str], int] = {}
    for running in running_jobs:
        payload = _model_payload(running)
        for rank_value in payload.get("ranks", []):
            rank = _mapping(rank_value, "active rank")
            shape = _mapping(rank.get("shape_json", {}), "rank shape")
            raw_env = shape.get("env") or shape.get("pool_id")
            if raw_env is None:
                continue
            env_key = _normalize_env_key(_env_key(raw_env), default_market)
            count = _positive_int(shape.get("count"), "active rank count")
            replicas = _positive_int(rank.get("n_replicas"), "active rank replicas")
            environment = resources.get(env_key)
            if environment is None:
                continue
            pool = _select_pool(
                environment,
                _optional_text(shape.get("instance_type")),
            )
            footprint = count * replicas
            if pool is not None:
                allocation_kind = str(
                    pool.get("allocation_kind")
                    or pool.get("allocation_unit")
                    or "instance"
                ).lower()
                if allocation_kind != "gpu":
                    footprint = replicas * _positive_int(
                        pool.get("gpus_per_instance"),
                        "gpus_per_instance",
                    )
                pool_key = (env_key, str(pool.get("instance_type")))
                used_by_pool[pool_key] = used_by_pool.get(pool_key, 0) + footprint
            used_by_env[env_key] = used_by_env.get(env_key, 0) + footprint

    for env_key, environment in resources.items():
        environment["free"] = max(
            0,
            int(environment.get("total", 0)) - used_by_env.get(env_key, 0),
        )
        pools = [dict(pool) for pool in environment.get("pools", [])]
        for pool in pools:
            pool_key = (env_key, str(pool.get("instance_type")))
            pool["free"] = max(
                0,
                int(pool.get("total", 0)) - used_by_pool.get(pool_key, 0),
            )
        environment["pools"] = pools
    return resources


def _running_job_payload(running_job: object) -> dict[str, Any]:
    """Flatten a Store running-job aggregate for the frozen snapshot."""
    payload = _model_payload(running_job)
    job = dict(_mapping(payload.get("job"), "running job"))
    job["active_ranks"] = payload.get("ranks", [])
    return job


def _model_payload(value: object) -> dict[str, Any]:
    """Serialize one Store model or aggregate."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="json"))
    if isinstance(value, Mapping):
        return dict(value)
    job = getattr(value, "job", None)
    ranks = getattr(value, "ranks", None)
    if job is not None and ranks is not None:
        return {
            "job": _model_payload(job),
            "ranks": [_model_payload(rank) for rank in ranks],
        }
    raise TypeError(f"Store value {type(value).__name__} is not serializable")


def _known_job_fields(store_job: Mapping[str, object]) -> dict[str, object]:
    """Flatten scheduler-visible fields from one Store job."""
    spec_value = store_job.get("spec_json")
    spec = spec_value if isinstance(spec_value, Mapping) else {}
    features_value = spec.get("job_features")
    features = features_value if isinstance(features_value, Mapping) else {}
    known: dict[str, object] = {}
    for source in (store_job, spec, features):
        for key, value in source.items():
            if value is not None:
                known.setdefault(str(key), value)
    return known


def _target_tps(known: Mapping[str, object]) -> float | None:
    """Derive output-token throughput from canonical Store job fields."""
    output_tokens = _number(known.get("osl_token_avg"))
    if output_tokens is None:
        return None
    kind = getattr(known.get("kind"), "value", known.get("kind"))
    try:
        if kind == "batch":
            token_budget = _positive_number(
                known.get("total_token_budget"),
                "total_token_budget",
            )
            deadline_hours = _positive_number(
                known.get("deadline_hrs"),
                "deadline_hrs",
            )
            input_tokens = _positive_number(
                known.get("isl_token_avg"),
                "isl_token_avg",
            )
            request_rate = token_budget / (
                (input_tokens + output_tokens) * deadline_hours * 3600
            )
        else:
            request_rate = _positive_number(
                known.get("request_arrival_rate"),
                "request_arrival_rate",
            )
    except ValueError:
        return None
    return round(request_rate * output_tokens, 3)


def _validate_complete_job_actions(
    snapshot: StoreSnapshot,
    actions: Sequence[Mapping[str, object]],
) -> None:
    """Require exactly one canonical action for every Store job."""
    store_ids: list[str] = []
    for jobs in (
        snapshot.running_jobs,
        snapshot.waiting_jobs,
        snapshot.paused_jobs,
        snapshot.finished_jobs,
    ):
        store_ids.extend(_text(job.get("job_id"), "Store job_id") for job in jobs)
    action_ids = [_text(action.get("job_id"), "plan job_id") for action in actions]
    if len(store_ids) != len(set(store_ids)):
        raise ValueError("Store snapshot contains duplicate job_id values")
    if len(action_ids) != len(set(action_ids)) or set(action_ids) != set(store_ids):
        raise ValueError("plan must contain exactly one action for every Store job")


def _tick_rationale(
    tick_number: int,
    actions: Sequence[Mapping[str, object]],
) -> str:
    """Summarize the deterministic action set persisted for one tick."""
    counts = {"place": 0, "defer": 0, "keep": 0, "preempt": 0}
    for action in actions:
        action_type = action.get("type")
        if action_type in counts:
            counts[action_type] += 1
    return (
        f"VRAM cluster simulator tick {tick_number}: "
        f"place={counts['place']}, defer={counts['defer']}, "
        f"keep={counts['keep']}, preempt={counts['preempt']}."
    )


def _select_pool(
    environment: Mapping[str, Any],
    instance_type: str | None,
) -> dict[str, Any] | None:
    """Resolve one environment pool from an active chain shape."""
    pools = environment.get("pools", [])
    if not isinstance(pools, list):
        raise ValueError("resource pools must be a list")
    typed_pools = []
    for value in pools:
        if not isinstance(value, dict):
            raise ValueError("resource pool must be an object")
        typed_pools.append(value)
    if not typed_pools:
        return None
    if instance_type is not None:
        return next(
            (
                pool
                for pool in typed_pools
                if pool.get("instance_type") == instance_type
            ),
            None,
        )
    return typed_pools[0] if len(typed_pools) == 1 else None


def _default_market(resource_map: ResourceMap) -> str:
    """Read the resource map's default market."""
    markets = getattr(resource_map, "market", None)
    if isinstance(markets, (list, tuple)) and markets:
        return str(markets[0])
    return str(markets or "reserved")


def _normalize_env_key(env_key: str, default_market: str) -> str:
    """Normalize legacy four-part environment keys."""
    parts = env_key.split("|")
    return "|".join((default_market, *parts)) if len(parts) == 4 else env_key


def _env_key(value: object) -> str:
    """Return one canonical pipe-delimited environment key."""
    if isinstance(value, (list, tuple)):
        return "|".join(str(part) for part in value)
    return str(value)


def _canonical_hardware(value: str) -> str:
    """Normalize benchmark and Store accelerator labels."""
    key = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return _HARDWARE_ALIASES.get(key, key)


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Require one mapping value."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


def _text(value: object, name: str) -> str:
    """Require one nonempty text value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: object) -> str | None:
    """Return optional nonempty text."""
    return None if value is None else _text(value, "optional text")


def _positive_int(value: object, name: str) -> int:
    """Require one positive integer value."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: object, name: str) -> float:
    """Require one positive numeric value."""
    number = _number(value)
    if number is None or number <= 0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _number(value: object) -> float | None:
    """Return one optional numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
