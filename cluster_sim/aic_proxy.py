"""Select AIC model proxies independently using Koi's joint-knn-v1 approach.

Reference: koi bca19fede5a393826ef7372e58014747407ab963, profile_search.py
and SurrogatePrediction._rank_aic_profiles. Matching is within one AIC system;
model size precedes structural distance. These signatures estimate performance,
not runtime memory fit.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

LOGGER = logging.getLogger(__name__)
PROXY_ALGORITHM_VERSION = "koi-joint-knn-v1-local-1"
MAX_PROXY_SPEED_RATIO = 20.0
_PROFILE_BACKENDS = frozenset({("vllm", "0.22.0"), ("sglang", "0.5.14")})
_DTYPE_BYTES = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "fp4": 0.5,
    "nf4": 0.5,
    "nvfp4": 9 / 16,
}


@dataclass(frozen=True)
class ModelProfile:
    """Structural target or proxy metadata, independent of AIC model support."""

    model_id: str
    architecture: str
    layers: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    kv_heads: int
    head_dim: int
    vocab_size: int
    num_experts: int
    active_experts: int
    expert_intermediate_size: int
    weight_dtype: str
    activation_dtype: str
    kv_dtype: str
    max_context: int
    multimodal: bool
    generative: bool
    mla_cache_elements: int | None
    parameter_count: float = 0.0


@dataclass(frozen=True)
class SupportedProfile:
    """One measured aggregate model/backend/version on a concrete AIC system."""

    model: ModelProfile
    system: str
    backend: str
    backend_version: str

    @property
    def profile_id(self) -> str:
        """Return Koi's deterministic support-point identity."""
        return (
            f"{self.backend}:{self.backend_version}:{self.system}:{self.model.model_id}"
        )


@dataclass(frozen=True)
class ProxyMatch:
    """One ranked prediction reference, retaining the original target identity."""

    requested_model_id: str
    profile: SupportedProfile
    distance: float
    reasons: tuple[str, ...]


def model_profile_from_config(
    model_id: str, config: Mapping[str, object]
) -> ModelProfile:
    """Read raw HF or parsed AIC metadata without building the model in AIC."""
    if not model_id or not isinstance(config, Mapping):
        raise ValueError("model identity and configuration are required")
    raw = config.get("raw_config", {})
    if not isinstance(raw, Mapping):
        raise ValueError("raw_config must be an object")
    values = {**raw, **config}
    text_config = values.get("text_config", {})
    if isinstance(text_config, Mapping):
        values = {**text_config, **values}
    architectures = values.get("architectures")
    architecture = values.get("architecture")
    if architecture is None and isinstance(architectures, list) and architectures:
        architecture = architectures[0]
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("model architecture is required")
    layers = _integer(values.get("layers", values.get("num_hidden_layers")), "layers")
    hidden = _integer(values.get("hidden_size"), "hidden_size")
    heads = _integer(values.get("n", values.get("num_attention_heads")), "heads")
    kv_heads = _integer(
        values.get("n_kv") or values.get("num_key_value_heads") or heads, "kv_heads"
    )
    intermediate = _integer(
        values.get("inter_size") or values.get("intermediate_size") or hidden * 4,
        "intermediate",
    )
    experts = _integer(
        values.get("num_experts")
        or values.get("num_local_experts")
        or values.get("n_routed_experts")
        or 0,
        "num_experts",
        minimum=0,
    )
    active = _integer(
        values.get("topk", values.get("num_experts_per_tok", 0)),
        "active_experts",
        minimum=0,
    )
    if active > experts or (experts > 0 and active == 0):
        raise ValueError("MoE metadata requires 1 <= active_experts <= num_experts")
    weight_dtype = values.get("weight_dtype")
    if "fp8" in str(values.get("weight_quantization_method", "")).lower():
        weight_dtype = "fp8"
    if weight_dtype is None:
        lowered = model_id.lower()
        if "fp8" in lowered:
            weight_dtype = "fp8"
        elif "int8" in lowered:
            weight_dtype = "int8"
        elif any(name in lowered for name in ("int4", "awq", "gptq")):
            weight_dtype = "int4"
        else:
            weight_dtype = values.get("torch_dtype", "bf16")
    mla = values.get("mla_cache_elements")
    if mla is None and values.get("kv_lora_rank") is not None:
        mla = _integer(values["kv_lora_rank"], "kv_lora_rank") + _integer(
            values.get("qk_rope_head_dim"), "qk_rope_head_dim"
        )
    profile = ModelProfile(
        model_id=model_id,
        architecture=architecture,
        layers=layers,
        hidden_size=hidden,
        intermediate_size=intermediate,
        attention_heads=heads,
        kv_heads=kv_heads,
        head_dim=_integer(
            values.get("d", values.get("head_dim", hidden // heads)), "head_dim"
        ),
        vocab_size=_integer(values.get("vocab", values.get("vocab_size")), "vocab"),
        num_experts=experts,
        active_experts=active,
        expert_intermediate_size=_integer(
            values.get("moe_inter_size")
            or values.get("moe_intermediate_size")
            or intermediate,
            "expert_intermediate_size",
        ),
        weight_dtype=_dtype(weight_dtype),
        activation_dtype=_dtype(
            values.get("fmha_quant_mode") or values.get("activation_dtype") or "bf16"
        ),
        kv_dtype=_dtype(
            values.get("kvcache_quant_mode") or values.get("kv_cache_dtype") or "bf16"
        ),
        max_context=_integer(
            values.get("context", values.get("max_position_embeddings", 8192)),
            "max_context",
        ),
        multimodal=bool(values.get("vision_config"))
        or "vl" in model_id.lower()
        or "vision" in model_id.lower(),
        generative=architecture.endswith(
            ("ForCausalLM", "ForConditionalGeneration", "LMHeadModel")
        ),
        mla_cache_elements=None if mla is None else _integer(mla, "mla_cache_elements"),
    )
    parameters = values.get("parameter_count")
    if parameters is None:
        billions = values.get("model_params_b", values.get("params_billion"))
        parameters = (
            _positive_number(billions, "model_params_b") * 1e9
            if billions is not None
            else sum(_parameter_components(profile))
        )
    return replace(profile, parameter_count=_positive_number(parameters, "parameters"))


@lru_cache(maxsize=128)
def load_model_profile(
    model_id: str, model_configs_root: Path | None = None
) -> ModelProfile:
    """Prefer local target JSON, including architectures AIC cannot parse."""
    model_path = Path(model_id)
    if not model_id or model_path.is_absolute() or ".." in model_path.parts:
        raise ValueError("model_id must be a relative model identifier")
    if model_configs_root is not None:
        path = model_configs_root / f"{model_id}.json"
        if path.is_file():
            return model_profile_from_config(model_id, json.loads(path.read_text()))
    from aiconfigurator.sdk.utils import get_model_config_from_model_path

    return model_profile_from_config(
        model_id, get_model_config_from_model_path(model_id)
    )


@lru_cache(maxsize=16)
def load_supported_profiles(system: str) -> tuple[SupportedProfile, ...]:
    """Load the same measured backend/version inventory used by reference Koi."""
    from aiconfigurator.sdk.common import get_support_matrix

    profiles = []
    seen = set()
    for row in get_support_matrix():
        if (
            row.get("System") != system
            or row.get("Mode") != "agg"
            or row.get("Status") != "PASS"
            or row.get("Source", "silicon") != "silicon"
            or (row.get("Backend"), row.get("Version")) not in _PROFILE_BACKENDS
        ):
            continue
        key = (row["HuggingFaceID"], row["Backend"], row["Version"])
        if key in seen:
            continue
        seen.add(key)
        try:
            model = load_model_profile(key[0])
        except Exception as error:
            LOGGER.warning("Skipping AIC proxy metadata for %s: %s", key[0], error)
            continue
        profiles.append(SupportedProfile(model, system, key[1], key[2]))
    return tuple(sorted(profiles, key=lambda profile: profile.profile_id))


def select_proxy_profiles(
    target: ModelProfile,
    available: Sequence[SupportedProfile],
    *,
    system: str,
    backend: str,
    tp: int,
    pp: int,
    input_tokens: int,
    output_tokens: int,
    batch_size: int,
) -> tuple[ProxyMatch, ...]:
    """Rank up to five compatible same-system proxies using Koi's ordering."""
    for name, value in (
        ("tp", tp),
        ("pp", pp),
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("batch_size", batch_size),
    ):
        _integer(value, name)
    if not _supports_topology(target, tp, pp):
        return ()
    target_signature = _operation_signature(
        target, tp, pp, input_tokens, output_tokens, batch_size
    )
    matches = []
    for profile in available:
        model = profile.model
        if (
            profile.system != system
            or bool(model.num_experts) != bool(target.num_experts)
            or model.multimodal != target.multimodal
            or model.generative != target.generative
            or not _supports_topology(model, tp, pp)
        ):
            continue
        signature = _operation_signature(
            model, tp, pp, input_tokens, output_tokens, batch_size
        )
        distance_squared = sum(
            weight * math.log(max(left, 1e-12) / max(right, 1e-12)) ** 2
            for left, right, weight in zip(
                target_signature, signature, (1, 1.5, 2, 2), strict=True
            )
        )
        reasons = ["model_operation_signature"]
        if _model_family(target) != _model_family(model):
            distance_squared += 1
            reasons.append("model_family_penalty")
        if target.architecture != model.architecture:
            distance_squared += 0.25
            reasons.append("architecture_penalty")
        for target_dtype, proxy_dtype in (
            (target.weight_dtype, model.weight_dtype),
            (target.activation_dtype, model.activation_dtype),
            (target.kv_dtype, model.kv_dtype),
        ):
            penalty, reason = _dtype_distance(target_dtype, proxy_dtype)
            distance_squared += penalty**2
            if reason is not None:
                reasons.append(reason)
        for ratio, reason in (
            (
                (input_tokens + output_tokens) / model.max_context,
                "context_extrapolation",
            ),
            (batch_size / 256, "batch_extrapolation"),
        ):
            if ratio > 1:
                distance_squared += math.log(ratio) ** 2
                reasons.append(reason)
        matches.append(
            ProxyMatch(
                target.model_id, profile, math.sqrt(distance_squared), tuple(reasons)
            )
        )
    matches.sort(
        key=lambda match: (
            match.profile.model.model_id != target.model_id,
            abs(math.log(match.profile.model.parameter_count / target.parameter_count)),
            match.profile.model.weight_dtype != target.weight_dtype,
            match.profile.backend != backend,
            match.distance,
            match.profile.profile_id,
        )
    )
    return tuple(matches[:5])


def _supports_topology(model: ModelProfile, tp: int, pp: int) -> bool:
    return (
        model.attention_heads % tp == 0
        and model.layers >= pp
        and (not model.num_experts or model.num_experts % tp == 0)
    )


def phase_correction_factors(
    target: ModelProfile,
    proxy: ModelProfile,
    *,
    system: str,
    tp: int,
    pp: int,
    input_tokens: int,
    output_tokens: int,
    batch_size: int,
) -> tuple[float, float]:
    """Estimate same-system prefill/decode speed ratios at one candidate point."""
    for name, value in (
        ("tp", tp),
        ("pp", pp),
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("batch_size", batch_size),
    ):
        _integer(value, name)
    if not _supports_topology(target, tp, pp) or not _supports_topology(proxy, tp, pp):
        raise ValueError("target and proxy must support the candidate topology")
    compute_rate, memory_rate, link_rate = _system_performance(system)
    times = []
    for model in (target, proxy):
        weights, kv_bytes, prefill, decode = _operation_signature(
            model, tp, pp, input_tokens, output_tokens, batch_size
        )
        width = model.hidden_size * _DTYPE_BYTES[model.weight_dtype]
        communication = 4.0 * model.layers * width * (tp - 1) / tp
        communication += (pp - 1) * width
        if model.num_experts:
            communication += 2.0 * model.layers * width * (tp - 1) / tp
        prefill_memory = weights + kv_bytes * input_tokens * batch_size
        context = input_tokens + max(1, output_tokens // 2)
        decode_memory = weights + kv_bytes * context * batch_size
        times.append(
            (
                max(prefill / compute_rate, prefill_memory / memory_rate)
                + communication * input_tokens * batch_size / link_rate,
                max(decode / compute_rate, decode_memory / memory_rate)
                + communication * batch_size / link_rate,
            )
        )
    prefill_ratio, decode_ratio = (
        max(0.05, min(MAX_PROXY_SPEED_RATIO, proxy_time / target_time))
        for target_time, proxy_time in zip(times[0], times[1], strict=True)
    )
    return prefill_ratio, decode_ratio


@lru_cache(maxsize=16)
def _system_performance(system: str) -> tuple[float, float, float]:
    from aiconfigurator.sdk.perf_database import load_system_spec

    spec = load_system_spec(system)
    gpu, node = spec["gpu"], spec["node"]
    return (
        _positive_number(
            gpu.get("bfloat16_tc_flops") or gpu.get("float16_tc_flops"), "GPU FLOPs/s"
        ),
        _positive_number(gpu.get("mem_bw"), "GPU memory bytes/s"),
        _positive_number(
            node.get("intra_node_bw") or node.get("pcie_bw"), "link bytes/s"
        ),
    )


def profile_for_aic_row(model: ModelProfile, row: Mapping[str, object]) -> ModelProfile:
    """Use the estimated row's effective quantization for performance transfer."""
    return replace(
        model,
        weight_dtype=_dtype(
            row.get("moe" if model.num_experts else "gemm", model.weight_dtype)
        ),
        activation_dtype=_dtype(row.get("fmha", model.activation_dtype)),
        kv_dtype=_dtype(row.get("kvcache", model.kv_dtype)),
    )


def correct_proxy_metrics(
    row: Mapping[str, object],
    prefill_ratio: float,
    decode_ratio: float,
    output_tokens: int,
) -> dict[str, object]:
    """Correct per-replica metrics before SLO filtering and replica sizing."""
    prefill_ratio = _positive_number(prefill_ratio, "prefill_ratio")
    decode_ratio = _positive_number(decode_ratio, "decode_ratio")
    output_tokens = _integer(output_tokens, "output_tokens")
    ttft = _positive_number(row.get("ttft"), "ttft")
    tpot = _positive_number(row.get("tpot"), "tpot")
    corrected_ttft, corrected_tpot = ttft / prefill_ratio, tpot / decode_ratio
    # Koi's throughput-transfer convention uses OSL; AIC request_latency uses OSL-1.
    throughput_ratio = (ttft + tpot * output_tokens) / (
        corrected_ttft + corrected_tpot * output_tokens
    )
    tokens = _positive_number(row.get("tokens/s"), "tokens/s") * throughput_ratio
    requests = _positive_number(row.get("seq/s"), "seq/s") * throughput_ratio
    devices = _integer(row.get("num_total_gpus"), "num_total_gpus")
    corrected = dict(row)
    for field in (
        "replicas_needed",
        "total_gpus_needed",
        "tokens/s/gpu_cluster",
        "load_served_pct",
        "_sla_exceeded",
    ):
        corrected.pop(field, None)
    corrected.update(
        {
            "ttft": corrected_ttft,
            "tpot": corrected_tpot,
            "request_latency": corrected_ttft
            + corrected_tpot * max(output_tokens - 1, 0),
            "tokens/s": tokens,
            "seq/s": requests,
            "request_rate": requests,
            "tokens/s/gpu": tokens / devices,
            "seq/s/gpu": requests / devices,
            "tokens/s/user": 1000 / corrected_tpot if output_tokens > 1 else 0.0,
        }
    )
    return corrected


def _parameter_components(model: ModelProfile) -> tuple[float, float]:
    hidden = model.hidden_size
    attention = 2.0 * hidden**2 + 2.0 * hidden * model.kv_heads * model.head_dim
    nonexpert = model.layers * (attention + 3.0 * hidden * model.intermediate_size)
    nonexpert += model.vocab_size * hidden
    expert = (
        model.layers * 3.0 * hidden * model.expert_intermediate_size * model.num_experts
    )
    return nonexpert, expert


def _operation_signature(
    model: ModelProfile, tp: int, pp: int, isl: int, osl: int, batch: int
) -> tuple[float, float, float, float]:
    """Return Koi's weight/KV traffic and prefill/decode work for local EP=TP."""
    nonexpert, expert = _parameter_components(model)
    scale = model.parameter_count / (nonexpert + expert)
    shards = tp * pp
    weights = model.parameter_count * _DTYPE_BYTES[model.weight_dtype] / shards
    active = (
        expert * model.active_experts / model.num_experts if model.num_experts else 0
    )
    active_parameters = (nonexpert + active) * scale / shards
    kv_bytes = (
        math.ceil(model.layers / pp) * model.mla_cache_elements
        if model.mla_cache_elements is not None
        else 2.0 * model.layers * model.kv_heads * model.head_dim / shards
    ) * _DTYPE_BYTES[model.kv_dtype]
    # shortcut: Koi's signature lacks linear-attention state; upgrade with calibration.
    prefill = 2.0 * active_parameters * isl * batch
    prefill += 4.0 * model.layers * isl**2 * model.hidden_size * batch / shards
    context = isl + max(1, osl // 2)
    decode = 2.0 * active_parameters * batch
    decode += 4.0 * model.layers * context * model.hidden_size * batch / shards
    return weights, kv_bytes, prefill, decode


def _model_family(model: ModelProfile) -> str:
    text = f"{model.model_id} {model.architecture}".lower()
    for family in ("qwen", "llama", "deepseek", "mistral", "gemma", "minimax"):
        if family in text:
            return family
    return model.architecture.lower()


def _dtype(value: object) -> str:
    text = str(value).lower().removeprefix("torch.")
    dtype = {
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "fp8_block": "fp8",
        "fp8_static": "fp8",
        "fp8_ootb": "fp8",
        "int8_wo": "int8",
        "sq": "int8",
        "int4_wo": "int4",
        "w4afp8": "int4",
        "w4a16_mxfp4": "fp4",
        "w4a8_mxfp4_mxfp8": "fp4",
        "w4a8_mxfp4_mxfp8_trtllm": "fp4",
        "w4a16_mxfp4_cutlass": "fp4",
    }.get(text, text)
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unsupported proxy dtype {value!r}")
    return dtype


def _dtype_distance(target: str, proxy: str) -> tuple[float, str | None]:
    if target == proxy:
        return 0.0, None
    if {target, proxy} == {"bf16", "fp16"}:
        return 0.25, "same_width_dtype_proxy"
    if target.startswith("fp8") and proxy == "bf16":
        return 0.75, "higher_precision_dtype_proxy"
    return 1.5, "dtype_family_penalty"


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or int(value) != value
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)
