"""Load model placement constraints from the cluster configuration policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

_POLICY_TEXT = """\
A100-40GB:
- BAAI/bge-reranker-v2-m3 → fp16 TP: 1
- deepseek-ai/DeepSeek-V3 → bf16-fp8 TP: 1, 8
- intfloat/multilingual-e5-large → fp16 TP: 1
- meta-llama/Llama-3.2-1B-Instruct → fp16 TP: 1, 2, 4, 8
- meta-llama/Llama-3.3-70B-Instruct → bf16 TP: 2, 4, 8
- microsoft/phi-4 → fp16 TP: 1, 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 → fp16 TP: 1, 2, 4, 8
- Qwen/Qwen2.5-Coder-32B-Instruct → fp16 TP: 1, 2, 4, 8
- Qwen/Qwen3-8B → bf16 TP: 1, 2
- Qwen/Qwen3-Next-80B-A3B-Thinking → bf16 TP: 1, 8
- zai-org/GLM-4.5-Air → bf16 TP: 1, 4, 8
A100-80GB:
- BAAI/bge-reranker-v2-m3 → fp16 TP: 1
- intfloat/multilingual-e5-large → fp16 TP: 1
- meta-llama/Llama-3.2-1B-Instruct → fp16 TP: 1, 2, 4, 8
- meta-llama/Llama-3.3-70B-Instruct → bf16 TP: 2, 4, 8
- microsoft/phi-4 → fp16 TP: 1, 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 → fp16 TP: 1, 2, 4, 8
- Qwen/Qwen2.5-Coder-32B-Instruct → fp16 TP: 1, 2, 4, 8
- Qwen/Qwen3-Next-80B-A3B-Thinking → bf16 TP: 4, 8
- zai-org/GLM-4.5-Air → bf16 TP: 4, 8
A10G:
- BAAI/bge-reranker-v2-m3 → fp16 TP: 1
- intfloat/multilingual-e5-large → fp16 TP: 1
- meta-llama/Llama-3.2-1B-Instruct → fp16 TP: 1, 2, 4
- meta-llama/Llama-3.3-70B-Instruct → bf16 TP: 4, 8
- microsoft/phi-4 → fp16 TP: 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 → fp16 TP: 4, 8
- Qwen/Qwen2.5-Coder-32B-Instruct → fp16 TP: 4, 8
- Qwen/Qwen3-Next-80B-A3B-Thinking → bf16 TP: 1, 8
- zai-org/GLM-4.5-Air → bf16 TP: 1, 8
H100:
- BAAI/bge-reranker-v2-m3 → fp16 TP: 1
- deepseek-ai/DeepSeek-V3 → bf16-fp8 TP: 1, 8
- intfloat/multilingual-e5-large → fp16 TP: 1
- meta-llama/Llama-3.1-8B → bf16 TP: 1, 2, 4
- meta-llama/Llama-3.2-1B-Instruct → bf16 TP: 1, 2, 4, 8
- meta-llama/Llama-3.3-70B-Instruct → bf16 TP: 2, 4, 8
- microsoft/phi-4 → bf16 TP: 1, 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 → bf16 TP: 1, 2, 4, 8
- mistralai/Mixtral-8x7B-v0.1 → bf16 TP: 1, 2
- moonshotai/Kimi-K2-Instruct → bf16-fp8 TP: 1, 4, 8
- Qwen/Qwen2.5-Coder-32B-Instruct → bf16 TP: 1, 2, 4, 8
- Qwen/Qwen3-235B-A22B → bf16 TP: 1, 8
- Qwen/Qwen3-30B-A3B-Instruct-2507 → bf16 TP: 1, 2, 4
- Qwen/Qwen3-Next-80B-A3B-Thinking → bf16 TP: 4, 8
- zai-org/GLM-4.5-Air → bf16 TP: 1, 4, 8
H200:
- mistralai/Mixtral-8x22B-v0.1 → bf16 TP: 1
- mistralai/Mixtral-8x7B-v0.1 → bf16 TP: 1
- Qwen/Qwen3-235B-A22B → bf16 TP: 1
- Qwen/Qwen3-30B-A3B-Instruct-2507 → bf16 TP: 1, 2
L4:
- meta-llama/Llama-3.2-1B-Instruct → bf16 TP: 1
L40S:
- BAAI/bge-reranker-v2-m3 → fp16 TP: 1
- deepseek-ai/DeepSeek-V3 → bf16-fp8 TP: 1, 8
- intfloat/multilingual-e5-large → fp16 TP: 1
- meta-llama/Llama-3.2-1B-Instruct → fp16 TP: 1, 2, 4, 8
- meta-llama/Llama-3.3-70B-Instruct → bf16 TP: 2, 4, 8
- microsoft/phi-4 → fp16 TP: 1, 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 → fp16 TP: 1, 2, 4, 8
- mistralai/Mixtral-8x7B-v0.1 → bf16 TP: 1, 2, 4
- moonshotai/Kimi-K2-Instruct → bf16-fp8 TP: 1, 4, 8
- Qwen/Qwen2.5-Coder-32B-Instruct → fp16 TP: 1, 2, 4, 8
- Qwen/Qwen3-Next-80B-A3B-Thinking → bf16 TP: 1, 4, 8
- zai-org/GLM-4.5-Air → bf16 TP: 1, 4, 8
MI300X:
- BAAI/bge-reranker-v2-m3 → fp16 TP: 1
- deepseek-ai/DeepSeek-V3 → bf16-fp8 TP: 1, 2, 4, 8
- intfloat/multilingual-e5-large → fp16 TP: 1
- meta-llama/Llama-3.2-1B-Instruct → fp16 TP: 1, 2, 4, 8
- meta-llama/Llama-3.3-70B-Instruct → bf16 TP: 1, 2, 4, 8
- microsoft/phi-4 → fp16 TP: 1, 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 → fp16 TP: 1, 2, 4, 8
- moonshotai/Kimi-K2-Instruct → bf16-fp8 TP: 1, 4, 8
- Qwen/Qwen2.5-Coder-32B-Instruct → fp16 TP: 1, 2, 4, 8
- Qwen/Qwen3-Next-80B-A3B-Thinking → bf16 TP: 1, 2, 4, 8
- zai-org/GLM-4.5-Air → bf16 TP: 1, 2, 4, 8
RTX3090:
- Qwen/Qwen3-4B → bf16 TP: 1, 2, 4, 8
- Qwen/Qwen3-8B → bf16 TP: 1, 2, 4, 8
RTX4090:
- meta-llama/Llama-3.1-8B → bf16 TP: 1
- Qwen/Qwen3-32B → bf16 TP: 1
RTXPRO6000:
- meta-llama/Llama-3.1-8B → bf16 TP: 1, 2
- meta-llama/Llama-3.2-1B-Instruct → bf16 TP: 1, 2, 4, 8
- meta-llama/Llama-3.3-70B-Instruct → bf16 TP: 2, 4, 8
- microsoft/phi-4 → bf16 TP: 1, 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 → bf16 TP: 1, 2, 4, 8
- Qwen/Qwen2.5-Coder-32B-Instruct → fp16 TP: 1, 2, 4, 8
- Qwen/Qwen3-30B-A3B-Instruct-2507 → bf16 TP: 1, 2
- Qwen/Qwen3-32B → bf16 TP: 1, 2
- Qwen/Qwen3-Next-80B-A3B-Thinking → bf16 TP: 1, 2, 4, 8
- zai-org/GLM-4.5-Air → bf16 TP: 1, 2, 4, 8
RTXPRO6000-RUNPOD:
- BAAI/bge-reranker-v2-m3 → fp16 TP: 1
- intfloat/multilingual-e5-large → fp16 TP: 1
- meta-llama/Llama-3.3-70B-Instruct → bf16 TP: 1, 2, 4, 8
- microsoft/phi-4 → fp16 TP: 1, 2
- mistralai/Mixtral-8x7B-Instruct-v0.1 → fp16 TP: 1, 2, 4, 8
- Qwen/Qwen2.5-Coder-32B-Instruct → fp16 TP: 1, 2, 4, 8
- Qwen/Qwen3-Next-80B-A3B-Thinking → bf16 TP: 1
- zai-org/GLM-4.5-Air → bf16 TP: 1, 2, 4, 8
"""

_REAL_POLICY_TEXT = """\
L4:
- meta-llama/Llama-3.2-1B-Instruct → fp16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- mistralai/Mixtral-8x7B-Instruct-v0.1 → fp16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- microsoft/phi-4 → fp16 TP: 1, 2 PP: 1, 2, 3, 4
- Qwen/Qwen2.5-Coder-32B-Instruct → fp16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- meta-llama/Llama-3.3-70B-Instruct → bf16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- Qwen/Qwen3-Next-80B-A3B-Thinking → bf16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- zai-org/GLM-4.5-Air → bf16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ → bf16-awq TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
A100:
- meta-llama/Llama-3.2-1B-Instruct → fp16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- mistralai/Mixtral-8x7B-Instruct-v0.1 → fp16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- microsoft/phi-4 → fp16 TP: 1, 2 PP: 1, 2, 3, 4
- Qwen/Qwen2.5-Coder-32B-Instruct → fp16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- meta-llama/Llama-3.3-70B-Instruct → bf16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- Qwen/Qwen3-Next-80B-A3B-Thinking → bf16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- zai-org/GLM-4.5-Air → bf16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ → bf16-awq TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
H100:
- meta-llama/Llama-3.2-1B-Instruct → fp16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- mistralai/Mixtral-8x7B-Instruct-v0.1 → fp16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- microsoft/phi-4 → fp16 TP: 1, 2 PP: 1, 2, 3, 4
- Qwen/Qwen2.5-Coder-32B-Instruct → fp16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- meta-llama/Llama-3.3-70B-Instruct → bf16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- Qwen/Qwen3-Next-80B-A3B-Thinking → bf16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- zai-org/GLM-4.5-Air → bf16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ → bf16-awq TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
RTX6000:
- meta-llama/Llama-3.2-1B-Instruct → fp16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- mistralai/Mixtral-8x7B-Instruct-v0.1 → fp16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- microsoft/phi-4 → fp16 TP: 1, 2 PP: 1, 2, 3, 4
- Qwen/Qwen2.5-Coder-32B-Instruct → fp16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- meta-llama/Llama-3.3-70B-Instruct → bf16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- Qwen/Qwen3-Next-80B-A3B-Thinking → bf16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- zai-org/GLM-4.5-Air → bf16 TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
- QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ → bf16-awq TP: 1, 2, 4, 8 PP: 1, 2, 3, 4
"""

_POLICY_ENTRY = re.compile(
    r"^- (?P<model>.+) → (?P<precision>\S+) TP: (?P<tp>\d+(?:, \d+)*)"
    r"(?: PP: (?P<pp>\d+(?:, \d+)*))?$"
)
VALID_TP_DEGREES = frozenset({1, 2, 4, 6, 8})
VALID_PP_DEGREES = frozenset({1, 2, 3, 4})
MODEL_MAX_TP = {"microsoft/phi-4": 2}
_POLICY_ENVIRONMENT = "simulation"
_HARDWARE_ALIASES = {
    "a100": "a100-80gb",
    "a100-pcie": "a100-40gb",
    "a100-pcie-40gb": "a100-40gb",
    "a100-sxm": "a100-80gb",
    "a100-sxm-80gb": "a100-80gb",
    "a10g-24gb": "a10g",
    "h100-sxm": "h100",
    "h100-sxm-80gb": "h100",
    "h200-sxm": "h200",
    "h200-sxm-141gb": "h200",
    "l4-24gb": "l4",
    "l40s-48gb": "l40s",
    "mi300": "mi300x",
    "mi300x-192gb": "mi300x",
}
_REAL_HARDWARE_ALIASES = {
    "a100-40gb": "a100",
    "a100-80gb": "a100",
    "a100-pcie": "a100",
    "a100-pcie-40gb": "a100",
    "a100-sxm": "a100",
    "a100-sxm-80gb": "a100",
    "h100-sxm": "h100",
    "h100-sxm-80gb": "h100",
    "l4-24gb": "l4",
    "rtx-pro-6000": "rtx6000",
    "rtxpro6000": "rtx6000",
}


@dataclass(frozen=True)
class ConfigPolicyRule:
    """Required precision and permitted tensor-parallel sizes."""

    precision: str
    allowed_tp: tuple[int, ...]
    allowed_pp: tuple[int, ...] | None = None

    def allows_parallelism(self, tp: int, pp: int) -> bool:
        """Return whether one TP and PP pair satisfies this rule."""
        return (
            tp in self.allowed_tp
            and pp in VALID_PP_DEGREES
            and (self.allowed_pp is None or pp in self.allowed_pp)
        )


class ConfigPolicy:
    """Resolve model placement constraints for canonical hardware names."""

    def __init__(
        self,
        rules: dict[tuple[str, str], ConfigPolicyRule],
    ) -> None:
        """Keep one immutable copy of the parsed policy rules."""
        self._rules = dict(rules)

    def rule_for(self, hardware: str, model_id: str) -> ConfigPolicyRule | None:
        """Return the policy rule for one hardware and model pair."""
        return self._rules.get((_canonical_hardware(hardware), model_id))


@cache
def load_config_policy() -> ConfigPolicy:
    """Parse and cache the policy for the configured environment."""
    rules: dict[tuple[str, str], ConfigPolicyRule] = {}
    hardware: str | None = None
    policy_text = (
        _REAL_POLICY_TEXT
        if _POLICY_ENVIRONMENT == "real_hardware"
        else _POLICY_TEXT
    )

    for line_number, raw_line in enumerate(policy_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":") and not line.startswith("- "):
            hardware = _canonical_hardware(line[:-1])
            continue
        if hardware is None:
            raise ValueError(f"policy entry before hardware at line {line_number}")
        match = _POLICY_ENTRY.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid config policy at line {line_number}: {line!r}")

        model_id = match.group("model")
        allowed_tp = tuple(int(value) for value in match.group("tp").split(", "))
        raw_pp = match.group("pp")
        allowed_pp = (
            tuple(sorted(VALID_PP_DEGREES))
            if raw_pp is None
            else tuple(int(value) for value in raw_pp.split(", "))
        )
        if any(tp < 1 for tp in allowed_tp):
            raise ValueError(f"TP values must be positive at policy line {line_number}")
        invalid_tp = tuple(tp for tp in allowed_tp if tp not in VALID_TP_DEGREES)
        if invalid_tp:
            raise ValueError(
                f"unsupported TP values {invalid_tp} at policy line {line_number}; "
                f"expected values from {sorted(VALID_TP_DEGREES)}"
            )
        if len(set(allowed_tp)) != len(allowed_tp):
            raise ValueError(f"duplicate TP value at policy line {line_number}")
        invalid_pp = tuple(pp for pp in allowed_pp if pp not in VALID_PP_DEGREES)
        if invalid_pp:
            raise ValueError(
                f"unsupported PP values {invalid_pp} at policy line {line_number}; "
                f"expected values from {sorted(VALID_PP_DEGREES)}"
            )
        if len(set(allowed_pp)) != len(allowed_pp):
            raise ValueError(f"duplicate PP value at policy line {line_number}")
        max_tp = MODEL_MAX_TP.get(model_id)
        if max_tp is not None and any(tp > max_tp for tp in allowed_tp):
            raise ValueError(
                f"{model_id} TP values must not exceed {max_tp} "
                f"at policy line {line_number}"
            )
        key = (hardware, model_id)
        if key in rules:
            raise ValueError(f"duplicate config policy entry at line {line_number}")
        rules[key] = ConfigPolicyRule(
            precision=match.group("precision"),
            allowed_tp=allowed_tp,
            allowed_pp=allowed_pp,
        )

    if not rules:
        raise ValueError("embedded config policy is empty")
    return ConfigPolicy(rules)


def configure_config_policy(environment: str) -> None:
    """Select the simulation or real-hardware policy for this run."""
    if environment not in {"simulation", "real_hardware"}:
        raise ValueError(f"unsupported policy environment {environment!r}")
    global _POLICY_ENVIRONMENT
    _POLICY_ENVIRONMENT = environment
    load_config_policy.cache_clear()


def _canonical_hardware(value: str) -> str:
    """Normalize Store and policy hardware labels to one lookup key."""
    normalized = value.strip().lower().replace("_", "-")
    aliases = (
        _REAL_HARDWARE_ALIASES
        if _POLICY_ENVIRONMENT == "real_hardware"
        else _HARDWARE_ALIASES
    )
    return aliases.get(normalized, normalized)
