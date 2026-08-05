"""Generate OpenAI-compatible batch JSONL from one batch job specification."""

import json
import logging
import math
import random
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TextIO, cast

import yaml
from aiperf.common import random_generator as aiperf_random
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.prompt import PromptGenerator as AIPerfPromptGenerator

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenDistribution:
    """Bounded token-length distribution."""

    kind: str
    minimum: int
    average: int
    maximum: int
    sigma: float = 0.8

    def sample(self, random_source: random.Random) -> int:
        """Sample one token length within the configured bounds."""
        if self.kind == "constant":
            return self.average

        location = _bounded_lognormal_location(
            self.minimum,
            self.average,
            self.maximum,
            self.sigma,
        )
        sampled = round(random_source.lognormvariate(location, self.sigma))
        return max(self.minimum, min(self.maximum, sampled))


@dataclass(frozen=True)
class BatchDatasetSpec:
    """Complete specification for one finite batch dataset."""

    dataset_id: str
    model: str
    endpoint: str
    record_count: int
    shard_count: int
    requests_per_shard: int
    seed: int
    request_format: str
    input_tokens: TokenDistribution
    output_tokens: TokenDistribution


@dataclass(frozen=True)
class DatasetSummary:
    """Result of generating one dataset."""

    dataset_id: str
    output_path: Path | None
    record_count: int
    average_input_tokens: float
    average_output_tokens: float


class SyntheticPromptGenerator:
    """Generate unique mock text at an exact token length using AIPerf."""

    def __init__(self, seed: int) -> None:
        aiperf_random.reset()
        aiperf_random.init(seed)
        tokenizer = Tokenizer.from_pretrained("builtin")
        self.generator = AIPerfPromptGenerator(
            prompts=None,
            prefix_prompts=None,
            tokenizer=tokenizer,
        )

    def generate(self, target_tokens: int) -> str:
        """Generate one synthetic prompt with exactly the requested length."""
        return cast(str, self.generator.generate_prompt(target_tokens))


def generate_batch_dataset(
    config_path: Path,
    output_path: Path | None = None,
    *,
    record_limit: int | None = None,
) -> DatasetSummary:
    """Stream one configured batch job to an API-ready JSONL file."""
    spec = load_batch_dataset_spec(config_path)
    record_count = spec.record_count if record_limit is None else record_limit
    if record_count <= 0 or record_count > spec.record_count:
        raise ValueError("record_limit must be between 1 and the configured count")

    random_source = random.Random(spec.seed)
    prompt_generator = SyntheticPromptGenerator(spec.seed)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    total_input_tokens = 0
    total_output_tokens = 0

    output_context = (
        output_path.open("w", encoding="utf-8")
        if output_path is not None
        else nullcontext(sys.stdout)
    )
    with output_context as output_file:
        output_file = cast(TextIO, output_file)
        for index in range(record_count):
            input_tokens = spec.input_tokens.sample(random_source)
            output_tokens = spec.output_tokens.sample(random_source)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            prompt = prompt_generator.generate(input_tokens)
            record = build_request_record(
                spec,
                index,
                prompt,
                output_tokens,
            )
            output_file.write(json.dumps(record, separators=(",", ":")))
            output_file.write("\n")

    average_input_tokens = total_input_tokens / record_count
    average_output_tokens = total_output_tokens / record_count
    LOGGER.info(
        "[batch-dataset] id=%s records=%d avg_isl=%.1f avg_osl=%.1f output=%s",
        spec.dataset_id,
        record_count,
        average_input_tokens,
        average_output_tokens,
        output_path or "stdout",
    )
    return DatasetSummary(
        dataset_id=spec.dataset_id,
        output_path=output_path,
        record_count=record_count,
        average_input_tokens=average_input_tokens,
        average_output_tokens=average_output_tokens,
    )


def build_request_record(
    spec: BatchDatasetSpec,
    index: int,
    prompt: str,
    output_tokens: int,
) -> dict[str, object]:
    """Build one OpenAI-compatible batch request."""
    shard_index = index // spec.requests_per_shard
    custom_id = f"{spec.dataset_id}/shard-{shard_index:04d}/request-{index:08d}"

    if spec.request_format == "embeddings":
        body: dict[str, object] = {
            "model": spec.model,
            "input": prompt,
        }
    else:
        body = {
            "model": spec.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_tokens,
        }

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": spec.endpoint,
        "body": body,
    }


def load_batch_dataset_spec(path: Path) -> BatchDatasetSpec:
    """Load batch generation settings from one canonical job YAML."""
    job_document = _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        "batch job",
    )
    if job_document.get("schema_version") != "benchmark-job/v1":
        raise ValueError(f"Unsupported benchmark job schema: {path}")
    job = _mapping(job_document.get("spec"), "spec")
    document = _mapping(job_document.get("workload"), "workload")
    if document.get("schema_version") != "batch-dataset/v1":
        raise ValueError(f"Unsupported batch dataset schema: {path}")

    dataset = _mapping(document.get("dataset"), "dataset")
    partitioning = _mapping(dataset.get("partitioning"), "dataset.partitioning")
    features = _mapping(job.get("features"), "spec.features")
    input_tokens = _load_job_distribution(features, "isl")
    output_tokens = _load_job_distribution(features, "osl")

    shard_count = _positive_integer(
        partitioning.get("shard_count"),
        "dataset.partitioning.shard_count",
    )
    requests_per_shard = _positive_integer(
        partitioning.get("requests_per_shard"),
        "dataset.partitioning.requests_per_shard",
    )
    record_count = _positive_integer(
        dataset.get("record_count"),
        "dataset.record_count",
    )
    if shard_count * requests_per_shard != record_count:
        raise ValueError(
            "dataset.record_count must equal shard_count * requests_per_shard"
        )

    request_format = _text(
        dataset.get("request_format"),
        "dataset.request_format",
    )
    if request_format not in {"chat", "embeddings"}:
        raise ValueError("dataset.request_format must be chat or embeddings")

    endpoint = _text(dataset.get("endpoint"), "dataset.endpoint")
    expected_endpoint = (
        "/v1/embeddings" if request_format == "embeddings" else "/v1/chat/completions"
    )
    if endpoint != expected_endpoint:
        raise ValueError(
            f"dataset.endpoint must be {expected_endpoint} for {request_format}"
        )

    return BatchDatasetSpec(
        dataset_id=_text(dataset.get("id"), "dataset.id"),
        model=_text(dataset.get("model"), "dataset.model"),
        endpoint=endpoint,
        record_count=record_count,
        shard_count=shard_count,
        requests_per_shard=requests_per_shard,
        seed=_nonnegative_integer(dataset.get("seed"), "dataset.seed"),
        request_format=request_format,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _load_job_distribution(
    features: Mapping[str, object],
    prefix: str,
) -> TokenDistribution:
    """Load one batch token distribution from canonical job features."""
    return _load_distribution(
        {
            "distribution": features.get(f"{prefix}_distribution_type"),
            "minimum": features.get(f"{prefix}_token_min"),
            "average": features.get(f"{prefix}_token_avg"),
            "maximum": features.get(f"{prefix}_token_max"),
        },
        f"job.features.{prefix}",
    )


def _load_distribution(value: object, name: str) -> TokenDistribution:
    definition = _mapping(value, name)
    kind = _text(definition.get("distribution"), f"{name}.distribution").lower()
    if kind not in {"constant", "lognormal"}:
        raise ValueError(f"{name}.distribution must be constant or lognormal")

    minimum = _positive_integer(definition.get("minimum"), f"{name}.minimum")
    average = _positive_integer(definition.get("average"), f"{name}.average")
    maximum = _positive_integer(definition.get("maximum"), f"{name}.maximum")
    if not minimum <= average <= maximum:
        raise ValueError(f"{name} must satisfy minimum <= average <= maximum")
    if kind == "constant" and not minimum == average == maximum:
        raise ValueError(f"{name} constant values must be equal")

    sigma_value = definition.get("sigma", 0.8)
    if isinstance(sigma_value, bool) or not isinstance(sigma_value, int | float):
        raise ValueError(f"{name}.sigma must be a number")
    sigma = float(sigma_value)
    if sigma <= 0:
        raise ValueError(f"{name}.sigma must be greater than zero")

    return TokenDistribution(
        kind=kind,
        minimum=minimum,
        average=average,
        maximum=maximum,
        sigma=sigma,
    )


@cache
def _bounded_lognormal_location(
    minimum: int,
    average: int,
    maximum: int,
    sigma: float,
) -> float:
    """Find the lognormal location whose clipped mean matches the target."""
    lower = math.log(minimum) - (4 * sigma)
    upper = math.log(maximum) + (4 * sigma)

    for _ in range(60):
        midpoint = (lower + upper) / 2
        if _bounded_lognormal_mean(midpoint, sigma, minimum, maximum) < average:
            lower = midpoint
        else:
            upper = midpoint

    return (lower + upper) / 2


def _bounded_lognormal_mean(
    location: float,
    sigma: float,
    minimum: int,
    maximum: int,
) -> float:
    """Return the analytic mean of a clipped lognormal distribution."""
    minimum_score = (math.log(minimum) - location) / sigma
    maximum_score = (math.log(maximum) - location) / sigma
    shifted_minimum_score = minimum_score - sigma
    shifted_maximum_score = maximum_score - sigma
    unbounded_mean = math.exp(location + (sigma**2 / 2))

    below_minimum = minimum * _normal_cdf(minimum_score)
    within_bounds = unbounded_mean * (
        _normal_cdf(shifted_maximum_score) - _normal_cdf(shifted_minimum_score)
    )
    above_maximum = maximum * (1 - _normal_cdf(maximum_score))
    return below_minimum + within_bounds + above_maximum


def _normal_cdf(value: float) -> float:
    """Return the standard normal cumulative distribution."""
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, object], value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value
