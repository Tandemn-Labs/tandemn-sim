"""Materialize one batch job as OpenAI batch JSONL with AIPerf prompts."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import yaml
from aiperf.common import random_generator as aiperf_random
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.prompt import PromptGenerator

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse dataset generation inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path, help="batch job YAML to materialize")
    parser.add_argument(
        "--turn-mode",
        choices=("single", "multi"),
        required=True,
        help="write independent requests or grouped conversations",
    )
    parser.add_argument("--output", type=Path, help="output JSONL path")
    parser.add_argument(
        "--tokenizer",
        help="AIPerf tokenizer name; defaults to the job model",
    )
    parser.add_argument("--turns-min", type=int, default=2)
    parser.add_argument("--turns-mean", type=float, default=4)
    parser.add_argument("--turns-max", type=int, default=8)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace output")
    return parser.parse_args(argv)


def _load_batch_job(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document["schedule"]["kind"] != "batch":
        raise ValueError(f"{path} is not a batch job")
    return document["spec"]["features"], document["workload"]["dataset"]


@cache
def _lognormal_location(
    minimum: int,
    average: int,
    maximum: int,
    sigma: float = 0.8,
) -> float:
    lower = math.log(minimum) - (4 * sigma)
    upper = math.log(maximum) + (4 * sigma)
    for _ in range(60):
        midpoint = (lower + upper) / 2
        minimum_score = (math.log(minimum) - midpoint) / sigma
        maximum_score = (math.log(maximum) - midpoint) / sigma
        unbounded_mean = math.exp(midpoint + (sigma**2 / 2))
        clipped_mean = (
            minimum * _normal_cdf(minimum_score)
            + unbounded_mean
            * (_normal_cdf(maximum_score - sigma) - _normal_cdf(minimum_score - sigma))
            + maximum * (1 - _normal_cdf(maximum_score))
        )
        if clipped_mean < average:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def _normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def _sample_tokens(
    features: dict[str, object], prefix: str, source: random.Random
) -> int:
    minimum = int(features[f"{prefix}_token_min"])
    average = int(features[f"{prefix}_token_avg"])
    maximum = int(features[f"{prefix}_token_max"])
    distribution = str(features[f"{prefix}_distribution_type"]).lower()
    if distribution == "constant":
        return average
    if distribution != "lognormal":
        raise ValueError(f"unsupported {prefix} distribution: {distribution}")
    location = _lognormal_location(minimum, average, maximum)
    return max(minimum, min(maximum, round(source.lognormvariate(location, 0.8))))


def _turn_counts(
    total: int,
    minimum: int,
    mean: float,
    maximum: int,
    source: random.Random,
) -> list[int]:
    if not 2 <= minimum <= mean <= maximum:
        raise ValueError("multi-turn bounds must satisfy 2 <= min <= mean <= max")
    if total < minimum:
        raise ValueError("record count is smaller than the minimum turn count")

    counts = []
    remaining = total
    deviation = max(0.5, (maximum - minimum) / 4)
    while remaining:
        if remaining <= maximum:
            counts.append(remaining)
            break
        count = max(minimum, min(maximum, round(source.gauss(mean, deviation))))
        if 0 < remaining - count < minimum:
            count = remaining - minimum
        counts.append(count)
        remaining -= count
    return counts


def _records(
    features: dict[str, object],
    record_count: int,
    dataset_id: str,
    model: str,
    endpoint: str,
    turn_mode: str,
    turns_min: int,
    turns_mean: float,
    turns_max: int,
    source: random.Random,
    prompts: PromptGenerator,
) -> Iterator[dict[str, object]]:
    def content(prefix: str, tokens: int | None = None) -> str:
        tokens = tokens or _sample_tokens(features, prefix, source)
        maximum_tokens = int(features[f"{prefix}_token_max"])
        text = prompts.generate(mean=tokens, stddev=0)
        encoded = prompts.tokenizer.encode(text)
        while len(encoded) > maximum_tokens:
            text = prompts.tokenizer.decode(encoded[: maximum_tokens - 1])
            encoded = prompts.tokenizer.encode(text)
        return text

    def request(
        custom_id: str,
        messages: list[dict[str, str]],
    ) -> dict[str, object]:
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": endpoint,
            "body": {
                "model": model,
                "messages": messages,
            },
        }

    if turn_mode == "single":
        for index in range(record_count):
            yield request(
                f"{dataset_id}-{index:08d}",
                [{"role": "user", "content": content("isl")}],
            )
        return

    counts = _turn_counts(
        record_count,
        turns_min,
        turns_mean,
        turns_max,
        source,
    )
    for session_index, count in enumerate(counts):
        messages = []
        for turn_index in range(count):
            messages.append({"role": "user", "content": content("isl")})
            output_tokens = _sample_tokens(features, "osl", source)
            yield request(
                f"{dataset_id}-{session_index:08d}-{turn_index:03d}",
                [dict(message) for message in messages],
            )
            messages.append(
                {"role": "assistant", "content": content("osl", output_tokens)}
            )


def generate_dataset(
    job_path: Path,
    output_path: Path,
    *,
    turn_mode: str,
    tokenizer_name: str | None,
    turns_min: int,
    turns_mean: float,
    turns_max: int,
    trust_remote_code: bool,
    force: bool,
) -> int:
    """Generate one JSONL dataset and return its API request count."""
    if output_path.exists() and not force:
        raise FileExistsError(
            f"{output_path} already exists; pass --force to replace it"
        )
    features, dataset = _load_batch_job(job_path)
    record_count = int(dataset["record_count"])
    seed = int(dataset["seed"])
    selected_tokenizer = tokenizer_name or str(dataset["model"])
    aiperf_random.init(seed)
    tokenizer = Tokenizer.from_pretrained(
        selected_tokenizer,
        trust_remote_code=trust_remote_code,
    )
    prompts = PromptGenerator(
        prompts=None,
        prefix_prompts=None,
        tokenizer=tokenizer,
    )
    rows = _records(
        features,
        record_count,
        str(dataset["id"]),
        str(dataset["model"]),
        str(dataset["endpoint"]),
        turn_mode,
        turns_min,
        turns_mean,
        turns_max,
        random.Random(seed),
        prompts,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")))
            output.write("\n")
    return record_count


def main(argv: list[str] | None = None) -> None:
    """Generate the requested batch dataset."""
    args = parse_args(argv)
    output = args.output or args.job.with_suffix(".jsonl")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    count = generate_dataset(
        args.job,
        output,
        turn_mode=args.turn_mode,
        tokenizer_name=args.tokenizer,
        turns_min=args.turns_min,
        turns_mean=args.turns_mean,
        turns_max=args.turns_max,
        trust_remote_code=args.trust_remote_code,
        force=args.force,
    )
    LOGGER.info("wrote %s requests to %s", count, output)


if __name__ == "__main__":
    main()
