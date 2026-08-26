"""Start one AIPerf process for a benchmark job."""

from __future__ import annotations

import argparse
import copy
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_ROOT = ROOT / "benchmarks"


def find_job(job_id: str) -> Path:
    """Return the unique benchmark job file with the requested ID."""
    matches = []
    for path in sorted(BENCHMARKS_ROOT.glob("*/jobs/*.yaml")):
        document = _load_yaml(path)
        if document.get("job_id") == job_id:
            matches.append(path)

    if not matches:
        raise ValueError(f"Unknown benchmark job: {job_id}")
    if len(matches) > 1:
        paths = ", ".join(str(path) for path in matches)
        raise ValueError(f"Benchmark job ID is not unique: {job_id} ({paths})")
    return matches[0]


def build_aiperf_config(
    job_path: Path,
    *,
    endpoint_url: str | None = None,
    session_header: str | None = None,
) -> dict[str, Any]:
    """Build the AIPerf config embedded in one benchmark job."""
    document = _load_yaml(job_path)
    job_id = _text(document.get("job_id"), f"job_id in {job_path}")
    schedule = _mapping(document.get("schedule"), f"schedule for {job_id}")
    if schedule.get("enabled") is not True:
        raise ValueError(f"Benchmark job is disabled: {job_id}")
    if schedule.get("driver") != "aiperf":
        raise ValueError(f"Benchmark job does not use AIPerf: {job_id}")

    workload = copy.deepcopy(
        _mapping(document.get("workload"), f"workload for {job_id}")
    )
    spec = _mapping(document.get("spec"), f"spec for {job_id}")
    features = _mapping(spec.get("features"), f"features for {job_id}")
    benchmark = _mapping(
        workload.get("benchmark"),
        f"AIPerf benchmark for {job_id}",
    )
    endpoint = _mapping(
        benchmark.get("endpoint"),
        f"AIPerf endpoint for {job_id}",
    )
    if endpoint_url is not None:
        endpoint["url"] = endpoint_url
        endpoint.pop("urls", None)
    elif "url" not in endpoint and "urls" not in endpoint:
        endpoint["url"] = "http://localhost:8000"
    if session_header is not None:
        endpoint["sessionHeader"] = session_header
    dataset = _mapping(
        benchmark.get("dataset"),
        f"AIPerf dataset for {job_id}",
    )

    multi_turn_ratio = _nonnegative_number(
        features.get("multi_turn_ratio"),
        f"multi-turn ratio for {job_id}",
    )
    if multi_turn_ratio > 1:
        raise ValueError(f"multi-turn ratio for {job_id} must be at most 1")
    if multi_turn_ratio == 0:
        dataset["turns"] = 1
    elif multi_turn_ratio == 1:
        dataset["turns"] = 4
    else:
        dataset["turns"] = {
            "points": [
                {"value": 1, "weight": round(100 * (1 - multi_turn_ratio), 10)},
                {"value": 4, "weight": round(100 * multi_turn_ratio, 10)},
            ]
        }

    rankings = dataset.get("rankings")
    if isinstance(rankings, dict):
        _apply_ranking_tokens(rankings, features, job_id)
    else:
        prompts_value = dataset.get("prompts")
        prompts = (
            {}
            if prompts_value is None
            else _mapping(prompts_value, f"AIPerf prompts for {job_id}")
        )
        prompts["isl"] = _token_distribution(features, "isl", job_id)
        prompts["osl"] = _token_distribution(features, "osl", job_id)
        dataset["prompts"] = prompts

    return workload


def aiperf_command(
    config_path: Path,
    *,
    headers: Mapping[str, str] | None = None,
    artifact_dir: Path | None = None,
) -> tuple[str, ...]:
    """Build the AIPerf profile command for a rendered config."""
    command = ["aiperf", "profile", "--config", str(config_path)]
    if artifact_dir is not None:
        command.extend(("--output-artifact-dir", str(artifact_dir)))
    for name, value in (headers or {}).items():
        command.extend(("-H", f"{name}:{value}"))
    return tuple(command)


def run_job(job_id: str) -> int:
    """Start AIPerf for one benchmark job and return its exit code."""
    config = build_aiperf_config(find_job(job_id))
    with tempfile.TemporaryDirectory(prefix="aiperf-job-") as directory:
        config_path = Path(directory) / "aiperf.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        completed = subprocess.run(
            aiperf_command(config_path),
            check=False,
        )
    return completed.returncode


def main() -> None:
    """Parse a benchmark job ID and run its AIPerf workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="job_id from a benchmark job YAML file")
    args = parser.parse_args()
    try:
        exit_code = run_job(args.job_id)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    raise SystemExit(exit_code)


def _apply_ranking_tokens(
    rankings: dict[str, Any],
    features: dict[str, Any],
    job_id: str,
) -> None:
    passages = _mapping(rankings.get("passages"), f"passages for {job_id}")
    queries = _mapping(rankings.get("queryTokens"), f"query tokens for {job_id}")
    rankings["passageTokens"] = {
        statistic: _ranking_passage_tokens(
            features,
            passages,
            queries,
            feature_suffix,
            statistic,
            job_id,
        )
        for statistic, feature_suffix in (
            ("mean", "avg"),
            ("min", "min"),
            ("max", "max"),
        )
    }


def _ranking_passage_tokens(
    features: Mapping[str, Any],
    passages: Mapping[str, Any],
    queries: Mapping[str, Any],
    feature_suffix: str,
    statistic: str,
    job_id: str,
) -> float:
    total = _positive_number(
        features.get(f"isl_token_{feature_suffix}"),
        f"ranking input {feature_suffix} for {job_id}",
    )
    query = _nonnegative_number(
        queries.get(statistic),
        f"ranking query {statistic} for {job_id}",
    )
    passage_count = _positive_number(
        passages.get(statistic),
        f"ranking passages {statistic} for {job_id}",
    )
    passage_tokens = (total - query) / passage_count
    if passage_tokens <= 0:
        raise ValueError(f"Invalid ranking token shape for {job_id}")
    return passage_tokens


def _token_distribution(
    features: Mapping[str, Any],
    prefix: str,
    job_id: str,
) -> dict[str, float]:
    return {
        statistic: _nonnegative_number(
            features.get(f"{prefix}_token_{feature_suffix}"),
            f"{prefix} {statistic} for {job_id}",
        )
        for statistic, feature_suffix in (
            ("mean", "avg"),
            ("min", "min"),
            ("max", "max"),
        )
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_number(value: object, name: str) -> float:
    number = _nonnegative_number(value, name)
    if number == 0:
        raise ValueError(f"{name} must be greater than zero")
    return number


def _nonnegative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"{name} must be a nonnegative number")
    return float(value)


if __name__ == "__main__":
    main()
