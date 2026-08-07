"""Load benchmark definitions from the repository catalog."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS_ROOT = ROOT / "benchmarks"
CATALOG_PATH = BENCHMARKS_ROOT / "catalog.yaml"


@dataclass(frozen=True)
class BenchmarkDefinition:
    """One benchmark entry and its executable-manifest status."""

    benchmark_id: str
    manifest: Path
    environment: str
    duration_hours: int
    status: str


def list_benchmarks() -> Mapping[str, BenchmarkDefinition]:
    """Return every benchmark declared in the authoritative catalog."""
    document = _mapping(
        yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")),
        "benchmark catalog",
    )
    if document.get("schema_version") != "benchmark-catalog/v1":
        raise ValueError("Unsupported benchmark catalog schema_version")
    entries = _mapping(document.get("benchmarks"), "benchmarks")
    definitions = (
        _load_definition(benchmark_id, value) for benchmark_id, value in entries.items()
    )
    return {definition.benchmark_id: definition for definition in definitions}


def resolve_benchmark(value: str) -> BenchmarkDefinition:
    """Resolve a catalog ID or explicit benchmark manifest path."""
    supplied_path = Path(value)
    if supplied_path.is_file():
        manifest = supplied_path.resolve()
        document = _mapping(
            yaml.safe_load(manifest.read_text(encoding="utf-8")),
            "benchmark manifest",
        )
        benchmark = _mapping(document.get("benchmark"), "benchmark")
        benchmark_id = _text(benchmark.get("id"), "benchmark.id")
        status_value = benchmark.get("status", "executable")
        return BenchmarkDefinition(
            benchmark_id=benchmark_id,
            manifest=manifest,
            environment=_text(
                benchmark.get("environment", "unspecified"),
                "benchmark.environment",
            ),
            duration_hours=max(
                1,
                round(_number(benchmark.get("duration_minutes", 60)) / 60),
            ),
            status=_text(status_value, "benchmark.status"),
        )
    try:
        return list_benchmarks()[value]
    except KeyError as error:
        raise ValueError(f"Unknown benchmark: {value}") from error


def _load_definition(benchmark_id: object, value: object) -> BenchmarkDefinition:
    identifier = _text(benchmark_id, "benchmark ID")
    definition = _mapping(value, f"benchmarks.{identifier}")
    return BenchmarkDefinition(
        benchmark_id=identifier,
        manifest=(
            BENCHMARKS_ROOT
            / _text(definition.get("manifest"), f"benchmarks.{identifier}.manifest")
        ).resolve(),
        environment=_text(
            definition.get("environment"),
            f"benchmarks.{identifier}.environment",
        ),
        duration_hours=_positive_int(
            definition.get("duration_hours"),
            f"benchmarks.{identifier}.duration_hours",
        ),
        status=_text(definition.get("status"), f"benchmarks.{identifier}.status"),
    )


def _mapping(value: object, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("duration_minutes must be numeric")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
