"""Load benchmark-method-metric matrices from experiment manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from data_generation.benchmarks.registry import list_benchmarks
from data_generation.methods.registry import list_methods
from data_generation.metrics.registry import load_metric_profile

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = ROOT / "experiments"
CATALOG_PATH = EXPERIMENT_ROOT / "catalog.yaml"


@dataclass(frozen=True)
class Experiment:
    """One authoritative benchmark-method-metric comparison matrix."""

    experiment_id: str
    benchmarks: tuple[str, ...]
    methods: tuple[str, ...]
    metric_profile: str
    metrics: tuple[str, ...]


def load_experiment(experiment_id: str) -> Experiment:
    """Load and cross-check one experiment manifest against all catalogs."""
    catalog = _mapping(
        yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")),
        "experiment catalog",
    )
    if catalog.get("schema_version") != "experiment-catalog/v1":
        raise ValueError("Unsupported experiment catalog schema_version")
    entries = _mapping(catalog.get("experiments"), "experiments")
    try:
        path_value = entries[experiment_id]
    except KeyError as error:
        raise ValueError(f"Unknown experiment: {experiment_id}") from error
    path = EXPERIMENT_ROOT / _text(path_value, f"experiments.{experiment_id}")
    document = _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        "experiment",
    )
    if document.get("schema_version") != "experiment/v1":
        raise ValueError("Unsupported experiment schema_version")
    definition = _mapping(document.get("experiment"), "experiment")
    declared_id = _text(definition.get("id"), "experiment.id")
    if declared_id != experiment_id:
        raise ValueError("Experiment ID does not match its catalog key")
    benchmarks = _text_list(definition.get("benchmarks"), "experiment.benchmarks")
    methods = _text_list(definition.get("methods"), "experiment.methods")
    benchmark_catalog = list_benchmarks()
    method_catalog = list_methods()
    unknown_benchmarks = set(benchmarks) - set(benchmark_catalog)
    unknown_methods = set(methods) - set(method_catalog)
    if unknown_benchmarks:
        raise ValueError(f"Unknown experiment benchmarks: {sorted(unknown_benchmarks)}")
    if unknown_methods:
        raise ValueError(f"Unknown experiment methods: {sorted(unknown_methods)}")
    metric_profile = _text(definition.get("metrics"), "experiment.metrics")
    return Experiment(
        experiment_id=experiment_id,
        benchmarks=benchmarks,
        methods=methods,
        metric_profile=metric_profile,
        metrics=load_metric_profile(metric_profile),
    )


def _mapping(value: object, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    return tuple(_text(item, name) for item in value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
