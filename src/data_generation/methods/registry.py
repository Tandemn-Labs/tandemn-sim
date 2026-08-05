"""Authoritative scheduling-method discovery and construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from data_generation.methods.dynamo_aic import load_dynamo_aic_method
from data_generation.methods.gpt_only import load_gpt_only_method
from data_generation.scheduling import SchedulingMethod

ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = ROOT / "configs" / "methods.yaml"

MethodLoader = Callable[[Path, Mapping[str, object]], SchedulingMethod]
METHOD_LOADERS: Mapping[str, MethodLoader] = {
    "dynamo_aic": load_dynamo_aic_method,
    "gpt_only": load_gpt_only_method,
}


@dataclass(frozen=True)
class MethodDefinition:
    """One paper method and its implementation status."""

    method_id: str
    display_name: str
    status: str
    settings: Mapping[str, object]


def list_methods() -> Mapping[str, MethodDefinition]:
    """Return every method declared in the paper-aligned catalog."""
    document = _mapping(
        yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")),
        "method catalog",
    )
    if document.get("schema_version") != "method-catalog/v2":
        raise ValueError("Unsupported method catalog schema_version")
    entries = _mapping(document.get("methods"), "methods")
    definitions = (
        _load_definition(method_id, value) for method_id, value in entries.items()
    )
    return {definition.method_id: definition for definition in definitions}


def load_method(method_id: str, benchmark_directory: Path) -> SchedulingMethod:
    """Build one implemented method without controller-specific branching."""
    methods = list_methods()
    try:
        definition = methods[method_id]
    except KeyError as error:
        raise ValueError(f"Unknown method: {method_id}") from error
    if definition.status != "implemented":
        raise ValueError(f"Method {method_id} is cataloged but not implemented")
    try:
        loader = METHOD_LOADERS[method_id]
    except KeyError as error:
        raise ValueError(f"Method {method_id} has no registered loader") from error
    return loader(benchmark_directory, definition.settings)


def _load_definition(method_id: object, value: object) -> MethodDefinition:
    identifier = _text(method_id, "method ID")
    definition = _mapping(value, f"methods.{identifier}")
    raw_settings = _mapping(
        definition.get("settings", {}),
        f"methods.{identifier}.settings",
    )
    return MethodDefinition(
        method_id=identifier,
        display_name=_text(definition.get("display_name"), "display_name"),
        status=_text(definition.get("status"), "status"),
        settings={str(key): item for key, item in raw_settings.items()},
    )


def _mapping(value: object, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
