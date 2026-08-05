"""Frozen benchmark settings for the Dynamo+AIC method."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from data_generation.scheduling import OptimizationGoal


@dataclass(frozen=True)
class DynamoAicSettings:
    """AIC engine and comparison settings shared by one benchmark run."""

    backend: str
    backend_version: str | None
    backend_versions: Mapping[str, str]
    database_mode: str
    serving_mode: str
    default_optimization_goal: OptimizationGoal


def load_dynamo_aic_settings(path: Path) -> DynamoAicSettings:
    """Load the frozen Dynamo+AIC settings for one benchmark."""
    document = _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        "Dynamo+AIC settings",
    )
    if document.get("schema_version") != "dynamo-aic/v1":
        raise ValueError("Unsupported Dynamo+AIC settings schema_version")
    serving_mode = _text(document.get("serving_mode"), "serving_mode")
    if serving_mode != "agg":
        raise ValueError("simulation_3h requires aggregate serving")
    version_value = document.get("backend_version")
    backend_version = (
        None if version_value is None else _text(version_value, "backend_version")
    )
    versions_value = _mapping(
        document.get("backend_versions", {}),
        "backend_versions",
    )
    backend_versions = {
        _text(pool_id, "backend_versions key"): _text(
            version,
            f"backend_versions.{pool_id}",
        )
        for pool_id, version in versions_value.items()
    }
    return DynamoAicSettings(
        backend=_text(document.get("backend"), "backend"),
        backend_version=backend_version,
        backend_versions=backend_versions,
        database_mode=_text(document.get("database_mode"), "database_mode"),
        serving_mode=serving_mode,
        default_optimization_goal=OptimizationGoal(
            _text(
                document.get("default_optimization_goal"),
                "default_optimization_goal",
            )
        ),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
