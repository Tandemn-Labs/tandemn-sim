"""Load paper metric profiles from their authoritative catalog."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
METRIC_ROOT = ROOT / "configs" / "metrics"
CATALOG_PATH = METRIC_ROOT / "catalog.yaml"


def load_metric_profile(profile_id: str) -> tuple[str, ...]:
    """Return the ordered metrics declared for one experiment profile."""
    catalog = _mapping(
        yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")),
        "metric catalog",
    )
    if catalog.get("schema_version") != "metric-catalog/v1":
        raise ValueError("Unsupported metric catalog schema_version")
    profiles = _mapping(catalog.get("profiles"), "profiles")
    try:
        path_value = profiles[profile_id]
    except KeyError as error:
        raise ValueError(f"Unknown metric profile: {profile_id}") from error
    path = METRIC_ROOT / _text(path_value, f"profiles.{profile_id}")
    document = _mapping(
        yaml.safe_load(path.read_text(encoding="utf-8")),
        "metric profile",
    )
    if document.get("schema_version") != "metric-profile/v1":
        raise ValueError("Unsupported metric profile schema_version")
    metrics = document.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"Metric profile {profile_id} must not be empty")
    return tuple(_text(metric, "metric") for metric in metrics)


def _mapping(value: object, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value
