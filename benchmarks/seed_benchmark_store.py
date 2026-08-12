"""Reset benchmark telemetry and seed Tandemn Store from benchmark catalog files."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete
from tandemn_system_data.clients import (
    ModelCatalogStore,
    PostgresClient,
    ResourceMapStore,
)
from tandemn_system_data.db.orm import (
    EvidenceRowRow,
    GpuMetricRow,
    ModelCatalogRow,
    ResourceMapRow,
    UserRow,
)
from tandemn_system_data.models.model_catalog import ModelCatalog
from tandemn_system_data.models.resource_map import ResourceMap

BENCHMARKS = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse optional benchmark input paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-catalog",
        type=Path,
        default=BENCHMARKS / "model_catalogs.csv",
    )
    parser.add_argument(
        "--gpu-profiles",
        type=Path,
        default=BENCHMARKS / "model_catalog_gpu_profiles.csv",
    )
    parser.add_argument(
        "--resource-map",
        type=Path,
        default=BENCHMARKS / "resource_map_benchmarking.jsonl",
    )
    return parser.parse_args(argv)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _value(value: str) -> str | None:
    return None if value in {"", "NA"} else value


def load_catalogs(catalog_path: Path, profiles_path: Path) -> list[ModelCatalog]:
    """Load the CSV representation into canonical Store model records."""
    catalog_rows = _rows(catalog_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in catalog_rows:
        grouped[row["model_id"]].append(row)

    selected = {
        model_id: next(
            (row for row in rows if row["source"] == "candidate_jsonl"), rows[0]
        )
        for model_id, rows in grouped.items()
    }
    profiles: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in _rows(profiles_path):
        model_id = row["model_id"]
        if model_id not in selected or row["source"] != selected[model_id]["source"]:
            continue
        value: str | int
        value = (
            row["value"]
            if row["profile_field"] == "kvcache_dtype"
            else int(row["value"])
        )
        profiles[model_id][row["profile_field"]].append(
            {"gpu_type": row["gpu_type"], "value": value}
        )

    catalogs = []
    for model_id, row in selected.items():
        body = {
            key: parsed
            for key, raw in row.items()
            if key not in {"source", "model_id", "model_name", "updated_at"}
            and (parsed := _value(raw)) is not None
        }
        body.update(profiles[model_id])
        catalogs.append(ModelCatalog.model_validate({"model_id": model_id, **body}))
    return catalogs


def load_resource_map(path: Path) -> tuple[str, ResourceMap]:
    """Read the single benchmarking ResourceMap JSONL record."""
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one JSONL record")
    return str(rows[0]["user_id"]), ResourceMap.model_validate(rows[0]["resource_map"])


def seed(args: argparse.Namespace) -> None:
    """Reset telemetry/evidence and make benchmark files the Store source of truth."""
    user_id, resource_map = load_resource_map(args.resource_map)
    catalogs = load_catalogs(args.model_catalog, args.gpu_profiles)
    client = PostgresClient()

    with client.begin() as session:
        session.execute(delete(GpuMetricRow))
        session.execute(delete(EvidenceRowRow))
        session.execute(delete(ResourceMapRow))
        session.execute(
            delete(ModelCatalogRow).where(
                ModelCatalogRow.model_id.not_in(
                    [catalog.model_id for catalog in catalogs]
                )
            )
        )
        if session.get(UserRow, user_id) is None:
            session.add(
                UserRow(
                    user_id=user_id,
                    name="benchmarking",
                    created_at=datetime.now(UTC),
                )
            )

    model_store = ModelCatalogStore(client)
    for catalog in catalogs:
        model_store.replace(catalog)
    ResourceMapStore(client, user_id=user_id).replace(resource_map)
    LOGGER.info(
        "seeded %d models and one 200-GPU reserved resource map for %s",
        len(catalogs),
        user_id,
    )


def main(argv: list[str] | None = None) -> None:
    """Configure logging and seed the local Tandemn Store."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    seed(parse_args(argv))


if __name__ == "__main__":
    main()
