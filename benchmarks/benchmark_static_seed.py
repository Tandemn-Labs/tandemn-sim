"""Destructively seed the local Store with static benchmark inputs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.engine import make_url
from tandemn_system_data.clients import (
    HardwareCatalogStore,
    ModelCatalogStore,
    PostgresClient,
    ResourceMapStore,
)
from tandemn_system_data.clients.postgres import DEFAULT_URL
from tandemn_system_data.db.orm import (
    CredentialsRow,
    EventConsumerOffsetRow,
    EventRow,
    EvidenceRowRow,
    GpuMetricRow,
    HardwareCatalogRow,
    JobRow,
    KoiCausalEdgeRow,
    KoiCausalMechanismRow,
    KoiCausalNodeRow,
    ModelCatalogRow,
    PlanRow,
    RankRow,
    ResourceMapRow,
    UserRow,
)
from tandemn_system_data.models.model_catalog import ModelCatalog
from tandemn_system_data.models.resource_map import ResourceMap

if __package__:
    from benchmarks.seed_benchmark_store import load_catalogs, load_resource_map
else:
    from seed_benchmark_store import load_catalogs, load_resource_map

BENCHMARKS = Path(__file__).resolve().parent
WORKSPACE_ROOT = BENCHMARKS.parent.parent
STORE_ROOT = WORKSPACE_ROOT / "tandemn-store"
CANONICAL_USER_ID = "usr_01KZSQ1M34PKS5XC9ZDR6DDSJ5"
AWS_CATALOG_KEY = "aws-accelerated-hardware-v2"
AZURE_CATALOG_KEY = "azure-accelerated-hardware-v1"
EXPECTED_POOL_COUNT = 8
EXPECTED_GPU_COUNT = 176
LOGGER = logging.getLogger(__name__)

EXPECTED_COUNTS: tuple[tuple[str, type[Any], int], ...] = (
    ("users", UserRow, 1),
    ("model_catalogs", ModelCatalogRow, 16),
    ("resource_maps", ResourceMapRow, 1),
    ("hardware_catalogs", HardwareCatalogRow, 2),
    ("jobs", JobRow, 0),
    ("plans", PlanRow, 0),
    ("ranks", RankRow, 0),
    ("gpu_metrics", GpuMetricRow, 0),
    ("evidence_rows", EvidenceRowRow, 0),
    ("events", EventRow, 0),
    ("event_consumer_offsets", EventConsumerOffsetRow, 0),
    ("credentials", CredentialsRow, 0),
    ("koi_causal_nodes", KoiCausalNodeRow, 0),
    ("koi_causal_edges", KoiCausalEdgeRow, 0),
    ("koi_causal_mechanisms", KoiCausalMechanismRow, 0),
)


@dataclass(frozen=True)
class StaticInputs:
    """Validated static benchmark files ready to write to Store."""

    catalogs: list[ModelCatalog]
    user_id: str
    resource_map: ResourceMap
    aws_catalog: dict[str, Any]
    azure_catalog: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse destructive static setup inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-reset",
        action="store_true",
        help="destroy the local Store Docker volumes before seeding",
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=BENCHMARKS / "catalog-snapshots",
        help="directory containing aws.json, azure.json, and SHA256SUMS",
    )
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


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_checksums(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid checksum line in {path}")
        filename = Path(parts[1].lstrip("*")).name
        expected[filename] = parts[0].lower()
    if set(expected) != {"aws.json", "azure.json"}:
        raise ValueError(
            f"{path} must contain checksums for aws.json and azure.json only"
        )
    return expected


def _read_catalog(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _validate_catalog(catalog: dict[str, Any], *, cloud: str, catalog_key: str) -> None:
    if catalog.get("catalog_version") != catalog_key:
        raise ValueError(f"{cloud} catalog_version must be {catalog_key}")
    if catalog.get("cloud") != cloud:
        raise ValueError(f"{cloud} catalog cloud must be {cloud}")
    if cloud == "aws":
        regions = catalog.get("regions")
        if not isinstance(regions, list) or not regions:
            raise ValueError("AWS catalog must contain non-empty regions")
        instance_lists = [region.get("instance_types") for region in regions]
    else:
        instance_lists = [catalog.get("instance_types")]
    if any(
        not isinstance(instances, list) or not instances for instances in instance_lists
    ):
        raise ValueError(f"{cloud} catalog must contain non-empty instance_types")
    instances = [instance for group in instance_lists for instance in group]
    if catalog.get("instance_type_count") != len(instances):
        raise ValueError(
            f"{cloud} catalog instance_type_count does not match its contents"
        )
    for instance in instances:
        if not isinstance(instance, dict):
            raise ValueError(f"{cloud} catalog contains a non-object instance type")
        if not isinstance(instance.get("instance_type"), str):
            raise ValueError(f"{cloud} catalog instance type is missing its name")
        if not isinstance(instance.get("accelerators"), list):
            raise ValueError(f"{cloud} catalog instance type is missing accelerators")
        if not isinstance(instance.get("offerings"), list):
            raise ValueError(f"{cloud} catalog instance type is missing offerings")


def _catalog_locations(
    catalog: dict[str, Any], cloud: str
) -> set[tuple[str, str, str]]:
    locations: set[tuple[str, str, str]] = set()
    regions = catalog["regions"] if cloud == "aws" else [catalog]
    for region_catalog in regions:
        region = region_catalog.get("region")
        instances = region_catalog.get("instance_types")
        if not isinstance(instances, list):
            continue
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            instance_type = instance.get("instance_type")
            offerings = instance.get("offerings")
            if not isinstance(instance_type, str) or not isinstance(offerings, list):
                continue
            for offering in offerings:
                if not isinstance(offering, dict):
                    continue
                offering_region = offering.get("region", region)
                for zone_key in ("zone_name", "zone_id"):
                    zone = offering.get(zone_key)
                    if isinstance(offering_region, str) and isinstance(zone, str):
                        locations.add((offering_region, zone, instance_type))
    return locations


def _validate_resource_map_coverage(
    resource_map: ResourceMap,
    aws_catalog: dict[str, Any],
    azure_catalog: dict[str, Any],
) -> None:
    catalogs = {
        "aws": _catalog_locations(aws_catalog, "aws"),
        "azure": _catalog_locations(azure_catalog, "azure"),
    }
    pools = [
        (cloud, region, zone, instance_type, pool)
        for cloud, region, zone, _, _, instance_type, pool in (
            resource_map.iter_machine_pools()
        )
        if pool.total_gpus > 0
    ]
    if len(pools) != EXPECTED_POOL_COUNT:
        raise ValueError(
            f"benchmark resource map must contain {EXPECTED_POOL_COUNT} non-empty pools"
        )
    if sum(pool.total_gpus for *_, pool in pools) != EXPECTED_GPU_COUNT:
        raise ValueError(
            f"benchmark resource map must contain {EXPECTED_GPU_COUNT} GPUs"
        )
    for cloud, region, zone, instance_type, _ in pools:
        if cloud not in catalogs:
            raise ValueError(f"unsupported benchmark cloud {cloud}")
        if (region, zone, instance_type) not in catalogs[cloud]:
            raise ValueError(
                "catalog does not cover benchmark pool "
                f"{cloud}/{region}/{zone}/{instance_type}"
            )


def load_inputs(args: argparse.Namespace) -> StaticInputs:
    """Load and validate all local benchmark inputs before reset."""
    catalog_dir = args.catalog_dir.resolve()
    aws_path = catalog_dir / "aws.json"
    azure_path = catalog_dir / "azure.json"
    checksums_path = catalog_dir / "SHA256SUMS"
    paths = (
        args.model_catalog,
        args.gpu_profiles,
        args.resource_map,
        aws_path,
        azure_path,
        checksums_path,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    checksums = _expected_checksums(checksums_path)
    for path in (aws_path, azure_path):
        if _checksum(path) != checksums[path.name]:
            raise ValueError(f"checksum mismatch for {path}")

    user_id, resource_map = load_resource_map(args.resource_map)
    if user_id != CANONICAL_USER_ID:
        raise ValueError(f"benchmark resource map user must be {CANONICAL_USER_ID}")
    catalogs = load_catalogs(args.model_catalog, args.gpu_profiles)
    if len(catalogs) != 16:
        raise ValueError("benchmark model catalog must contain 16 models")
    aws_catalog = _read_catalog(aws_path)
    azure_catalog = _read_catalog(azure_path)
    _validate_catalog(aws_catalog, cloud="aws", catalog_key=AWS_CATALOG_KEY)
    _validate_catalog(azure_catalog, cloud="azure", catalog_key=AZURE_CATALOG_KEY)
    _validate_resource_map_coverage(resource_map, aws_catalog, azure_catalog)
    return StaticInputs(catalogs, user_id, resource_map, aws_catalog, azure_catalog)


def _validate_local_store() -> None:
    for name in ("docker-compose.yml", "alembic.ini", "pyproject.toml"):
        if not (STORE_ROOT / name).is_file():
            raise FileNotFoundError(STORE_ROOT / name)

    url = make_url(os.getenv("TANDEMN_POSTGRES_URL", DEFAULT_URL))
    if (
        url.get_backend_name() != "postgresql"
        or url.host not in {"localhost", "127.0.0.1"}
        or url.port != 55432
        or url.database != "tandemn"
    ):
        raise ValueError("--full-reset only permits the local tandemn Store database")


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=STORE_ROOT, check=True)


def reset_store() -> None:
    """Destroy and recreate the local Store containers, then migrate them."""
    _run(["docker", "compose", "down", "--volumes", "--remove-orphans"])
    _run(["docker", "compose", "up", "-d", "--wait"])
    _run([sys.executable, "-m", "alembic", "upgrade", "head"])


def seed_static(inputs: StaticInputs) -> None:
    """Write the validated, shared benchmark inputs to the empty Store."""
    client = PostgresClient()
    with client.begin() as session:
        session.add(
            UserRow(
                user_id=inputs.user_id,
                name="benchmarking",
                created_at=datetime.now(UTC),
            )
        )

    model_store = ModelCatalogStore(client)
    for catalog in inputs.catalogs:
        model_store.replace(catalog)
    ResourceMapStore(client, user_id=inputs.user_id).replace(inputs.resource_map)
    hardware_store = HardwareCatalogStore(client)
    hardware_store.replace(inputs.aws_catalog, AWS_CATALOG_KEY)
    hardware_store.replace(inputs.azure_catalog, AZURE_CATALOG_KEY)


def verify_static_state(client: PostgresClient) -> None:
    """Fail unless the Store has exactly the canonical static benchmark state."""
    with client.session() as session:
        actual_counts = {
            name: session.scalar(select(func.count()).select_from(table))
            for name, table, _ in EXPECTED_COUNTS
        }
        expected_counts = {name: count for name, _, count in EXPECTED_COUNTS}
        canonical_user = session.get(UserRow, CANONICAL_USER_ID)
        catalog_keys = set(
            session.scalars(select(HardwareCatalogRow.catalog_key)).all()
        )
    if actual_counts != expected_counts:
        raise RuntimeError(f"unexpected Store counts: {actual_counts}")
    if canonical_user is None:
        raise RuntimeError(f"missing canonical user {CANONICAL_USER_ID}")
    if catalog_keys != {AWS_CATALOG_KEY, AZURE_CATALOG_KEY}:
        raise RuntimeError(f"unexpected hardware catalog keys: {sorted(catalog_keys)}")


def main(argv: list[str] | None = None) -> None:
    """Perform one validated, destructive local static benchmark setup."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    if not args.full_reset:
        raise ValueError("--full-reset is required")
    inputs = load_inputs(args)
    _validate_local_store()
    LOGGER.info("validated static benchmark inputs; resetting local Store")
    reset_store()
    seed_static(inputs)
    verify_static_state(PostgresClient())
    LOGGER.info("seeded 16 models, two hardware catalogs, and one 176-GPU resource map")


if __name__ == "__main__":
    main()
