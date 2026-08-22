"""Create or reset one isolated benchmark baseline user."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, or_, select
from tandemn_system_data.clients import PostgresClient
from tandemn_system_data.db.orm import (
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
from tandemn_system_data.ids import new_user_id
from tandemn_system_data.models._base import utc_now
from tandemn_system_data.models.resource_map import ResourceMap

if __package__:
    from benchmarks.seed_benchmark_store import DEFAULT_BENCHMARK_ID, load_resource_map
else:
    from seed_benchmark_store import DEFAULT_BENCHMARK_ID, load_resource_map

BENCHMARKS = Path(__file__).resolve().parent
BASELINES = (
    "koi",
    "dynamo_aic",
    "bestfit",
    "vram",
    "edf",
    "prism",
    "cp_sat",
    "gpt_only",
    "openevolve",
    "contextual_bandit",
    "evox",
)
AWS_CATALOG_KEY = "aws-accelerated-hardware-v2"
AZURE_CATALOG_KEY = "azure-accelerated-hardware-v1"
EXPECTED_KOI_COUNTS = (123, 2222, 89)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one baseline identity and optional Koi source root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=BASELINES)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--koi-root", type=Path)
    parser.add_argument(
        "--resource-map",
        type=Path,
        default=BENCHMARKS / "resource_map_benchmarking.jsonl",
    )
    parser.add_argument("--benchmark-id", default=DEFAULT_BENCHMARK_ID)
    return parser.parse_args(argv)


def baseline_name(baseline: str, run_id: str) -> str:
    """Build the stable Store name for one baseline run."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "--run-id may contain only letters, numbers, dots, underscores, and hyphens"
        )
    return f"benchmark-baseline:{baseline}:{run_id}"


def resource_map_body(resource_map: ResourceMap) -> dict[str, Any]:
    """Keep only the canonical resource-map body, never its source user or version."""
    return {
        "market": list(resource_map.market),
        "clouds": {
            cloud_id: cloud.model_dump(mode="json")
            for cloud_id, cloud in resource_map.clouds.items()
        },
    }


def require_static_seed(client: PostgresClient, canonical_user_id: str) -> None:
    """Require the global benchmark state before creating an isolated user."""
    with client.session() as session:
        canonical_user = session.get(UserRow, canonical_user_id)
        canonical_map = session.get(ResourceMapRow, canonical_user_id)
        model_count = session.scalar(select(func.count()).select_from(ModelCatalogRow))
        catalog_keys = set(
            session.scalars(select(HardwareCatalogRow.catalog_key)).all()
        )
    if canonical_user is None or canonical_map is None:
        raise RuntimeError(
            "run benchmark_static_seed.py before creating baseline users"
        )
    if model_count != 16 or catalog_keys != {AWS_CATALOG_KEY, AZURE_CATALOG_KEY}:
        raise RuntimeError("static benchmark catalogs are incomplete")


def reset_user(
    client: PostgresClient,
    *,
    name: str,
    resource_map: ResourceMap,
) -> str:
    """Atomically replace one named user's mutable benchmark state and resource map."""
    with client.begin() as session:
        users = list(session.scalars(select(UserRow).where(UserRow.name == name)).all())
        if len(users) > 1:
            raise RuntimeError(f"multiple Store users are named {name!r}")
        if users:
            user_id = users[0].user_id
        else:
            user_id = new_user_id()
            session.add(UserRow(user_id=user_id, name=name, created_at=utc_now()))
            session.flush()

        job_ids = list(
            session.scalars(
                select(JobRow.job_id).where(JobRow.user_id == user_id)
            ).all()
        )
        rank_ids = list(
            session.scalars(
                select(RankRow.rank_id)
                .join(JobRow, RankRow.job_id == JobRow.job_id)
                .where(JobRow.user_id == user_id)
            ).all()
        )
        metric_scopes = []
        event_scopes = [EventRow.user_id == user_id]
        if job_ids:
            metric_scopes.append(GpuMetricRow.job_id.in_(job_ids))
            event_scopes.append(EventRow.job_id.in_(job_ids))
        if rank_ids:
            metric_scopes.append(GpuMetricRow.rank_id.in_(rank_ids))
            event_scopes.append(EventRow.rank_id.in_(rank_ids))
        if metric_scopes:
            session.execute(delete(GpuMetricRow).where(or_(*metric_scopes)))
        session.execute(delete(EventRow).where(or_(*event_scopes)))
        session.execute(delete(EvidenceRowRow).where(EvidenceRowRow.user_id == user_id))
        session.execute(
            delete(KoiCausalMechanismRow).where(
                KoiCausalMechanismRow.user_id == user_id
            )
        )
        session.execute(
            delete(KoiCausalEdgeRow).where(KoiCausalEdgeRow.user_id == user_id)
        )
        session.execute(
            delete(KoiCausalNodeRow).where(KoiCausalNodeRow.user_id == user_id)
        )
        session.execute(delete(PlanRow).where(PlanRow.user_id == user_id))
        session.execute(delete(ResourceMapRow).where(ResourceMapRow.user_id == user_id))
        session.execute(delete(JobRow).where(JobRow.user_id == user_id))
        session.add(
            ResourceMapRow(
                user_id=user_id,
                version=1,
                pools_json=resource_map_body(resource_map),
                updated_at=utc_now(),
            )
        )
    return user_id


def _graph_counts(client: PostgresClient, user_id: str) -> tuple[int, int, int]:
    with client.session() as session:
        return (
            session.scalar(
                select(func.count())
                .select_from(KoiCausalNodeRow)
                .where(KoiCausalNodeRow.user_id == user_id)
            ),
            session.scalar(
                select(func.count())
                .select_from(KoiCausalEdgeRow)
                .where(KoiCausalEdgeRow.user_id == user_id)
            ),
            session.scalar(
                select(func.count())
                .select_from(KoiCausalMechanismRow)
                .where(KoiCausalMechanismRow.user_id == user_id)
            ),
        )


def initialize_koi(client: PostgresClient, user_id: str, koi_root: Path | None) -> None:
    """Import Koi's fixed causal graph for this user and verify its shape."""
    if koi_root is None:
        raise ValueError("baseline 'koi' requires --koi-root")
    koi_root = koi_root.resolve()
    if not (koi_root / "src/bootstrap/initialization.py").is_file():
        raise FileNotFoundError(koi_root / "src/bootstrap/initialization.py")
    sys.path.insert(0, str(koi_root))
    from src.bootstrap.initialization import init_causal_graph

    init_causal_graph(user_id, postgres_client=client)
    if _graph_counts(client, user_id) != EXPECTED_KOI_COUNTS:
        raise RuntimeError(f"unexpected Koi graph counts for {user_id}")


def main(argv: list[str] | None = None) -> None:
    """Reset one baseline user and emit its Store identity as JSON."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    name = baseline_name(args.baseline, args.run_id)
    canonical_user_id, resource_map = load_resource_map(
        args.resource_map,
        args.benchmark_id,
    )
    client = PostgresClient()
    require_static_seed(client, canonical_user_id)
    user_id = reset_user(client, name=name, resource_map=resource_map)
    if args.baseline == "koi":
        initialize_koi(client, user_id, args.koi_root)
    elif _graph_counts(client, user_id) != (0, 0, 0):
        raise RuntimeError(f"non-Koi baseline {args.baseline} has a causal graph")
    LOGGER.info("prepared %s for %s", user_id, name)
    print(
        json.dumps(
            {"baseline": args.baseline, "run_id": args.run_id, "user_id": user_id}
        )
    )


if __name__ == "__main__":
    main()
