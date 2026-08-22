"""Test static benchmark seed validation without resetting Docker."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.benchmark_static_seed import (
    AWS_CATALOG_KEY,
    AZURE_CATALOG_KEY,
    _validate_catalog,
    _validate_local_store,
    _validate_resource_map_coverage,
    load_inputs,
    parse_args,
    reset_store,
)
from benchmarks.seed_benchmark_store import load_resource_map

ROOT = Path(__file__).resolve().parents[1]


def _aws_catalog() -> dict[str, object]:
    return {
        "catalog_version": AWS_CATALOG_KEY,
        "cloud": "aws",
        "instance_type_count": 6,
        "regions": [
            {
                "region": "us-east-1",
                "instance_types": [
                    {
                        "instance_type": instance_type,
                        "accelerators": [],
                        "offerings": [{"zone_name": "us-east-1a"}],
                    }
                    for instance_type in ("p5.48xlarge", "g5.48xlarge", "g6e.48xlarge")
                ],
            },
            {
                "region": "ap-northeast-1",
                "instance_types": [
                    {
                        "instance_type": instance_type,
                        "accelerators": [],
                        "offerings": [{"zone_id": "apne1-az1"}],
                    }
                    for instance_type in ("p5.48xlarge", "g5.48xlarge", "g6e.48xlarge")
                ],
            },
        ],
    }


def _azure_catalog() -> dict[str, object]:
    return {
        "catalog_version": AZURE_CATALOG_KEY,
        "cloud": "azure",
        "instance_type_count": 3,
        "instance_types": [
            {
                "instance_type": "Standard_ND96amsr_A100_v4",
                "accelerators": [],
                "offerings": [{"region": "westus2", "zone_name": "1"}],
            },
            {
                "instance_type": "Standard_NC96ads_A100_v4",
                "accelerators": [],
                "offerings": [{"region": "westus2", "zone_name": "1"}],
            },
            {
                "instance_type": "Standard_ND96isr_MI300X_v5",
                "accelerators": [],
                "offerings": [{"region": "westus", "zone_name": "default"}],
            },
        ],
    }


class StaticSeedTest(unittest.TestCase):
    """Verify static inputs before a destructive Store reset."""

    def test_loads_the_fixed_model_and_resource_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            catalog_dir = Path(temporary)
            aws_path = catalog_dir / "aws.json"
            azure_path = catalog_dir / "azure.json"
            aws_path.write_text(json.dumps(_aws_catalog()), encoding="utf-8")
            azure_path.write_text(json.dumps(_azure_catalog()), encoding="utf-8")
            checksums = "\n".join(
                " ".join(
                    (
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                        f"snapshots/{path.name}",
                    )
                )
                for path in (aws_path, azure_path)
            )
            (catalog_dir / "SHA256SUMS").write_text(checksums + "\n", encoding="utf-8")

            inputs = load_inputs(
                SimpleNamespace(
                    catalog_dir=catalog_dir,
                    model_catalog=ROOT / "benchmarks/model_catalogs.csv",
                    gpu_profiles=ROOT / "benchmarks/model_catalog_gpu_profiles.csv",
                    resource_map=ROOT / "benchmarks/resource_map_benchmarking.jsonl",
                    benchmark_id="simulation_3h",
                )
            )

        self.assertEqual(len(inputs.catalogs), 16)
        self.assertEqual(
            sum(
                pool.total_gpus for *_, pool in inputs.resource_map.iter_machine_pools()
            ),
            200,
        )

    def test_uses_the_repository_catalog_snapshot_directory_by_default(self) -> None:
        self.assertEqual(
            parse_args(["--benchmark-id", "simulation_3h"]).catalog_dir,
            ROOT / "benchmarks/catalog-snapshots",
        )

    def test_selects_resource_map_by_benchmark_id(self) -> None:
        _, resource_map = load_resource_map(
            ROOT / "benchmarks/resource_map_benchmarking.jsonl",
            "simulation_12h",
        )

        self.assertEqual(
            sum(
                pool.total_gpus
                for *_, pool in resource_map.iter_machine_pools()
            ),
            400,
        )

    def test_rejects_catalog_without_a_benchmark_pool(self) -> None:
        _, resource_map = load_resource_map(
            ROOT / "benchmarks/resource_map_benchmarking.jsonl"
        )
        aws_catalog = _aws_catalog()
        aws_catalog["regions"] = [aws_catalog["regions"][0]]

        with self.assertRaisesRegex(ValueError, "does not cover"):
            _validate_resource_map_coverage(
                resource_map,
                aws_catalog,
                _azure_catalog(),
                "simulation_3h",
            )

    def test_rejects_wrong_catalog_version(self) -> None:
        catalog = _azure_catalog()
        catalog["catalog_version"] = "wrong"

        with self.assertRaisesRegex(ValueError, "catalog_version"):
            _validate_catalog(catalog, cloud="azure", catalog_key=AZURE_CATALOG_KEY)

    def test_rejects_remote_database_url(self) -> None:
        with patch.dict(
            "os.environ",
            {"TANDEMN_POSTGRES_URL": "postgresql://tandemn:tandemn@db.example/tandemn"},
        ):
            with self.assertRaisesRegex(ValueError, "local tandemn"):
                _validate_local_store()

    def test_resets_then_migrates(self) -> None:
        with patch("benchmarks.benchmark_static_seed._run") as run:
            reset_store()

        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                ["docker", "compose", "down", "--volumes", "--remove-orphans"],
                ["docker", "compose", "up", "-d", "--wait"],
                [sys.executable, "-m", "alembic", "upgrade", "head"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
