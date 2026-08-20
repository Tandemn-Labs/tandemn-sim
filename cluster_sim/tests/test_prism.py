"""Test Prism terminal-job handling."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from policy import load_config_policy
from prism import (
    PlacementCandidate,
    PrismPlannerDriver,
    StoreSnapshot,
    build_prism_plan,
    verify_place_plan,
)
from serving import build_rank_cluster
from tandemn_system_data.ids import is_valid_id
from tandemn_system_data.models import Plan
from vram import ResourceTarget

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
_SERVINGSIM_HARDWARE = {
    "profile": "H100",
    "memory_gb": 80,
    "memory_bandwidth_gb_s": 3350,
    "link_bandwidth_gb_s": 900,
}


class PrismPlannerDriverTest(unittest.TestCase):
    """Verify Prism omits terminal jobs from later plans."""

    def test_terminal_jobs_produce_no_actions(self) -> None:
        """Keep terminal jobs out of future placement plans."""
        driver = object.__new__(PrismPlannerDriver)
        driver.benchmark_name = "test"
        driver.benchmark_job_ids = {}
        driver.config_policy = object()
        driver._select_placements = lambda problems, capacities: {}
        snapshot = StoreSnapshot(
            resources={},
            running_jobs=(),
            waiting_jobs=(),
            paused_jobs=(),
            finished_jobs=({"job_id": "finished"},),
            model_catalogs={},
        )

        self.assertEqual(driver._build_actions(snapshot), [])

    def test_snapshot_reads_each_relevant_model_catalog_once(self) -> None:
        """Freeze distinct Store catalog records with the planning tick."""
        driver = object.__new__(PrismPlannerDriver)
        driver.user_id = "user"
        waiting = (_store_job("first"), _store_job("second"))
        requested = []
        driver.job_store = SimpleNamespace(
            running_jobs=lambda _user_id: (),
            waiting_jobs=lambda _user_id: waiting,
            paused_jobs=lambda _user_id: (),
        )
        driver.resource_map_store = SimpleNamespace(
            get=lambda: SimpleNamespace(
                market="reserved",
                scheduling_summary=lambda: {},
            )
        )
        driver.model_catalog_store = SimpleNamespace(
            get=lambda model_id: (
                requested.append(model_id)
                or SimpleNamespace(model_dump=lambda **_kwargs: _catalogs()[model_id])
            )
        )

        snapshot = driver._read_store_snapshot()

        self.assertEqual(requested, [MODEL_ID])
        self.assertEqual(snapshot.model_catalogs, _catalogs())

    def test_selected_candidate_uses_exact_gpu_catalog_profile(self) -> None:
        """Materialize Store launch fields only after Prism selection."""
        candidate = PlacementCandidate(
            candidate_id="candidate",
            job_id="job",
            target=ResourceTarget(
                environment_key="reserved|aws|us-east|use1-az1|L40S",
                environment={},
                pool=None,
                gpu_type="L40S",
                instance_type=None,
            ),
            tp=1,
            pp=1,
            dp=3,
            footprint=3,
            memory_gib=48.0,
            required_vram_gib=1.0,
            model_weight_gib=1.0,
            kv_demand=0.0,
            price_per_hour=1.0,
        )
        catalogs = _catalogs()
        policy = load_config_policy()

        plan = build_prism_plan(
            _store_job("job"),
            candidate,
            catalogs,
            policy,
        )

        config = plan["ladder"][0]["config"]
        self.assertEqual(config["max_num_seq"], 256)
        self.assertEqual(config["count"], 1)
        self.assertEqual(plan["ladder"][0]["n_replicas"], 3)
        for field, value in _launch_config().items():
            self.assertEqual(config[field], value)
        self.assertTrue(is_valid_id(plan["ladder"][0]["rank_id"], "rank"))
        verify_place_plan(plan, candidate, catalogs, policy)
        Plan(user_id="tenant", actions=[plan])
        cluster = build_rank_cluster(
            SimpleNamespace(
                rank_name=plan["ladder"][0]["rank_id"],
                model_name=config["model_id"],
                shape=config,
                chains=(object(),) * 3,
            ),
            _SERVINGSIM_HARDWARE,
        )
        self.assertEqual(cluster["nodes"][0]["instances"][0]["num_npus"], 1)
        self.assertEqual(cluster["nodes"][0]["num_instances"], 3)


def _store_job(job_id: str) -> dict[str, object]:
    """Create one canonical waiting Store job."""
    return {
        "job_id": job_id,
        "user_id": "tenant",
        "spec_json": {"model_id": MODEL_ID},
    }


def _launch_config() -> dict[str, object]:
    """Return one resolved launch profile from Tandemn Store."""
    return {
        "engine_name": "vllm",
        "engine_version": "0.16.0",
        "weight_dtype": "float16",
        "weight_quantization_method": "none",
        "weight_quantization_bits": 16,
        "activation_dtype": "float16",
        "gpu_mem_util": 0.9,
        "max_model_len": 131072,
        "prefix_cache_enabled": False,
        "chunked_prefill_enable": True,
        "chunk_size": 100,
        "scheduling_policy": "fcfs",
        "preemption_policy": "recompute",
        "router_policy": "kv_router",
        "max_num_seq": 256,
        "max_num_batched_tokens": 8192,
        "block_size": 16,
        "kvcache_dtype": "auto",
    }


def _catalogs() -> dict[str, dict[str, object]]:
    """Return one raw Store catalog with target-specific GPU profiles."""
    catalog = _launch_config()
    for field in (
        "max_num_seq",
        "max_num_batched_tokens",
        "block_size",
        "kvcache_dtype",
    ):
        catalog[field] = [
            {"gpu_type": "H100", "value": "not-selected"},
            {"gpu_type": "L40S", "value": catalog[field]},
        ]
    return {MODEL_ID: catalog}


if __name__ == "__main__":
    unittest.main()
