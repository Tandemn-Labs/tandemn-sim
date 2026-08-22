"""Test the Store-backed cluster simulator best-fit adapter."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from bestfit import (
    BEST_FIT_CONFIGURATIONS,
    BestFitPlannerDriver,
    StoreSnapshot,
    _place_action,
)
from models import OnlineJob, Scenario
from policy import VALID_TP_DEGREES, load_config_policy
from serving import build_rank_cluster
from tandemn_system_data.models import Plan
from vram import VRAM_REQUIREMENTS_GIB, ResourceTarget

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
_SERVINGSIM_HARDWARE = {
    "profile": "H100",
    "memory_gb": 80,
    "memory_bandwidth_gb_s": 3350,
    "link_bandwidth_gb_s": 900,
}


class BestFitPlannerDriverTest(unittest.TestCase):
    """Verify preferred lookup, VRAM fallback, and greedy capacity."""

    def test_mapping_covers_all_benchmarks(self) -> None:
        """Keep one positive hardware configuration for every mapped job."""
        self.assertEqual(
            {name: len(jobs) for name, jobs in BEST_FIT_CONFIGURATIONS.items()},
            {
                "real_life_12h": 32,
                "real_life_24h": 41,
                "real_life_3h": 19,
                "real_3h_homogenous": 25,
                "15min_real": 1,
                "simulation_12h": 37,
                "simulation_24h": 48,
                "simulation_3h": 25,
            },
        )
        for jobs in BEST_FIT_CONFIGURATIONS.values():
            for placement in jobs.values():
                self.assertTrue(placement["hardware"])
                configuration = placement["configuration"]
                self.assertEqual(set(configuration), {"tp", "pp", "dp"})
                self.assertTrue(all(value > 0 for value in configuration.values()))
                self.assertIn(configuration["tp"], VALID_TP_DEGREES)

    def test_simulation_3h_mapping_obeys_policy_and_resource_map(self) -> None:
        """Fit every scheduled shape within policy, market, and region limits."""
        benchmark_root = ROOT / "benchmarks" / "simulation_3h"
        resource_document = yaml.safe_load(
            (benchmark_root / "resources.yaml").read_text(encoding="utf-8")
        )
        resources = {
            resource["accelerator"]: resource
            for resource in resource_document["resources"]
        }
        jobs = {}
        for path in sorted((benchmark_root / "jobs").glob("*.yaml")):
            job = yaml.safe_load(path.read_text(encoding="utf-8"))
            schedule = job["schedule"]
            workload = job["workload"]
            source = workload.get("benchmark") or workload.get("dataset")
            jobs[job["job_id"]] = {
                "model": source["model"],
                "start": schedule["start_at_offset_minutes"],
                "end": schedule.get("stop_at_offset_minutes")
                or schedule["hard_stop_at_offset_minutes"],
            }

        mappings = BEST_FIT_CONFIGURATIONS["simulation_3h"]
        self.assertEqual(set(mappings), set(jobs))
        policy = load_config_policy()
        footprints = {}
        for job_id, mapping in mappings.items():
            hardware = mapping["hardware"]
            configuration = mapping["configuration"]
            rule = policy.rule_for(hardware, jobs[job_id]["model"])
            self.assertIsNotNone(rule, f"unsupported mapping for {job_id}")
            assert rule is not None
            self.assertIn(configuration["tp"], rule.allowed_tp)
            footprints[job_id] = (
                configuration["tp"] * configuration["pp"] * configuration["dp"]
            )

        boundaries = sorted(
            {offset for job in jobs.values() for offset in (job["start"], job["end"])}
        )
        for offset in boundaries:
            active_by_hardware: dict[str, list[int]] = defaultdict(list)
            for job_id, job in jobs.items():
                if job["start"] <= offset < job["end"]:
                    hardware = mappings[job_id]["hardware"]
                    active_by_hardware[hardware].append(footprints[job_id])

            for hardware, shapes in active_by_hardware.items():
                resource = resources[hardware]
                region_limits = tuple(resource["regions"].values())
                market_limits = tuple(resource["markets"].values())
                self.assertTrue(
                    _fits_capacity_bins(shapes, region_limits),
                    f"{hardware} exceeds region capacity at minute {offset}",
                )
                self.assertTrue(
                    _fits_capacity_bins(shapes, market_limits),
                    f"{hardware} exceeds market capacity at minute {offset}",
                )

    def test_simulation_mappings_match_jobs_models_and_resources(self) -> None:
        """Cross-reference every simulation job with both static baselines."""
        policy = load_config_policy()
        for benchmark in ("simulation_3h", "simulation_12h", "simulation_24h"):
            root = ROOT / "benchmarks" / benchmark
            resources = {
                resource["accelerator"]
                for resource in yaml.safe_load(
                    (root / "resources.yaml").read_text(encoding="utf-8")
                )["resources"]
            }
            jobs = {}
            for path in sorted((root / "jobs").glob("*.yaml")):
                job = yaml.safe_load(path.read_text(encoding="utf-8"))
                source = job["workload"].get("benchmark") or job["workload"]["dataset"]
                jobs[job["job_id"]] = source["model"]

            mappings = BEST_FIT_CONFIGURATIONS[benchmark]
            self.assertEqual(set(mappings), set(jobs))
            self.assertEqual(set(VRAM_REQUIREMENTS_GIB[benchmark]), set(jobs))
            for job_id, model_id in jobs.items():
                mapping = mappings[job_id]
                hardware = mapping["hardware"]
                self.assertIn(hardware, resources)
                rule = policy.rule_for(hardware, model_id)
                self.assertIsNotNone(rule, f"unsupported mapping for {job_id}")
                assert rule is not None
                self.assertIn(mapping["configuration"]["tp"], rule.allowed_tp)

    def test_waiting_jobs_exclude_terminal_jobs(self) -> None:
        """Defer a later mapped job without an action for a terminal job."""
        driver = _driver()
        first = "job-benchmark_timing_real_12hr_192gpu-0-1"
        second = "job-benchmark_timing_real_12hr_192gpu-0-2"
        snapshot = StoreSnapshot(
            resources={
                "reserved|aws|us-east|use1-az1|L40S": {
                    "gpu_type": "L40S",
                    "total": 1,
                    "free": 1,
                    "pools": [],
                }
            },
            running_jobs=({"job_id": "running"},),
            waiting_jobs=(_store_job(first), _store_job(second)),
            paused_jobs=({"job_id": "paused"},),
            finished_jobs=({"job_id": "finished"},),
            model_catalogs=_catalogs(),
        )

        actions = driver._build_actions(snapshot)

        self.assertEqual(
            [(action["job_id"], action["type"]) for action in actions],
            [
                ("running", "keep"),
                (first, "place"),
                (second, "defer"),
                ("paused", "defer"),
            ],
        )
        placed = actions[1]
        rank_config = placed["ladder"][0]["config"]
        self.assertEqual(rank_config["model_id"], MODEL_ID)
        self.assertEqual(rank_config["precision"], "fp16")
        self.assertEqual(rank_config["dp"], 1)
        self.assertEqual(
            rank_config["count"],
            rank_config["gpu_count"],
        )
        self.assertEqual(rank_config["ep"], 1)
        for field, value in _launch_config().items():
            self.assertEqual(rank_config[field], value)
        self.assertRegex(
            placed["ladder"][0]["rank_id"], r"^rank_[0-7][0-9A-HJKMNP-TV-Z]{25}$"
        )

    def test_snapshot_reads_each_relevant_model_catalog_once(self) -> None:
        """Freeze Store catalog records with the rest of the planning tick."""
        driver = _driver()
        requested = []
        driver.job_store = SimpleNamespace(
            running_jobs=lambda _user_id: (),
            waiting_jobs=lambda _user_id: (
                _store_job("first"),
                _store_job("second"),
            ),
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

    def test_unavailable_fixed_mapping_uses_vram_fallback(self) -> None:
        """Use memory best-fit until the preferred hardware becomes free."""
        driver = _driver()
        job_id = "job-benchmark_timing_real_12hr_192gpu-0-1"
        model_id = MODEL_ID
        unavailable = _snapshot(
            job_id,
            l40s_free=0,
            h100_free=8,
            model_id=model_id,
        )

        first_tick = driver._build_actions(unavailable)
        second_tick = driver._build_actions(
            _snapshot(job_id, l40s_free=1, h100_free=8, model_id=model_id)
        )

        first_config = first_tick[0]["ladder"][0]["config"]
        self.assertEqual(first_tick[0]["type"], "place")
        self.assertEqual(first_config["gpu_type"], "H100")
        self.assertEqual(first_config["gpu_count"], 8)
        self.assertEqual(first_config["tp"], 8)
        self.assertEqual(second_tick[0]["type"], "place")
        self.assertEqual(
            second_tick[0]["ladder"][0]["config"]["gpu_type"],
            "L40S",
        )

    def test_vram_fallback_reserves_capacity_greedily(self) -> None:
        """Let an earlier fallback reduce capacity for later jobs in the tick."""
        driver = _driver()
        first = "job-benchmark_timing_real_12hr_192gpu-0-1"
        second = "job-benchmark_timing_real_12hr_192gpu-0-2"
        snapshot = StoreSnapshot(
            resources={
                "reserved|aws|us-east|use1-az1|H100": {
                    "gpu_type": "H100",
                    "total": 8,
                    "free": 8,
                    "pools": [],
                }
            },
            running_jobs=(),
            waiting_jobs=(
                _store_job(first, MODEL_ID),
                _store_job(second, MODEL_ID),
            ),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=_catalogs(),
        )

        actions = driver._build_actions(snapshot)

        self.assertEqual([action["type"] for action in actions], ["place", "defer"])
        self.assertEqual(actions[0]["ladder"][0]["config"]["gpu_count"], 8)
        self.assertEqual(actions[0]["ladder"][0]["config"]["tp"], 8)

    def test_job_defers_when_primary_and_vram_fallback_do_not_fit(self) -> None:
        """Defer only after exhausting the fixed and memory-based choices."""
        driver = _driver()
        job_id = "job-benchmark_timing_real_12hr_192gpu-0-1"

        actions = driver._build_actions(_snapshot(job_id, l40s_free=0, h100_free=0))

        self.assertEqual(actions, [{"job_id": job_id, "type": "defer"}])

    def test_fresh_store_id_resolves_fixed_and_fallback_mappings(self) -> None:
        """Preserve benchmark identity after the CLI replaces Store job IDs."""
        benchmark_job_id = "job-benchmark_timing_real_12hr_192gpu-0-1"
        store_job_id = "job-fresh-store-id"
        scenario_job = OnlineJob(
            name=benchmark_job_id,
            store_job_id=store_job_id,
            model_name=MODEL_ID,
            starts_at_ns=0,
            stops_at_ns=1,
            workload={},
        )
        driver = _driver((scenario_job,))

        actions = driver._build_actions(
            _snapshot(store_job_id, l40s_free=1, h100_free=0)
        )

        self.assertEqual(actions[0]["type"], "place")
        self.assertEqual(actions[0]["ladder"][0]["config"]["gpu_type"], "L40S")

    def test_data_parallelism_creates_per_replica_rank_shapes(self) -> None:
        """Keep DP in replica count rather than one ServingSim rank shape."""
        action = _place_action(
            "job",
            _store_job("job"),
            ResourceTarget(
                environment_key="reserved|aws|us-east|use1-az1|H100",
                environment={},
                pool=None,
                gpu_type="H100",
                instance_type=None,
            ),
            1,
            1,
            4,
            _catalogs(),
            load_config_policy(),
        )

        shape = action["ladder"][0]["config"]
        self.assertEqual(shape["count"], 1)
        self.assertEqual(shape["gpu_count"], 1)
        self.assertEqual(action["ladder"][0]["n_replicas"], 4)
        Plan(user_id="tenant", actions=[action])
        cluster = build_rank_cluster(
            SimpleNamespace(
                rank_name=action["ladder"][0]["rank_id"],
                model_name=shape["model_id"],
                shape=shape,
                chains=(object(),) * 4,
            ),
            _SERVINGSIM_HARDWARE,
        )
        self.assertEqual(cluster["nodes"][0]["instances"][0]["num_npus"], 1)
        self.assertEqual(cluster["nodes"][0]["num_instances"], 4)


def _driver(jobs: tuple[OnlineJob, ...] = ()) -> BestFitPlannerDriver:
    """Build a best-fit driver without external Store clients."""
    scenario = Scenario("real_life_12h", 1, 1, jobs, ())
    return BestFitPlannerDriver(
        user_id="user",
        scenario=scenario,
        job_store=object(),
        resource_map_store=object(),
        model_catalog_store=object(),
        plan_store=object(),
        backend="vllm",
    )


def _store_job(job_id: str, model_id: str = MODEL_ID) -> dict[str, object]:
    """Create one canonical waiting Store job payload."""
    return {
        "job_id": job_id,
        "user_id": "tenant",
        "kind": "online",
        "spec_json": {
            "model_id": model_id,
            "job_features": {
                "target_p99_ttft_ms": 700,
                "target_p99_tpot_ms": 55,
            },
        },
    }


def _snapshot(
    job_id: str,
    l40s_free: int,
    h100_free: int,
    model_id: str = MODEL_ID,
) -> StoreSnapshot:
    """Create one tick snapshot with independently controlled hardware."""
    return StoreSnapshot(
        resources={
            "reserved|aws|us-east|use1-az1|L40S": {
                "gpu_type": "L40S",
                "total": l40s_free,
                "free": l40s_free,
                "pools": [],
            },
            "reserved|aws|us-east|use1-az1|H100": {
                "gpu_type": "H100",
                "total": h100_free,
                "free": h100_free,
                "pools": [],
            },
        },
        running_jobs=(),
        waiting_jobs=(_store_job(job_id, model_id),),
        paused_jobs=(),
        finished_jobs=(),
        model_catalogs=_catalogs(),
    )


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
    """Return one raw Store catalog with exact target-GPU profiles."""
    catalog = _launch_config()
    for field in (
        "max_num_seq",
        "max_num_batched_tokens",
        "block_size",
        "kvcache_dtype",
    ):
        catalog[field] = [
            {"gpu_type": gpu_type, "value": catalog[field]}
            for gpu_type in ("H100", "L40S")
        ]
    return {MODEL_ID: catalog}


def _fits_capacity_bins(shapes: list[int], capacities: tuple[int, ...]) -> bool:
    """Return whether indivisible placement shapes fit the supplied capacities."""
    ordered_shapes = sorted(shapes, reverse=True)

    def place(index: int, remaining: tuple[int, ...]) -> bool:
        if index == len(ordered_shapes):
            return True
        shape = ordered_shapes[index]
        attempted = set()
        for bin_index, capacity in enumerate(remaining):
            if capacity < shape or capacity in attempted:
                continue
            attempted.add(capacity)
            updated = list(remaining)
            updated[bin_index] -= shape
            if place(index + 1, tuple(sorted(updated, reverse=True))):
                return True
        return False

    return place(0, tuple(sorted(capacities, reverse=True)))


if __name__ == "__main__":
    unittest.main()
