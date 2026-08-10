"""Test the workload-derived maximum-VRAM cluster simulator adapter."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import OnlineJob, Scenario
from vram import (
    MODEL_ARCHITECTURES,
    VRAM_REQUIREMENTS_GIB,
    StoreSnapshot,
    VramPlannerDriver,
    generate_vram_requirements,
)


class VramPlannerDriverTest(unittest.TestCase):
    """Verify generated requirements and capacity-aware placement."""

    def test_mapping_covers_all_six_benchmarks(self) -> None:
        """Keep one positive VRAM requirement for every benchmark job."""
        self.assertEqual(
            {name: len(jobs) for name, jobs in VRAM_REQUIREMENTS_GIB.items()},
            {
                "real_life_12h": 33,
                "real_life_24h": 42,
                "real_life_3h": 20,
                "simulation_12h": 38,
                "simulation_24h": 49,
                "simulation_3h": 26,
            },
        )
        for jobs in VRAM_REQUIREMENTS_GIB.values():
            self.assertTrue(all(required_gib > 0 for required_gib in jobs.values()))

    def test_mapping_is_reproducible_from_benchmark_workloads(self) -> None:
        """Recalculate all requirements from architecture and workload inputs."""
        self.assertEqual(
            generate_vram_requirements(ROOT / "benchmarks"),
            VRAM_REQUIREMENTS_GIB,
        )

    def test_known_vram_results_match_benchmark_inputs(self) -> None:
        """Preserve generated encoder and frontier-model VRAM outcomes."""
        mappings = VRAM_REQUIREMENTS_GIB["simulation_3h"]
        encoder = mappings["job-benchmark_timing_3hr-0-5"]
        kimi = mappings["job-benchmark_timing_3hr-67-1"]

        self.assertEqual(encoder, 1.053)
        self.assertEqual(kimi, 2206.908)
        self.assertEqual(
            MODEL_ARCHITECTURES["moonshotai/Kimi-K2-Instruct"].mla_cache_elements, 576
        )

    def test_placement_uses_best_available_memory_fit(self) -> None:
        """Use another fitting GPU when the smallest accelerator is full."""
        driver = _driver()
        job_id = "job-benchmark_timing_3hr-0-5"

        first_tick = driver._build_actions(_snapshot(job_id, a10g_free=0, h100_free=8))
        second_tick = driver._build_actions(_snapshot(job_id, a10g_free=1, h100_free=8))

        self.assertEqual(first_tick[0]["type"], "place")
        self.assertEqual(first_tick[0]["ladder"][0]["config"]["gpu_type"], "H100")
        self.assertEqual(second_tick[0]["type"], "place")
        self.assertEqual(
            second_tick[0]["ladder"][0]["config"]["gpu_type"],
            "A10G",
        )

    def test_placement_minimizes_total_provisioned_vram(self) -> None:
        """Prefer seven 80 GiB GPUs over three 192 GiB GPUs."""
        driver = _driver(benchmark_name="real_life_12h")
        job_id = "job-benchmark_timing_real_12hr_192gpu-0-3"

        actions = driver._build_actions(
            _snapshot(
                job_id,
                a10g_free=0,
                h100_free=8,
                mi300x_free=8,
            )
        )

        config = actions[0]["ladder"][0]["config"]
        self.assertEqual(config["gpu_type"], "H100")
        self.assertEqual(config["gpu_count"], 7)

    def test_fresh_store_id_resolves_to_benchmark_requirement(self) -> None:
        """Look up requirements by benchmark identity after CLI ID replacement."""
        benchmark_job_id = "job-benchmark_timing_3hr-0-5"
        store_job_id = "job-fresh-store-id"
        scenario_job = OnlineJob(
            name=benchmark_job_id,
            store_job_id=store_job_id,
            model_name="intfloat/multilingual-e5-large",
            starts_at_ns=0,
            stops_at_ns=1,
            workload={},
        )
        driver = _driver((scenario_job,))

        actions = driver._build_actions(
            _snapshot(store_job_id, a10g_free=1, h100_free=0)
        )

        self.assertEqual(actions[0]["type"], "place")
        self.assertEqual(actions[0]["ladder"][0]["config"]["gpu_type"], "A10G")


def _driver(
    jobs: tuple[OnlineJob, ...] = (),
    benchmark_name: str = "simulation_3h",
) -> VramPlannerDriver:
    """Build a VRAM driver without external Store clients."""
    scenario = Scenario(benchmark_name, 1, 1, jobs, ())
    return VramPlannerDriver(
        user_id="user",
        scenario=scenario,
        job_store=object(),
        resource_map_store=object(),
        plan_store=object(),
        backend="vllm",
    )


def _snapshot(
    job_id: str,
    a10g_free: int,
    h100_free: int,
    mi300x_free: int = 0,
) -> StoreSnapshot:
    """Create one tick snapshot with independently controlled hardware."""
    return StoreSnapshot(
        resources={
            "reserved|aws|us-east|use1-az1|A10G": {
                "gpu_type": "A10G",
                "total": a10g_free,
                "free": a10g_free,
                "pools": [],
            },
            "reserved|aws|us-east|use1-az1|H100": {
                "gpu_type": "H100",
                "total": h100_free,
                "free": h100_free,
                "pools": [],
            },
            "reserved|aws|us-east|use1-az1|MI300X": {
                "gpu_type": "MI300X",
                "total": mi300x_free,
                "free": mi300x_free,
                "pools": [],
            },
        },
        running_jobs=(),
        waiting_jobs=(
            {
                "job_id": job_id,
                "user_id": "tenant",
                "kind": "online",
                "spec_json": {
                    "model_id": "intfloat/multilingual-e5-large",
                    "job_features": {
                        "target_p99_ttft_ms": 120,
                        "target_p99_tpot_ms": 1,
                    },
                },
            },
        ),
        paused_jobs=(),
        finished_jobs=(),
    )


if __name__ == "__main__":
    unittest.main()
