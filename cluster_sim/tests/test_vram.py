"""Test the fixed maximum-VRAM cluster simulator adapter."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import Scenario
from vram import VRAM_CONFIGURATIONS, StoreSnapshot, VramPlannerDriver


class VramPlannerDriverTest(unittest.TestCase):
    """Verify generated VRAM mappings and fixed-placement behavior."""

    def test_mapping_covers_all_six_benchmarks(self) -> None:
        """Keep one positive hardware configuration for every benchmark job."""
        self.assertEqual(
            {name: len(jobs) for name, jobs in VRAM_CONFIGURATIONS.items()},
            {
                "real_life_12h": 33,
                "real_life_24h": 42,
                "real_life_3h": 20,
                "simulation_12h": 38,
                "simulation_24h": 49,
                "simulation_3h": 26,
            },
        )
        for jobs in VRAM_CONFIGURATIONS.values():
            for placement in jobs.values():
                self.assertTrue(placement["hardware"])
                configuration = placement["configuration"]
                self.assertEqual(set(configuration), {"tp", "pp", "dp"})
                self.assertTrue(all(value > 0 for value in configuration.values()))

    def test_known_vram_results_match_benchmark_inputs(self) -> None:
        """Preserve generated encoder and frontier-model VRAM outcomes."""
        mappings = VRAM_CONFIGURATIONS["simulation_3h"]
        encoder = mappings["job-benchmark_timing_3hr-0-5"]
        kimi = mappings["job-benchmark_timing_3hr-67-1"]

        self.assertEqual(encoder["hardware"], "A10G-24GB")
        self.assertEqual(encoder["configuration"], {"tp": 1, "pp": 1, "dp": 1})
        self.assertEqual(kimi["hardware"], "MI300X-192GB")
        self.assertEqual(kimi["configuration"], {"tp": 8, "pp": 1, "dp": 3})

    def test_deferred_job_retries_only_its_vram_mapping(self) -> None:
        """Wait for mapped hardware instead of falling back to another GPU."""
        driver = _driver()
        job_id = "job-benchmark_timing_3hr-0-5"

        first_tick = driver._build_actions(_snapshot(job_id, a10g_free=0, h100_free=8))
        second_tick = driver._build_actions(_snapshot(job_id, a10g_free=1, h100_free=8))

        self.assertEqual(first_tick, [{"job_id": job_id, "type": "defer"}])
        self.assertEqual(second_tick[0]["type"], "place")
        self.assertEqual(
            second_tick[0]["ladder"][0]["config"]["gpu_type"],
            "A10G",
        )


def _driver() -> VramPlannerDriver:
    """Build a VRAM driver without external Store clients."""
    scenario = Scenario("simulation_3h", 1, 1, (), ())
    return VramPlannerDriver(
        user_id="user",
        scenario=scenario,
        job_store=object(),
        resource_map_store=object(),
        plan_store=object(),
        backend="vllm",
    )


def _snapshot(job_id: str, a10g_free: int, h100_free: int) -> StoreSnapshot:
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
