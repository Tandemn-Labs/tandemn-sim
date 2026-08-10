"""Test the Store-backed cluster simulator best-fit adapter."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from bestfit import BEST_FIT_CONFIGURATIONS, BestFitPlannerDriver, StoreSnapshot
from models import OnlineJob, Scenario


class BestFitPlannerDriverTest(unittest.TestCase):
    """Verify preferred lookup, VRAM fallback, and greedy capacity."""

    def test_mapping_covers_all_six_benchmarks(self) -> None:
        """Keep one positive hardware configuration for every mapped job."""
        self.assertEqual(
            {name: len(jobs) for name, jobs in BEST_FIT_CONFIGURATIONS.items()},
            {
                "real_life_12h": 33,
                "real_life_24h": 42,
                "real_life_3h": 20,
                "simulation_12h": 38,
                "simulation_24h": 49,
                "simulation_3h": 26,
            },
        )
        for jobs in BEST_FIT_CONFIGURATIONS.values():
            for placement in jobs.values():
                self.assertTrue(placement["hardware"])
                configuration = placement["configuration"]
                self.assertEqual(set(configuration), {"tp", "pp", "dp"})
                self.assertTrue(all(value > 0 for value in configuration.values()))

    def test_waiting_jobs_reserve_capacity_greedily(self) -> None:
        """Defer a later mapped job after the first consumes its hardware."""
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
        )

        actions = driver._build_actions(snapshot)

        self.assertEqual(
            [(action["job_id"], action["type"]) for action in actions],
            [
                ("running", "keep"),
                (first, "place"),
                (second, "defer"),
                ("paused", "defer"),
                ("finished", "preempt"),
            ],
        )
        placed = actions[1]
        self.assertEqual(placed["tenant_id"], "tenant")
        self.assertEqual(placed["ladder"][0]["config"]["model_id"], "model")
        self.assertEqual(placed["ladder"][0]["config"]["dp"], 1)
        self.assertIsNone(placed["ladder"][0]["config"]["ep"])

    def test_unavailable_fixed_mapping_uses_vram_fallback(self) -> None:
        """Use memory best-fit until the preferred hardware becomes free."""
        driver = _driver()
        job_id = "job-benchmark_timing_real_12hr_192gpu-0-1"
        unavailable = _snapshot(job_id, l40s_free=0, h100_free=8)

        first_tick = driver._build_actions(unavailable)
        second_tick = driver._build_actions(_snapshot(job_id, l40s_free=1, h100_free=8))

        first_config = first_tick[0]["ladder"][0]["config"]
        self.assertEqual(first_tick[0]["type"], "place")
        self.assertEqual(first_config["gpu_type"], "H100")
        self.assertEqual(first_config["gpu_count"], 5)
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
                    "total": 7,
                    "free": 7,
                    "pools": [],
                }
            },
            running_jobs=(),
            waiting_jobs=(_store_job(first), _store_job(second)),
            paused_jobs=(),
            finished_jobs=(),
        )

        actions = driver._build_actions(snapshot)

        self.assertEqual([action["type"] for action in actions], ["place", "defer"])
        self.assertEqual(actions[0]["ladder"][0]["config"]["gpu_count"], 5)

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
            model_name="model",
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


def _driver(jobs: tuple[OnlineJob, ...] = ()) -> BestFitPlannerDriver:
    """Build a best-fit driver without external Store clients."""
    scenario = Scenario("real_life_12h", 1, 1, jobs, ())
    return BestFitPlannerDriver(
        user_id="user",
        scenario=scenario,
        job_store=object(),
        resource_map_store=object(),
        plan_store=object(),
        backend="vllm",
    )


def _store_job(job_id: str) -> dict[str, object]:
    """Create one canonical waiting Store job payload."""
    return {
        "job_id": job_id,
        "user_id": "tenant",
        "kind": "online",
        "spec_json": {
            "model_id": "model",
            "job_features": {
                "target_p99_ttft_ms": 700,
                "target_p99_tpot_ms": 55,
            },
        },
    }


def _snapshot(job_id: str, l40s_free: int, h100_free: int) -> StoreSnapshot:
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
        waiting_jobs=(_store_job(job_id),),
        paused_jobs=(),
        finished_jobs=(),
    )


if __name__ == "__main__":
    unittest.main()
