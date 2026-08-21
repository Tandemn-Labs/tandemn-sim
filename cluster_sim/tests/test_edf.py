"""Test earliest-end-time VRAM placement."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from edf import EdfPlannerDriver
from models import BatchJob, OnlineJob, Scenario
from vram import ResourceTarget, StoreSnapshot


class EdfPlannerDriverTest(unittest.TestCase):
    """Verify EDF reorders the full Store waiting queue every tick."""

    def test_deferred_jobs_retry_before_later_ending_new_arrivals(self) -> None:
        """Give old deferred work its EDF position on the next tick."""
        driver = _driver()

        with (
            patch("edf.select_vram_placement", side_effect=_select_one_gpu),
            patch("edf._place_action", side_effect=_place_action),
        ):
            first_tick = driver._build_actions(_snapshot(("late", "early"), free=0))
            second_tick = driver._build_actions(
                _snapshot(("middle", "late", "early"), free=1)
            )

        self.assertEqual(
            [(action["job_id"], action["type"]) for action in first_tick],
            [("early", "defer"), ("late", "defer")],
        )
        self.assertEqual(
            [(action["job_id"], action["type"]) for action in second_tick],
            [
                ("early", "place"),
                ("middle", "defer"),
                ("late", "defer"),
            ],
        )


def _driver() -> EdfPlannerDriver:
    online_jobs = tuple(
        OnlineJob(
            name=benchmark_id,
            store_job_id=store_id,
            model_name="model",
            starts_at_ns=0,
            stops_at_ns=end_time,
            workload={},
        )
        for store_id, benchmark_id, end_time in (
            ("middle", "job-benchmark_timing_3hr-0-4", 20),
            ("late", "job-benchmark_timing_3hr-0-1", 30),
        )
    )
    batch_jobs = (
        BatchJob(
            name="job-benchmark_timing_3hr-0-5",
            store_job_id="early",
            model_name="model",
            releases_at_ns=0,
            deadline_at_ns=10,
            hard_stop_at_ns=40,
            workload={},
        ),
    )
    return EdfPlannerDriver(
        user_id="user",
        scenario=Scenario("simulation_3h", 1, 1, online_jobs, batch_jobs),
        job_store=object(),
        resource_map_store=object(),
        model_catalog_store=object(),
        plan_store=object(),
        backend="vllm",
        config_policy=object(),
    )


def _snapshot(job_ids: tuple[str, ...], free: int) -> StoreSnapshot:
    return StoreSnapshot(
        resources={
            "reserved|aws|us-east|use1-az1|A10G": {
                "gpu_type": "A10G",
                "total": free,
                "free": free,
                "pools": [],
            }
        },
        running_jobs=(),
        waiting_jobs=tuple(
            {
                "job_id": job_id,
                "spec_json": {"model_id": "model"},
            }
            for job_id in job_ids
        ),
        paused_jobs=(),
        finished_jobs=(),
        model_catalogs={},
    )


def _select_one_gpu(
    _required_vram_gib: float,
    resources: dict[str, dict[str, object]],
    **_kwargs: object,
) -> tuple[ResourceTarget, int, int, int] | None:
    environment_key, environment = next(iter(resources.items()))
    if environment["free"] == 0:
        return None
    return (
        ResourceTarget(
            environment_key=environment_key,
            environment=environment,
            pool=None,
            gpu_type="A10G",
            instance_type=None,
        ),
        1,
        1,
        1,
    )


def _place_action(job_id: str, *_args: object) -> dict[str, object]:
    return {"job_id": job_id, "type": "place"}


if __name__ == "__main__":
    unittest.main()
