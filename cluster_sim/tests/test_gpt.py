"""Test the Store-backed cluster simulator GPT adapter."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from gpt import GptPlacement, GptPlannerDriver, StoreSnapshot
from models import OnlineJob, Scenario


class GptPlannerDriverTest(unittest.TestCase):
    """Verify GPT decisions become complete canonical Store actions."""

    def test_builds_complete_lifecycle_actions(self) -> None:
        """Combine GPT placement fields with deterministic Store data."""
        driver = _driver()
        snapshot = StoreSnapshot(
            resources={
                "reserved|aws|us-east|use1-az1|H100": {
                    "gpu_type": "H100",
                    "total": 1,
                    "free": 1,
                    "pools": [],
                }
            },
            running_jobs=({"job_id": "running"},),
            waiting_jobs=(_store_job(),),
            paused_jobs=({"job_id": "paused"},),
            finished_jobs=({"job_id": "finished"},),
        )
        placement = GptPlacement(
            decision="place",
            environment="reserved|aws|us-east|use1-az1|H100",
            instance_type=None,
            configuration={"replicas_needed": 1, "tp": 1, "pp": 1, "dp": 1},
        )

        actions = driver._build_actions(snapshot, (placement,))

        self.assertEqual(
            [(action["job_id"], action["type"]) for action in actions],
            [
                ("running", "keep"),
                ("waiting", "place"),
                ("paused", "defer"),
                ("finished", "preempt"),
            ],
        )
        placed = actions[1]
        self.assertEqual(placed["tenant_id"], "tenant")
        self.assertEqual(placed["target_p99_ttft_ms"], 700.0)
        self.assertEqual(placed["ladder"][0]["config"]["model_id"], "model")
        self.assertEqual(placed["ladder"][0]["config"]["tp"], 1)
        self.assertIsNone(placed["ladder"][0]["config"]["ep"])

    def test_requires_one_decision_per_known_waiting_job(self) -> None:
        """Reject incomplete GPT output before creating partial actions."""
        driver = _driver()
        snapshot = StoreSnapshot(
            resources={},
            running_jobs=(),
            waiting_jobs=(_store_job(),),
            paused_jobs=(),
            finished_jobs=(),
        )

        with self.assertRaisesRegex(ValueError, "decide every waiting job"):
            driver._build_actions(snapshot, ())


def _driver() -> GptPlannerDriver:
    """Build a GPT driver without external Store or OpenAI clients."""
    job = OnlineJob(
        name="waiting",
        store_job_id="waiting",
        model_name="model",
        starts_at_ns=0,
        stops_at_ns=1,
        workload={
            "features": {
                "isl_token_avg": 100,
                "osl_token_avg": 20,
            },
            "phases": [{"rate": 1.0}],
        },
    )
    scenario = Scenario("test", 1, 1, (job,), ())
    return GptPlannerDriver(
        user_id="user",
        scenario=scenario,
        job_store=object(),
        resource_map_store=object(),
        plan_store=object(),
        model="gpt-test",
        max_tokens=100,
        backend="vllm",
        client=object(),
    )


def _store_job() -> dict[str, object]:
    """Create one canonical waiting Store job payload."""
    return {
        "job_id": "waiting",
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


if __name__ == "__main__":
    unittest.main()
