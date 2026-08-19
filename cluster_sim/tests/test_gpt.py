"""Test the Store-backed cluster simulator GPT adapter."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from gpt import (
    SYSTEM_PROMPT,
    GptPlacement,
    GptPlannerDriver,
    StoreSnapshot,
    _build_user_prompt,
    _resources_with_free_capacity,
)
from models import OnlineJob, Scenario
from policy import load_config_policy
from serving import build_rank_cluster
from tandemn_orca.orca import ladder_to_ranks
from tandemn_system_data.models import Plan


class GptPlannerDriverTest(unittest.TestCase):
    """Verify GPT decisions become complete canonical Store actions."""

    def test_builds_actions_without_terminal_jobs(self) -> None:
        """Combine GPT placement fields without acting on terminal jobs."""
        driver = _driver()
        snapshot = StoreSnapshot(
            resources={
                "reserved|aws|us-east|use1-az1|H100": {
                    "gpu_type": "H100",
                    "total": 8,
                    "free": 8,
                    "pools": [
                        {
                            "instance_type": "p5.48xlarge",
                            "gpus_per_instance": 8,
                            "total": 8,
                            "free": 8,
                        }
                    ],
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
            instance_type="p5.48xlarge",
            configuration={"replicas_needed": 1, "tp": 1, "pp": 1, "dp": 1},
        )

        actions = driver._build_actions(snapshot, (placement,))

        self.assertEqual(
            [(action["job_id"], action["type"]) for action in actions],
            [
                ("running", "keep"),
                ("waiting", "place"),
                ("paused", "defer"),
            ],
        )
        placed = actions[1]
        self.assertEqual(placed["target_p99_ttft_ms"], 700.0)
        self.assertEqual(
            placed["ladder"][0]["config"]["model_id"],
            "meta-llama/Llama-3.2-1B-Instruct",
        )
        self.assertEqual(placed["ladder"][0]["config"]["tp"], 1)
        self.assertEqual(placed["ladder"][0]["config"]["precision"], "bf16")
        self.assertEqual(placed["ladder"][0]["config"]["ep"], 1)
        plan = Plan(user_id="user", actions=actions)
        self.assertEqual(plan.actions[1].job_id, "waiting")
        ranks = ladder_to_ranks(
            plan.actions[1].ladder,
            job_id="waiting",
            plan_id=plan.plan_id,
            job_spec=_store_job()["spec_json"],
        )
        self.assertEqual(len(ranks), 1)
        self.assertEqual(ranks[0].rank_id, placed["ladder"][0]["rank_id"])
        cluster = build_rank_cluster(
            SimpleNamespace(
                model_name="meta-llama/Llama-3.2-1B-Instruct",
                shape=ranks[0].shape_json,
                chains=(object(),),
            ),
            {
                "profile": "H100",
                "memory_gb": 80,
                "memory_bandwidth_gb_s": 3350,
                "link_bandwidth_gb_s": 900,
            },
        )
        self.assertEqual(cluster["nodes"][0]["instances"][0]["ep_size"], 1)

    def test_prompt_contains_relevant_config_policy(self) -> None:
        """Give GPT the supported configurations for this tick."""
        driver = _driver()
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
            waiting_jobs=(_store_job(),),
            paused_jobs=(),
            finished_jobs=(),
        )

        prompt = _build_user_prompt(
            1,
            datetime(2026, 1, 1, tzinfo=UTC),
            snapshot,
            tuple(driver.jobs.values()),
            load_config_policy(),
        )

        document = json.loads(prompt)
        self.assertEqual(
            document["config_policy"],
            [
                {
                    "allowed_tp": [1, 2, 4, 8],
                    "gpu_type": "H100",
                    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                    "precision": "bf16",
                }
            ],
        )
        self.assertIn("global constraint across every placement", SYSTEM_PROMPT)
        self.assertIn("replica consumes one whole instance", SYSTEM_PROMPT)

    def test_defers_only_placement_that_exceeds_cumulative_capacity(self) -> None:
        """Keep an earlier placement and defer a later pool oversubscription."""
        driver = _driver(("first", "second"))
        environment = "reserved|aws|us-east|use1-az1|H100"
        snapshot = StoreSnapshot(
            resources={
                environment: {
                    "gpu_type": "H100",
                    "total": 8,
                    "free": 8,
                    "pools": [
                        {
                            "instance_type": "p5.48xlarge",
                            "gpus_per_instance": 8,
                            "total": 8,
                            "free": 8,
                        }
                    ],
                }
            },
            running_jobs=(),
            waiting_jobs=(_store_job("first"), _store_job("second")),
            paused_jobs=(),
            finished_jobs=(),
        )
        placements = tuple(
            GptPlacement(
                decision="place",
                environment=environment,
                instance_type="p5.48xlarge",
                configuration={"replicas_needed": 1, "tp": 1, "pp": 1, "dp": 1},
            )
            for _ in range(2)
        )

        actions = driver._build_actions(snapshot, placements)

        self.assertEqual(
            [(action["job_id"], action["type"]) for action in actions],
            [("first", "place"), ("second", "defer")],
        )

    def test_rejects_policy_invalid_tp(self) -> None:
        """Reject a positive TP value that the selected policy disallows."""
        driver = _driver()
        snapshot = StoreSnapshot(
            resources={
                "reserved|aws|us-east|use1-az1|H100": {
                    "gpu_type": "H100",
                    "total": 3,
                    "free": 3,
                    "pools": [],
                }
            },
            running_jobs=(),
            waiting_jobs=(_store_job(),),
            paused_jobs=(),
            finished_jobs=(),
        )
        placement = GptPlacement(
            decision="place",
            environment="reserved|aws|us-east|use1-az1|H100",
            instance_type=None,
            configuration={"replicas_needed": 1, "tp": 3, "pp": 1, "dp": 1},
        )

        with self.assertRaisesRegex(ValueError, "tp=3 is not allowed"):
            driver._build_actions(snapshot, (placement,))

    def test_gpt_ladder_uses_per_replica_gpu_count(self) -> None:
        """Serialize GPT DP width as canonical Koi/Orca replicas."""
        driver = _driver()
        environment = "reserved|aws|us-east|use1-az1|H100"
        snapshot = StoreSnapshot(
            resources={
                environment: {
                    "gpu_type": "H100",
                    "total": 24,
                    "free": 24,
                    "pools": [
                        {
                            "allocation_kind": "gpu",
                            "instance_type": "p5.48xlarge",
                            "gpus_per_instance": 8,
                            "total": 24,
                            "free": 24,
                        }
                    ],
                }
            },
            running_jobs=(),
            waiting_jobs=(_store_job(),),
            paused_jobs=(),
            finished_jobs=(),
        )
        placement = GptPlacement(
            decision="place",
            environment=environment,
            instance_type="p5.48xlarge",
            configuration={"replicas_needed": 2, "tp": 2, "pp": 1, "dp": 3},
        )

        ladder = driver._build_actions(snapshot, (placement,))[0]["ladder"]
        rank = ladder_to_ranks(ladder, job_id="waiting", plan_id="plan-1")[0]

        self.assertEqual(ladder[0]["config"]["count"], 2)
        self.assertEqual(ladder[0]["config"]["gpu_count"], 2)
        self.assertEqual(ladder[0]["n_replicas"], 6)
        self.assertEqual(rank.shape_json["count"], 2)
        self.assertEqual(rank.n_replicas, 6)

    def test_active_replicas_reduce_gpt_gpu_pool_capacity(self) -> None:
        """Subtract every active Koi/Orca replica from GPT's free GPU pool."""
        environment = "reserved|aws|us-east|use1-az1|H100"
        resource_map = SimpleNamespace(
            market="reserved",
            scheduling_summary=lambda: {
                environment: {
                    "total": 24,
                    "pools": [
                        {
                            "allocation_kind": "gpu",
                            "instance_type": "p5.48xlarge",
                            "total": 24,
                        }
                    ],
                }
            },
        )
        running = {
            "ranks": [
                {
                    "n_replicas": 3,
                    "shape_json": {
                        "env": environment.split("|"),
                        "count": 2,
                        "instance_type": "p5.48xlarge",
                    },
                }
            ]
        }

        resources = _resources_with_free_capacity(resource_map, (running,))

        self.assertEqual(resources[environment]["free"], 18)
        self.assertEqual(resources[environment]["pools"][0]["free"], 18)

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


def _driver(job_ids: tuple[str, ...] = ("waiting",)) -> GptPlannerDriver:
    """Build a GPT driver without external Store or OpenAI clients."""
    jobs = tuple(
        OnlineJob(
            name=job_id,
            store_job_id=job_id,
            model_name="meta-llama/Llama-3.2-1B-Instruct",
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
        for job_id in job_ids
    )
    scenario = Scenario("test", 1, 1, jobs, ())
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


def _store_job(job_id: str = "waiting") -> dict[str, object]:
    """Create one canonical waiting Store job payload."""
    return {
        "job_id": job_id,
        "user_id": "tenant",
        "kind": "online",
        "spec_json": {
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
            "job_features": {
                "target_p99_ttft_ms": 700,
                "target_p99_tpot_ms": 55,
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
