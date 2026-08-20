"""Test the Store-backed cluster simulator GPT adapter."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from gpt import (
    PLACEMENT_SCHEMA,
    SYSTEM_PROMPT,
    GptPlacement,
    GptPlannerDriver,
    OpenAIResponsesClient,
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

    def test_client_returns_provider_token_usage(self) -> None:
        """Keep exact Responses API usage with its simulation tick."""
        response = SimpleNamespace(
            status="completed",
            output_text=json.dumps(
                {
                    "placements": [
                        {
                            "decision": "defer",
                            "environment": None,
                            "instance_type": None,
                            "configuration": None,
                        }
                    ]
                }
            ),
            model="gpt-used",
            usage=SimpleNamespace(input_tokens=90, output_tokens=10),
        )
        sdk = SimpleNamespace(
            OpenAI=lambda: SimpleNamespace(
                responses=SimpleNamespace(create=lambda **_kwargs: response)
            )
        )

        with patch("gpt.import_module", return_value=sdk):
            placements, usage = OpenAIResponsesClient().place(
                2,
                "gpt-requested",
                100,
                "system",
                "prompt",
            )

        self.assertEqual(placements[0].decision, "defer")
        self.assertEqual(usage[0].tick_number, 2)
        self.assertEqual(usage[0].input_tokens, 90)
        self.assertEqual(usage[0].output_tokens, 10)
        self.assertEqual(usage[0].model, "gpt-used")

    def test_schema_requires_every_catalog_field(self) -> None:
        """Keep the OpenAI strict schema aligned with the rank launch config."""
        placement = PLACEMENT_SCHEMA["properties"]["placements"]["items"]
        configuration = placement["properties"]["configuration"]["anyOf"][0]

        self.assertEqual(
            set(configuration["required"]),
            set(configuration["properties"]),
        )
        self.assertFalse(configuration["additionalProperties"])

    def test_snapshot_reads_each_relevant_model_catalog_once(self) -> None:
        """Use the existing planner Store connection for tick catalog context."""
        driver = _driver(("first", "second"))
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
            get=lambda: SimpleNamespace(market="reserved", scheduling_summary=dict)
        )
        driver.model_catalog_store = SimpleNamespace(
            get=lambda model_id: (
                requested.append(model_id)
                or SimpleNamespace(model_dump=lambda **_kwargs: _catalogs()[model_id])
            )
        )

        snapshot = driver._read_store_snapshot()

        self.assertEqual(
            requested,
            ["meta-llama/Llama-3.2-1B-Instruct"],
        )
        self.assertEqual(snapshot.model_catalogs, _catalogs())

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
            model_catalogs=_catalogs(),
        )
        placement = GptPlacement(
            decision="place",
            environment="reserved|aws|us-east|use1-az1|H100",
            instance_type="p5.48xlarge",
            configuration=_configuration(),
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
        self.assertEqual(placed["ladder"][0]["config"]["engine_version"], "0.16.0")
        self.assertEqual(placed["ladder"][0]["config"]["max_num_seq"], 256)
        self.assertEqual(placed["ladder"][0]["config"]["kvcache_dtype"], "auto")
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
            model_catalogs=_catalogs(),
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
        self.assertEqual(
            document["model_catalog"],
            [
                {
                    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                    "gpu_type": "H100",
                    **_launch_config(),
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
            model_catalogs=_catalogs(),
        )
        placements = tuple(
            GptPlacement(
                decision="place",
                environment=environment,
                instance_type="p5.48xlarge",
                configuration=_configuration(),
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
            model_catalogs=_catalogs(),
        )
        placement = GptPlacement(
            decision="place",
            environment="reserved|aws|us-east|use1-az1|H100",
            instance_type=None,
            configuration=_configuration(tp=3),
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
            model_catalogs=_catalogs(),
        )
        placement = GptPlacement(
            decision="place",
            environment=environment,
            instance_type="p5.48xlarge",
            configuration=_configuration(replicas_needed=2, tp=2, dp=3),
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
            model_catalogs=_catalogs(),
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
        model_catalog_store=object(),
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


def _launch_config() -> dict[str, object]:
    """Return one resolved H100 launch profile from Tandemn Store."""
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
    """Return one raw Store model catalog keyed by model ID."""
    catalog = _launch_config()
    for field in (
        "max_num_seq",
        "max_num_batched_tokens",
        "block_size",
        "kvcache_dtype",
    ):
        catalog[field] = [{"gpu_type": "H100", "value": catalog[field]}]
    return {"meta-llama/Llama-3.2-1B-Instruct": catalog}


def _configuration(
    *,
    replicas_needed: int = 1,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
) -> dict[str, object]:
    """Return GPT output that copies its supplied Store launch profile."""
    return {
        "replicas_needed": replicas_needed,
        "tp": tp,
        "pp": pp,
        "dp": dp,
        **_launch_config(),
    }


if __name__ == "__main__":
    unittest.main()
