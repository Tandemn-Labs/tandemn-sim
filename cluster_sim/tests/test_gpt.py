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
    _parse_placements,
    _resources_with_free_capacity,
)
from models import OnlineJob, Scenario
from policy import load_config_policy
from serving import build_rank_cluster
from simulation_orca import ladder_to_ranks
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
                            "lifecycle": "waiting",
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

    def test_parser_rejects_an_invalid_lifecycle_decision(self) -> None:
        """Do not accept a waiting action for a running lifecycle entry."""
        with self.assertRaisesRegex(
            ValueError,
            "running jobs cannot use place decisions",
        ):
            _parse_placements(
                {
                    "placements": [
                        {
                            "lifecycle": "running",
                            "decision": "place",
                            "environment": None,
                            "instance_type": None,
                            "configuration": None,
                        }
                    ]
                }
            )

    def test_schema_requires_every_catalog_field(self) -> None:
        """Keep the OpenAI strict schema aligned with the rank launch config."""
        variants = PLACEMENT_SCHEMA["properties"]["placements"]["items"][
            "anyOf"
        ]

        self.assertEqual(len(variants), 4)
        self.assertEqual(
            {
                (
                    variant["properties"]["lifecycle"]["enum"][0],
                    variant["properties"]["decision"]["enum"][0],
                )
                for variant in variants
            },
            {
                ("waiting", "place"),
                ("waiting", "defer"),
                ("running", "keep"),
                ("running", "swap"),
            },
        )
        for variant in variants:
            self.assertEqual(
                set(variant["required"]),
                set(variant["properties"]),
            )
            self.assertFalse(variant["additionalProperties"])
        for variant in (variants[0], variants[3]):
            configuration = variant["properties"]["configuration"]
            self.assertEqual(
                set(configuration["required"]),
                set(configuration["properties"]),
            )
            self.assertFalse(configuration["additionalProperties"])

    def test_snapshot_reads_active_candidate_model_catalog(self) -> None:
        """Load model context for a running scenario job without waiting jobs."""
        driver = _driver(("first",))
        requested = []
        driver.job_store = SimpleNamespace(
            running_jobs=lambda _user_id: (_running_job_aggregate("first"),),
            waiting_jobs=lambda _user_id: (),
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
            lifecycle="waiting",
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
            ],
        )
        placed = actions[1]
        self.assertEqual(placed["target_p99_ttft_ms"], 700.0)
        self.assertEqual(
            placed["ladder"][0]["config"]["model_id"],
            "meta-llama/Llama-3.2-1B-Instruct",
        )
        self.assertEqual(placed["ladder"][0]["config"]["tp"], 1)
        self.assertEqual(placed["ladder"][0]["config"]["precision"], "fp16")
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
            tuple(("waiting", job) for job in driver.jobs.values()),
            load_config_policy(),
        )

        document = json.loads(prompt)
        self.assertEqual(
            document["config_policy"],
            [
                {
                    "allowed_pp": [1, 2, 3, 4],
                    "allowed_tp": [1, 2, 4, 8],
                    "gpu_type": "H100",
                    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                    "precision": "fp16",
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
        self.assertIn("PP must never exceed 4", SYSTEM_PROMPT)
        self.assertEqual(document["jobs_for_tick"][0]["lifecycle"], "waiting")
        pp_schema = PLACEMENT_SCHEMA["properties"]["placements"]["items"][
            "anyOf"
        ][0]["properties"]["configuration"]["properties"]["pp"]
        self.assertEqual(pp_schema["maximum"], 4)

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
                lifecycle="waiting",
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
            lifecycle="waiting",
            decision="place",
            environment="reserved|aws|us-east|use1-az1|H100",
            instance_type=None,
            configuration=_configuration(tp=3),
        )

        with self.assertRaisesRegex(ValueError, "tp=3 is not allowed"):
            driver._build_actions(snapshot, (placement,))

    def test_gpt_ladder_uses_dp_as_canonical_replica_count(self) -> None:
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
            lifecycle="waiting",
            decision="place",
            environment=environment,
            instance_type="p5.48xlarge",
            configuration=_configuration(tp=2, dp=3),
        )

        ladder = driver._build_actions(snapshot, (placement,))[0]["ladder"]
        rank = ladder_to_ranks(ladder, job_id="waiting", plan_id="plan-1")[0]

        self.assertEqual(ladder[0]["config"]["count"], 2)
        self.assertEqual(ladder[0]["config"]["gpu_count"], 2)
        self.assertEqual(ladder[0]["n_replicas"], 3)
        self.assertEqual(rank.shape_json["count"], 2)
        self.assertEqual(rank.n_replicas, 3)

    def test_swap_retains_the_active_rank_id(self) -> None:
        """Build a full replacement ladder for a changed running job."""
        driver = _driver(("running",))
        environment = "reserved|aws|us-east|use1-az1|H100"
        snapshot = StoreSnapshot(
            resources={
                environment: {
                    "gpu_type": "H100",
                    "total": 16,
                    "free": 8,
                    "pools": [
                        {
                            "allocation_kind": "gpu",
                            "instance_type": "p5.48xlarge",
                            "gpus_per_instance": 8,
                            "total": 16,
                            "free": 8,
                        }
                    ],
                }
            },
            running_jobs=(_running_job("running"),),
            waiting_jobs=(),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=_catalogs(),
        )
        placement = GptPlacement(
            lifecycle="running",
            decision="swap",
            environment=environment,
            instance_type="p5.48xlarge",
            configuration=_configuration(tp=2),
        )

        action = driver._build_actions(snapshot, (placement,))[0]

        self.assertEqual(action["type"], "swap")
        self.assertEqual(len(action["ladder"]), 1)
        self.assertEqual(action["ladder"][0]["rank_id"], _ACTIVE_RANK_ID)
        self.assertEqual(action["ladder"][0]["config"]["tp"], 2)
        self.assertEqual(action["ladder"][0]["n_replicas"], 1)

    def test_running_keep_and_waiting_defer_follow_candidate_order(self) -> None:
        """Accept the lifecycle-specific no-op decisions in order."""
        driver = _driver(("running", "waiting"))
        snapshot = StoreSnapshot(
            resources={},
            running_jobs=(_running_job("running"),),
            waiting_jobs=(_store_job("waiting"),),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=_catalogs(),
        )
        placements = (
            GptPlacement("running", "keep", None, None, None),
            GptPlacement("waiting", "defer", None, None, None),
        )

        actions = driver._build_actions(snapshot, placements)

        self.assertEqual(
            [(action["job_id"], action["type"]) for action in actions],
            [("running", "keep"), ("waiting", "defer")],
        )

    def test_noop_swap_is_normalized_to_keep_without_free_capacity(self) -> None:
        """Avoid submitting a replacement that preserves the active shape."""
        driver = _driver(("running",))
        environment = "reserved|aws|us-east|use1-az1|H100"
        snapshot = StoreSnapshot(
            resources={
                environment: {
                    "gpu_type": "H100",
                    "total": 8,
                    "free": 0,
                    "pools": [
                        {
                            "allocation_kind": "gpu",
                            "instance_type": "p5.48xlarge",
                            "gpus_per_instance": 8,
                            "total": 8,
                            "free": 0,
                        }
                    ],
                }
            },
            running_jobs=(_running_job("running"),),
            waiting_jobs=(),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=_catalogs(),
        )
        placement = GptPlacement(
            lifecycle="running",
            decision="swap",
            environment=environment,
            instance_type="p5.48xlarge",
            configuration=_configuration(),
        )

        actions = driver._build_actions(snapshot, (placement,))

        self.assertEqual(actions, [{"job_id": "running", "type": "keep"}])

    def test_rejects_a_lifecycle_mismatched_decision(self) -> None:
        """Do not apply a running decision to a waiting Store job."""
        driver = _driver()
        snapshot = StoreSnapshot(
            resources={},
            running_jobs=(),
            waiting_jobs=(_store_job(),),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=_catalogs(),
        )

        with self.assertRaisesRegex(ValueError, "lifecycle does not match"):
            driver._build_actions(
                snapshot,
                (GptPlacement("running", "keep", None, None, None),),
            )

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

        with self.assertRaisesRegex(ValueError, "decide every actionable job"):
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


_ACTIVE_RANK_ID = "rank_01ARZ3NDEKTSV4RRFFQ69G5FAV"


def _running_job(job_id: str = "running") -> dict[str, object]:
    """Create one flattened running Store job with its active rank."""
    environment = "reserved|aws|us-east|use1-az1|H100"
    return {
        **_store_job(job_id),
        "active_ranks": [
            {
                "rank_id": _ACTIVE_RANK_ID,
                "n_replicas": 1,
                "shape_json": {
                    "env": environment.split("|"),
                    "instance_type": "p5.48xlarge",
                    "gpu_type": "H100",
                    "gpu_count": 1,
                    "count": 1,
                    "tp": 1,
                    "pp": 1,
                    "dp": 1,
                },
            }
        ],
    }


def _running_job_aggregate(job_id: str) -> dict[str, object]:
    """Create the Store aggregate used while reading a running job snapshot."""
    return {
        "job": _store_job(job_id),
        "ranks": _running_job(job_id)["active_ranks"],
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
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
) -> dict[str, object]:
    """Return GPT output that copies its supplied Store launch profile."""
    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        **_launch_config(),
    }


if __name__ == "__main__":
    unittest.main()
