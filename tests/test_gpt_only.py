"""Pure GPT placement method checks."""

import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from data_generation.controller import build_plan_action, load_benchmark, scheduler_job
from data_generation.methods.gpt_only import (
    PLACEMENT_RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    GptPlacement,
    GptRequest,
    OpenAIResponsesClient,
    load_gpt_only_method,
)
from data_generation.scheduling import (
    DecisionFailureCategory,
    DecisionStatus,
    OptimizationGoal,
    SchedulerJob,
    SchedulerWorkload,
)
from data_generation.tandemn_store import ClusterStateSnapshot

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIRECTORY = ROOT / "benchmarks" / "simulation_3h"
METHOD_SETTINGS: dict[str, object] = {
    "model": "gpt-5.6-luna",
    "max_tokens": 4096,
    "backend": "vllm",
    "default_optimization_goal": "throughput",
}
ENVIRONMENT = "reserved|aws|us-east|use1-az1|H100-SXM-80GB"
INSTANCE_TYPE = "h100-node"


class GptOnlyMethodTests(unittest.TestCase):
    """Check GPT prompting, decisions, and capacity mutation."""

    def test_loader_does_not_read_benchmark_resource_yaml(self) -> None:
        method = load_gpt_only_method(
            Path("/benchmark-with-no-resource-yaml"),
            METHOD_SETTINGS,
            client=FakeGptClient(accepted_placement()),
        )

        decision = method.schedule(
            scheduler_test_job("store-only-job"),
            cluster_state=store_state(),
        )

        self.assertEqual(decision.status, DecisionStatus.ACCEPTED)

    def test_accepted_placement_receives_job_and_live_resource_map(self) -> None:
        client = FakeGptClient(accepted_placement())
        method = load_gpt_only_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=client,
        )

        decision = method.schedule(
            scheduler_test_job("gpt-job"),
            cluster_state=store_state(),
        )

        self.assertEqual(decision.status, DecisionStatus.ACCEPTED)
        self.assertEqual(decision.method, "gpt_only")
        self.assertEqual(decision.pool_id, ENVIRONMENT)
        self.assertEqual(decision.accelerator, "H100-SXM-80GB")
        self.assertEqual(decision.devices, 4)
        self.assertIsNone(decision.reason)
        self.assertIsNone(decision.ttft_ms)
        self.assertIsNone(decision.throughput_tokens_per_second)
        self.assertEqual(len(decision.candidates), 1)

        request = client.requests[0]
        self.assertEqual(request.model, "gpt-5.6-luna")
        self.assertEqual(request.system_prompt, SYSTEM_PROMPT)
        prompt = json.loads(request.user_prompt)
        self.assertEqual(prompt["jobs_for_tick"][0]["job_id"], "gpt-job")
        self.assertEqual(
            prompt["jobs_for_tick"][0]["workload"]["request_rate"],
            37.081,
        )
        self.assertEqual(
            prompt["jobs_for_tick"][0]["optimization_goal"],
            "latency",
        )
        self.assertEqual(
            prompt["benchmark_jobs_seen"][0]["job_id"],
            "gpt-job",
        )
        self.assertEqual(prompt["tandemn_state"]["resource_map"]["version"], 3)
        self.assertIsNone(prompt["past_plan"])
        self.assertNotIn("benchmark_resource_map", prompt)
        live_h100 = prompt["tandemn_state"]["resource_map"]["environments"][ENVIRONMENT]
        self.assertEqual(live_h100["free"], 40)

        allocated_h100 = method.resource_snapshot()["environments"][ENVIRONMENT]
        self.assertEqual(allocated_h100["free"], 36)
        method.release("gpt-job")
        released_h100 = method.resource_snapshot()["environments"][ENVIRONMENT]
        self.assertEqual(released_h100["free"], 36)

    def test_next_request_observes_previous_allocation(self) -> None:
        client = FakeGptClient(accepted_placement())
        method = load_gpt_only_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=client,
        )

        method.schedule(
            scheduler_test_job("first-job"),
            cluster_state=store_state(),
        )
        past_plan = {"actions": [past_place_action("first-job")]}
        method.schedule(
            scheduler_test_job("second-job"),
            past_plan,
            cluster_state=store_state(),
        )

        second_prompt = json.loads(client.requests[1].user_prompt)
        live_h100 = second_prompt["tandemn_state"]["resource_map"]["environments"][
            ENVIRONMENT
        ]
        self.assertEqual(live_h100["free"], 36)
        self.assertEqual(second_prompt["past_plan"], {"actions": past_plan["actions"]})

    def test_tick_sends_all_jobs_in_one_openai_request(self) -> None:
        client = FakeGptClient(accepted_placement())
        method = load_gpt_only_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=client,
        )
        jobs = (
            scheduler_test_job("first-job"),
            scheduler_test_job("second-job"),
        )

        decisions = method.schedule_tick(jobs, cluster_state=store_state())

        self.assertEqual(len(client.requests), 1)
        prompt = json.loads(client.requests[0].user_prompt)
        self.assertEqual(
            [job["job_id"] for job in prompt["jobs_for_tick"]],
            ["first-job", "second-job"],
        )
        self.assertEqual(set(decisions), {"first-job", "second-job"})
        self.assertTrue(
            all(
                decision.status == DecisionStatus.ACCEPTED
                for decision in decisions.values()
            )
        )

    def test_tick_with_no_unresolved_jobs_skips_openai(self) -> None:
        client = FakeGptClient(accepted_placement())
        method = load_gpt_only_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=client,
        )
        running_job = scheduler_test_job("running-job")
        cluster = store_state(
            running_jobs=(
                {
                    "job_id": "running-job",
                    "status": "running",
                    "active_chains": [],
                },
            )
        )

        decisions = method.schedule_tick(
            (),
            cluster_state=cluster,
            seen_jobs=(running_job,),
        )

        self.assertEqual(decisions, {})
        self.assertEqual(client.requests, [])

    def test_placement_uses_exact_store_environment(self) -> None:
        environment = "on_demand|gcp|apac|apac-a|H100-SXM-80GB"
        placement = accepted_placement(environment=environment)
        method = load_gpt_only_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=FakeGptClient(placement),
        )

        decision = method.schedule(
            scheduler_test_job("coordinate-job"),
            cluster_state=store_state(resources=store_resources(environment)),
        )

        self.assertEqual(decision.region, "apac")
        self.assertEqual(decision.market, "on_demand")
        self.assertEqual(decision.pool_id, environment)

    def test_defer_placement_uses_shared_failure_contract(self) -> None:
        client = FakeGptClient(deferred_placement())
        method = load_gpt_only_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=client,
        )

        decision = method.schedule(
            scheduler_test_job("rejected-job"),
            cluster_state=store_state(),
        )

        self.assertEqual(decision.status, DecisionStatus.REJECTED)
        self.assertEqual(
            decision.failure_category,
            DecisionFailureCategory.LOAD_INFEASIBLE,
        )
        self.assertEqual(decision.reason, "GPT deferred the job")
        self.assertEqual(
            decision.resource_snapshot_before,
            decision.resource_snapshot_after,
        )

    def test_invalid_capacity_is_rejected_without_reservation(self) -> None:
        placement = accepted_placement(replicas=8, devices_per_replica=8)
        method = load_gpt_only_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=FakeGptClient(placement),
        )

        decision = method.schedule(
            scheduler_test_job("oversized-job"),
            cluster_state=store_state(),
        )

        self.assertEqual(decision.status, DecisionStatus.REJECTED)
        self.assertEqual(
            decision.failure_category,
            DecisionFailureCategory.METHOD_ERROR,
        )
        self.assertIn("has 40 free GPUs, not 64", decision.reason or "")
        h100 = method.resource_snapshot()["environments"][ENVIRONMENT]
        self.assertEqual(h100["free"], 40)

    def test_common_plan_is_derived_from_job_resource_and_placement(self) -> None:
        benchmark = load_benchmark(BENCHMARK_DIRECTORY / "benchmark.yaml")
        job = benchmark.jobs[0]
        method = load_gpt_only_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=FakeGptClient(accepted_placement()),
        )

        action = build_plan_action(
            job,
            method.schedule(
                scheduler_job(job),
                cluster_state=store_state(),
            ),
        )
        ladder = action["ladder"][0]
        configuration = ladder["config"]

        self.assertEqual(action["job_id"], job.job_id)
        self.assertEqual(action["tenant_id"], "tenant_chat_free")
        self.assertEqual(action["target_p99_ttft_ms"], 700.0)
        self.assertEqual(action["target_p99_tpot_ms"], 55.0)
        self.assertIsNone(action["mechanism_id"])
        self.assertIsNone(action["budget_ref"])
        self.assertEqual(ladder["rank_id"], ENVIRONMENT)
        self.assertEqual(
            ladder["env"],
            ENVIRONMENT.split("|"),
        )
        self.assertEqual(
            configuration["model_id"],
            "meta-llama/Llama-3.2-1B-Instruct",
        )
        self.assertEqual(configuration["gpu_type"], "H100-SXM-80GB")
        self.assertEqual(configuration["gpu_count"], 4)
        self.assertEqual(configuration["count"], 1)
        self.assertEqual(configuration["engine_name"], "vllm")
        self.assertEqual(configuration["instance_type"], INSTANCE_TYPE)
        self.assertEqual(configuration["max_num_seq"], 64)
        self.assertEqual(configuration["max_num_batched_tokens"], 32768)
        self.assertEqual(configuration["gpu_mem_util"], 0.9)
        self.assertTrue(configuration["prefix_cache_enabled"])
        self.assertIsNone(configuration["ep"])
        self.assertIsNone(ladder["mechanism_id"])

    def test_response_schema_contains_only_placement_information(self) -> None:
        self.assertEqual(
            set(PLACEMENT_RESPONSE_SCHEMA["properties"]),
            {"placements"},
        )
        item_schema = PLACEMENT_RESPONSE_SCHEMA["properties"]["placements"]["items"]
        self.assertEqual(
            set(item_schema["properties"]),
            {
                "job_id",
                "placement_decision",
                "environment",
                "instance_type",
                "configuration",
            },
        )
        configuration_schema = item_schema["properties"]["configuration"]["anyOf"][0]
        self.assertEqual(
            set(configuration_schema["properties"]),
            {
                "replicas_needed",
                "tp",
                "pp",
                "dp",
                "max_num_seq",
                "max_num_batched_tokens",
                "gpu_mem_util",
                "prefix_cache_enabled",
            },
        )

    def test_openai_client_uses_responses_structured_output(self) -> None:
        calls: list[dict[str, object]] = []
        placement = accepted_placement(job_id="api-job")
        placement_document = {"placements": [asdict(placement)]}
        response = SimpleNamespace(
            status="completed",
            output_text=json.dumps(placement_document),
        )

        def create(**kwargs: object) -> object:
            calls.append(kwargs)
            return response

        sdk = SimpleNamespace(
            OpenAI=lambda: SimpleNamespace(
                responses=SimpleNamespace(create=create),
            )
        )
        request = GptRequest(
            model="gpt-5.6-luna",
            max_tokens=4096,
            system_prompt=SYSTEM_PROMPT,
            user_prompt="{}",
        )

        with mock.patch(
            "data_generation.methods.gpt_only.import_module",
            return_value=sdk,
        ):
            placement = OpenAIResponsesClient().evaluate(request)

        self.assertEqual(placement, (accepted_placement(job_id="api-job"),))
        self.assertEqual(calls[0]["model"], "gpt-5.6-luna")
        self.assertEqual(calls[0]["max_output_tokens"], 4096)
        self.assertEqual(calls[0]["input"][0]["content"], SYSTEM_PROMPT)
        text_config = calls[0]["text"]
        self.assertIsInstance(text_config, dict)
        self.assertEqual(text_config["format"]["type"], "json_schema")
        self.assertTrue(text_config["format"]["strict"])


class FakeGptClient:
    """Return one stable placement while recording each prompt."""

    def __init__(self, placement: GptPlacement) -> None:
        self.placement = placement
        self.requests: list[GptRequest] = []

    def evaluate(self, request: GptRequest) -> tuple[GptPlacement, ...]:
        self.requests.append(request)
        prompt = json.loads(request.user_prompt)
        return tuple(
            replace(self.placement, job_id=job["job_id"])
            for job in prompt["jobs_for_tick"]
        )


def accepted_placement(
    *,
    job_id: str = "",
    replicas: int = 1,
    devices_per_replica: int = 4,
    tensor_parallel: int | None = None,
    environment: str = ENVIRONMENT,
) -> GptPlacement:
    """Build one accepted GPT response."""
    return GptPlacement(
        job_id=job_id,
        placement_decision="place",
        environment=environment,
        instance_type=INSTANCE_TYPE,
        configuration={
            "replicas_needed": replicas,
            "tp": tensor_parallel or devices_per_replica,
            "pp": 1,
            "dp": 1,
            "max_num_seq": 64,
            "max_num_batched_tokens": 32768,
            "gpu_mem_util": 0.9,
            "prefix_cache_enabled": True,
        },
    )


def deferred_placement() -> GptPlacement:
    """Build one deferred GPT response."""
    return GptPlacement(
        job_id="",
        placement_decision="defer",
        environment=None,
        instance_type=None,
        configuration=None,
    )


def scheduler_test_job(job_id: str) -> SchedulerJob:
    """Build one scheduler-visible online job."""
    return SchedulerJob(
        job_id=job_id,
        kind="online",
        spec={
            "model_id": "meta-llama/Llama-3.3-70B-Instruct",
            "precision": "fp16",
            "job_features": {
                "target_p99_ttft_ms": 700,
                "target_p99_tpot_ms": 55,
            },
        },
        workload=SchedulerWorkload(
            input_tokens=900,
            output_tokens=300,
            request_rate=37.081,
            max_concurrency=12000,
        ),
        optimization_goal=OptimizationGoal.LATENCY,
    )


def store_state(
    *,
    running_jobs: tuple[dict[str, object], ...] = (),
    resources: dict[str, dict[str, object]] | None = None,
) -> ClusterStateSnapshot:
    """Return one empty but versioned Tandemn cluster snapshot."""
    return ClusterStateSnapshot(
        user_id="usr-test",
        resource_map_version=3,
        resource_map_updated_at="2026-08-07T18:00:00+00:00",
        resources=resources or store_resources(),
        running_jobs=running_jobs,
        waiting_jobs=(),
        paused_jobs=(),
        finished_jobs=(),
    )


def store_resources(
    environment: str = ENVIRONMENT,
) -> dict[str, dict[str, object]]:
    """Return one Store-owned environment with GPU-granular capacity."""
    market, cloud, region, zone, gpu_type = environment.split("|")
    return {
        environment: {
            "total": 40,
            "used": 0,
            "free": 40,
            "gpu_type": gpu_type,
            "market": market,
            "cloud": cloud,
            "region": region,
            "zone": zone,
            "pools": [
                {
                    "instance_type": INSTANCE_TYPE,
                    "allocation_kind": "gpu",
                    "total": 40,
                    "used": 0,
                    "free": 40,
                    "gpus_per_instance": 8,
                }
            ],
        }
    }


def past_place_action(job_id: str) -> dict[str, object]:
    """Return one pending four-GPU place action."""
    return {
        "job_id": job_id,
        "type": "place",
        "ladder": [
            {
                "env": ENVIRONMENT.split("|"),
                "n_replicas": 1,
                "config": {
                    "instance_type": INSTANCE_TYPE,
                    "gpu_count": 4,
                    "count": 1,
                },
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
