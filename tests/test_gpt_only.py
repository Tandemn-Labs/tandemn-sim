"""Pure GPT placement method checks."""

import json
import unittest
from dataclasses import asdict
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

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIRECTORY = ROOT / "benchmarks" / "simulation_3h"
METHOD_SETTINGS: dict[str, object] = {
    "model": "gpt-5.6-luna",
    "max_tokens": 4096,
    "backend": "vllm",
    "default_optimization_goal": "throughput",
}


class GptOnlyMethodTests(unittest.TestCase):
    """Check GPT prompting, decisions, and capacity mutation."""

    def test_accepted_placement_receives_job_and_live_resource_map(self) -> None:
        client = FakeGptClient(accepted_placement())
        method = load_gpt_only_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=client,
        )

        decision = method.schedule(test_job("gpt-job"))

        self.assertEqual(decision.status, DecisionStatus.ACCEPTED)
        self.assertEqual(decision.method, "gpt_only")
        self.assertEqual(decision.pool_id, "h100-sxm-80gb")
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
        self.assertEqual(prompt["job"]["job_id"], "gpt-job")
        self.assertEqual(prompt["job"]["workload"]["request_rate"], 37.081)
        self.assertEqual(prompt["job"]["optimization_goal"], "latency")
        self.assertEqual(len(prompt["resource_map"]["hardware"]), 7)
        h100 = prompt["resource_map"]["hardware"][0]
        self.assertEqual(h100["memory_gb_per_device"], 80)
        live_h100 = prompt["resource_map"]["live_capacity"]["pools"][0]
        self.assertEqual(live_h100["available_devices"], 32)

        allocated_h100 = method.resource_snapshot()["pools"][0]
        self.assertEqual(allocated_h100["available_devices"], 28)
        method.release("gpt-job")
        released_h100 = method.resource_snapshot()["pools"][0]
        self.assertEqual(released_h100["available_devices"], 32)

    def test_next_request_observes_previous_allocation(self) -> None:
        client = FakeGptClient(accepted_placement())
        method = load_gpt_only_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=client,
        )

        method.schedule(test_job("first-job"))
        method.schedule(test_job("second-job"))

        second_prompt = json.loads(client.requests[1].user_prompt)
        live_h100 = second_prompt["resource_map"]["live_capacity"]["pools"][0]
        self.assertEqual(live_h100["available_devices"], 28)

    def test_null_placement_uses_shared_failure_contract(self) -> None:
        client = FakeGptClient(None)
        method = load_gpt_only_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=client,
        )

        decision = method.schedule(test_job("rejected-job"))

        self.assertEqual(decision.status, DecisionStatus.REJECTED)
        self.assertEqual(
            decision.failure_category,
            DecisionFailureCategory.LOAD_INFEASIBLE,
        )
        self.assertEqual(decision.reason, "GPT returned no feasible placement")
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

        decision = method.schedule(test_job("oversized-job"))

        self.assertEqual(decision.status, DecisionStatus.REJECTED)
        self.assertEqual(
            decision.failure_category,
            DecisionFailureCategory.METHOD_ERROR,
        )
        self.assertIn("has 32 devices, not 64", decision.reason or "")
        h100 = method.resource_snapshot()["pools"][0]
        self.assertEqual(h100["available_devices"], 32)

    def test_invalid_expert_parallelism_is_rejected(self) -> None:
        placement = accepted_placement(tensor_parallel=2, expert_parallel=4)
        method = load_gpt_only_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=FakeGptClient(placement),
        )

        decision = method.schedule(test_job("inconsistent-job"))

        self.assertEqual(decision.status, DecisionStatus.REJECTED)
        self.assertIn("ep cannot exceed GPUs per replica", decision.reason or "")

    def test_common_plan_is_derived_from_job_resource_and_placement(self) -> None:
        benchmark = load_benchmark(BENCHMARK_DIRECTORY / "benchmark.yaml")
        job = benchmark.jobs[0]
        method = load_gpt_only_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=FakeGptClient(accepted_placement()),
        )

        action = build_plan_action(job, method.schedule(scheduler_job(job)))
        ladder = action["ladder"][0]
        configuration = ladder["config"]

        self.assertEqual(action["job_id"], job.job_id)
        self.assertEqual(action["tenant_id"], "tenant_chat_free")
        self.assertEqual(action["target_p99_ttft_ms"], 700.0)
        self.assertEqual(action["target_p99_tpot_ms"], 55.0)
        self.assertEqual(action["mechanism_id"], "gpt_only")
        self.assertEqual(ladder["rank_id"], "h100-sxm-80gb:us-east")
        self.assertEqual(
            ladder["env"],
            ["reserved", "us-east", "h100-sxm-80gb", "H100-SXM-80GB"],
        )
        self.assertEqual(
            configuration["model_id"],
            "meta-llama/Llama-3.2-1B-Instruct",
        )
        self.assertEqual(configuration["gpu_type"], "H100-SXM-80GB")
        self.assertEqual(configuration["gpu_count"], 4)
        self.assertEqual(configuration["count"], 1)
        self.assertEqual(configuration["engine_name"], "vllm")

    def test_response_schema_contains_only_placement_information(self) -> None:
        self.assertEqual(
            set(PLACEMENT_RESPONSE_SCHEMA["properties"]),
            {"placement"},
        )
        placement_schema = PLACEMENT_RESPONSE_SCHEMA["properties"]["placement"]
        accepted_schema = placement_schema["anyOf"][0]
        self.assertEqual(
            set(accepted_schema["properties"]),
            {"gpu_type", "configuration"},
        )
        configuration_schema = accepted_schema["properties"]["configuration"]
        self.assertEqual(
            set(configuration_schema["properties"]),
            {"replicas_needed", "tp", "pp", "dp", "ep"},
        )

    def test_openai_client_uses_responses_structured_output(self) -> None:
        calls: list[dict[str, object]] = []
        placement_document = {"placement": asdict(accepted_placement())}
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

        self.assertEqual(placement, accepted_placement())
        self.assertEqual(calls[0]["model"], "gpt-5.6-luna")
        self.assertEqual(calls[0]["max_output_tokens"], 4096)
        self.assertEqual(calls[0]["input"][0]["content"], SYSTEM_PROMPT)
        text_config = calls[0]["text"]
        self.assertIsInstance(text_config, dict)
        self.assertEqual(text_config["format"]["type"], "json_schema")
        self.assertTrue(text_config["format"]["strict"])


class FakeGptClient:
    """Return one stable placement while recording each prompt."""

    def __init__(self, placement: GptPlacement | None) -> None:
        self.placement = placement
        self.requests: list[GptRequest] = []

    def evaluate(self, request: GptRequest) -> GptPlacement | None:
        self.requests.append(request)
        return self.placement


def accepted_placement(
    *,
    replicas: int = 1,
    devices_per_replica: int = 4,
    tensor_parallel: int | None = None,
    expert_parallel: int = 1,
) -> GptPlacement:
    """Build one accepted GPT response."""
    return GptPlacement(
        gpu_type="H100-SXM-80GB",
        configuration={
            "replicas_needed": replicas,
            "tp": tensor_parallel or devices_per_replica,
            "pp": 1,
            "dp": 1,
            "ep": expert_parallel,
        },
    )


def test_job(job_id: str) -> SchedulerJob:
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


if __name__ == "__main__":
    unittest.main()
