"""Dynamo+AIC placement and resource-accounting checks."""

import json
import sys
import tempfile
import types
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from data_generation.controller import (
    build_plan_action,
    load_benchmark,
    run_placement_only,
    scheduler_job,
)
from data_generation.methods.dynamo_aic import (
    AicEvaluationError,
    AicRequest,
    AicResult,
    DynamoAicMethod,
    NvidiaAicClient,
    load_dynamo_aic_method,
)
from data_generation.scheduling import (
    CandidateStatus,
    DecisionFailureCategory,
    DecisionStatus,
    OptimizationGoal,
    SchedulerJob,
    SchedulerWorkload,
    SchedulingDecision,
    load_resource_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIRECTORY = ROOT / "benchmarks" / "simulation_3h"
METHOD_SETTINGS: dict[str, object] = {
    "backend": "vllm",
    "backend_version": None,
    "backend_versions": {
        "h100-sxm-80gb": "0.24.0",
        "a100-sxm-80gb": "0.22.0",
        "l40s-48gb": "0.24.0",
    },
    "database_mode": "HYBRID",
    "serving_mode": "agg",
    "default_optimization_goal": "throughput",
}


class DynamoAicTests(unittest.TestCase):
    """Check AIC adaptation, candidate choice, and capacity mutation."""

    def test_three_hour_resource_map_has_all_six_hardware_types(self) -> None:
        inventory = load_resource_inventory(BENCHMARK_DIRECTORY / "resources.yaml")

        self.assertEqual(len(inventory.pools), 6)
        self.assertEqual(
            sum(pool.total_devices for pool in inventory.pools),
            200,
        )
        self.assertEqual(
            sum(pool.aic_system is not None for pool in inventory.pools),
            3,
        )
        self.assertEqual(
            sum(sum(pool.markets.values()) for pool in inventory.pools),
            200,
        )
        h100 = inventory.pools[0]
        self.assertEqual(h100.markets, {"reserved": 32, "on_demand": 8})
        self.assertEqual(
            h100.regions,
            {"us-east": 16, "eu-west": 8, "apac": 16},
        )

    def test_inventory_tracks_region_and_market_pressure_together(self) -> None:
        inventory = load_resource_inventory(BENCHMARK_DIRECTORY / "resources.yaml")

        inventory.reserve(
            "east-job",
            "h100-sxm-80gb",
            "us-east",
            "reserved",
            16,
        )
        inventory.reserve(
            "west-job",
            "h100-sxm-80gb",
            "eu-west",
            "on_demand",
            8,
        )

        h100_regions = {
            candidate.region: candidate.available_devices
            for candidate in inventory.regional_pools()
            if candidate.pool.pool_id == "h100-sxm-80gb"
        }
        self.assertEqual(
            h100_regions,
            {"us-east": 0, "eu-west": 0, "apac": 16},
        )
        h100 = inventory.snapshot()["pools"][0]
        self.assertEqual(h100["available_devices"], 16)
        self.assertEqual(
            h100["markets"]["reserved"]["available_devices"],
            16,
        )
        self.assertEqual(
            h100["markets"]["on_demand"]["available_devices"],
            0,
        )

        inventory.release("east-job")
        inventory.release("west-job")
        restored_h100 = inventory.snapshot()["pools"][0]
        self.assertEqual(restored_h100["available_devices"], 40)

    def test_three_hour_jobs_explicitly_define_policy_by_kind(self) -> None:
        benchmark = load_benchmark(BENCHMARK_DIRECTORY / "benchmark.yaml")

        for job in benchmark.jobs:
            expected = (
                OptimizationGoal.LATENCY
                if job.kind.value == "online"
                else OptimizationGoal.THROUGHPUT
            )
            self.assertEqual(
                job.optimization_goal,
                expected,
                job.job_id,
            )

    def test_throughput_goal_selects_best_feasible_aic_result(self) -> None:
        client = FakeAicClient()
        method = load_dynamo_aic_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=client,
        )

        decision = method.schedule(test_job("throughput-job"))

        self.assertEqual(decision.status, DecisionStatus.ACCEPTED)
        self.assertEqual(decision.pool_id, "h100-sxm-80gb")
        self.assertEqual(decision.devices, 8)
        self.assertIn(decision.region, {"us-east", "eu-west", "apac"})
        self.assertIn(decision.market, {"reserved", "on_demand"})
        self.assertEqual(len(decision.candidates), 16)
        self.assertEqual(len(client.requests), 9)
        self.assertEqual(
            sum(
                candidate.status == CandidateStatus.UNSUPPORTED
                for candidate in decision.candidates
            ),
            7,
        )
        h100 = decision.resource_snapshot_after["pools"][0]
        self.assertEqual(h100["available_devices"], 32)
        self.assertEqual(
            h100["regions"][decision.region]["available_devices"],
            h100["regions"][decision.region]["total_devices"] - 8,
        )
        self.assertEqual(
            h100["markets"][decision.market]["available_devices"],
            h100["markets"][decision.market]["total_devices"] - 8,
        )
        allocation = decision.resource_snapshot_after["allocations"]["throughput-job"]
        self.assertEqual(allocation["region"], decision.region)
        self.assertEqual(allocation["market"], decision.market)

        method.release("throughput-job")
        released_h100 = method.resource_snapshot()["pools"][0]
        self.assertEqual(released_h100["available_devices"], 40)
        self.assertEqual(
            released_h100["regions"][decision.region]["available_devices"],
            released_h100["regions"][decision.region]["total_devices"],
        )
        self.assertEqual(
            released_h100["markets"][decision.market]["available_devices"],
            released_h100["markets"][decision.market]["total_devices"],
        )

    def test_rejection_documents_insufficient_capacity(self) -> None:
        method = load_dynamo_aic_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=FakeAicClient(devices=64),
        )

        decision = method.schedule(test_job("oversized-job"))

        self.assertEqual(decision.status, DecisionStatus.REJECTED)
        self.assertEqual(
            decision.failure_category,
            DecisionFailureCategory.INSUFFICIENT_CAPACITY,
        )
        self.assertEqual(
            decision.to_dict()["failure_category"],
            "insufficient_capacity",
        )
        self.assertIn("requires 64 devices", decision.reason or "")

    def test_rejection_documents_unsupported_model(self) -> None:
        method = load_dynamo_aic_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=FailingAicClient(
                "The model's architecture Phi3ForCausalLM is not supported"
            ),
        )

        decision = method.schedule(test_job("unsupported-model-job"))

        self.assertEqual(
            decision.failure_category,
            DecisionFailureCategory.MODEL_UNSUPPORTED,
        )
        self.assertEqual(
            decision.reason,
            "AIC does not support model architecture Phi3ForCausalLM",
        )

    def test_empty_aic_result_is_load_infeasible(self) -> None:
        method = load_dynamo_aic_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=FailingAicClient("AIC returned no aggregate configuration"),
        )

        decision = method.schedule(test_job("infeasible-job"))

        self.assertEqual(
            decision.failure_category,
            DecisionFailureCategory.LOAD_INFEASIBLE,
        )
        self.assertEqual(
            decision.reason,
            "AIC found no configuration meeting the job load and SLO",
        )

    def test_first_job_uses_canonical_rate_and_token_shape(self) -> None:
        benchmark = load_benchmark(BENCHMARK_DIRECTORY / "benchmark.yaml")
        first_job = benchmark.jobs[0]
        client = FakeAicClient(devices=1)
        method = load_dynamo_aic_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=client,
        )

        method.schedule(scheduler_job(first_job))

        request = client.requests[0]
        self.assertEqual(request.target_request_rate, 37.081)
        self.assertIsNone(request.target_concurrency)
        self.assertEqual(request.input_tokens, 900)
        self.assertEqual(request.output_tokens, 300)
        self.assertEqual(request.available_devices, 16)
        self.assertEqual(
            [request.available_devices for request in client.requests[:3]],
            [16, 8, 16],
        )
        self.assertEqual(
            request.optimization_goal,
            OptimizationGoal.LATENCY,
        )

    def test_batch_job_uses_deadline_load_and_throughput_policy(self) -> None:
        benchmark = load_benchmark(BENCHMARK_DIRECTORY / "benchmark.yaml")
        batch_job = next(job for job in benchmark.jobs if job.kind.value == "batch")
        client = FakeAicClient(devices=1)
        method = load_dynamo_aic_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=client,
        )

        method.schedule(scheduler_job(batch_job))

        request = client.requests[0]
        self.assertEqual(request.input_tokens, 1800)
        self.assertEqual(request.output_tokens, 1200)
        self.assertAlmostEqual(request.target_request_rate or 0, 14400 / (2.25 * 3600))
        self.assertEqual(request.optimization_goal, OptimizationGoal.THROUGHPUT)

    def test_latency_goal_selects_lowest_aic_ttft(self) -> None:
        method = load_dynamo_aic_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=FakeAicClient(a100_ttft_ms=0.5),
        )

        decision = method.schedule(test_job("latency-job", OptimizationGoal.LATENCY))

        self.assertEqual(decision.pool_id, "a100-sxm-80gb")
        self.assertEqual(decision.latency_ms, 80)

    def test_nvidia_client_applies_each_job_policy(self) -> None:
        recommend_calls: list[dict[str, object]] = []
        picker_calls: list[dict[str, object]] = []
        frame = pd.DataFrame(
            [
                aic_row(ttft=10.0, tpot=20.0, throughput=100.0),
                aic_row(ttft=20.0, tpot=10.0, throughput=200.0),
                aic_row(
                    ttft=1.0,
                    tpot=1.0,
                    throughput=300.0,
                    total_gpus=64,
                ),
            ]
        )

        def recommend(**kwargs: object) -> SimpleNamespace:
            recommend_calls.append(kwargs)
            return SimpleNamespace(best_configs={"agg": frame})

        def pick(**kwargs: object) -> dict[str, pd.DataFrame]:
            picker_calls.append(kwargs)
            candidates = kwargs["pareto_df"]
            assert isinstance(candidates, pd.DataFrame)
            goal = kwargs["optimization_type"]
            column = "tpot" if goal == "latency" else "tokens/s/gpu"
            ascending = goal == "latency"
            return {
                "best_config_df": candidates.sort_values(
                    column,
                    ascending=ascending,
                ).head(1)
            }

        modules = fake_aic_modules(recommend, pick)
        with mock.patch.dict(sys.modules, modules):
            latency_result = NvidiaAicClient().evaluate(
                test_aic_request(OptimizationGoal.LATENCY)
            )
            throughput_result = NvidiaAicClient().evaluate(
                test_aic_request(OptimizationGoal.THROUGHPUT)
            )

        self.assertEqual(recommend_calls[0]["ttft"], 700)
        self.assertEqual(recommend_calls[0]["tpot"], 55)
        self.assertEqual(recommend_calls[0]["target_request_rate"], 37.081)
        self.assertEqual(picker_calls[0]["optimization_type"], "throughput")
        self.assertEqual(picker_calls[0]["total_gpus"], 32)
        picker_frame = picker_calls[0]["pareto_df"]
        self.assertIsInstance(picker_frame, pd.DataFrame)
        self.assertEqual(len(picker_frame), 2)
        self.assertEqual(latency_result.ttft_ms, 10)
        self.assertEqual(
            throughput_result.throughput_tokens_per_second,
            200,
        )

    def test_next_job_observes_the_updated_capacity(self) -> None:
        method = load_dynamo_aic_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=FakeAicClient(devices=8),
        )

        first = method.schedule(test_job("first-job"))
        second = method.schedule(test_job("second-job"))

        self.assertEqual(first.pool_id, "h100-sxm-80gb")
        self.assertEqual(second.pool_id, "h100-sxm-80gb")
        second_h100 = next(
            candidate
            for candidate in second.candidates
            if candidate.pool_id == "h100-sxm-80gb" and candidate.region == first.region
        )
        region_total = next(
            pool for pool in method.inventory.pools if pool.pool_id == "h100-sxm-80gb"
        ).regions[first.region or ""]
        self.assertEqual(second_h100.available_devices, region_total - 8)

    def test_placement_only_simulates_without_starting_data_senders(self) -> None:
        benchmark = load_benchmark(BENCHMARK_DIRECTORY / "benchmark.yaml")
        method = load_dynamo_aic_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=FakeAicClient(devices=1),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_root = Path(temporary_directory)

            with self.assertLogs(
                "data_generation.controller",
                level="INFO",
            ) as captured_logs:
                summary = run_placement_only(
                    benchmark,
                    artifact_root,
                    method,
                    run_id="placement-test",
                )

            run_directory = (
                artifact_root
                / "adhoc"
                / "simulation_3h"
                / "dynamo_aic"
                / "placement-test"
            )
            plans = json.loads(
                (run_directory / "placement.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(summary.decisions), 26)
            self.assertEqual(
                sorted(path.name for path in run_directory.iterdir()),
                ["placement.json"],
            )
            self.assertEqual(plans["schema_version"], "cluster-plans/v1")
            self.assertEqual(len(plans["ticks"]), 20)
            self.assertEqual(len(plans["ticks"][0]["plan"]["actions"]), 5)
            self.assertEqual(len(plans["ticks"][-1]["plan"]["actions"]), 26)
            final_h100 = method.resource_snapshot()["pools"][0]
            self.assertEqual(final_h100["available_devices"], 40)
            placement_logs = [line for line in captured_logs.output if "[plan]" in line]
            self.assertEqual(len(placement_logs), 20)
            self.assertIn("T+0m", placement_logs[0])
            self.assertIn('"type": "place"', placement_logs[0])
            self.assertIn('"tp": 1', placement_logs[0])
            self.assertIn('"target_p99_ttft_ms": 700.0', placement_logs[0])
            first_action = plans["ticks"][0]["plan"]["actions"][0]
            self.assertEqual(first_action["type"], "place")
            self.assertEqual(first_action["tenant_id"], "tenant_chat_free")
            self.assertEqual(first_action["target_tps"], 11124.3)
            self.assertEqual(first_action["target_p99_ttft_ms"], 700.0)
            self.assertEqual(first_action["ladder"][0]["config"]["gpu_count"], 1)
            second_tick_actions = plans["ticks"][1]["plan"]["actions"]
            self.assertEqual(
                [action["type"] for action in second_tick_actions[:5]],
                ["keep"] * 5,
            )
            self.assertEqual(
                [action["type"] for action in second_tick_actions[5:]],
                ["place"] * 3,
            )
            final_actions = plans["ticks"][-1]["plan"]["actions"]
            self.assertEqual(
                {action["type"] for action in final_actions},
                {"preempt"},
            )
            batch_action = next(
                action
                for action in final_actions
                if action["job_id"] == "job-benchmark_timing_3hr-15-1"
            )
            self.assertAlmostEqual(batch_action["target_tps"], 2133.333)
            first_decision = next(iter(summary.decisions.values()))
            self.assertEqual(first_decision.ttft_ms, 1.0)
            self.assertEqual(first_decision.tpot_ms, 0.5)
            batch_release_tick = next(
                tick for tick in plans["ticks"] if tick["offset_minutes"] == 150
            )
            batch_release_action = next(
                action
                for action in batch_release_tick["plan"]["actions"]
                if action["job_id"] == "job-benchmark_timing_3hr-15-1"
            )
            self.assertEqual(batch_release_action["type"], "preempt")

    def test_rejected_decision_becomes_a_deferred_plan_action(self) -> None:
        benchmark = load_benchmark(BENCHMARK_DIRECTORY / "benchmark.yaml")
        method = load_dynamo_aic_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=FakeAicClient(devices=64),
        )
        job = benchmark.jobs[0]

        action = build_plan_action(job, method.schedule(scheduler_job(job)))

        self.assertEqual(action["type"], "defer")
        self.assertEqual(action["ladder"], [])
        self.assertIn("Deferred (insufficient_capacity)", action["rationale"])

    def test_placement_only_checkpoints_each_completed_decision(self) -> None:
        benchmark = load_benchmark(BENCHMARK_DIRECTORY / "benchmark.yaml")
        delegate = load_dynamo_aic_method(
            BENCHMARK_DIRECTORY,
            METHOD_SETTINGS,
            client=FakeAicClient(devices=1),
        )
        method = FailingAfterOneMethod(delegate)
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_root = Path(temporary_directory)

            with self.assertRaisesRegex(RuntimeError, "test interruption"):
                run_placement_only(
                    benchmark,
                    artifact_root,
                    method,
                    run_id="interrupted-placement-test",
                )

            placement_path = (
                artifact_root
                / "adhoc"
                / "simulation_3h"
                / "failing_after_one"
                / "interrupted-placement-test"
                / "placement.json"
            )
            document = json.loads(placement_path.read_text(encoding="utf-8"))
            self.assertEqual(
                len(document["ticks"][0]["plan"]["actions"]),
                1,
            )


class FailingAfterOneMethod:
    """Interrupt placement after one successfully checkpointed decision."""

    method_name = "failing_after_one"

    def __init__(self, delegate: DynamoAicMethod) -> None:
        self.delegate = delegate
        self.schedule_calls = 0

    def schedule(
        self,
        job: SchedulerJob,
        past_plan: Mapping[str, object] | None = None,
    ) -> SchedulingDecision:
        if self.schedule_calls == 1:
            raise RuntimeError("test interruption")
        self.schedule_calls += 1
        return self.delegate.schedule(job, past_plan)

    def release(self, job_id: str) -> None:
        self.delegate.release(job_id)

    def resource_snapshot(self) -> dict[str, object]:
        return self.delegate.resource_snapshot()


class FakeAicClient:
    """Return stable per-system predictions while recording AIC inputs."""

    def __init__(
        self,
        *,
        devices: int = 8,
        h100_devices: int | None = None,
        a100_ttft_ms: float = 1,
        a100_tpot_ms: float = 0.5,
    ) -> None:
        self.devices = devices
        self.h100_devices = h100_devices or devices
        self.a100_ttft_ms = a100_ttft_ms
        self.a100_tpot_ms = a100_tpot_ms
        self.requests: list[AicRequest] = []

    def evaluate(self, request: AicRequest) -> AicResult:
        self.requests.append(request)
        metrics = {
            "h100_sxm": (100.0, 300.0, self.h100_devices, 1.0, 0.5),
            "a100_sxm": (
                80.0,
                250.0,
                self.devices,
                self.a100_ttft_ms,
                self.a100_tpot_ms,
            ),
            "l40s": (90.0, 200.0, self.devices, 1.0, 0.5),
        }
        latency, throughput, devices, ttft_ms, tpot_ms = metrics[request.system]
        return AicResult(
            devices=devices,
            replicas=1,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            request_latency_ms=latency,
            throughput_tokens_per_second=throughput,
            configuration={"system": request.system, "parallel": "tp1"},
        )


class FailingAicClient:
    """Fail every AIC evaluation with one stable reason."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def evaluate(self, request: AicRequest) -> AicResult:
        raise AicEvaluationError(self.reason)


def test_job(
    job_id: str,
    goal: OptimizationGoal | None = None,
) -> SchedulerJob:
    """Build one scheduler-visible online job."""
    return SchedulerJob(
        job_id=job_id,
        kind="online",
        workload=SchedulerWorkload(
            input_tokens=1685.737,
            output_tokens=107.797,
            request_rate=37.081,
            max_concurrency=12000,
        ),
        optimization_goal=goal,
        spec={
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
            "job_features": {
                "isl_token_avg": 900,
                "osl_token_avg": 300,
                "max_concurrent_streaming": 12000,
                "target_p99_ttft_ms": 700,
                "target_p99_tpot_ms": 55,
            },
        },
    )


def test_aic_request(goal: OptimizationGoal) -> AicRequest:
    """Build one complete request for the real AIC client boundary."""
    return AicRequest(
        model="test-model",
        system="h100_sxm",
        available_devices=32,
        backend="vllm",
        backend_version="0.24.0",
        database_mode="HYBRID",
        serving_mode="agg",
        input_tokens=1686,
        output_tokens=108,
        target_ttft_ms=700,
        target_tpot_ms=55,
        optimization_goal=goal,
        target_request_rate=37.081,
    )


def aic_row(
    *,
    ttft: float,
    tpot: float,
    throughput: float,
    total_gpus: int = 1,
) -> dict[str, object]:
    """Build one normalized AIC load-matched result row."""
    return {
        "replicas_needed": 1,
        "total_gpus_needed": total_gpus,
        "num_total_gpus": 1,
        "tokens/s": throughput,
        "tokens/s/gpu": throughput,
        "ttft": ttft,
        "tpot": tpot,
        "request_latency": 100.0,
        "parallel": "tp1pp1dp1",
    }


def fake_aic_modules(
    recommend: object,
    pick: object,
) -> dict[str, types.ModuleType]:
    """Expose the two AIC functions imported inside the production client."""
    package = types.ModuleType("aiconfigurator")
    cli_package = types.ModuleType("aiconfigurator.cli")
    cli_api = types.ModuleType("aiconfigurator.cli.api")
    sdk_package = types.ModuleType("aiconfigurator.sdk")
    picking = types.ModuleType("aiconfigurator.sdk.picking")
    cli_api.cli_recommend = recommend  # type: ignore[attr-defined]
    picking.pick_optimization_type = pick  # type: ignore[attr-defined]
    return {
        "aiconfigurator": package,
        "aiconfigurator.cli": cli_package,
        "aiconfigurator.cli.api": cli_api,
        "aiconfigurator.sdk": sdk_package,
        "aiconfigurator.sdk.picking": picking,
    }


if __name__ == "__main__":
    unittest.main()
