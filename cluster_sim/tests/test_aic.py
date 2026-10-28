"""Test the Store-backed cluster simulator AIC adapter."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from aic import (
    AicEvaluationError,
    AicPlannerDriver,
    AicRequest,
    AicResult,
    AicSettings,
    HardwareCandidate,
    NvidiaAicClient,
    ResourceTarget,
    StoreSnapshot,
    _aic_system,
    _evaluate_aic_request,
    _evaluate_job,
    _normalize_result,
    _parallelism,
    _policy_hardware_candidates,
    _runtime_parallelism,
)
from models import OnlineJob, Scenario
from serving import build_rank_cluster
from tandemn_system_data.ids import is_valid_id
from tandemn_system_data.models import Plan

_SERVINGSIM_HARDWARE = {
    "profile": "H100",
    "memory_gb": 80,
    "memory_bandwidth_gb_s": 3350,
    "link_bandwidth_gb_s": 900,
}


class AicPlannerDriverTest(unittest.TestCase):
    """Verify greedy AIC placement and canonical lifecycle actions."""

    def test_aic_uses_sol_then_empirical_once_after_primary_failure(self) -> None:
        """Use one SOL/EMPIRICAL fallback round after all primary calls fail."""
        client = FallbackAicClient()

        placement = _evaluate_job(
            _store_job({"job_id": "job_1"}),
            _snapshot(h100_free=8).resources,
            _driver_settings(),
            client,
            float("inf"),
        )

        self.assertIsNotNone(placement)
        self.assertEqual(
            [request.database_mode for request in client.requests],
            ["HYBRID", "SOL", "EMPIRICAL"],
        )

    def test_aic_process_timeout_terminates_the_worker(self) -> None:
        """Stop AIConfigurator without waiting for its worker threads."""
        request = AicRequest(
            model="model",
            system="h100_sxm",
            available_devices=8,
            backend="vllm",
            backend_version=None,
            database_mode="HYBRID",
            serving_mode="agg",
            input_tokens=1,
            output_tokens=1,
            target_ttft_ms=10,
            target_tpot_ms=10,
            optimization_goal="latency",
            target_request_rate=1,
            target_concurrency=None,
            parallelism_candidates=((1, 1),),
        )

        context = TimeoutContext()
        with patch("aic.multiprocessing.get_context", return_value=context):
            with self.assertRaisesRegex(AicEvaluationError, "timed out"):
                NvidiaAicClient().evaluate(request, timeout_seconds=0.01)
        self.assertTrue(context.process.terminated)

    def test_aic_process_returns_successful_result(self) -> None:
        """Return the child process result."""
        request = AicRequest(
            model="model",
            system="h100_sxm",
            available_devices=8,
            backend="vllm",
            backend_version=None,
            database_mode="HYBRID",
            serving_mode="agg",
            input_tokens=1,
            output_tokens=1,
            target_ttft_ms=10,
            target_tpot_ms=10,
            optimization_goal="latency",
            target_request_rate=1,
            target_concurrency=None,
            parallelism_candidates=((1, 1),),
        )
        result = AicResult(
            devices=1,
            replicas=1,
            ttft_ms=1,
            tpot_ms=1,
            request_latency_ms=1,
            throughput_tokens_per_second=1,
            configuration={},
        )

        with patch("aic.multiprocessing.get_context", return_value=SuccessContext()):
            with patch("aic._evaluate_aic_request", return_value=result):
                self.assertIs(NvidiaAicClient().evaluate(request), result)

    def test_filters_pp_above_limit_before_selecting_aic_result(self) -> None:
        """Select the best remaining result after removing PP above four."""
        frame = pd.DataFrame(
            [
                {
                    "parallel": "tp1pp5dp1",
                    "num_total_gpus": 5,
                    "seq/s": 500,
                    "tokens/s": 500,
                    "tokens/s/gpu": 100,
                    "tokens/s/user": 1000,
                    "ttft": 1,
                    "tpot": 1,
                    "bs": 1,
                },
                {
                    "parallel": "tp1pp4dp1",
                    "num_total_gpus": 4,
                    "seq/s": 400,
                    "tokens/s": 400,
                    "tokens/s/gpu": 100,
                    "tokens/s/user": 500,
                    "ttft": 2,
                    "tpot": 2,
                    "bs": 1,
                },
            ]
        )
        request = AicRequest(
            model="model",
            system="h100_sxm",
            available_devices=8,
            backend="vllm",
            backend_version=None,
            database_mode="HYBRID",
            serving_mode="agg",
            input_tokens=1,
            output_tokens=1,
            target_ttft_ms=10,
            target_tpot_ms=10,
            optimization_goal="latency",
            target_request_rate=1,
            target_concurrency=None,
            parallelism_candidates=((1, 4),),
        )

        with (
            patch(
                "aiconfigurator.sdk.utils.get_model_config_from_model_path",
                return_value={"layers": 32, "num_experts": 0},
            ),
            patch("aiconfigurator.sdk.task_v2.Task") as task,
        ):
            task.return_value.run.return_value = frame
            result = _evaluate_aic_request(request)

        self.assertEqual(result.configuration["parallel"], "tp1pp4dp1")
        self.assertEqual(task.call_args.kwargs["agg_pp_candidates"], [4])

    def test_moe_search_constrains_layout_before_evaluation(self) -> None:
        """Select a valid expert layout and size independent replicas from load."""
        request = _moe_request()
        supported = _moe_row()
        frame = pd.DataFrame(
            [
                {**supported, "moe_tp": 2, "moe_ep": 2, "ttft": 1},
                {**supported, "dp": 2, "moe_ep": 8, "ttft": 2},
                {**supported, "cp": 2, "ttft": 3},
                {**supported, "ttft": 10_000},
                supported,
            ]
        )

        with (
            patch(
                "aiconfigurator.sdk.utils.get_model_config_from_model_path",
                return_value={"layers": 32, "num_experts": 8},
            ),
            patch("aiconfigurator.sdk.task_v2.Task") as task,
        ):
            task.return_value.run.return_value = frame
            result = _evaluate_aic_request(request)

        kwargs = task.call_args.kwargs
        self.assertTrue(kwargs["enable_chunked_prefill"])
        self.assertEqual(kwargs["agg_moe_tp_candidates"], [1])
        self.assertEqual(kwargs["agg_dp_candidates"], [1])
        self.assertEqual(kwargs["agg_cp_candidates"], [1])
        self.assertEqual(kwargs["agg_tp_candidates"], [4])
        self.assertEqual(kwargs["agg_moe_ep_candidates"], [4])
        self.assertEqual(kwargs["agg_pp_candidates"], [1, 2])
        self.assertEqual(kwargs["agg_num_gpu_candidates"], [4, 8])
        self.assertEqual(result.configuration["moe_ep"], 4)
        self.assertEqual(result.configuration["dp"], 1)
        self.assertEqual(result.replicas, 2)
        self.assertEqual(result.devices, 8)
        self.assertEqual(result.throughput_tokens_per_second, 2000)
        self.assertEqual(result.num_experts, 8)

    def test_moe_concurrency_load_matching_uses_independent_replicas(self) -> None:
        """Keep concurrent-stream sizing distinct from request-rate sizing."""
        request = replace(
            _moe_request(),
            target_request_rate=None,
            target_concurrency=17,
            available_devices=16,
        )
        with (
            patch(
                "aiconfigurator.sdk.utils.get_model_config_from_model_path",
                return_value={"layers": 32, "num_experts": 8},
            ),
            patch("aiconfigurator.sdk.task_v2.Task") as task,
        ):
            task.return_value.run.return_value = pd.DataFrame([_moe_row()])
            result = _evaluate_aic_request(request)

        self.assertEqual(result.replicas, 3)
        self.assertEqual(result.devices, 12)

    def test_moe_excludes_nondivisible_expert_layouts_from_search(self) -> None:
        """Filter expert divisibility and layer count before creating an AIC task."""
        request = replace(
            _moe_request(),
            available_devices=32,
            parallelism_candidates=((4, 1), (4, 4), (6, 1)),
        )
        with (
            patch(
                "aiconfigurator.sdk.utils.get_model_config_from_model_path",
                return_value={"layers": 2, "num_experts": 8},
            ),
            patch("aiconfigurator.sdk.task_v2.Task") as task,
        ):
            task.return_value.run.return_value = pd.DataFrame([_moe_row()])
            _evaluate_aic_request(request)

        self.assertEqual(task.call_args.kwargs["agg_tp_candidates"], [4])
        self.assertEqual(task.call_args.kwargs["agg_pp_candidates"], [1])

    def test_compact_expert_tp_does_not_overwrite_attention_tp(self) -> None:
        """Parse compact AIC strings without mistaking etp for attention tp."""
        config = {"parallel": "tp4pp2dp1etp1ep4cp1"}
        self.assertEqual(_parallelism(config), (4, 2, 1))
        self.assertEqual(_runtime_parallelism(config, _moe_request(), 8), (4, 2, 1, 4))
        self.assertEqual(_parallelism({**config, "tp": 2}), (2, 2, 1))

    def test_normalization_rejects_unrepresentable_moe_layouts(self) -> None:
        """Never silently reinterpret expert TP, grouped DP, or missing EP."""
        target = ResourceTarget("h100", "H100", None, {"free": 16}, None)
        hardware = HardwareCandidate("h100_sxm", (target,), 16)
        invalid = (
            {**_moe_row(), "moe_tp": 2, "moe_ep": 2},
            {**_moe_row(), "dp": 2, "moe_ep": 8},
            {**_moe_row(), "moe_ep": 1},
            {**_moe_row(), "cp": 2},
            {**_moe_row(), "num_total_gpus": 8},
            {"tp": 4, "pp": 1, "dp": 1, "bs": 8},
        )
        for config in invalid:
            with self.subTest(config=config):
                result = AicResult(4, 1, 10, 5, 100, 1000, config, num_experts=8)
                with self.assertRaises(AicEvaluationError):
                    _normalize_result(result, _moe_request(), hardware)

    def test_moe_ep_survives_deployment_and_replica_accounting(self) -> None:
        """Launch the selected whole-expert layout on each independent replica."""
        client = FakeAicClient(tp=4, pp=2, replicas=2, num_experts=8)
        snapshot = _snapshot(waiting=({"job_id": "job_1"},), h100_free=16)
        snapshot.waiting_jobs[0]["spec_json"]["model_id"] = _moe_request().model
        snapshot.waiting_jobs[0]["spec_json"]["ep"] = 1

        actions = _driver((_job("job_1"),), client)._build_actions(snapshot)

        rank = actions[0]["ladder"][0]
        shape = rank["config"]
        self.assertEqual(shape["ep"], 4)
        self.assertTrue(shape["chunked_prefill_enable"])
        self.assertEqual(shape["tp"], 4)
        self.assertEqual(shape["pp"], 2)
        self.assertEqual(shape["count"], 8)
        self.assertEqual(shape["dp"], 2)
        self.assertEqual(rank["n_replicas"], 2)
        self.assertEqual(shape["model_id"], _moe_request().model)
        Plan(user_id="tenant", actions=actions)
        cluster = build_rank_cluster(
            SimpleNamespace(
                rank_name=rank["rank_id"],
                model_name=shape["model_id"],
                shape=shape,
                chains=(object(), object()),
            ),
            _SERVINGSIM_HARDWARE,
        )
        instances = cluster["nodes"][0]["instances"]
        self.assertEqual(len(instances), 2)
        self.assertTrue(all(instance["ep_size"] == 4 for instance in instances))
        self.assertEqual(sum(instance["num_npus"] for instance in instances), 16)

    def test_greedy_jobs_observe_updated_capacity(self) -> None:
        """Reserve the first placement before evaluating the second job."""
        jobs = (_job("job_1"), _job("job_2"))
        client = FakeAicClient()
        driver = _driver(jobs, client)

        actions = driver._build_actions(
            _snapshot(
                waiting=({"job_id": "job_1"}, {"job_id": "job_2"}),
                h100_free=2,
            )
        )

        self.assertEqual([action["type"] for action in actions], ["place", "place"])
        self.assertEqual(
            [request.available_devices for request in client.requests],
            [2, 1],
        )
        self.assertEqual(
            client.requests[0].model,
            "meta-llama/Llama-3.2-1B-Instruct",
        )
        self.assertEqual(client.requests[0].target_request_rate, 2.5)
        self.assertEqual(actions[0]["ladder"][0]["config"]["tp"], 1)

    def test_search_shapes_respect_per_instance_capacity(self) -> None:
        """A pool's free GPU total cannot enlarge an individual replica."""
        client = FakeAicClient()
        _driver((_job("job_1"),), client)._build_actions(
            _snapshot(
                waiting=({"job_id": "job_1"},),
                h100_free=32,
                h100_instance_pool=True,
            )
        )
        self.assertIn((4, 2), client.requests[0].parallelism_candidates)
        self.assertNotIn((8, 2), client.requests[0].parallelism_candidates)

    def test_unsupported_model_defers_without_stopping_tick(self) -> None:
        """Defer one unsupported model and continue evaluating later jobs."""
        jobs = (_job("unsupported"), _job("supported"))
        client = FakeAicClient(unsupported_model="unsupported-model")
        driver = _driver(jobs, client)

        actions = driver._build_actions(
            _snapshot(
                waiting=(
                    {"job_id": "unsupported"},
                    {"job_id": "supported"},
                ),
                h100_free=2,
            )
        )

        self.assertEqual([action["type"] for action in actions], ["defer", "place"])

    def test_online_concurrency_mode_uses_aic_concurrency(self) -> None:
        """Preserve Store's authoritative online concurrency intent."""
        job = _job("job_1")
        client = FakeAicClient()
        driver = _driver((job,), client)
        snapshot = _snapshot(waiting=({"job_id": "job_1"},), h100_free=2)
        features = snapshot.waiting_jobs[0]["spec_json"]["job_features"]
        features["_traffic_mode"] = "concurrency"
        features["max_concurrent_streaming"] = 42

        actions = driver._build_actions(snapshot)

        self.assertEqual(actions[0]["type"], "place")
        self.assertEqual(client.requests[0].optimization_goal, "latency")
        self.assertIsNone(client.requests[0].target_request_rate)
        self.assertEqual(client.requests[0].target_concurrency, 42)

    def test_max_num_seq_uses_aic_batch_size(self) -> None:
        """Use the AIC-selected batch size instead of a Store value."""
        client = FakeAicClient(batch_size=32)
        snapshot = _snapshot(waiting=({"job_id": "job_1"},), h100_free=2)
        snapshot.waiting_jobs[0]["spec_json"]["max_num_seq"] = 999

        actions = _driver((_job("job_1"),), client)._build_actions(snapshot)

        self.assertEqual(actions[0]["ladder"][0]["config"]["max_num_seq"], 32)

    def test_runtime_launch_settings_come_from_target_gpu_catalog(self) -> None:
        """Use H100 catalog token/chunk settings when Store job has no override."""
        client = FakeAicClient(batch_size=512)
        snapshot = _snapshot(waiting=({"job_id": "job_1"},), h100_free=8)
        snapshot.waiting_jobs[0]["_model_catalog"] = {
            "chunked_prefill_enable": True,
            "gpu_mem_util": 0.9,
            "prefix_cache_enabled": False,
            "max_num_seq": [{"gpu_type": "H100", "value": 1024}],
            "max_num_batched_tokens": [{"gpu_type": "H100", "value": 8192}],
            "block_size": [{"gpu_type": "H100", "value": 32}],
            "kvcache_dtype": [{"gpu_type": "H100", "value": "fp8"}],
        }

        action = _driver((_job("job_1"),), client)._build_actions(snapshot)[0]
        config = action["ladder"][0]["config"]

        self.assertEqual(config["max_num_seq"], 512)
        self.assertEqual(config["max_num_batched_tokens"], 8192)
        self.assertTrue(config["chunked_prefill_enable"])
        self.assertEqual(config["block_size"], 32)
        self.assertEqual(config["kvcache_dtype"], "fp8")

    def test_explicit_job_chunked_prefill_override_wins_over_catalog(self) -> None:
        """Preserve an explicit Store job runtime override."""
        snapshot = _snapshot(waiting=({"job_id": "job_1"},), h100_free=8)
        waiting = snapshot.waiting_jobs[0]
        waiting["_traffic_mode"] = "request_rate"
        waiting["spec_json"]["job_features"]["chunked_prefill_enable"] = False
        waiting["_model_catalog"] = {"chunked_prefill_enable": True}

        action = _driver((_job("job_1"),), FakeAicClient())._build_actions(snapshot)[0]

        self.assertFalse(action["ladder"][0]["config"]["chunked_prefill_enable"])

    def test_unavailable_aic_parallelism_defers(self) -> None:
        """Reject an exact TP/PP/DP footprint that Store cannot provide."""
        job = _job("job_1")
        client = FakeAicClient(tp=4)
        driver = _driver((job,), client)

        actions = driver._build_actions(
            _snapshot(waiting=({"job_id": "job_1"},), h100_free=2)
        )

        self.assertEqual(actions, [{"job_id": "job_1", "type": "defer"}])

    def test_missing_store_workload_does_not_fall_back_to_scenario(self) -> None:
        """Defer when Store lacks scheduler inputs available in Scenario."""
        job = _job("job_1")
        client = FakeAicClient()
        driver = _driver((job,), client)
        base = _snapshot(h100_free=2)
        snapshot = StoreSnapshot(
            resources=base.resources,
            running_jobs=(),
            waiting_jobs=(
                {
                    "job_id": "job_1",
                    "kind": "online",
                    "spec_json": {
                        "model_id": "store-model-job_1",
                        "job_features": {
                            "target_p99_ttft_ms": 700,
                            "target_p99_tpot_ms": 55,
                        },
                    },
                },
            ),
            paused_jobs=(),
            finished_jobs=(),
        )

        actions = driver._build_actions(snapshot)

        self.assertEqual(actions, [{"job_id": "job_1", "type": "defer"}])
        self.assertEqual(client.requests, [])

    def test_best_hardware_excludes_terminal_jobs(self) -> None:
        """Pick hardware without producing an action for a terminal job."""
        job = _job("waiting")
        client = FakeAicClient(throughput={"h100_sxm": 300, "a100_sxm": 200})
        driver = _driver((job,), client)
        snapshot = _snapshot(
            running=({"job_id": "running"},),
            waiting=(
                {
                    "job_id": "waiting",
                    "user_id": "tenant",
                    "kind": "batch",
                },
            ),
            paused=({"job_id": "paused"},),
            finished=({"job_id": "finished"},),
            h100_free=2,
            a100_free=2,
        )

        actions = driver._build_actions(snapshot)

        self.assertEqual(
            [(action["job_id"], action["type"]) for action in actions],
            [
                ("running", "keep"),
                ("waiting", "place"),
            ],
        )
        placed = actions[1]
        self.assertEqual(placed["ladder"][0]["config"]["gpu_type"], "H100")
        self.assertTrue(
            all(
                request.optimization_goal == "throughput" for request in client.requests
            )
        )
        self.assertIsNone(placed["ladder"][0]["mechanism_id"])
        self.assertTrue(is_valid_id(placed["ladder"][0]["rank_id"], "rank"))
        self.assertEqual(placed["ladder"][0]["config"]["count"], 1)
        self.assertEqual(placed["ladder"][0]["config"]["ep"], 1)
        self.assertEqual(placed["ladder"][0]["config"]["precision"], "fp16")
        Plan(user_id="tenant", actions=[placed])

    def test_azure_a100_80gb_is_an_aic_candidate(self) -> None:
        """Map Store's cloud-neutral A100 80 GB label to AIC's A100 system."""
        client = FakeAicClient(throughput={"a100_sxm": 200})
        snapshot = StoreSnapshot(
            resources={
                "reserved|azure|westus2|1|A100-80GB": {
                    "gpu_type": "A100-80GB",
                    "total": 8,
                    "free": 8,
                    "pools": [],
                }
            },
            running_jobs=(),
            waiting_jobs=(_store_job({"job_id": "job_1"}),),
            paused_jobs=(),
            finished_jobs=(),
        )

        actions = _driver((_job("job_1"),), client)._build_actions(snapshot)

        self.assertEqual(actions[0]["type"], "place")
        self.assertEqual(client.requests[0].system, "a100_sxm")
        self.assertEqual(actions[0]["ladder"][0]["env"][1], "azure")

    def test_maps_canonical_hardware_to_aic_systems(self) -> None:
        """Translate supported canonical labels to AIC system names."""
        expected_systems = {
            "H100": "h100_sxm",
            "A100-40GB": "a100_sxm",
            "A100-80GB": "a100_sxm",
            "A10G": "l4",
            "L40S": "l40s",
            "RTXPRO6000": "rtx_pro_6000_server",
            "RTX-PRO-6000": "rtx_pro_6000_server",
            "RTX-PRO-6000-BSE": "rtx_pro_6000_server",
        }

        for hardware, expected in expected_systems.items():
            with self.subTest(hardware=hardware):
                self.assertEqual(_aic_system({}, None, hardware, {}), expected)

    def test_quantization_policy_filters_unsupported_gpu_without_checking_tp(
        self,
    ) -> None:
        """Keep AIC on profiled GPUs while leaving parallelism to AIC."""
        h100 = ResourceTarget("h100", "H100", None, {"free": 8}, None)
        l40s = ResourceTarget("l40s", "L40S", None, {"free": 8}, None)
        candidates = (
            HardwareCandidate("h100_sxm", (h100,), 8),
            HardwareCandidate("l40s", (l40s,), 8),
        )

        filtered = _policy_hardware_candidates(
            candidates,
            "moonshotai/Kimi-K2-Instruct",
        )

        self.assertEqual(filtered, (candidates[0],))

    def test_data_parallelism_creates_per_replica_rank_shapes(self) -> None:
        """Keep DP in replica count rather than one ServingSim rank shape."""
        client = FakeAicClient(tp=2, dp=3)
        actions = _driver((_job("job_1"),), client)._build_actions(
            _snapshot(waiting=({"job_id": "job_1"},), h100_free=6)
        )

        placed = actions[0]
        shape = placed["ladder"][0]["config"]
        self.assertEqual(shape["count"], 2)
        self.assertEqual(shape["gpu_count"], 2)
        self.assertEqual(placed["ladder"][0]["n_replicas"], 3)
        Plan(user_id="tenant", actions=[placed])
        cluster = build_rank_cluster(
            SimpleNamespace(
                rank_name=placed["ladder"][0]["rank_id"],
                model_name=shape["model_id"],
                shape=shape,
                chains=(object(),) * 3,
            ),
            _SERVINGSIM_HARDWARE,
        )
        self.assertEqual(cluster["nodes"][0]["instances"][0]["num_npus"], 2)
        self.assertEqual(cluster["nodes"][0]["num_instances"], 3)

    def test_instance_pool_reserves_every_aic_replica(self) -> None:
        """Require one complete instance for each flattened AIC replica."""
        client = FakeAicClient(tp=2, dp=2, replicas=2)
        driver = _driver((_job("job_1"),), client)

        insufficient = driver._build_actions(
            _snapshot(
                waiting=({"job_id": "job_1"},),
                h100_free=16,
                h100_instance_pool=True,
            )
        )
        self.assertEqual(insufficient, [{"job_id": "job_1", "type": "defer"}])

        actions = _driver(
            (_job("job_1"), _job("job_2")),
            FakeAicClient(tp=2, dp=2, replicas=2),
        )._build_actions(
            _snapshot(
                waiting=({"job_id": "job_1"}, {"job_id": "job_2"}),
                h100_free=32,
                h100_instance_pool=True,
            )
        )
        self.assertEqual([action["type"] for action in actions], ["place", "defer"])
        rank = actions[0]["ladder"][0]
        self.assertEqual(rank["config"]["count"], 2)
        self.assertEqual(rank["config"]["gpu_count"], 2)
        self.assertEqual(rank["config"]["dp"], 4)
        self.assertEqual(rank["n_replicas"], 4)


class FakeAicClient:
    """Return deterministic per-system AIC predictions."""

    def __init__(
        self,
        *,
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
        num_experts: int = 0,
        batch_size: int = 128,
        replicas: int = 1,
        unsupported_model: str | None = None,
        throughput: dict[str, float] | None = None,
    ) -> None:
        self.tp = tp
        self.pp = pp
        self.dp = dp
        self.num_experts = num_experts
        self.batch_size = batch_size
        self.replicas = replicas
        self.unsupported_model = unsupported_model
        self.throughput = throughput or {"h100_sxm": 300}
        self.requests: list[AicRequest] = []

    def evaluate(
        self, request: AicRequest, timeout_seconds: float | None = None
    ) -> AicResult:
        self.requests.append(request)
        if request.model == self.unsupported_model:
            raise AicEvaluationError(
                "The model's architecture TestModel is not supported"
            )
        return AicResult(
            devices=self.tp * self.pp * self.dp * self.replicas,
            replicas=self.replicas,
            ttft_ms=10,
            tpot_ms=5,
            request_latency_ms=100,
            throughput_tokens_per_second=self.throughput[request.system],
            configuration={
                "parallel": f"tp{self.tp}pp{self.pp}dp{self.dp}",
                "moe_tp": 1,
                "moe_ep": self.tp if self.num_experts else 1,
                "bs": self.batch_size,
            },
            num_experts=self.num_experts,
        )


class FallbackAicClient(FakeAicClient):
    """Fail primary and SOL modes before accepting EMPIRICAL."""

    def evaluate(
        self, request: AicRequest, timeout_seconds: float | None = None
    ) -> AicResult:
        self.requests.append(request)
        if request.database_mode == "HYBRID":
            raise AicEvaluationError("primary unavailable")
        if request.database_mode == "SOL":
            raise AicEvaluationError("SOL unavailable")
        if request.database_mode != "EMPIRICAL":
            raise AssertionError(f"unexpected fallback mode {request.database_mode}")
        return AicResult(
            devices=1,
            replicas=1,
            ttft_ms=10,
            tpot_ms=5,
            request_latency_ms=100,
            throughput_tokens_per_second=300,
            configuration={"parallel": "tp1pp1dp1", "bs": 128},
        )


def _moe_request() -> AicRequest:
    """Describe a native MoE workload with two supported pipeline layouts."""
    return AicRequest(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        system="h100_sxm",
        available_devices=8,
        backend="vllm",
        backend_version=None,
        database_mode="HYBRID",
        serving_mode="agg",
        input_tokens=1000,
        output_tokens=100,
        target_ttft_ms=1000,
        target_tpot_ms=100,
        optimization_goal="throughput",
        target_request_rate=15,
        target_concurrency=None,
        parallelism_candidates=((4, 1), (4, 2)),
    )


def _moe_row() -> dict[str, object]:
    """Return an AIC aggregate worker row before independent load matching."""
    return {
        "parallel": "tp4pp1dp1etp1ep4",
        "tp": 4,
        "pp": 1,
        "dp": 1,
        "cp": 1,
        "moe_tp": 1,
        "moe_ep": 4,
        "num_total_gpus": 4,
        "seq/s": 10,
        "tokens/s": 1000,
        "tokens/s/gpu": 250,
        "tokens/s/user": 200,
        "ttft": 10,
        "tpot": 5,
        "concurrency": 8,
        "bs": 8,
    }


def _driver(
    jobs: tuple[OnlineJob, ...],
    client: FakeAicClient,
) -> AicPlannerDriver:
    """Build an AIC driver without external Store clients."""
    scenario = Scenario("test", 1, 1, jobs, ())
    return AicPlannerDriver(
        user_id="user",
        scenario=scenario,
        job_store=object(),
        resource_map_store=object(),
        plan_store=object(),
        settings=_driver_settings(),
        client=client,
    )


def _driver_settings() -> AicSettings:
    """Return bounded default settings for local AIC adapter tests."""
    return AicSettings(
        backend="vllm",
        backend_version=None,
        backend_versions={},
        database_mode="HYBRID",
        serving_mode="agg",
        system_mappings={},
        call_timeout_seconds=20.0,
        tick_timeout_seconds=280.0,
    )


class TimeoutProcess:
    """Represent an AIC child that exceeds its deadline."""

    def __init__(self) -> None:
        self.terminated = False

    def start(self) -> None:
        pass

    def join(self, timeout: float | None = None) -> None:
        pass

    def is_alive(self) -> bool:
        return not self.terminated

    def terminate(self) -> None:
        self.terminated = True


class TimeoutContext:
    """Supply a deterministic timeout process without spawning a worker."""

    def __init__(self) -> None:
        self.process = TimeoutProcess()

    def Queue(self, maxsize: int) -> object:
        return object()

    def Process(self, target: object, args: tuple[object, ...]) -> TimeoutProcess:
        return self.process


class SuccessQueue:
    """Hold one child result for the synchronous process test."""

    def __init__(self) -> None:
        self.value: tuple[object, object] | None = None

    def put(self, value: tuple[object, object]) -> None:
        self.value = value

    def get(self, timeout: float) -> tuple[object, object]:
        assert self.value is not None
        return self.value

    def close(self) -> None:
        pass


class SuccessProcess:
    """Run the supplied child target synchronously for protocol coverage."""

    def __init__(self, target: object, args: tuple[object, ...]) -> None:
        self.target = target
        self.args = args

    def start(self) -> None:
        self.target(*self.args)

    def join(self, timeout: float | None = None) -> None:
        pass

    def is_alive(self) -> bool:
        return False


class SuccessContext:
    """Supply a synchronous successful child process."""

    def __init__(self) -> None:
        self.queue = SuccessQueue()

    def Queue(self, maxsize: int) -> SuccessQueue:
        return self.queue

    def Process(self, target: object, args: tuple[object, ...]) -> SuccessProcess:
        return SuccessProcess(target, args)


def _job(job_id: str) -> OnlineJob:
    """Build one scheduler-visible online job."""
    model = (
        "unsupported-model"
        if job_id == "unsupported"
        else "meta-llama/Llama-3.2-1B-Instruct"
    )
    return OnlineJob(
        name=job_id,
        store_job_id=job_id,
        model_name=model,
        starts_at_ns=0,
        stops_at_ns=1,
        workload={
            "features": {
                "isl_token_avg": 900,
                "osl_token_avg": 300,
                "target_p99_ttft_ms": 700,
                "target_p99_tpot_ms": 55,
            },
            "phases": [{"rate": 1.0}],
        },
    )


def _snapshot(
    *,
    running: tuple[dict[str, object], ...] = (),
    waiting: tuple[dict[str, object], ...] = (),
    paused: tuple[dict[str, object], ...] = (),
    finished: tuple[dict[str, object], ...] = (),
    h100_free: int = 0,
    h100_instance_pool: bool = False,
    a100_free: int = 0,
) -> StoreSnapshot:
    """Build one normalized Store snapshot."""
    resources = {}
    if h100_free:
        resources["reserved|aws|us-east-1|az1|H100"] = {
            "gpu_type": "H100",
            "total": h100_free,
            "free": h100_free,
            "pools": (
                [
                    {
                        "instance_type": "p5.48xlarge",
                        "gpus_per_instance": 8,
                        "total": h100_free,
                        "free": h100_free,
                    }
                ]
                if h100_instance_pool
                else []
            ),
        }
    if a100_free:
        resources["reserved|aws|us-east-1|az1|A100"] = {
            "gpu_type": "A100",
            "total": a100_free,
            "free": a100_free,
            "pools": [],
        }
    normalized_waiting = tuple(_store_job(job) for job in waiting)
    return StoreSnapshot(resources, running, normalized_waiting, paused, finished)


def _store_job(job: dict[str, object]) -> dict[str, object]:
    """Attach canonical job features persisted with one Store job."""
    payload = dict(job)
    job_id = str(payload["job_id"])
    kind = str(payload.setdefault("kind", "online"))
    model_id = (
        "unsupported-model"
        if job_id == "unsupported"
        else "meta-llama/Llama-3.2-1B-Instruct"
    )
    payload["spec_json"] = {
        "model_id": model_id,
        "job_features": {
            "isl_token_avg": 900,
            "osl_token_avg": 300,
            "request_arrival_rate": 2.5,
            "max_concurrent_streaming": 10,
            "_traffic_mode": "request_rate",
            "target_p99_ttft_ms": 700,
            "target_p99_tpot_ms": 55,
            "deadline_hrs": 1 if kind == "batch" else 0,
            "total_token_budget": 1_000_000,
        },
    }
    return payload


if __name__ == "__main__":
    unittest.main()
