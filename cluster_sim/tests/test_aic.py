"""Test the Store-backed cluster simulator AIC adapter."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
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
    NvidiaAicClient,
    StoreSnapshot,
    _aic_system,
    _evaluate_aic_request,
    _evaluate_job,
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
                    "replicas_needed": 1,
                    "total_gpus_needed": 5,
                    "tokens/s": 500,
                    "ttft": 1,
                    "tpot": 1,
                    "bs": 1,
                },
                {
                    "parallel": "tp1pp4dp1",
                    "replicas_needed": 1,
                    "total_gpus_needed": 4,
                    "tokens/s": 400,
                    "ttft": 2,
                    "tpot": 2,
                    "bs": 1,
                },
            ]
        )
        recommendation = SimpleNamespace(best_configs={"agg": frame})
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
        )

        with patch(
            "aiconfigurator.cli.api.cli_recommend",
            return_value=recommendation,
        ) as recommend:
            result = _evaluate_aic_request(request)

        self.assertEqual(result.configuration["parallel"], "tp1pp4dp1")
        self.assertEqual(recommend.call_args.kwargs["top_n"], 10)

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
        self.assertEqual(client.requests[0].model, "store-model-job_1")
        self.assertEqual(client.requests[0].target_request_rate, 2.5)
        self.assertEqual(actions[0]["ladder"][0]["config"]["tp"], 1)

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
                ("paused", "defer"),
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
        }

        for hardware, expected in expected_systems.items():
            with self.subTest(hardware=hardware):
                self.assertEqual(_aic_system({}, None, hardware, {}), expected)

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
        dp: int = 1,
        batch_size: int = 128,
        replicas: int = 1,
        unsupported_model: str | None = None,
        throughput: dict[str, float] | None = None,
    ) -> None:
        self.tp = tp
        self.dp = dp
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
            devices=self.tp * self.dp * self.replicas,
            replicas=self.replicas,
            ttft_ms=10,
            tpot_ms=5,
            request_latency_ms=100,
            throughput_tokens_per_second=self.throughput[request.system],
            configuration={
                "parallel": f"tp{self.tp}pp1dp{self.dp}",
                "bs": self.batch_size,
            },
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
    model = "unsupported-model" if job_id == "unsupported" else "supported-model"
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
        "unsupported-model" if job_id == "unsupported" else f"store-model-{job_id}"
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
