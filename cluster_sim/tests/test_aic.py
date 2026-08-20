"""Test the Store-backed cluster simulator AIC adapter."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from aic import (
    AicEvaluationError,
    AicPlannerDriver,
    AicRequest,
    AicResult,
    AicSettings,
    StoreSnapshot,
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


class FakeAicClient:
    """Return deterministic per-system AIC predictions."""

    def __init__(
        self,
        *,
        tp: int = 1,
        dp: int = 1,
        replicas: int = 1,
        unsupported_model: str | None = None,
        throughput: dict[str, float] | None = None,
    ) -> None:
        self.tp = tp
        self.dp = dp
        self.replicas = replicas
        self.unsupported_model = unsupported_model
        self.throughput = throughput or {"h100_sxm": 300}
        self.requests: list[AicRequest] = []

    def evaluate(self, request: AicRequest) -> AicResult:
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
            configuration={"parallel": f"tp{self.tp}pp1dp{self.dp}"},
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
        settings=AicSettings(
            backend="vllm",
            backend_version=None,
            backend_versions={},
            database_mode="HYBRID",
            serving_mode="agg",
            system_mappings={},
        ),
        client=client,
    )


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
    a100_free: int = 0,
) -> StoreSnapshot:
    """Build one normalized Store snapshot."""
    resources = {}
    if h100_free:
        resources["reserved|aws|us-east-1|az1|H100"] = {
            "gpu_type": "H100",
            "total": h100_free,
            "free": h100_free,
            "pools": [],
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
