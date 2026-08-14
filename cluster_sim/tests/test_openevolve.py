"""Test the Store-backed OpenEvolve scheduler adapter."""

# ruff: noqa: E402

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import OnlineJob, Scenario
from openevolve import (
    OPENEOLVE_INITIAL_PROGRAM,
    OnlineOpenEvolveClient,
    OpenEvolvePlacement,
    OpenEvolvePlannerDriver,
    StoreSnapshot,
    _evaluator_source,
    load_policy_source,
)


class OpenEvolvePlannerTest(unittest.TestCase):
    """Verify OpenEvolve returns narrow placements for the shared plan adapter."""

    def test_driver_uses_openevolve_method_name(self) -> None:
        """Identify persisted plans as the OpenEvolve baseline."""
        self.assertEqual(OpenEvolvePlannerDriver.method_name, "openevolve")

    def test_module_does_not_import_peer_scheduler_implementations(self) -> None:
        """Keep OpenEvolve independent from other planner method files."""
        source = (ROOT / "cluster_sim" / "openevolve.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }

        self.assertTrue({"gpt", "agentic"}.isdisjoint(imports))

    def test_client_evolves_and_validates_one_tick_policy(self) -> None:
        """Run the winning policy only after the frozen tick validates it."""
        calls = []
        client = OnlineOpenEvolveClient(
            iterations=3,
            output_dir=Path("outputs/test-openevolve"),
            runner=lambda *args: (
                calls.append(args) or SimpleNamespace(best_code=_VALID_POLICY)
            ),
        )

        placements = client.place(
            "model-test",
            123,
            "system",
            json.dumps(_payload()),
        )

        self.assertEqual(placements[0].decision, "place")
        self.assertEqual(placements[0].environment, "env|H100")
        self.assertEqual(
            placements[0].configuration,
            {"replicas_needed": 1, "tp": 1, "pp": 1, "dp": 1},
        )
        self.assertEqual(calls[0][3], 3)
        self.assertEqual(client.tick_number, 1)
        self.assertEqual(client.initial_program, _VALID_POLICY)

    def test_driver_builds_complete_store_actions(self) -> None:
        """Own canonical lifecycle action construction inside this planner."""
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
        placement = OpenEvolvePlacement(
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
        self.assertEqual(placed["ladder"][0]["config"]["precision"], "bf16")
        self.assertEqual(placed["ladder"][0]["config"]["tp"], 1)

    def test_seed_policy_reserves_tensor_parallel_footprint(self) -> None:
        """Defer when the smallest allowed TP shape cannot fit the pool."""
        payload = _payload()
        payload["config_policy"][0]["allowed_tp"] = [2]

        response = load_policy_source(OPENEOLVE_INITIAL_PROGRAM)(payload)

        self.assertEqual(response["placements"][0]["decision"], "defer")

    def test_seed_policy_selects_and_reserves_an_instance_pool(self) -> None:
        """Select an exact pool and reserve its whole-instance footprint."""
        payload = _payload()
        payload["tandemn_state"]["resources"]["env|H100"] = {
            "gpu_type": "H100",
            "total": 16,
            "free": 16,
            "pools": [
                {
                    "instance_type": "h100.8gpu-a",
                    "gpus_per_instance": 8,
                    "total": 8,
                    "free": 8,
                },
                {
                    "instance_type": "h100.8gpu-b",
                    "gpus_per_instance": 8,
                    "total": 8,
                    "free": 8,
                },
            ],
        }
        payload["jobs_for_tick"] = [
            {"model_name": "model"},
            {"model_name": "model"},
            {"model_name": "model"},
        ]

        response = load_policy_source(OPENEOLVE_INITIAL_PROGRAM)(payload)

        self.assertEqual(
            [placement["decision"] for placement in response["placements"]],
            ["place", "place", "defer"],
        )
        self.assertEqual(
            [placement["instance_type"] for placement in response["placements"][:2]],
            ["h100.8gpu-a", "h100.8gpu-b"],
        )

        namespace: dict[str, object] = {}
        exec(_evaluator_source(payload), namespace)
        validation = namespace["_validate"](payload, response)
        self.assertEqual(
            validation,
            {"valid": True, "gpu_footprint": 16},
        )


def _payload() -> dict[str, object]:
    """Create one minimal public-facts placement problem."""
    return {
        "tandemn_state": {
            "resources": {
                "env|H100": {
                    "gpu_type": "H100",
                    "total": 1,
                    "free": 1,
                    "pools": [],
                }
            },
            "active_jobs": [],
            "pending_jobs": [],
        },
        "config_policy": [
            {
                "model_id": "model",
                "gpu_type": "H100",
                "precision": "bf16",
                "allowed_tp": [1],
            }
        ],
        "jobs_for_tick": [{"model_name": "model"}],
    }


def _driver() -> OpenEvolvePlannerDriver:
    """Build an OpenEvolve driver without external Store or LLM clients."""
    job = OnlineJob(
        name="waiting",
        store_job_id="waiting",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        starts_at_ns=0,
        stops_at_ns=1,
        workload={
            "features": {"isl_token_avg": 100, "osl_token_avg": 20},
            "phases": [{"rate": 1.0}],
        },
    )
    return OpenEvolvePlannerDriver(
        user_id="user",
        scenario=Scenario("test", 1, 1, (job,), ()),
        job_store=object(),
        resource_map_store=object(),
        plan_store=object(),
        model="model-test",
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
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
            "job_features": {
                "target_p99_ttft_ms": 700,
                "target_p99_tpot_ms": 55,
            },
        },
    }


_VALID_POLICY = """\
def compute_placement(payload):
    return {
        "placements": [
            {
                "decision": "place",
                "environment": "env|H100",
                "instance_type": None,
                "configuration": {
                    "replicas_needed": 1,
                    "tp": 1,
                    "pp": 1,
                    "dp": 1,
                },
            }
        ]
    }
"""
