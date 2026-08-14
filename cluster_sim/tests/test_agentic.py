"""Test the Claude Agent SDK cluster placement adapter."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from agentic import (
    AgenticPlacement,
    AgenticPlannerDriver,
    ClaudeAgentPlacementClient,
    StoreSnapshot,
    _validate_tick_plan,
)
from models import OnlineJob, Scenario


class AgenticPlannerTest(unittest.TestCase):
    """Verify the agentic method keeps the shared planner contract."""

    def test_driver_uses_agentic_method_name(self) -> None:
        """Identify persisted plans as the agentic scheduler method."""
        self.assertEqual(AgenticPlannerDriver.method_name, "agentic")
        self.assertEqual(AgenticPlannerDriver.__bases__, (object,))

    def test_method_is_in_method_catalog(self) -> None:
        """Expose the independent driver through the method catalog."""
        methods = yaml.safe_load(
            (ROOT / "configs" / "methods.yaml").read_text(encoding="utf-8")
        )

        self.assertEqual(methods["methods"]["agentic"]["status"], "implemented")

    def test_driver_builds_actions_without_another_scheduler(self) -> None:
        """Create canonical actions through the independent agentic driver."""
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
        placement = AgenticPlacement(
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
        self.assertEqual(actions[1]["ladder"][0]["config"]["precision"], "bf16")

    def test_client_requires_positive_max_turns(self) -> None:
        """Reject an agent loop with no permitted tool-use turns."""
        with self.assertRaisesRegex(ValueError, "max_turns"):
            ClaudeAgentPlacementClient(0)

    def test_client_runs_bounded_tool_loop_with_structured_output(self) -> None:
        """Configure the SDK tools, limits, and shared placement schema."""
        sdk = _FakeAgentSdk()
        client = ClaudeAgentPlacementClient(max_turns=6)

        with patch("agentic._load_agent_sdk", return_value=sdk):
            placements = client.place(
                model="sonnet",
                max_tokens=12000,
                system_prompt="system",
                user_prompt=json.dumps(_scheduler_document()),
            )

        self.assertEqual(
            [placement.decision for placement in placements], ["defer"] * 2
        )
        self.assertEqual(sdk.options["model"], "sonnet")
        self.assertEqual(sdk.options["max_turns"], 6)
        self.assertEqual(sdk.options["task_budget"], {"total": 12000})
        self.assertEqual(sdk.options["tools"], [])
        self.assertTrue(sdk.options["strict_mcp_config"])
        self.assertEqual(sdk.options["output_format"]["type"], "json_schema")
        self.assertEqual(
            sdk.options["env"],
            {
                "CLAUDE_CODE_USE_BEDROCK": "1",
                "AWS_PROFILE": "default",
                "AWS_REGION": "us-east-2",
                "AWS_BEARER_TOKEN_BEDROCK": "",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": (
                    "us.anthropic.claude-haiku-4-5-20251001-v1:0"
                ),
            },
        )

    def test_tick_validation_reserves_shared_capacity(self) -> None:
        """Reject later placements that reuse capacity from earlier jobs."""
        document = _scheduler_document()
        placement = {
            "decision": "place",
            "environment": "reserved|aws|us-east|use1-az1|H100",
            "instance_type": None,
            "configuration": {
                "replicas_needed": 1,
                "tp": 1,
                "pp": 1,
                "dp": 1,
            },
        }

        result = _validate_tick_plan(
            document,
            {"placements": [placement, placement]},
        )

        self.assertFalse(result["valid"])
        self.assertIn("only 0 are free", result["errors"][0])

    def test_client_omits_unsupported_haiku_task_budget(self) -> None:
        """Bound Haiku by turns without sending its unsupported task budget."""
        sdk = _FakeAgentSdk()
        client = ClaudeAgentPlacementClient(max_turns=6)

        with patch("agentic._load_agent_sdk", return_value=sdk):
            client.place(
                model="haiku",
                max_tokens=12000,
                system_prompt="system",
                user_prompt=json.dumps(_scheduler_document()),
            )

        self.assertNotIn("task_budget", sdk.options)

    def test_tick_validation_accepts_defer(self) -> None:
        """Allow a deferred job without consuming resource capacity."""
        document = _scheduler_document()

        result = _validate_tick_plan(
            document,
            {
                "placements": [
                    {
                        "decision": "defer",
                        "environment": None,
                        "instance_type": None,
                        "configuration": None,
                    },
                    {
                        "decision": "place",
                        "environment": "reserved|aws|us-east|use1-az1|H100",
                        "instance_type": None,
                        "configuration": {
                            "replicas_needed": 1,
                            "tp": 1,
                            "pp": 1,
                            "dp": 1,
                        },
                    },
                ]
            },
        )

        self.assertTrue(result["valid"])


def _scheduler_document() -> dict[str, object]:
    """Create a two-job frozen scheduler prompt with one free GPU."""
    job = {"model_name": "meta-llama/Llama-3.2-1B-Instruct"}
    return {
        "tandemn_state": {
            "resources": {
                "reserved|aws|us-east|use1-az1|H100": {
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
                "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                "gpu_type": "H100",
                "precision": "bf16",
                "allowed_tp": [1],
            }
        ],
        "jobs_for_tick": [job, job],
    }


def _driver() -> AgenticPlannerDriver:
    """Build an independent agentic driver without external clients."""
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
    return AgenticPlannerDriver(
        user_id="user",
        scenario=Scenario("test", 1, 1, (job,), ()),
        job_store=object(),
        resource_map_store=object(),
        plan_store=object(),
        model="sonnet",
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


class _FakeResultMessage:
    """Represent one successful Agent SDK result in the client test."""

    subtype = "success"
    is_error = False
    errors = None
    result = None
    structured_output = {
        "placements": [
            {
                "decision": "defer",
                "environment": None,
                "instance_type": None,
                "configuration": None,
            },
            {
                "decision": "defer",
                "environment": None,
                "instance_type": None,
                "configuration": None,
            },
        ]
    }


class _FakeAgentSdk:
    """Capture Agent SDK configuration without making a Claude request."""

    ResultMessage = _FakeResultMessage

    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def tool(self, *args: object, **kwargs: object) -> object:
        """Return a decorator that preserves the in-process handler."""
        del args, kwargs

        def decorate(handler: object) -> object:
            return handler

        return decorate

    def create_sdk_mcp_server(self, **kwargs: object) -> dict[str, object]:
        """Return the captured server configuration."""
        return dict(kwargs)

    def ClaudeAgentOptions(self, **kwargs: object) -> dict[str, object]:
        """Capture and return the query options."""
        self.options = dict(kwargs)
        return self.options

    async def query(self, **kwargs: object) -> object:
        """Yield one final structured result."""
        del kwargs
        yield _FakeResultMessage()


if __name__ == "__main__":
    unittest.main()
