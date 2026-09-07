"""Test the Claude Agent SDK cluster placement adapter."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from agentic import (
    AgenticPlacement,
    AgenticPlannerDriver,
    ClaudeAgentPlacementClient,
    StoreSnapshot,
    _build_user_prompt,
    _validate_tick_plan,
)
from llm_usage import LlmUsage
from models import OnlineJob, Scenario
from policy import load_config_policy
from tandemn_system_data.models import Plan


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

    def test_snapshot_reads_each_relevant_model_catalog_once(self) -> None:
        """Use the existing planner Store connection for tick catalog context."""
        driver = _driver()
        requested = []
        driver.job_store = SimpleNamespace(
            running_jobs=lambda _user_id: (),
            waiting_jobs=lambda _user_id: (_store_job(),),
            paused_jobs=lambda _user_id: (),
        )
        driver.resource_map_store = SimpleNamespace(
            get=lambda: SimpleNamespace(
                market="reserved",
                scheduling_summary=lambda: {
                    "reserved|aws|us-east|use1-az1|H100": {
                        "gpu_type": "H100",
                        "total": 1,
                        "pools": [],
                    }
                },
            )
        )
        driver.model_catalog_store = SimpleNamespace(
            get=lambda model_id: (
                requested.append(model_id)
                or SimpleNamespace(model_dump=lambda **_kwargs: _catalogs()[model_id])
            )
        )

        snapshot = driver._read_store_snapshot()

        self.assertEqual(requested, ["meta-llama/Llama-3.2-1B-Instruct"])
        self.assertEqual(snapshot.model_catalogs, _catalogs())
        prompt = json.loads(
            _build_user_prompt(
                1,
                datetime(2026, 1, 1, tzinfo=UTC),
                snapshot,
                tuple(driver.jobs.values()),
                load_config_policy(),
            )
        )
        self.assertEqual(
            prompt["model_catalog"],
            [
                {
                    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                    "gpu_type": "H100",
                    **_launch_config(),
                }
            ],
        )

    def test_driver_excludes_terminal_jobs(self) -> None:
        """Create actions without acting on terminal jobs."""
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
            model_catalogs=_catalogs(),
        )
        placement = AgenticPlacement(
            decision="place",
            environment="reserved|aws|us-east|use1-az1|H100",
            instance_type=None,
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
        self.assertEqual(actions[1]["ladder"][0]["config"]["precision"], "fp16")
        for field, value in _launch_config().items():
            self.assertEqual(actions[1]["ladder"][0]["config"][field], value)
        Plan(user_id="tenant", actions=[actions[1]])

    def test_prompt_includes_prior_outcome_feedback(self) -> None:
        snapshot = StoreSnapshot(
            resources={},
            running_jobs=(),
            waiting_jobs=(),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs={},
        )

        prompt = json.loads(
            _build_user_prompt(
                1,
                datetime(2026, 1, 1, tzinfo=UTC),
                snapshot,
                (),
                load_config_policy(),
                ({"job_id": "waiting", "reward": 0.5},),
            )
        )

        self.assertEqual(prompt["outcome_feedback"], [{"job_id": "waiting", "reward": 0.5}])

    def test_driver_records_persisted_place_feedback(self) -> None:
        driver = _driver()
        driver.feedback = Mock()
        snapshot = StoreSnapshot(
            resources={
                "reserved|aws|us-east|use1-az1|H100": {
                    "gpu_type": "H100",
                    "total": 1,
                    "free": 1,
                    "pools": [],
                }
            },
            running_jobs=(),
            waiting_jobs=(_store_job(),),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=_catalogs(),
        )
        action = driver._build_actions(
            snapshot,
            (
                AgenticPlacement(
                    decision="place",
                    environment="reserved|aws|us-east|use1-az1|H100",
                    instance_type=None,
                    configuration=_configuration(),
                ),
            ),
        )[0]

        driver._record_action_feedback([action], datetime(2026, 1, 1, tzinfo=UTC))

        driver.feedback.record.assert_called_once()
        self.assertEqual(driver.feedback.record.call_args.kwargs["fingerprint"].n_replicas, 1)

    def test_driver_passes_prior_outcomes_to_agent_prompt(self) -> None:
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
            running_jobs=(),
            waiting_jobs=(_store_job(),),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=_catalogs(),
        )
        outcome = {
            "job_id": "waiting",
            "environment": "reserved|aws|us-east|use1-az1|H100",
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "n_replicas": 1,
            "reward": 0.5,
        }
        prompts = []
        driver._read_store_snapshot = lambda: snapshot
        driver.feedback = Mock()
        driver.feedback.outcomes_for_job_ids.return_value = (outcome,)
        driver.client = SimpleNamespace(
            place=lambda _tick, _model, _tokens, _system, prompt: (
                prompts.append(prompt)
                or ((AgenticPlacement("defer", None, None, None),), ())
            )
        )
        driver.plan_store = SimpleNamespace(create=lambda plan: None)

        driver.create_plans(1, datetime(2026, 1, 1, tzinfo=UTC))

        self.assertEqual(driver.feedback.observe.call_args.args, (datetime(2026, 1, 1, tzinfo=UTC),))
        self.assertEqual(driver.feedback.outcomes_for_job_ids.call_args.args, (("waiting",),))
        self.assertEqual(json.loads(prompts[0])["outcome_feedback"], [outcome])

    def test_client_requires_positive_max_turns(self) -> None:
        """Reject an agent loop with no permitted tool-use turns."""
        with self.assertRaisesRegex(ValueError, "max_turns"):
            ClaudeAgentPlacementClient(0)

    def test_client_runs_bounded_tool_loop_with_structured_output(self) -> None:
        """Configure the SDK tools, limits, and shared placement schema."""
        sdk = _FakeAgentSdk()
        client = ClaudeAgentPlacementClient(max_turns=6)

        with (
            patch("agentic._load_agent_sdk", return_value=sdk),
            patch(
                "agentic._bedrock_environment",
                return_value={
                    "CLAUDE_CODE_USE_BEDROCK": "1",
                    "AWS_PROFILE": "default",
                    "AWS_REGION": "us-east-1",
                    "AWS_BEARER_TOKEN_BEDROCK": "",
                    "ANTHROPIC_DEFAULT_HAIKU_MODEL": (
                        "global.anthropic.claude-haiku-4-5-20251001-v1:0"
                    ),
                },
            ),
        ):
            placements, usage = client.place(
                7,
                model="sonnet",
                max_tokens=12000,
                system_prompt="system",
                user_prompt=json.dumps(_scheduler_document()),
            )

        self.assertEqual(
            [placement.decision for placement in placements], ["defer"] * 2
        )
        self.assertEqual(usage[0].tick_number, 7)
        self.assertEqual(usage[0].input_tokens, 120)
        self.assertEqual(usage[0].output_tokens, 30)
        self.assertEqual(usage[0].model, "claude-sonnet-test")
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
                "AWS_REGION": "us-east-1",
                "AWS_BEARER_TOKEN_BEDROCK": "",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": (
                    "global.anthropic.claude-haiku-4-5-20251001-v1:0"
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
            "configuration": _configuration(),
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
                7,
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
                        "configuration": _configuration(),
                    },
                ]
            },
        )

        self.assertTrue(result["valid"])

    def test_tick_validation_rejects_changed_catalog_fields(self) -> None:
        """Reject a final Claude response that changes Store launch values."""
        configuration = _configuration()
        configuration["max_num_seq"] = 999

        result = _validate_tick_plan(
            _scheduler_document(),
            {
                "placements": [
                    {
                        "decision": "place",
                        "environment": "reserved|aws|us-east|use1-az1|H100",
                        "instance_type": None,
                        "configuration": configuration,
                    },
                    {
                        "decision": "defer",
                        "environment": None,
                        "instance_type": None,
                        "configuration": None,
                    },
                ]
            },
        )

        self.assertFalse(result["valid"])
        self.assertIn("must contain only", result["errors"][0])

    def test_driver_preserves_agent_usage_in_planner_result(self) -> None:
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
            running_jobs=(),
            waiting_jobs=(_store_job(),),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=_catalogs(),
        )
        driver._read_store_snapshot = lambda: snapshot
        usage = LlmUsage(7, 120, 30, "claude-haiku-test")
        driver.client = SimpleNamespace(
            place=lambda *args: ((AgenticPlacement("defer", None, None, None),), (usage,))
        )
        driver.plan_store = SimpleNamespace(create=lambda plan: None)

        result = driver.create_plans(7, datetime(2026, 1, 1, tzinfo=UTC))

        self.assertEqual(result.llm_usage, (usage,))


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
        "model_catalog": [
            {
                "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                "gpu_type": "H100",
                **_launch_config(),
            }
        ],
        "jobs_for_tick": [job, job],
        "outcome_feedback": [],
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
        model_catalog_store=object(),
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
    """Return the only placement fields Claude is allowed to choose."""
    return {
        "replicas_needed": replicas_needed,
        "tp": tp,
        "pp": pp,
        "dp": dp,
    }


class _FakeResultMessage:
    """Represent one successful Agent SDK result in the client test."""

    subtype = "success"
    is_error = False
    errors = None
    result = None
    usage = None
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
    model_usage = {
        "sonnet": {
            "inputTokens": 120,
            "outputTokens": 30,
            "canonicalModel": "claude-sonnet-test",
        }
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
