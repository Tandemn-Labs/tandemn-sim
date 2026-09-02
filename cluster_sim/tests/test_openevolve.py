"""Test the Store-backed OpenEvolve scheduler adapter."""

# ruff: noqa: E402

from __future__ import annotations

import ast
import json
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import OnlineJob, Scenario
from openevolve import (
    OPENEOLVE_INITIAL_PROGRAM,
    OPENEOLVE_SYSTEM_PROMPT,
    OnlineOpenEvolveClient,
    OpenEvolvePlacement,
    OpenEvolvePlannerDriver,
    StoreSnapshot,
    _build_user_prompt,
    _evaluator_source,
    _job_required_vram_gib,
    _place_action,
    _resource_prompt_payload,
    _validate_policy_response,
    load_policy_source,
)
from policy import load_config_policy
from serving import build_rank_cluster
from tandemn_system_data.ids import is_valid_id
from tandemn_system_data.models import Plan

_SERVINGSIM_HARDWARE = {
    "profile": "H100",
    "memory_gb": 80,
    "memory_bandwidth_gb_s": 3350,
    "link_bandwidth_gb_s": 900,
}


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
                calls.append(args)
                or SimpleNamespace(
                    best_code=_VALID_POLICY,
                    llm_usage=[
                        {
                            "input_tokens": 80,
                            "output_tokens": 20,
                            "model": "model-used",
                        }
                    ],
                )
            ),
        )

        placements, usage = client.place(
            4,
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
        system_message = calls[0][2]["prompt"]["system_message"]
        self.assertTrue(system_message.startswith("system"))
        self.assertIn('"engine_version":"0.16.0"', system_message)
        self.assertEqual(client.tick_number, 1)
        self.assertEqual(client.initial_program, _VALID_POLICY)
        self.assertEqual(usage[0].tick_number, 4)
        self.assertEqual(usage[0].input_tokens, 80)
        self.assertEqual(usage[0].output_tokens, 20)
        self.assertEqual(usage[0].model, "model-used")

    def test_snapshot_reads_each_relevant_model_catalog_once(self) -> None:
        """Freeze resolved Store launch profiles for the tick and LLM context."""
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
        prompt = json.loads(
            _build_user_prompt(
                1,
                datetime(2026, 1, 1, tzinfo=UTC),
                snapshot,
                tuple(driver.jobs.values()),
                load_config_policy(),
            )
        )

        self.assertEqual(requested, ["meta-llama/Llama-3.2-1B-Instruct"])
        self.assertEqual(snapshot.model_catalogs, _catalogs())
        self.assertEqual(prompt["config_policy"][0]["allowed_pp"], [1, 2, 3, 4])
        self.assertIn("PP must never exceed 4", OPENEOLVE_SYSTEM_PROMPT)
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

    def test_resource_payload_resolves_store_mi300_memory(self) -> None:
        """Accept the MI300 spelling used by Store resource maps and profiles."""
        payload = _resource_prompt_payload(
            {
                "reserved|runpod|AU|default|MI300": {
                    "gpu_type": "MI300",
                    "pools": [],
                }
            }
        )

        self.assertEqual(
            payload["reserved|runpod|AU|default|MI300"]["gpu_memory_gib"],
            192.0,
        )

    def test_driver_excludes_terminal_jobs(self) -> None:
        """Own lifecycle actions without acting on terminal jobs."""
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
            ],
        )
        placed = actions[1]
        self.assertEqual(placed["target_p99_ttft_ms"], 700.0)
        self.assertTrue(is_valid_id(placed["ladder"][0]["rank_id"], "rank"))
        rank_config = placed["ladder"][0]["config"]
        self.assertEqual(rank_config["precision"], "bf16")
        self.assertEqual(rank_config["tp"], 1)
        for field, value in _launch_config().items():
            self.assertEqual(rank_config[field], value)
        Plan(user_id="tenant", actions=[placed])

    def test_data_parallelism_creates_per_replica_rank_shapes(self) -> None:
        """Keep DP in replica count rather than one ServingSim rank shape."""
        driver = _driver()
        job = driver.jobs["waiting"]
        action = _place_action(
            "waiting",
            _store_job(),
            job,
            OpenEvolvePlacement(
                decision="place",
                environment="reserved|aws|us-east|use1-az1|H100",
                instance_type=None,
                configuration={"replicas_needed": 2, "tp": 2, "pp": 1, "dp": 3},
            ),
            {
                "reserved|aws|us-east|use1-az1|H100": {
                    "gpu_type": "H100",
                    "total": 12,
                    "free": 12,
                    "pools": [],
                }
            },
            _catalogs(),
        )

        shape = action["ladder"][0]["config"]
        self.assertEqual(shape["count"], 2)
        self.assertEqual(shape["gpu_count"], 2)
        self.assertEqual(action["ladder"][0]["n_replicas"], 6)
        Plan(user_id="tenant", actions=[action])
        cluster = build_rank_cluster(
            SimpleNamespace(
                rank_name=action["ladder"][0]["rank_id"],
                model_name=shape["model_id"],
                shape=shape,
                chains=(object(),) * 6,
            ),
            _SERVINGSIM_HARDWARE,
        )
        self.assertEqual(cluster["nodes"][0]["instances"][0]["num_npus"], 2)
        self.assertEqual(cluster["nodes"][0]["num_instances"], 6)

    def test_vram_requirement_uses_benchmark_name_after_store_id_refresh(self) -> None:
        """Keep OpenEvolve's VRAM lookup stable across repeated CLI runs."""
        job = OnlineJob(
            name="job-benchmark_timing_3hr-0-5",
            store_job_id="job-fresh-store-id",
            model_name="intfloat/multilingual-e5-large",
            starts_at_ns=0,
            stops_at_ns=1,
            workload={},
        )

        self.assertEqual(_job_required_vram_gib(job), 2.458)

    def test_seed_policy_reserves_tensor_parallel_footprint(self) -> None:
        """Defer when the smallest allowed TP shape cannot fit the pool."""
        payload = _payload()
        payload["config_policy"][0]["allowed_tp"] = [2]

        response = load_policy_source(OPENEOLVE_INITIAL_PROGRAM)(payload)

        self.assertEqual(response["placements"][0]["decision"], "defer")

    def test_prompt_exposes_workload_and_resource_vram(self) -> None:
        """Give evolved policies the memory facts needed for valid placement."""
        driver = _driver()
        snapshot = StoreSnapshot(
            resources={
                "env|H100-SXM-80GB": {
                    "gpu_type": "H100-SXM-80GB",
                    "total": 8,
                    "free": 8,
                    "pools": [],
                }
            },
            running_jobs=(),
            waiting_jobs=(_store_job(),),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=_catalogs("H100-SXM-80GB"),
        )

        prompt = json.loads(
            _build_user_prompt(
                0,
                datetime(2026, 1, 1, tzinfo=UTC),
                snapshot,
                (driver.jobs["waiting"],),
                load_config_policy(),
            )
        )

        self.assertEqual(prompt["jobs_for_tick"][0]["required_vram_gib"], 7.966)
        self.assertEqual(
            prompt["tandemn_state"]["resources"]["env|H100-SXM-80GB"]["gpu_memory_gib"],
            80,
        )

    def test_resource_prompt_normalizes_azure_a100_memory(self) -> None:
        """Resolve Store's Azure A100_80GB label without pool memory fields."""
        payload = _resource_prompt_payload(
            {
                "reserved|azure|westus2|1|A100_80GB": {
                    "gpu_type": "A100_80GB",
                    "total": 8,
                    "free": 8,
                    "pools": [
                        {
                            "instance_type": "Standard_ND96amsr_A100_v4",
                            "total": 8,
                            "free": 8,
                        }
                    ],
                }
            }
        )

        resource = payload["reserved|azure|westus2|1|A100_80GB"]
        self.assertEqual(resource["gpu_memory_gib"], 80)
        self.assertEqual(resource["pools"][0]["gpu_memory_gib"], 80)

    def test_evaluator_and_final_validator_reject_insufficient_vram(self) -> None:
        """Reject a policy whose TP/PP copy cannot hold the workload."""
        payload = _payload()
        payload["jobs_for_tick"][0]["required_vram_gib"] = 171.67
        payload["tandemn_state"]["resources"]["env|H100"]["gpu_memory_gib"] = 40
        payload["tandemn_state"]["resources"]["env|H100"]["total"] = 8
        payload["tandemn_state"]["resources"]["env|H100"]["free"] = 8
        payload["config_policy"][0]["allowed_tp"] = [2]
        response = _placement_response(tp=2)
        namespace: dict[str, object] = {}
        exec(_evaluator_source(payload), namespace)

        self.assertEqual(namespace["_validate"](payload, response), {"valid": False})
        with self.assertRaisesRegex(ValueError, "needs 171.67 GiB"):
            _validate_policy_response(payload, response)

    def test_seed_policy_selects_and_reserves_an_instance_pool(self) -> None:
        """Select an exact pool and reserve its whole-instance footprint."""
        payload = _payload()
        payload["tandemn_state"]["resources"]["env|H100"] = {
            "gpu_type": "H100",
            "gpu_memory_gib": 80,
            "total": 16,
            "free": 16,
            "pools": [
                {
                    "instance_type": "h100.8gpu-a",
                    "gpus_per_instance": 8,
                    "gpu_memory_gib": 80,
                    "total": 8,
                    "free": 8,
                },
                {
                    "instance_type": "h100.8gpu-b",
                    "gpus_per_instance": 8,
                    "gpu_memory_gib": 80,
                    "total": 8,
                    "free": 8,
                },
            ],
        }
        payload["jobs_for_tick"] = [
            {"model_name": "model", "required_vram_gib": 1.0},
            {"model_name": "model", "required_vram_gib": 1.0},
            {"model_name": "model", "required_vram_gib": 1.0},
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
                    "gpu_memory_gib": 80,
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
        "model_catalog": [
            {
                "model_id": "model",
                "gpu_type": "H100",
                **_launch_config(),
            }
        ],
        "jobs_for_tick": [{"model_name": "model", "required_vram_gib": 1.0}],
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
            "features": {
                "isl_token_avg": 100,
                "osl_token_avg": 20,
                "required_vram_gib": 7.966,
            },
            "phases": [{"rate": 1.0}],
        },
    )
    return OpenEvolvePlannerDriver(
        user_id="user",
        scenario=Scenario("test", 1, 1, (job,), ()),
        job_store=object(),
        resource_map_store=object(),
        model_catalog_store=object(),
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


def _catalogs(gpu_type: str = "H100") -> dict[str, dict[str, object]]:
    """Return one raw Store model catalog keyed by model ID."""
    catalog = _launch_config()
    for field in (
        "max_num_seq",
        "max_num_batched_tokens",
        "block_size",
        "kvcache_dtype",
    ):
        catalog[field] = [{"gpu_type": gpu_type, "value": catalog[field]}]
    return {"meta-llama/Llama-3.2-1B-Instruct": catalog}


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


def _placement_response(tp: int) -> dict[str, object]:
    """Return one complete placement response for evaluator tests."""
    return {
        "placements": [
            {
                "decision": "place",
                "environment": "env|H100",
                "instance_type": None,
                "configuration": {
                    "replicas_needed": 1,
                    "tp": tp,
                    "pp": 1,
                    "dp": 1,
                },
            }
        ]
    }
