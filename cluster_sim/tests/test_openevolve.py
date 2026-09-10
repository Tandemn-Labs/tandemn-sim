"""Test the Store-backed OpenEvolve scheduler adapter."""

# ruff: noqa: E402

from __future__ import annotations

import ast
import json
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import BatchJob, OnlineJob, Scenario
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
    _outcome_feedback_payload,
    _place_action,
    _resource_prompt_payload,
    _resources_with_free_capacity,
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

_ACTIVE_RANK_ID = "rank_01ARZ3NDEKTSV4RRFFQ69G5FAV"


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
            {"tp": 1, "pp": 1, "dp": 1},
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
        driver.jobs["running"] = _online_job("running")
        requested = []
        driver.job_store = SimpleNamespace(
            running_jobs=lambda _user_id: (
                {
                    "job": _store_job_for("running"),
                    "ranks": [
                        {
                            "rank_id": _ACTIVE_RANK_ID,
                            "n_replicas": 1,
                            "shape_json": {
                                "env": "reserved|aws|us-east|use1-az1|H100",
                                "count": 1,
                            },
                        }
                    ],
                },
            ),
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
        self.assertEqual(snapshot.running_jobs[0]["job_id"], "running")
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
            lifecycle="waiting",
            decision="place",
            environment="reserved|aws|us-east|use1-az1|H100",
            instance_type=None,
            configuration={"tp": 1, "pp": 1, "dp": 1},
        )

        actions = driver._build_actions(snapshot, (placement,))

        self.assertEqual(
            [(action["job_id"], action["type"]) for action in actions],
            [
                ("running", "keep"),
                ("waiting", "place"),
            ],
        )
        placed = actions[1]
        self.assertEqual(placed["target_p99_ttft_ms"], 700.0)
        self.assertTrue(is_valid_id(placed["ladder"][0]["rank_id"], "rank"))
        rank_config = placed["ladder"][0]["config"]
        self.assertEqual(rank_config["precision"], "fp16")
        self.assertEqual(rank_config["tp"], 1)
        for field, value in _launch_config().items():
            self.assertEqual(rank_config[field], value)
        Plan(user_id="tenant", actions=[placed])

    def test_running_candidates_swap_with_fresh_replacement_rank_id(self) -> None:
        """Use active placement context but give its replacement a fresh identity."""
        driver = _driver()
        driver.jobs["running"] = _online_job("running")
        active_rank = _ACTIVE_RANK_ID
        snapshot = StoreSnapshot(
            resources={
                "env-a|H100": {
                    "gpu_type": "H100",
                    "total": 1,
                    "free": 0,
                    "pools": [],
                },
                "env-b|H100": {
                    "gpu_type": "H100",
                    "total": 1,
                    "free": 1,
                    "pools": [],
                },
            },
            running_jobs=(_running_store_job("running", active_rank, "env-a|H100"),),
            waiting_jobs=(_store_job(),),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=_catalogs(),
        )
        prompt = json.loads(
            _build_user_prompt(
                1,
                datetime(2026, 1, 1, tzinfo=UTC),
                snapshot,
                (driver.jobs["running"], driver.jobs["waiting"]),
                load_config_policy(),
            )
        )
        placements = (
            OpenEvolvePlacement(
                lifecycle="running",
                decision="swap",
                environment="env-b|H100",
                instance_type=None,
                configuration={"tp": 1, "pp": 1, "dp": 1},
            ),
            OpenEvolvePlacement("waiting", "defer", None, None, None),
        )

        actions = driver._build_actions(snapshot, placements)

        self.assertEqual(
            [job["lifecycle"] for job in prompt["jobs_for_tick"]],
            ["running", "waiting"],
        )
        self.assertEqual(
            prompt["jobs_for_tick"][0]["active_placement"]["environment"],
            "env-a|H100",
        )
        self.assertEqual([action["type"] for action in actions], ["swap", "defer"])
        replacement_rank_id = actions[0]["ladder"][0]["rank_id"]
        self.assertTrue(is_valid_id(replacement_rank_id, "rank"))
        self.assertNotEqual(replacement_rank_id, active_rank)
        with self.assertRaisesRegex(ValueError, "swaps require a rank ID"):
            _place_action(
                "running",
                snapshot.running_jobs[0],
                driver.jobs["running"],
                placements[0],
                snapshot.resources,
                snapshot.model_catalogs,
                action_type="swap",
            )

    def test_noop_swap_is_kept(self) -> None:
        """Do not materialize a swap whose physical ladder is unchanged."""
        driver = _driver()
        driver.jobs["running"] = _online_job("running")
        snapshot = StoreSnapshot(
            resources={
                "env|H100": {
                    "gpu_type": "H100",
                    "total": 2,
                    "free": 1,
                    "pools": [],
                }
            },
            running_jobs=(
                _running_store_job("running", _ACTIVE_RANK_ID, "env|H100"),
            ),
            waiting_jobs=(),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=_catalogs(),
        )

        actions = driver._build_actions(
            snapshot,
            (
                OpenEvolvePlacement(
                    lifecycle="running",
                    decision="swap",
                    environment="env|H100",
                    instance_type=None,
                    configuration={"tp": 1, "pp": 1, "dp": 1},
                ),
            ),
        )

        self.assertEqual(actions, [{"job_id": "running", "type": "keep"}])

    def test_dp_only_swap_retains_the_active_rank_id(self) -> None:
        """Resize replicas without replacing the deployment identity."""
        driver = _driver()
        driver.jobs["running"] = _online_job("running")
        snapshot = StoreSnapshot(
            resources={
                "env|H100": {
                    "gpu_type": "H100",
                    "total": 3,
                    "free": 2,
                    "pools": [],
                }
            },
            running_jobs=(
                _running_store_job("running", _ACTIVE_RANK_ID, "env|H100"),
            ),
            waiting_jobs=(),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=_catalogs(),
        )

        actions = driver._build_actions(
            snapshot,
            (
                OpenEvolvePlacement(
                    lifecycle="running",
                    decision="swap",
                    environment="env|H100",
                    instance_type=None,
                    configuration={"tp": 1, "pp": 1, "dp": 2},
                ),
            ),
        )

        self.assertEqual(actions[0]["type"], "swap")
        self.assertEqual(actions[0]["ladder"][0]["rank_id"], _ACTIVE_RANK_ID)
        self.assertEqual(actions[0]["ladder"][0]["n_replicas"], 2)

    def test_validators_enforce_lifecycle_and_reject_noop_swaps(self) -> None:
        """Accept only running keep or changed swap decisions."""
        payload = _payload()
        job = payload["jobs_for_tick"][0]
        job["lifecycle"] = "running"
        job["active_placement"] = {
            "environment": "env|H100",
            "instance_type": None,
            "gpu_count": 1,
            "tp": 1,
            "pp": 1,
            "n_replicas": 1,
        }
        keep = {
            "placements": [
                {
                    "lifecycle": "running",
                    "decision": "keep",
                    "environment": None,
                    "instance_type": None,
                    "configuration": None,
                }
            ]
        }
        namespace: dict[str, object] = {}
        exec(_evaluator_source(payload), namespace)

        self.assertEqual(
            namespace["_validate"](payload, keep),
            {"valid": True, "gpu_footprint": 0, "hourly_price": 0.0},
        )
        _validate_policy_response(payload, keep)

        noop = {
            "placements": [
                {
                    "lifecycle": "running",
                    "decision": "swap",
                    "environment": "env|H100",
                    "instance_type": None,
                    "configuration": {"tp": 1, "pp": 1, "dp": 1},
                }
            ]
        }
        self.assertEqual(namespace["_validate"](payload, noop), {"valid": False})
        with self.assertRaisesRegex(ValueError, "different physical placement"):
            _validate_policy_response(payload, noop)

    def test_data_parallelism_creates_per_replica_rank_shapes(self) -> None:
        """Keep DP in replica count rather than one ServingSim rank shape."""
        driver = _driver()
        job = driver.jobs["waiting"]
        action = _place_action(
            "waiting",
            _store_job(),
            job,
            OpenEvolvePlacement(
                lifecycle="waiting",
                decision="place",
                environment="reserved|aws|us-east|use1-az1|H100",
                instance_type=None,
                configuration={"tp": 2, "pp": 1, "dp": 3},
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
        self.assertEqual(action["ladder"][0]["n_replicas"], 3)
        Plan(user_id="tenant", actions=[action])
        cluster = build_rank_cluster(
            SimpleNamespace(
                rank_name=action["ladder"][0]["rank_id"],
                model_name=shape["model_id"],
                shape=shape,
                chains=(object(),) * 3,
            ),
            _SERVINGSIM_HARDWARE,
        )
        self.assertEqual(cluster["nodes"][0]["instances"][0]["num_npus"], 2)
        self.assertEqual(cluster["nodes"][0]["num_instances"], 3)

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

    def test_evaluator_and_final_validator_use_canonical_dp_footprints(self) -> None:
        """Charge DP copies by GPU or whole-instance replicas consistently."""
        payload = _payload()
        resource = payload["tandemn_state"]["resources"]["env|H100"]
        resource.update({"total": 6, "free": 6})
        resource["pools"] = [
            {
                "instance_type": "gpu-pool",
                "allocation_kind": "gpu",
                "total": 6,
                "free": 6,
            }
        ]
        payload["config_policy"][0]["allowed_tp"] = [2]
        response = _placement_response(tp=2, dp=3, instance_type="gpu-pool")
        namespace: dict[str, object] = {}
        exec(_evaluator_source(payload), namespace)

        self.assertEqual(
            namespace["_validate"](payload, response),
            {"valid": True, "gpu_footprint": 6, "hourly_price": 0.0},
        )
        _validate_policy_response(payload, response)

        resource["free"] = 5
        resource["pools"][0]["free"] = 5
        namespace = {}
        exec(_evaluator_source(payload), namespace)
        self.assertEqual(namespace["_validate"](payload, response), {"valid": False})
        with self.assertRaisesRegex(ValueError, "needs 6 GPUs"):
            _validate_policy_response(payload, response)

        resource.update({"total": 16, "free": 16})
        resource["pools"] = [
            {
                "instance_type": "instance-pool",
                "gpus_per_instance": 8,
                "total": 16,
                "free": 16,
            }
        ]
        response = _placement_response(tp=2, dp=2, instance_type="instance-pool")
        namespace = {}
        exec(_evaluator_source(payload), namespace)

        self.assertEqual(
            namespace["_validate"](payload, response),
            {"valid": True, "gpu_footprint": 16, "hourly_price": 0.0},
        )
        _validate_policy_response(payload, response)

        oversized = _placement_response(tp=16, dp=1, instance_type="instance-pool")
        payload["config_policy"][0]["allowed_tp"] = [16]
        namespace = {}
        exec(_evaluator_source(payload), namespace)
        self.assertEqual(namespace["_validate"](payload, oversized), {"valid": False})
        with self.assertRaisesRegex(ValueError, "needs 16 GPUs per replica"):
            _validate_policy_response(payload, oversized)

    def test_active_gpu_pool_accounting_includes_all_replicas(self) -> None:
        """Subtract each active GPU-pool rank replica from local free capacity."""
        resources = _resources_with_free_capacity(
            SimpleNamespace(
                market="reserved",
                scheduling_summary=lambda: {
                    "reserved|aws|us-east|use1-az1|H100": {
                        "gpu_type": "H100",
                        "total": 8,
                        "pools": [
                            {
                                "instance_type": "gpu-pool",
                                "allocation_kind": "gpu",
                                "total": 8,
                            }
                        ],
                    }
                },
            ),
            (
                {
                    "ranks": [
                        {
                            "n_replicas": 3,
                            "shape_json": {
                                "env": "reserved|aws|us-east|use1-az1|H100",
                                "instance_type": "gpu-pool",
                                "count": 2,
                            },
                        }
                    ]
                },
            ),
        )

        resource = resources["reserved|aws|us-east|use1-az1|H100"]
        self.assertEqual(resource["free"], 2)
        self.assertEqual(resource["pools"][0]["free"], 2)

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
            {
                "model_name": "model",
                "required_vram_gib": 1.0,
                "lifecycle": "waiting",
            },
            {
                "model_name": "model",
                "required_vram_gib": 1.0,
                "lifecycle": "waiting",
            },
            {
                "model_name": "model",
                "required_vram_gib": 1.0,
                "lifecycle": "waiting",
            },
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
            {"valid": True, "gpu_footprint": 16, "hourly_price": 0.0},
        )

    def test_prompt_includes_canonical_outcome_feedback_evidence(self) -> None:
        """Expose only matching JSON-native placement outcomes to the policy."""
        driver = _driver()
        snapshot = StoreSnapshot(
            resources={
                "env|H100": {
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
        evidence = {
            "job_id": "waiting",
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
            "environment": "env|H100",
            "instance_type": None,
            "gpu_type": "H100",
            "tp": 1,
            "pp": 1,
            "dp": 1,
            "reward": 0.75,
        }
        feedback = SimpleNamespace(
            outcomes_for_job_ids=lambda job_ids: (
                (evidence,) if tuple(job_ids) == ("waiting",) else ()
            )
        )

        feedback_payload = _outcome_feedback_payload(
            feedback,
            snapshot,
            (driver.jobs["waiting"],),
            load_config_policy(),
        )
        prompt = json.loads(
            _build_user_prompt(
                1,
                datetime(2026, 1, 1, tzinfo=UTC),
                snapshot,
                (driver.jobs["waiting"],),
                load_config_policy(),
                feedback_payload,
            )
        )

        self.assertEqual(prompt["jobs_for_tick"][0]["job_id"], "waiting")
        self.assertEqual(
            prompt["jobs_for_tick"][0]["model_id"],
            "meta-llama/Llama-3.2-1B-Instruct",
        )
        self.assertEqual(prompt["jobs_for_tick"][0]["priority"], 2)
        self.assertEqual(prompt["outcome_feedback"], [evidence])

    def test_evaluator_matches_evox_fitness(self) -> None:
        """Score a valid placement by priority, outcome, price, and footprint."""
        response = _placement_response(tp=1)
        source = "def compute_placement(payload):\n    return " + repr(response) + "\n"
        rewarded_payload = _payload()
        rewarded_payload["jobs_for_tick"][0].update(
            {"job_id": "waiting", "model_id": "model", "priority": 3}
        )
        rewarded_payload["tandemn_state"]["resources"]["env|H100"][
            "price_per_gpu_hour"
        ] = 2.0
        rewarded_payload["outcome_feedback"] = [
            {
                "job_id": "waiting",
                "model_id": "model",
                "environment": "env|H100",
                "instance_type": None,
                "gpu_type": "H100",
                "tp": 1,
                "pp": 1,
                "dp": 1,
                "reward": 0.8,
            }
        ]
        neutral_payload = _payload()
        neutral_payload["jobs_for_tick"][0].update(
            {"job_id": "waiting", "model_id": "model", "priority": 3}
        )
        neutral_payload["tandemn_state"]["resources"]["env|H100"][
            "price_per_gpu_hour"
        ] = 2.0
        namespace: dict[str, object] = {}
        rewarded_namespace: dict[str, object] = {}
        exec(_evaluator_source(neutral_payload), namespace)
        exec(_evaluator_source(rewarded_payload), rewarded_namespace)
        with TemporaryDirectory() as directory:
            policy_path = Path(directory) / "candidate.py"
            policy_path.write_text(source, encoding="utf-8")
            neutral_score = namespace["evaluate"](policy_path)
            rewarded_score = rewarded_namespace["evaluate"](policy_path)

        self.assertEqual(neutral_score["priority_score"], 4.0)
        self.assertEqual(rewarded_score["priority_score"], 4.0)
        self.assertEqual(neutral_score["mean_observed_reward"], 0.0)
        self.assertEqual(rewarded_score["mean_observed_reward"], 0.8)
        self.assertEqual(rewarded_score["hourly_price"], 2.0)
        self.assertEqual(rewarded_score["gpu_footprint"], 1.0)
        self.assertAlmostEqual(
            rewarded_score["combined_score"],
            4 * 4 + 0.8 + 1 / 3 + 1 / 2_000,
        )
        self.assertGreater(
            rewarded_score["combined_score"],
            neutral_score["combined_score"],
        )
        self.assertEqual(
            rewarded_score["admitted_jobs"],
            neutral_score["admitted_jobs"],
        )

    def test_pending_running_batch_is_kept_while_waiting_batch_can_place(self) -> None:
        """Do not offer a swap until a batch placement has a terminal outcome."""
        driver = _driver()
        driver.jobs["running"] = _batch_job("running")
        driver.jobs["waiting"] = _batch_job("waiting")
        driver.feedback = SimpleNamespace(
            has_pending_batch=lambda job_id: job_id == "running"
        )
        snapshot = StoreSnapshot(
            resources={
                "env|H100": {
                    "gpu_type": "H100",
                    "total": 2,
                    "free": 1,
                    "pools": [],
                }
            },
            running_jobs=(
                _running_store_job("running", _ACTIVE_RANK_ID, "env|H100"),
            ),
            waiting_jobs=(_store_job(),),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=_catalogs(),
        )
        place_waiting = OpenEvolvePlacement(
            lifecycle="waiting",
            decision="place",
            environment="env|H100",
            instance_type=None,
            configuration={"tp": 1, "pp": 1, "dp": 1},
        )

        self.assertEqual(
            driver._actionable_job_ids(snapshot),
            (("waiting", "waiting"),),
        )
        self.assertEqual(
            [
                action["type"]
                for action in driver._build_actions(snapshot, (place_waiting,))
            ],
            ["keep", "place"],
        )

    def test_create_plans_records_canonical_feedback_with_telemetry_targets(
        self,
    ) -> None:
        """Track exactly the canonical ladder after its plan persists."""
        driver = _driver()
        recorded = []
        observed = []
        driver.feedback = SimpleNamespace(
            observe=lambda logical_time: observed.append(logical_time),
            has_pending_batch=lambda _job_id: False,
            outcomes_for_job_ids=lambda _job_ids: (),
            record=lambda **kwargs: recorded.append(kwargs),
        )
        snapshot = StoreSnapshot(
            resources={
                "env|H100": {
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
        placement = OpenEvolvePlacement(
            lifecycle="waiting",
            decision="place",
            environment="env|H100",
            instance_type=None,
            configuration={"tp": 1, "pp": 1, "dp": 1},
        )
        logical_time = datetime(2026, 1, 1, tzinfo=UTC)
        created = []
        driver._read_store_snapshot = lambda: snapshot
        driver.client = SimpleNamespace(place=lambda *_args: ((placement,), ()))
        driver.plan_store = SimpleNamespace(create=lambda plan: created.append(plan))

        result = driver.create_plans(1, logical_time)

        self.assertEqual(observed, [logical_time])
        self.assertEqual(len(created), 1)
        self.assertFalse(result.degraded)
        self.assertEqual(len(recorded), 1)
        record = recorded[0]
        self.assertEqual(record["fingerprint"].job_id, "waiting")
        self.assertEqual(record["fingerprint"].environment, "env|H100")
        self.assertTrue(record["rank_id"].startswith("rank_"))
        self.assertEqual(record["window_start"], logical_time)
        self.assertFalse(record["is_batch"])
        self.assertEqual(record["required_tps"], 20.0)
        self.assertEqual(record["ttft_target_ms"], 700.0)
        self.assertEqual(record["tpot_target_ms"], 55.0)


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
        "jobs_for_tick": [
            {
                "model_name": "model",
                "required_vram_gib": 1.0,
                "lifecycle": "waiting",
            }
        ],
    }


def _driver() -> OpenEvolvePlannerDriver:
    """Build an OpenEvolve driver without external Store or LLM clients."""
    job = _online_job("waiting")
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


def _online_job(job_id: str) -> OnlineJob:
    """Create one scenario job with the shared test model and workload."""
    return OnlineJob(
        name=job_id,
        store_job_id=job_id,
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


def _batch_job(job_id: str) -> BatchJob:
    """Create one batch job with an explicit memory requirement."""
    return BatchJob(
        name=job_id,
        store_job_id=job_id,
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        releases_at_ns=0,
        deadline_at_ns=1_000_000_000,
        hard_stop_at_ns=2_000_000_000,
        workload={"features": {"required_vram_gib": 7.966}},
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


def _store_job_for(job_id: str) -> dict[str, object]:
    """Create one canonical Store job payload for an arbitrary job ID."""
    job = _store_job()
    job["job_id"] = job_id
    return job


def _running_store_job(
    job_id: str,
    rank_id: str,
    environment: str,
) -> dict[str, object]:
    """Create one running Store payload with a single active rank."""
    job = _store_job_for(job_id)
    job["active_ranks"] = [
        {
            "rank_id": rank_id,
            "role": "aggregate",
            "n_replicas": 1,
            "shape_json": {
                **_launch_config(),
                "env": environment,
                "instance_type": None,
                "gpu_type": "H100",
                "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                "precision": "fp16",
                "gpu_count": 1,
                "count": 1,
                "tp": 1,
                "pp": 1,
                "dp": 1,
                "ep": 1,
                "sp": 1,
                "cp": 1,
                "rank_traffic_share": 1.0,
                "target_p99_ttft_ms": 700.0,
                "target_p99_tpot_ms": 55.0,
            },
        }
    ]
    return job


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
                "lifecycle": "waiting",
                "decision": "place",
                "environment": "env|H100",
                "instance_type": None,
                "configuration": {
                    "tp": 1,
                    "pp": 1,
                    "dp": 1,
                },
            }
        ]
    }
"""


def _placement_response(
    tp: int,
    dp: int = 1,
    instance_type: str | None = None,
) -> dict[str, object]:
    """Return one complete placement response for evaluator tests."""
    return {
        "placements": [
            {
                "lifecycle": "waiting",
                "decision": "place",
                "environment": "env|H100",
                "instance_type": instance_type,
                "configuration": {
                    "tp": tp,
                    "pp": 1,
                    "dp": dp,
                },
            }
        ]
    }
