"""Test contextual-bandit terminal-job handling."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from contextual_bandit import (
    ContextualBanditPlannerDriver,
    PlacementCandidate,
    StoreSnapshot,
    _features,
    _PendingObservation,
    _target_memory_gib,
    build_contextual_bandit_plan,
    build_placement_problems,
    capacity_limits,
    enumerate_candidates,
    verify_place_plan,
)
from policy import ConfigPolicy, ConfigPolicyRule, load_config_policy
from vram import ResourceTarget

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
RUNNING_MODEL_ID = "running-model"
_H100_ENVIRONMENT = "reserved|aws|us-east|use1-az1|H100"


class ContextualBanditPlannerDriverTest(unittest.TestCase):
    """Verify Contextual Bandit omits terminal jobs from later plans."""

    def test_canonical_hardware_and_pp_limit_reach_candidate_enumeration(
        self,
    ) -> None:
        """Resolve benchmark GPU names and discard a PP-five candidate."""
        expected_memory = {
            "H100": 80,
            "A100-80GB": 80,
            "A100-40GB": 40,
            "A10G": 24,
            "L40S": 48,
            "MI300": 192,
            "L4": 24,
            "RTXPRO6000": 96,
        }
        for hardware, memory_gib in expected_memory.items():
            target = ResourceTarget(
                environment_key=f"reserved|test|region|zone|{hardware}",
                environment={"free": 8},
                pool=None,
                gpu_type=hardware,
                instance_type=None,
            )
            with self.subTest(hardware=hardware):
                self.assertEqual(_target_memory_gib(target), memory_gib)

        resources = {
            "reserved|test|region|zone|H100": {
                "gpu_type": "H100",
                "free": 8,
                "pools": [],
            }
        }
        policy = ConfigPolicy({("h100", MODEL_ID): ConfigPolicyRule("bf16", (1,))})
        self.assertEqual(
            enumerate_candidates(
                "job",
                MODEL_ID,
                {},
                400,
                resources,
                policy,
            ),
            (),
        )

    def test_terminal_jobs_produce_no_actions(self) -> None:
        """Keep terminal jobs out of future placement plans."""
        driver = object.__new__(ContextualBanditPlannerDriver)
        driver.benchmark_name = "test"
        driver.benchmark_job_ids = {}
        driver.config_policy = object()
        driver._select_placements = lambda problems, capacities: {}
        snapshot = StoreSnapshot(
            resources={},
            running_jobs=(),
            waiting_jobs=(),
            paused_jobs=(),
            finished_jobs=({"job_id": "finished"},),
            model_catalogs={},
        )

        self.assertEqual(driver._build_actions(snapshot), [])

    def test_snapshot_reads_each_relevant_model_catalog_once(self) -> None:
        """Freeze distinct Store catalog records with the planning tick."""
        driver = object.__new__(ContextualBanditPlannerDriver)
        driver.user_id = "user"
        waiting = (_store_job("first"), _store_job("second"))
        requested = []
        driver.job_store = SimpleNamespace(
            running_jobs=lambda _user_id: (),
            waiting_jobs=lambda _user_id: waiting,
            paused_jobs=lambda _user_id: (),
        )
        driver.resource_map_store = SimpleNamespace(
            get=lambda: SimpleNamespace(
                market="reserved",
                scheduling_summary=lambda: {},
            )
        )
        driver.model_catalog_store = SimpleNamespace(
            get=lambda model_id: (
                requested.append(model_id)
                or SimpleNamespace(model_dump=lambda **_kwargs: _catalogs()[model_id])
            )
        )

        snapshot = driver._read_store_snapshot()

        self.assertEqual(requested, [MODEL_ID])
        self.assertEqual(snapshot.model_catalogs, _catalogs())

    def test_snapshot_reads_running_job_model_catalogs(self) -> None:
        """Freeze catalogs for both waiting placements and running-job swaps."""
        driver = object.__new__(ContextualBanditPlannerDriver)
        driver.user_id = "user"
        catalogs = {
            MODEL_ID: _catalogs()[MODEL_ID],
            RUNNING_MODEL_ID: _catalogs()[MODEL_ID],
        }
        requested = []
        driver.job_store = SimpleNamespace(
            running_jobs=lambda _user_id: (
                {
                    "job": _store_job("running", RUNNING_MODEL_ID),
                    "ranks": [_active_rank("rank-running")],
                },
            ),
            waiting_jobs=lambda _user_id: (_store_job("waiting"),),
            paused_jobs=lambda _user_id: (),
        )
        driver.resource_map_store = SimpleNamespace(
            get=lambda: SimpleNamespace(
                market="reserved",
                scheduling_summary=lambda: {},
            )
        )
        driver.model_catalog_store = SimpleNamespace(
            get=lambda model_id: (
                requested.append(model_id)
                or SimpleNamespace(model_dump=lambda **_kwargs: catalogs[model_id])
            )
        )

        snapshot = driver._read_store_snapshot()

        self.assertEqual(requested, [MODEL_ID, RUNNING_MODEL_ID])
        self.assertEqual(snapshot.model_catalogs, catalogs)

    def test_selected_candidate_uses_exact_gpu_catalog_profile(self) -> None:
        """Materialize Store launch fields only after bandit selection."""
        candidate = PlacementCandidate(
            candidate_id="candidate",
            job_id="job",
            target=ResourceTarget(
                environment_key="reserved|aws|us-east|use1-az1|L40S",
                environment={},
                pool=None,
                gpu_type="L40S",
                instance_type=None,
            ),
            tp=1,
            pp=1,
            dp=1,
            footprint=1,
            memory_gib=48.0,
            required_vram_gib=1.0,
            model_weight_gib=1.0,
            kv_demand=0.0,
            price_per_hour=1.0,
        )
        catalogs = _catalogs()
        policy = load_config_policy()

        plan = build_contextual_bandit_plan(
            _store_job("job"),
            candidate,
            catalogs,
            policy,
        )

        config = plan["ladder"][0]["config"]
        self.assertEqual(config["max_num_seq"], 256)
        self.assertEqual(config["count"], 1)
        self.assertNotIn("tenant_id", plan)
        self.assertNotIn("mechanism_id", plan)
        self.assertNotIn("budget_ref", plan)
        self.assertNotIn("rationale", plan)
        self.assertTrue(plan["ladder"][0]["rank_id"].startswith("rank_"))
        for field, value in _launch_config().items():
            self.assertEqual(config[field], value)
        verify_place_plan(plan, candidate, catalogs, policy)

    def test_data_parallel_candidates_use_per_replica_ladder_shape(self) -> None:
        """Represent DP as replicas while reserving the total GPU footprint."""
        policy = ConfigPolicy({("h100", MODEL_ID): ConfigPolicyRule("bf16", (1,))})
        candidates = enumerate_candidates(
            "job",
            MODEL_ID,
            {},
            1,
            _resources(3),
            policy,
        )

        self.assertEqual(
            [
                (candidate.dp, candidate.gpu_count, candidate.footprint)
                for candidate in candidates
            ],
            [(1, 1, 1), (2, 1, 2), (3, 1, 3)],
        )
        candidate = candidates[-1]
        plan = build_contextual_bandit_plan(
            _store_job("job"),
            candidate,
            _catalogs(),
            policy,
        )

        rank = plan["ladder"][0]
        self.assertEqual(rank["n_replicas"], 3)
        self.assertEqual(rank["config"]["gpu_count"], 1)
        self.assertEqual(rank["config"]["count"], 1)
        verify_place_plan(plan, candidate, _catalogs(), policy)

    def test_selected_running_candidate_swaps_with_active_rank_id(self) -> None:
        """Keep the active rank identity and selected context through a swap."""
        now = datetime(2026, 1, 1, tzinfo=UTC)
        policy = ConfigPolicy({("h100", MODEL_ID): ConfigPolicyRule("bf16", (1,))})
        running = _running_job("job", "rank-active")
        snapshot = StoreSnapshot(
            resources=_resources(2),
            running_jobs=(running,),
            waiting_jobs=(),
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=_catalogs(),
        )
        driver = _planning_driver(policy, now)
        driver._sample_weight = lambda: (0.0,) * 11

        actions = driver._build_actions(snapshot)

        self.assertEqual(actions[0]["type"], "swap")
        self.assertEqual(actions[0]["ladder"][0]["rank_id"], "rank-active")
        self.assertEqual(actions[0]["ladder"][0]["n_replicas"], 2)
        self.assertEqual(driver._pending["job"].rank_id, "rank-active")
        problem = build_placement_problems(
            (),
            _resources(2),
            "test",
            {},
            policy,
            running_jobs=(running,),
        )[0]
        candidate = next(
            candidate for candidate in problem.candidates if candidate.dp == 2
        )
        self.assertEqual(
            driver._pending["job"].features,
            _features(
                problem,
                candidate,
                capacity_limits(_resources(2)),
                1,
            ),
        )

    def test_unselected_running_candidate_keeps_current_rank(self) -> None:
        """Use keep rather than defer when a running candidate is not selected."""
        policy = ConfigPolicy({("h100", MODEL_ID): ConfigPolicyRule("bf16", (1,))})
        running = _running_job(
            "job",
            "rank-active",
            count=2,
            tp=1,
            pp=2,
        )
        driver = _planning_driver(policy, datetime(2026, 1, 1, tzinfo=UTC))
        driver._select_placements = lambda _problems, _capacities: {}

        actions = driver._build_actions(
            StoreSnapshot(
                resources=_resources(1),
                running_jobs=(running,),
                waiting_jobs=(),
                paused_jobs=(),
                finished_jobs=(),
                model_catalogs=_catalogs(),
            )
        )

        self.assertEqual(actions, [{"job_id": "job", "type": "keep"}])

    def test_unchanged_and_non_single_running_ranks_keep(self) -> None:
        """Skip no-op swaps and leave no-rank or multi-rank jobs untouched."""
        policy = ConfigPolicy({("h100", MODEL_ID): ConfigPolicyRule("bf16", (1,))})
        unchanged = _running_job("unchanged", "rank-unchanged")
        no_rank = _store_job("no-rank")
        multiple_ranks = _running_job(
            "multiple",
            "rank-first",
            extra_ranks=(_active_rank("rank-second"),),
        )
        problems = build_placement_problems(
            (),
            _resources(1),
            "test",
            {},
            policy,
            running_jobs=(unchanged, no_rank, multiple_ranks),
        )

        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0].job_id, "unchanged")
        self.assertEqual(problems[0].action_type, "swap")
        self.assertEqual(problems[0].candidates, ())
        driver = _planning_driver(policy, datetime(2026, 1, 1, tzinfo=UTC))
        driver._select_placements = lambda _problems, _capacities: {}
        actions = driver._build_actions(
            StoreSnapshot(
                resources=_resources(1),
                running_jobs=(unchanged, no_rank, multiple_ranks),
                waiting_jobs=(),
                paused_jobs=(),
                finished_jobs=(),
                model_catalogs=_catalogs(),
            )
        )

        self.assertEqual(
            actions,
            [
                {"job_id": "unchanged", "type": "keep"},
                {"job_id": "no-rank", "type": "keep"},
                {"job_id": "multiple", "type": "keep"},
            ],
        )

    def test_pending_rank_id_matches_the_placement_plan(self) -> None:
        """Query Store telemetry using the rank identity Orca materializes."""
        candidate = _candidate()
        driver = _reward_driver()
        driver.config_policy = load_config_policy()
        driver._pending["job"] = _observation(
            datetime(2026, 1, 1, tzinfo=UTC),
        )
        problem = SimpleNamespace(store_job=_store_job("job"))

        plan = driver._build_place_plan(problem, candidate, _catalogs())

        self.assertEqual(
            plan["ladder"][0]["rank_id"],
            driver._pending["job"].rank_id,
        )

    def test_online_reward_uses_store_aggregate_metrics(self) -> None:
        """Learn one weighted online reward from public rank metrics."""
        driver = _reward_driver()
        now = datetime(2026, 1, 1, tzinfo=UTC)
        driver._pending["job"] = _observation(now)
        driver.event_log = SimpleNamespace(read_after=lambda *_args, **_kwargs: ())
        driver._event_cursor = None
        driver.metric_store = SimpleNamespace(
            rows_for_rank_window=lambda *_args: (
                SimpleNamespace(
                    ts=now,
                    p99_ttft_ms=250.0,
                    p99_tpot_ms=25.0,
                    throughput_token_per_sec=150.0,
                ),
            )
        )

        driver._observe_store_results(now)

        self.assertEqual(driver.observation_count, 1)
        self.assertAlmostEqual(driver.last_reward, 0.5)
        self.assertNotIn("job", driver._pending)

    def test_batch_reward_prioritizes_deadline(self) -> None:
        """Keep deadline success positive and deadline misses negative."""
        now = datetime(2026, 1, 1, tzinfo=UTC)
        driver = _reward_driver()
        driver._pending["met"] = _observation(now, is_batch=True)
        driver.observe_results(
            "met",
            p99_ttft_ms=None,
            p99_tpot_ms=None,
            throughput_token_per_sec=100.0,
            batch_deadline_met=True,
        )
        self.assertAlmostEqual(driver.last_reward, 0.6)

        driver._pending["missed"] = _observation(now, is_batch=True)
        driver.observe_results(
            "missed",
            p99_ttft_ms=None,
            p99_tpot_ms=None,
            throughput_token_per_sec=100.0,
            batch_deadline_met=False,
        )
        self.assertAlmostEqual(driver.last_reward, -0.6)


def _store_job(
    job_id: str,
    model_id: str = MODEL_ID,
) -> dict[str, object]:
    """Create one canonical waiting Store job."""
    return {
        "job_id": job_id,
        "user_id": "tenant",
        "spec_json": {"model_id": model_id, "required_vram_gib": 1},
    }


def _resources(free: int) -> dict[str, dict[str, object]]:
    """Create one H100 resource target with the requested free capacity."""
    return {
        _H100_ENVIRONMENT: {
            "gpu_type": "H100",
            "free": free,
            "pools": [],
        }
    }


def _active_rank(
    rank_id: str,
    *,
    count: int = 1,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
) -> dict[str, object]:
    """Create one canonical active rank shape for a running Store job."""
    return {
        "rank_id": rank_id,
        "n_replicas": dp,
        "shape_json": {
            "env": _H100_ENVIRONMENT.split("|"),
            "instance_type": None,
            "gpu_type": "H100",
            "gpu_count": count,
            "count": count,
            "tp": tp,
            "pp": pp,
            "dp": dp,
        },
    }


def _running_job(
    job_id: str,
    rank_id: str,
    model_id: str = MODEL_ID,
    *,
    count: int = 1,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
    extra_ranks: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    """Create one Store running job with canonical active-rank payloads."""
    return {
        **_store_job(job_id, model_id),
        "active_ranks": [
            _active_rank(rank_id, count=count, tp=tp, pp=pp, dp=dp),
            *extra_ranks,
        ],
    }


def _planning_driver(
    policy: ConfigPolicy,
    feedback_start: datetime,
) -> ContextualBanditPlannerDriver:
    """Build only the bandit state required to construct actions."""
    driver = object.__new__(ContextualBanditPlannerDriver)
    driver.benchmark_name = "test"
    driver.benchmark_job_ids = {}
    driver.config_policy = policy
    driver._pending = {}
    driver._feedback_start = feedback_start
    return driver


def _reward_driver() -> ContextualBanditPlannerDriver:
    """Build only the Thompson state needed for reward observations."""
    driver = object.__new__(ContextualBanditPlannerDriver)
    driver._precision = [
        [1.0 if row == column else 0.0 for column in range(11)] for row in range(11)
    ]
    driver._reward_weight = [0.0] * 11
    driver._pending = {}
    driver.user_id = "user"
    driver.observation_count = 0
    driver.last_reward = None
    return driver


def _observation(
    now: datetime,
    *,
    is_batch: bool = False,
) -> _PendingObservation:
    """Return one public action context with unit throughput target."""
    return _PendingObservation(
        features=(1.0,) * 11,
        rank_id="rank_test",
        window_start=now,
        required_tps=100.0,
        is_batch=is_batch,
        ttft_target_ms=500.0,
        tpot_target_ms=50.0,
    )


def _launch_config() -> dict[str, object]:
    """Return one resolved launch profile from Tandemn Store."""
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


def _candidate() -> PlacementCandidate:
    """Return one policy-compliant L40S candidate."""
    return PlacementCandidate(
        candidate_id="candidate",
        job_id="job",
        target=ResourceTarget(
            environment_key="reserved|aws|us-east|use1-az1|L40S",
            environment={},
            pool=None,
            gpu_type="L40S",
            instance_type=None,
        ),
        tp=1,
        pp=1,
        dp=1,
        footprint=1,
        memory_gib=48.0,
        required_vram_gib=1.0,
        model_weight_gib=1.0,
        kv_demand=0.0,
        price_per_hour=1.0,
    )


def _catalogs() -> dict[str, dict[str, object]]:
    """Return one raw Store catalog with target-specific GPU profiles."""
    catalog = _launch_config()
    for field in (
        "max_num_seq",
        "max_num_batched_tokens",
        "block_size",
        "kvcache_dtype",
    ):
        catalog[field] = [
            {"gpu_type": "H100", "value": "not-selected"},
            {"gpu_type": "L40S", "value": catalog[field]},
        ]
    return {MODEL_ID: catalog}


if __name__ == "__main__":
    unittest.main()
