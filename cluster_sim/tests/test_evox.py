"""Test the Store-backed EvoX scheduler adapter."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from evox import (
    EVOX_INITIAL_PROGRAM,
    EvoxDeadlineExceeded,
    EvoxPlannerDriver,
    OnlineEvoxEvaluator,
    OnlineSkyDiscoverEvoxClient,
    StoreSnapshot,
    build_evox_payload,
    build_evox_plan,
    build_placement_problems,
    capacity_limits,
    run_evox_discovery,
    select_evox_placements,
    verify_place_plan,
)
from llm_usage import LlmUsage
from outcome_feedback import fingerprint
from policy import load_config_policy


class EvoxPlannerDriverTest(unittest.TestCase):
    """Verify EvoX catalog context and canonical plan behavior."""

    def test_terminal_jobs_produce_no_actions(self) -> None:
        """Keep terminal jobs out of future placement plans."""
        driver = object.__new__(EvoxPlannerDriver)
        driver.benchmark_name = "test"
        driver.benchmark_job_ids = {}
        driver.config_policy = object()
        driver._select_placements = lambda problems, capacities: {}
        snapshot = StoreSnapshot({}, (), (), (), ({"job_id": "finished"},), {})

        self.assertEqual(driver._build_actions(snapshot), [])

    def test_no_winner_keeps_running_and_defers_remaining_jobs(self) -> None:
        """Emit a safe complete plan when EvoX produces no valid placement."""
        driver = object.__new__(EvoxPlannerDriver)
        driver.benchmark_name = "test"
        driver.benchmark_job_ids = {}
        driver.config_policy = load_config_policy()
        driver._select_placements = lambda problems, capacities: {}
        snapshot = StoreSnapshot(
            _resources(),
            ({"job_id": "running"},),
            (_store_job(),),
            ({"job_id": "paused"},),
            (),
            _catalogs(),
        )

        self.assertEqual(
            driver._build_actions(snapshot),
            [
                {"job_id": "running", "type": "keep"},
                {"job_id": "waiting", "type": "defer"},
            ],
        )

    def test_timeout_persists_one_safe_fallback_plan(self) -> None:
        """Persist keep/defer when EvoX reaches its deadline without a winner."""
        plans = []
        client = FallbackClient()
        driver = object.__new__(EvoxPlannerDriver)
        driver.user_id = "user"
        driver.benchmark_name = "test"
        driver.benchmark_job_ids = {}
        driver._job_is_batch = {}
        driver.backend = "fallback"
        driver.config_policy = load_config_policy()
        driver.client = client
        driver.plan_store = SimpleNamespace(create=plans.append)
        driver.feedback = SimpleNamespace(
            observe=lambda _logical_time: None,
            has_pending_batch=lambda _job_id: False,
            for_fingerprint=lambda _placement: None,
            record=lambda **_kwargs: None,
        )
        driver._read_store_snapshot = lambda: StoreSnapshot(
            _resources(),
            ({"job_id": "running"},),
            (_store_job(),),
            (),
            (),
            _catalogs(),
        )

        result = driver.create_plans(3, datetime(2026, 1, 1, tzinfo=UTC))

        self.assertEqual(len(plans), 1)
        self.assertEqual(
            [action.type for action in plans[0].actions],
            ["keep", "defer"],
        )
        self.assertEqual(len(result.plans), 1)
        self.assertTrue(result.timed_out)
        self.assertTrue(result.degraded)
        self.assertEqual(result.selection_source, "safe_fallback")

    def test_snapshot_reads_each_relevant_model_catalog_once(self) -> None:
        """Freeze waiting and running model catalogs once for one tick."""
        requested = []
        running = {"job": _store_job("running"), "ranks": [_active_rank("rank-active")]}
        driver = object.__new__(EvoxPlannerDriver)
        driver.user_id = "user"
        driver.job_store = SimpleNamespace(
            running_jobs=lambda _user_id: (running,),
            waiting_jobs=lambda _user_id: (_store_job(),),
            paused_jobs=lambda _user_id: (),
        )
        driver.resource_map_store = SimpleNamespace(
            get=lambda: SimpleNamespace(
                market="reserved",
                scheduling_summary=lambda: _resources(),
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
        self.assertEqual(snapshot.running_jobs[0]["job_id"], "running")

    def test_catalog_reaches_evolution_context_and_final_plan(self) -> None:
        """Use one resolved candidate profile for LLM context and Store output."""
        policy = load_config_policy()
        resources = _resources()
        problems = build_placement_problems(
            (_store_job(),),
            resources,
            "test",
            {},
            policy,
            _catalogs(),
        )
        capacities = capacity_limits(resources)
        payload = build_evox_payload(problems, capacities)
        candidate = problems[0].candidates[0]
        client = OnlineSkyDiscoverEvoxClient(
            model="model-test",
            iterations=2,
            output_dir=Path("outputs/test-evox"),
        )

        def discovery_side_effect(evaluator, *_args, **_kwargs):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "program.py"
                path.write_text(EVOX_INITIAL_PROGRAM, encoding="utf-8")
                evaluator(path)
            return SimpleNamespace(best_solution=EVOX_INITIAL_PROGRAM)

        with (
            patch("evox.multiprocessing.get_context", return_value=SyncContext()),
            patch(
                "evox.run_evox_discovery",
                side_effect=discovery_side_effect,
            ) as discovery,
        ):
            selected, usage = client.select(5, problems, capacities)

        self.assertEqual(selected, {"waiting": candidate})
        self.assertEqual(usage, ())
        self.assertEqual(discovery.call_args.kwargs["tick_number"], 5)
        self.assertEqual(discovery.call_args.kwargs["iterations"], 2)
        self.assertEqual(
            payload["model_catalog"],
            [
                {
                    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                    "gpu_type": "H100",
                    **_launch_config(),
                }
            ],
        )
        system_prompt = discovery.call_args.kwargs["system_prompt"]
        self.assertIn('"engine_version":"0.16.0"', system_prompt)

        plan = build_evox_plan(_store_job(), candidate, "fallback", policy)
        rank_config = plan["ladder"][0]["config"]
        self.assertNotIn("tenant_id", plan)
        self.assertNotIn("mechanism_id", plan)
        self.assertNotIn("budget_ref", plan)
        self.assertNotIn("rationale", plan)
        self.assertTrue(plan["ladder"][0]["rank_id"].startswith("rank_"))
        for field, value in _launch_config().items():
            self.assertEqual(rank_config[field], value)

    def test_data_parallelism_uses_per_replica_rank_shapes(self) -> None:
        """Materialize DP as replicas while keeping rank GPU shape canonical."""
        policy = load_config_policy()
        resources = _resources(free=3)
        problem = build_placement_problems(
            (_store_job(),),
            resources,
            "test",
            {},
            policy,
            _catalogs(),
        )[0]
        candidate = next(
            value
            for value in problem.candidates
            if value.tp == 1 and value.pp == 1 and value.dp == 3
        )

        plan = build_evox_plan(_store_job(), candidate, "fallback", policy)

        rank = plan["ladder"][0]
        self.assertEqual(candidate.gpu_count, 1)
        self.assertEqual(candidate.footprint, 3)
        self.assertEqual(rank["n_replicas"], 3)
        self.assertEqual(rank["config"]["gpu_count"], 1)
        self.assertEqual(rank["config"]["count"], 1)
        verify_place_plan(plan, candidate, policy)

    def test_v3_payload_maps_compact_decisions_to_place_and_swap(self) -> None:
        """Map candidate-or-null policy output according to each job lifecycle."""
        policy = load_config_policy()
        resources = _resources()
        running = _running_store_job()
        problems = build_placement_problems(
            (_store_job(),),
            resources,
            "test",
            {},
            policy,
            _catalogs(),
            running_jobs=(running,),
        )
        capacities = capacity_limits(resources)
        waiting = next(problem for problem in problems if problem.job_id == "waiting")
        running_problem = next(
            problem for problem in problems if problem.job_id == "running"
        )
        waiting_candidate = next(
            candidate
            for candidate in waiting.candidates
            if candidate.tp == 1 and candidate.pp == 1 and candidate.dp == 1
        )
        running_candidate = next(
            candidate
            for candidate in running_problem.candidates
            if candidate.tp == 1 and candidate.pp == 1 and candidate.dp == 2
        )
        payload = build_evox_payload(problems, capacities)

        self.assertEqual(payload["schema_version"], "tandemn-evox-placement/v3")
        jobs = {job["job_id"]: job for job in payload["jobs"]}
        self.assertEqual(jobs["waiting"]["lifecycle"], "waiting")
        self.assertIsNone(jobs["waiting"]["current_deployment"])
        self.assertEqual(jobs["running"]["lifecycle"], "running")
        self.assertEqual(jobs["running"]["active_rank_id"], "rank-active")
        self.assertEqual(
            jobs["running"]["current_deployment"]["environment"],
            "reserved|aws|us-east|use1-az1|H100",
        )

        selected = select_evox_placements(
            lambda _payload: {
                "placements": [
                    {
                        "job_id": "waiting",
                        "candidate_id": waiting_candidate.candidate_id,
                    },
                    {
                        "job_id": "running",
                        "candidate_id": running_candidate.candidate_id,
                    },
                ]
            },
            problems,
            capacities,
        )
        selected_ids = {
            job_id: candidate.candidate_id for job_id, candidate in selected.items()
        }
        driver = _driver(policy)
        driver._select_placements = lambda runtime_problems, _capacities: {
            problem.job_id: next(
                candidate
                for candidate in problem.candidates
                if candidate.candidate_id == selected_ids[problem.job_id]
            )
            for problem in runtime_problems
        }

        actions = driver._build_actions(
            StoreSnapshot(
                resources=resources,
                running_jobs=(running,),
                waiting_jobs=(_store_job(),),
                paused_jobs=(),
                finished_jobs=(),
                model_catalogs=_catalogs(),
            )
        )

        self.assertEqual([action["type"] for action in actions], ["swap", "place"])
        self.assertEqual(actions[0]["ladder"][0]["rank_id"], "rank-active")
        self.assertEqual(actions[0]["ladder"][0]["n_replicas"], 2)

    def test_null_running_decision_keeps_active_rank(self) -> None:
        """Treat a null candidate for an eligible running job as keep."""
        policy = load_config_policy()
        resources = _resources()
        running = _running_store_job()
        problems = build_placement_problems(
            (),
            resources,
            "test",
            {},
            policy,
            _catalogs(),
            running_jobs=(running,),
        )
        selected = select_evox_placements(
            lambda _payload: {
                "placements": [{"job_id": "running", "candidate_id": None}]
            },
            problems,
            capacity_limits(resources),
        )
        driver = _driver(policy)
        driver._select_placements = lambda _problems, _capacities: selected

        actions = driver._build_actions(
            StoreSnapshot(
                resources=resources,
                running_jobs=(running,),
                waiting_jobs=(),
                paused_jobs=(),
                finished_jobs=(),
                model_catalogs=_catalogs(),
            )
        )

        self.assertEqual(actions, [{"job_id": "running", "type": "keep"}])

    def test_noop_and_multi_rank_running_jobs_are_not_swap_candidates(self) -> None:
        """Refuse no-op and non-single-rank changes while retaining running jobs."""
        policy = load_config_policy()
        unchanged = _running_store_job("unchanged")
        multiple_ranks = _running_store_job(
            "multiple",
            extra_ranks=(_active_rank("rank-second"),),
        )
        problems = build_placement_problems(
            (),
            _resources(free=1),
            "test",
            {},
            policy,
            _catalogs(),
            running_jobs=(unchanged, multiple_ranks),
        )

        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0].job_id, "unchanged")
        self.assertEqual(problems[0].action_type, "swap")
        self.assertEqual(problems[0].candidates, ())

        driver = _driver(policy)
        driver._select_placements = lambda _problems, _capacities: {}
        actions = driver._build_actions(
            StoreSnapshot(
                resources=_resources(free=1),
                running_jobs=(unchanged, multiple_ranks),
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
                {"job_id": "multiple", "type": "keep"},
            ],
        )

    def test_v3_payload_serializes_feedback_for_exact_candidate(self) -> None:
        """Expose a known outcome only on its matching placement candidate."""
        policy = load_config_policy()
        resources = _resources()
        problem = build_placement_problems(
            (_store_job(),),
            resources,
            "test",
            {},
            policy,
            _catalogs(),
        )[0]
        candidate = problem.candidates[0]
        outcome = {"reward": 0.75, "sample_count": 3}
        feedback = {
            _candidate_fingerprint(problem, candidate): outcome,
        }

        payload = build_evox_payload(
            (problem,),
            capacity_limits(resources),
            feedback,
        )

        serialized = next(
            value
            for value in payload["jobs"][0]["candidates"]
            if value["candidate_id"] == candidate.candidate_id
        )
        self.assertEqual(payload["schema_version"], "tandemn-evox-placement/v3")
        self.assertEqual(serialized["feedback"], outcome)

    def test_evaluator_uses_observed_reward_after_priority_admission(self) -> None:
        """Prefer an observed positive outcome over an otherwise cheaper shape."""
        policy = load_config_policy()
        resources = _resources()
        problem = build_placement_problems(
            (_store_job(),),
            resources,
            "test",
            {},
            policy,
            _catalogs(),
        )[0]
        lower_footprint = next(
            candidate for candidate in problem.candidates if candidate.dp == 1
        )
        rewarded = next(
            candidate for candidate in problem.candidates if candidate.dp == 2
        )
        feedback = {
            _candidate_fingerprint(problem, rewarded): {"reward": 0.8},
        }
        evaluator = OnlineEvoxEvaluator(
            (problem,),
            capacity_limits(resources),
            feedback,
        )

        with tempfile.TemporaryDirectory() as directory:
            lower_path = Path(directory) / "lower.py"
            rewarded_path = Path(directory) / "rewarded.py"
            lower_path.write_text(
                _program_for_candidate("waiting", lower_footprint.candidate_id),
                encoding="utf-8",
            )
            rewarded_path.write_text(
                _program_for_candidate("waiting", rewarded.candidate_id),
                encoding="utf-8",
            )
            lower_metrics = evaluator.evaluate(str(lower_path))
            rewarded_metrics = evaluator.evaluate(str(rewarded_path))

        self.assertGreater(
            rewarded_metrics["combined_score"],
            lower_metrics["combined_score"],
        )
        self.assertEqual(rewarded_metrics["mean_observed_reward"], 0.8)
        self.assertEqual(rewarded_metrics["observed_feedback_count"], 1.0)
        self.assertEqual(rewarded_metrics["feedback"], {"waiting": {"reward": 0.8}})

    def test_pending_batch_running_job_keeps_without_swap_candidates(self) -> None:
        """Do not replace a batch rank before its terminal outcome is observed."""
        policy = load_config_policy()
        running = _running_store_job()
        waiting = _store_job("waiting-batch")
        running["kind"] = "batch"
        waiting["kind"] = "batch"
        driver = _driver(policy)
        driver.feedback = SimpleNamespace(
            has_pending_batch=lambda job_id: job_id == "running",
        )
        seen = []

        def select(problems, _capacities):
            seen.extend(problems)
            return {}

        driver._select_placements = select

        actions = driver._build_actions(
            StoreSnapshot(
                resources=_resources(),
                running_jobs=(running,),
                waiting_jobs=(waiting,),
                paused_jobs=(),
                finished_jobs=(),
                model_catalogs=_catalogs(),
            )
        )

        self.assertEqual([problem.job_id for problem in seen], ["waiting-batch"])
        self.assertEqual(seen[0].action_type, "place")
        self.assertTrue(seen[0].candidates)
        self.assertEqual(
            actions,
            [
                {"job_id": "running", "type": "keep"},
                {"job_id": "waiting-batch", "type": "defer"},
            ],
        )

    def test_final_placement_records_outcome_feedback(self) -> None:
        """Record the exact verified rank configuration after a place action."""
        policy = load_config_policy()
        resources = _resources()
        driver = _driver(policy)
        records = []
        driver.feedback = SimpleNamespace(
            has_pending_batch=lambda _job_id: False,
            record=lambda **kwargs: records.append(kwargs),
        )

        def select(problems, _capacities):
            problem = problems[0]
            candidate = next(value for value in problem.candidates if value.dp == 1)
            return {problem.job_id: candidate}

        driver._select_placements = select

        actions = driver._build_actions(
            StoreSnapshot(
                resources=resources,
                running_jobs=(),
                waiting_jobs=(_store_job(),),
                paused_jobs=(),
                finished_jobs=(),
                model_catalogs=_catalogs(),
            )
        )

        self.assertEqual(actions[0]["type"], "place")
        driver._record_action_feedback(
            actions,
            datetime(2026, 1, 1, tzinfo=UTC),
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["fingerprint"].job_id, "waiting")
        self.assertEqual(record["rank_id"], actions[0]["ladder"][0]["rank_id"])
        self.assertEqual(record["window_start"], datetime(2026, 1, 1, tzinfo=UTC))
        self.assertFalse(record["is_batch"])
        self.assertIsNone(record["required_tps"])
        self.assertEqual(record["ttft_target_ms"], 700.0)
        self.assertEqual(record["tpot_target_ms"], 55.0)

    def test_timeout_without_valid_program_returns_no_placements(self) -> None:
        """Let the driver keep running jobs and defer the rest after timeout."""
        policy = load_config_policy()
        resources = _resources()
        problems = build_placement_problems(
            (_store_job(),), resources, "test", {}, policy, _catalogs()
        )
        capacities = capacity_limits(resources)
        usage = LlmUsage(5, 60, 15, "model-used")
        client = OnlineSkyDiscoverEvoxClient(
            model="model-test",
            iterations=2,
            output_dir=Path("outputs/test-evox"),
        )

        with (
            patch("evox.multiprocessing.get_context", return_value=SyncContext()),
            patch(
                "evox.run_evox_discovery",
                side_effect=EvoxDeadlineExceeded((usage,)),
            ),
        ):
            selected, actual_usage = client.select(5, problems, capacities)

        self.assertEqual(selected, {})
        self.assertEqual(actual_usage, ())
        self.assertTrue(client.timed_out)
        self.assertEqual(client.initial_program, EVOX_INITIAL_PROGRAM)
        self.assertEqual(client.tick_number, 1)

    def test_timeout_uses_best_valid_program_from_current_tick(self) -> None:
        """Keep a validated same-tick program when discovery reaches its deadline."""
        policy = load_config_policy()
        resources = _resources()
        problems = build_placement_problems(
            (_store_job(),), resources, "test", {}, policy, _catalogs()
        )
        capacities = capacity_limits(resources)
        client = OnlineSkyDiscoverEvoxClient("model-test", 2, Path("outputs/test-evox"))
        evolved = EVOX_INITIAL_PROGRAM.replace(
            "chosen = {}", "chosen = {}\n    # validated same-tick winner"
        )

        def timeout_after_valid(evaluator, *_args, **_kwargs):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "program.py"
                path.write_text(evolved, encoding="utf-8")
                evaluator(path)
            raise EvoxDeadlineExceeded

        with (
            patch("evox.multiprocessing.get_context", return_value=SyncContext()),
            patch("evox.run_evox_discovery", side_effect=timeout_after_valid),
        ):
            selected, _ = client.select(5, problems, capacities)

        self.assertEqual(selected, {"waiting": problems[0].candidates[0]})
        self.assertEqual(client.initial_program, evolved)
        self.assertTrue(client.timed_out)
        self.assertEqual(client.selection_source, "evox_best_so_far")

    def test_hard_deadline_uses_published_best_and_kills_worker(self) -> None:
        """Return a validated candidate without waiting for a hung child."""
        policy = load_config_policy()
        resources = _resources()
        problems = build_placement_problems(
            (_store_job(),), resources, "test", {}, policy, _catalogs()
        )
        capacities = capacity_limits(resources)
        context = HangingContext(problems[0].candidates[0])
        client = OnlineSkyDiscoverEvoxClient("model-test", 8, Path("outputs/test"))

        with patch("evox.multiprocessing.get_context", return_value=context):
            selected, _ = client.select(
                0,
                problems,
                capacities,
                deadline=monotonic() + 20.01,
            )

        self.assertEqual(selected, {"waiting": problems[0].candidates[0]})
        self.assertTrue(context.process.killed)
        self.assertTrue(client.timed_out)
        self.assertEqual(client.selection_source, "evox_best_so_far")
        self.assertIsNone(client.error)

    def test_discovery_enforces_total_and_request_timeouts(self) -> None:
        """Interrupt discovery at its wall budget while allowing long LLM calls."""
        captured = {}

        class Config:
            @staticmethod
            def from_dict(value):
                captured.update(value)
                return SimpleNamespace(llm=SimpleNamespace(models=[]))

        class LLMPool:
            async def check_availability(self):
                return True

        def run_discovery(**_kwargs):
            try:
                time.sleep(1)
            except Exception:
                return SimpleNamespace(best_solution=EVOX_INITIAL_PROGRAM)

        modules = {
            "skydiscover": SimpleNamespace(run_discovery=run_discovery),
            "skydiscover.config": SimpleNamespace(Config=Config),
            "skydiscover.llm.llm_pool": SimpleNamespace(LLMPool=LLMPool),
            "skydiscover.llm.openai": SimpleNamespace(OpenAILLM=object),
        }
        with patch.dict(sys.modules, modules):
            with self.assertRaises(EvoxDeadlineExceeded):
                run_evox_discovery(
                    lambda _path: {},
                    Path("outputs/test-evox"),
                    time_budget_seconds=0.01,
                )

        self.assertEqual(captured["llm"]["timeout"], 250)


def _driver(policy) -> EvoxPlannerDriver:
    """Build an EvoX driver with only the fields used by action construction."""
    driver = object.__new__(EvoxPlannerDriver)
    driver.benchmark_name = "test"
    driver.benchmark_job_ids = {}
    driver.backend = "fallback"
    driver.config_policy = policy
    return driver


class FallbackClient:
    timed_out = True
    selection_source = "safe_fallback"
    completed_candidates = 0
    error = None

    def select(self, *_args, **_kwargs):
        return {}, ()


class PipeState:
    """Hold process messages for synchronous worker tests."""

    def __init__(self) -> None:
        self.messages: deque[tuple[object, ...]] = deque()


class ReceivePipe:
    def __init__(self, state: PipeState) -> None:
        self.state = state

    def poll(self, _timeout: float = 0.0) -> bool:
        return bool(self.state.messages)

    def recv(self) -> tuple[object, ...]:
        return self.state.messages.popleft()

    def close(self) -> None:
        pass


class SendPipe:
    def __init__(self, state: PipeState) -> None:
        self.state = state

    def send(self, message: tuple[object, ...]) -> None:
        self.state.messages.append(message)

    def close(self) -> None:
        pass


class SyncProcess:
    """Run one process target inline for protocol tests."""

    def __init__(self, target, args) -> None:
        self.target = target
        self.args = args

    def start(self) -> None:
        self.target(*self.args)

    def is_alive(self) -> bool:
        return False

    def join(self) -> None:
        pass

    def kill(self) -> None:
        pass


class SyncContext:
    def Pipe(self, *, duplex: bool) -> tuple[ReceivePipe, SendPipe]:
        assert not duplex
        state = PipeState()
        return ReceivePipe(state), SendPipe(state)

    def Process(self, target, args) -> SyncProcess:
        return SyncProcess(target, args)


class HangingProcess:
    """Publish one best candidate and remain alive until killed."""

    def __init__(self, args, candidate) -> None:
        self.messages = args[3]
        self.candidate = candidate
        self.killed = False

    def start(self) -> None:
        self.messages.send(
            (
                "best",
                1.0,
                EVOX_INITIAL_PROGRAM,
                {self.candidate.job_id: self.candidate.candidate_id},
                1,
            )
        )

    def is_alive(self) -> bool:
        return not self.killed

    def join(self) -> None:
        pass

    def kill(self) -> None:
        self.killed = True


class HangingContext(SyncContext):
    def __init__(self, candidate) -> None:
        self.candidate = candidate
        self.process = None

    def Process(self, target, args) -> HangingProcess:
        del target
        self.process = HangingProcess(args, self.candidate)
        return self.process


def _store_job(job_id: str = "waiting") -> dict[str, object]:
    """Return one Store job with explicit placement demand."""
    return {
        "job_id": job_id,
        "user_id": "tenant",
        "kind": "online",
        "spec_json": {
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
            "job_features": {
                "required_vram_gib": 1.0,
                "target_p99_ttft_ms": 700,
                "target_p99_tpot_ms": 55,
            },
        },
    }


def _resources(free: int = 8) -> dict[str, dict[str, object]]:
    """Return one available exact-GPU Store resource target."""
    return {
        "reserved|aws|us-east|use1-az1|H100": {
            "gpu_type": "H100",
            "gpu_memory_gib": 80,
            "total": 8,
            "free": free,
            "pools": [],
        }
    }


def _active_rank(
    rank_id: str,
    *,
    count: int = 1,
    replicas: int = 1,
) -> dict[str, object]:
    """Return one active rank shape for a running-job lifecycle fixture."""
    return {
        "rank_id": rank_id,
        "n_replicas": replicas,
        "shape_json": {
            "env": ["reserved", "aws", "us-east", "use1-az1", "H100"],
            "instance_type": None,
            "gpu_type": "H100",
            "gpu_count": count,
            "count": count,
            "tp": 1,
            "pp": count,
            "dp": replicas,
        },
    }


def _running_store_job(
    job_id: str = "running",
    *,
    extra_ranks: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    """Return one flattened running Store job with its active rank payload."""
    return {
        **_store_job(job_id),
        "active_ranks": [_active_rank("rank-active"), *extra_ranks],
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
    """Return one raw Store catalog with exact H100 profile lists."""
    catalog = _launch_config()
    for field in (
        "max_num_seq",
        "max_num_batched_tokens",
        "block_size",
        "kvcache_dtype",
    ):
        catalog[field] = [{"gpu_type": "H100", "value": catalog[field]}]
    return {"meta-llama/Llama-3.2-1B-Instruct": catalog}


def _candidate_fingerprint(problem, candidate):
    """Return the feedback identity for one fixture candidate."""
    return fingerprint(
        job_id=problem.job_id,
        model_id=problem.model_id,
        environment=candidate.target.environment_key,
        instance_type=candidate.target.instance_type,
        gpu_type=candidate.target.gpu_type,
        tp=candidate.tp,
        pp=candidate.pp,
        dp=candidate.dp,
    )


def _program_for_candidate(job_id: str, candidate_id: str) -> str:
    """Return a minimal valid evolved policy selecting one candidate."""
    return (
        "def compute_placement(payload):\n"
        "    return {\n"
        "        'placements': [\n"
        f"            {{'job_id': {job_id!r}, 'candidate_id': {candidate_id!r}}}\n"
        "        ]\n"
        "    }\n"
    )


if __name__ == "__main__":
    unittest.main()
