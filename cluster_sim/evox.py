"""Evolve placement programs online with EvoX for each control tick."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import multiprocessing
import os
import signal
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from types import ModuleType, SimpleNamespace
from typing import Any, Protocol, cast

from llm_usage import LlmUsage, OpenAIUsageClient
from models import BatchJob, CreatedPlan, PlannerResult, Scenario
from outcome_feedback import OutcomeFeedback, PlacementFingerprint, fingerprint
from policy import ConfigPolicy, load_config_policy
from rank_ids import active_rank_id, rank_id
from vram import (
    _HARDWARE_MEMORY_GIB,
    MODEL_ARCHITECTURES,
    VRAM_REQUIREMENTS_GIB,
    ResourceTarget,
    _cache_elements_per_token,
    _canonical_hardware,
    _known_job_fields,
    _model_payload,
    _number,
    _resource_targets,
    _resources_with_free_capacity,
    _running_job_payload,
    _target_available,
    _target_footprint,
    _target_tps,
    reserve_vram_placement,
)

_PRIORITY_LEVELS = {
    "BEST_EFFORT": 1,
    "LOW": 1,
    "NORMAL": 2,
    "STANDARD": 2,
    "HIGH": 3,
    "CRITICAL": 4,
    "P0": 4,
}

_EVOX_TICK_BUDGET_SECONDS = 300.0
_EVOX_FINALIZATION_RESERVE_SECONDS = 20.0

LAUNCH_FIELDS = (
    "engine_name",
    "engine_version",
    "weight_dtype",
    "weight_quantization_method",
    "weight_quantization_bits",
    "activation_dtype",
    "gpu_mem_util",
    "max_model_len",
    "prefix_cache_enabled",
    "chunked_prefill_enable",
    "chunk_size",
    "scheduling_policy",
    "preemption_policy",
    "router_policy",
    "max_num_seq",
    "max_num_batched_tokens",
    "block_size",
    "kvcache_dtype",
)
EVOX_SYSTEM_PROMPT = """You are evolving a deterministic GPU placement program
online for the current simulator control tick.

Return a usable program within the strict five-minute planning budget.

Improve only compute_placement(payload). The input uses schema
tandemn-evox-placement/v3 and contains public static facts: jobs, each job's
lifecycle, active rank identity and current deployment when it is running,
priority and feasible policy-compliant candidates, and nested resource
capacities. Each candidate includes exact capacity consumption, its per-replica
GPU shape, replica count, total reservation footprint, VRAM demand,
model-weight memory, estimated KV demand, and hourly price. The model_catalog
contains authoritative Store runtime and weight configuration resolved for each
candidate model and exact target GPU. Use it as supplementary model workload
context, but return only candidate IDs. Each candidate's feedback is null when
no outcome is known, or contains the latest Store-observed telemetry outcome
and reward for that exact job placement. Use observed feedback to prefer proven
outcomes while treating unknown feedback as neutral.

Return {"placements": [{"job_id": str, "candidate_id": str | null}, ...]} with
exactly one entry for every input job. A non-null candidate must belong to that
job. Lifecycle maps the compact protocol to Store actions: a waiting job maps
candidate/null to place/defer, and a running job maps candidate/null to
swap/keep. A swap always retains the supplied active_rank_id. Across selected
candidates, consumption for every capacity key must not exceed its available
value. The program must be deterministic, self-contained, side-effect free,
and use no third-party packages or hidden runtime state.

The evaluator rejects malformed or infeasible output and exposes a
larger-is-better combined_score. Preserve high-priority admission before
improving lower-priority admission, then improve mean known observed reward,
then seek lower cost and capacity usage. Explore better global assignment,
packing, ordering, and tie-breaking rather than adding prose or external calls.
"""

EVOX_INITIAL_PROGRAM = '''\
# EVOLVE-BLOCK-START
def compute_placement(payload):
    """Choose one feasible candidate or defer for every public-facts job."""
    capacities = {
        item["capacity_id"]: int(item["available"])
        for item in payload["capacities"]
    }
    chosen = {}
    ordered_jobs = sorted(
        payload["jobs"],
        key=lambda job: (-int(job["priority"]), job["job_id"]),
    )
    for job in ordered_jobs:
        feasible = []
        for candidate in job["candidates"]:
            if all(
                capacities.get(capacity_id, 0) >= int(amount)
                for capacity_id, amount in candidate["consumption"].items()
            ):
                feasible.append(candidate)
        if not feasible:
            chosen[job["job_id"]] = None
            continue
        candidate = min(
            feasible,
            key=lambda item: (
                float(item["price_per_hour"]),
                int(item["footprint"]),
                int(item["pp"]),
                -int(item["tp"]),
                item["candidate_id"],
            ),
        )
        chosen[job["job_id"]] = candidate["candidate_id"]
        for capacity_id, amount in candidate["consumption"].items():
            capacities[capacity_id] -= int(amount)
    return {
        "placements": [
            {"job_id": job["job_id"], "candidate_id": chosen[job["job_id"]]}
            for job in payload["jobs"]
        ]
    }
# EVOLVE-BLOCK-END
'''


@dataclass(frozen=True)
class StoreSnapshot:
    """Freeze Store resources and lifecycle groups for one EvoX tick."""

    resources: dict[str, dict[str, Any]]
    running_jobs: tuple[dict[str, Any], ...]
    waiting_jobs: tuple[dict[str, Any], ...]
    paused_jobs: tuple[dict[str, Any], ...]
    finished_jobs: tuple[dict[str, Any], ...]
    model_catalogs: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class PlacementCandidate:
    """Represent one policy-compliant placement or swap configuration."""

    candidate_id: str
    job_id: str
    target: ResourceTarget
    tp: int
    pp: int
    dp: int
    footprint: int
    memory_gib: float
    required_vram_gib: float
    model_weight_gib: float
    kv_demand: float
    price_per_hour: float
    launch_config: dict[str, object]

    @property
    def gpu_count(self) -> int:
        """Return the accelerator count for each serving replica."""
        return self.tp * self.pp

    @property
    def target_id(self) -> str:
        """Return a stable resource-pool identity for optimization."""
        pool = self.target.instance_type or "aggregate"
        return f"{self.target.environment_key}::{pool}"

    @property
    def capacity_keys(self) -> tuple[str, ...]:
        """Return every nested Store capacity consumed by this placement."""
        environment = f"environment::{self.target.environment_key}"
        if self.target.pool is None:
            return (environment,)
        return (environment, f"pool::{self.target_id}")


@dataclass(frozen=True)
class PlacementProblem:
    """Hold public facts and feasible placements for one Store job."""

    job_id: str
    model_id: str
    priority: int
    priority_class: str
    store_job: dict[str, Any]
    candidates: tuple[PlacementCandidate, ...]
    action_type: str = "place"
    rank_id: str | None = None

    @property
    def lifecycle(self) -> str:
        """Return the Store lifecycle represented by this placement problem."""
        return "running" if self.action_type == "swap" else "waiting"

    @property
    def active_rank_id(self) -> str | None:
        """Return the rank identity which a running-job swap must retain."""
        return self.rank_id


class JobStoreClient(Protocol):
    """Read canonical lifecycle groups from Tandemn Store."""

    def running_jobs(self, user_id: str) -> Sequence[object]: ...

    def waiting_jobs(self, user_id: str) -> Sequence[object]: ...

    def paused_jobs(self, user_id: str) -> Sequence[object]: ...


class ResourceMap(Protocol):
    """Expose Store resource-map fields used by the EvoX adapter."""

    market: object

    def scheduling_summary(self) -> Mapping[str, Mapping[str, Any]]: ...


class ResourceMapStoreClient(Protocol):
    """Read one current Store resource map."""

    def get(self) -> ResourceMap: ...


class ModelCatalogStoreClient(Protocol):
    """Read one model catalog by canonical model ID."""

    def get(self, model_id: str) -> object | None: ...


class PlanStoreClient(Protocol):
    """Persist one canonical Store plan."""

    def create(self, plan: object) -> object: ...


class EvoxPolicy(Protocol):
    """Run one evolved placement program over a JSON-native payload."""

    def __call__(self, payload: dict[str, object]) -> Mapping[str, object]: ...


class EvoxClient(Protocol):
    """Evolve and select placements online for one control tick."""

    def select(
        self,
        tick_number: int,
        problems: Sequence[PlacementProblem],
        capacities: Mapping[str, int],
        feedback: Mapping[object, Mapping[str, object] | None] | None = None,
        deadline: float | None = None,
    ) -> tuple[Mapping[str, PlacementCandidate], tuple[LlmUsage, ...]]: ...


class EvoxDeadlineExceeded(TimeoutError):
    """Carry partial usage when one online EvoX search exceeds five minutes."""

    def __init__(self, llm_usage: tuple[LlmUsage, ...] = ()) -> None:
        super().__init__("EvoX exceeded its five-minute planning budget")
        self.llm_usage = llm_usage


class _EvoxDeadlineSignal(BaseException):
    """Escape SkyDiscover immediately when the wall-clock budget expires.

    This deliberately inherits directly from ``BaseException``. SkyDiscover
    catches ``Exception`` around LLM calls, retries, evaluations, and whole
    iterations; raising a normal timeout there converts the deadline into a
    failed attempt and lets another iteration start after the one-shot alarm
    has fired. This private signal crosses those recovery boundaries, is
    converted back to public ``EvoxDeadlineExceeded`` at our integration
    boundary, and must never be used for ordinary errors.
    """


class OnlineSkyDiscoverEvoxClient:
    """Run one warm-started SkyDiscover EvoX search per control tick."""

    def __init__(
        self,
        model: str,
        iterations: int,
        output_dir: Path,
        api_base: str | None = None,
    ) -> None:
        """Keep online search settings and the previous tick's winner."""
        if iterations < 1:
            raise ValueError("EvoX iterations must be positive")
        self.model = model
        self.iterations = iterations
        self.output_dir = output_dir
        self.api_base = api_base
        self.initial_program = EVOX_INITIAL_PROGRAM
        self.tick_number = 0
        self.timed_out = False
        self.selection_source = "safe_fallback"
        self.completed_candidates = 0
        self.error: str | None = None
        self.worker_completed = False

    def select(
        self,
        tick_number: int,
        problems: Sequence[PlacementProblem],
        capacities: Mapping[str, int],
        feedback: Mapping[object, Mapping[str, object] | None] | None = None,
        deadline: float | None = None,
    ) -> tuple[Mapping[str, PlacementCandidate], tuple[LlmUsage, ...]]:
        """Evolve, verify, and execute the current tick's winning program."""
        self.timed_out = False
        self.selection_source = "safe_fallback"
        self.completed_candidates = 0
        self.error = None
        self.worker_completed = False
        deadline = deadline or monotonic() + _EVOX_TICK_BUDGET_SECONDS
        payload = build_evox_payload(problems, capacities, feedback)
        catalog_json = json.dumps(
            payload["model_catalog"], sort_keys=True, separators=(",", ":")
        )
        system_prompt = (
            f"{EVOX_SYSTEM_PROMPT}\n\nFrozen resolved model catalog for this tick:\n"
            f"{catalog_json}"
        )
        search_seconds = deadline - _EVOX_FINALIZATION_RESERVE_SECONDS - monotonic()
        if search_seconds <= 0:
            self.timed_out = True
            self.tick_number += 1
            return {}, ()

        context = multiprocessing.get_context("spawn")
        messages, worker_messages = context.Pipe(duplex=False)
        process = context.Process(
            target=_run_evox_worker,
            args=(
                tuple(problems),
                dict(capacities),
                {} if feedback is None else dict(feedback),
                worker_messages,
                self.output_dir / f"tick-{self.tick_number:06d}",
                self.model,
                self.iterations,
                self.initial_program,
                self.api_base,
                system_prompt,
                tick_number,
                search_seconds,
            ),
        )
        best: tuple[float, str, dict[str, str]] | None = None
        llm_usage: list[LlmUsage] = []
        try:
            process.start()
            worker_messages.close()
        except Exception as error:
            messages.close()
            worker_messages.close()
            self.error = repr(error)
            self.tick_number += 1
            return {}, ()
        try:
            search_deadline = deadline - _EVOX_FINALIZATION_RESERVE_SECONDS
            while process.is_alive() and monotonic() < search_deadline:
                wait = min(0.1, max(0.0, search_deadline - monotonic()))
                if not messages.poll(wait):
                    continue
                try:
                    message = messages.recv()
                except (EOFError, OSError) as error:
                    if not self.timed_out and not self.worker_completed:
                        self.error = repr(error)
                    break
                best = _consume_worker_message(message, best, llm_usage, self)

            if monotonic() >= search_deadline:
                self.timed_out = True
            if process.is_alive():
                process.kill()
            process.join()
            while messages.poll():
                try:
                    message = messages.recv()
                except (EOFError, OSError) as error:
                    if not self.timed_out and not self.worker_completed:
                        self.error = repr(error)
                    break
                best = _consume_worker_message(message, best, llm_usage, self)
            if (
                getattr(process, "exitcode", 0) not in {0, None}
                and not self.timed_out
                and self.error is None
            ):
                self.error = f"EvoX worker exited with code {process.exitcode}"
        except Exception as error:
            self.error = repr(error)
        finally:
            if process.is_alive():
                process.kill()
                process.join()
            messages.close()

        self.tick_number += 1
        if best is None:
            if self.error is None and not self.timed_out:
                self.error = "EvoX worker exited without a valid candidate"
            return {}, tuple(llm_usage)

        _, best_solution, selected_ids = best
        try:
            selected = _selected_candidates(problems, selected_ids)
            validate_selection(problems, selected, capacities)
        except Exception as error:
            self.error = repr(error)
            return {}, tuple(llm_usage)
        self.initial_program = best_solution
        self.selection_source = (
            "evox_best_so_far"
            if self.timed_out or self.error is not None or not self.worker_completed
            else "evox_completed"
        )
        return selected, tuple(llm_usage)


class OnlineEvoxEvaluator:
    """Score online candidate programs using verified static placement facts."""

    def __init__(
        self,
        problems: Sequence[PlacementProblem],
        capacities: Mapping[str, int],
        feedback: Mapping[object, Mapping[str, object] | None] | None = None,
        on_best: Callable[[float, str, Mapping[str, PlacementCandidate], int], None]
        | None = None,
        on_progress: Callable[[int], None] | None = None,
    ) -> None:
        """Freeze the current tick throughout one online evolution run."""
        self.problems = tuple(problems)
        self.capacities = dict(capacities)
        self.feedback = {} if feedback is None else feedback
        self.best_solution: str | None = None
        self.best_score = -math.inf
        self.completed_candidates = 0
        self.on_best = on_best
        self.on_progress = on_progress

    def evaluate(self, program_path: str) -> dict[str, object]:
        """Return a larger-is-better score after strict plan verification."""
        try:
            policy = load_policy_file(Path(program_path))
            selected = select_evox_placements(
                policy,
                self.problems,
                self.capacities,
                self.feedback,
            )
        except Exception:
            return {
                "combined_score": 0.0,
                "feasible": 0.0,
                "invalid_program": 1.0,
                "mean_observed_reward": 0.0,
                "observed_feedback_count": 0.0,
                "feedback": {},
            }

        priority_base = len(self.problems) + 1
        priority_by_job = {
            problem.job_id: problem.priority for problem in self.problems
        }
        priority_score = sum(
            priority_base ** (priority_by_job[job_id] - 1) for job_id in selected
        )
        price = sum(candidate.price_per_hour for candidate in selected.values())
        footprint = sum(candidate.footprint for candidate in selected.values())
        problem_by_id = {problem.job_id: problem for problem in self.problems}
        selected_feedback = {
            job_id: _candidate_feedback(
                self.feedback,
                problem_by_id[job_id],
                candidate,
            )
            for job_id, candidate in selected.items()
        }
        observed_rewards = [
            reward
            for outcome in selected_feedback.values()
            if (reward := _observed_reward(outcome)) is not None
        ]
        mean_observed_reward = (
            sum(observed_rewards) / len(observed_rewards) if observed_rewards else 0.0
        )
        tie_break = 0.0
        if selected:
            tie_break = 1 / (1 + price) + 1 / (1_000 * (1 + footprint))
        metrics = {
            "combined_score": float(
                4 * priority_score + mean_observed_reward + tie_break
            ),
            "feasible": 1.0,
            "admitted_jobs": float(len(selected)),
            "priority_score": float(priority_score),
            "mean_observed_reward": float(mean_observed_reward),
            "observed_feedback_count": float(len(observed_rewards)),
            "hourly_price": float(price),
            "gpu_footprint": float(footprint),
            "feedback": selected_feedback,
        }
        self.completed_candidates += 1
        if self.on_progress is not None:
            self.on_progress(self.completed_candidates)
        if metrics["combined_score"] > self.best_score:
            source = Path(program_path).read_text(encoding="utf-8")
            self.best_solution = source
            self.best_score = metrics["combined_score"]
            if self.on_best is not None:
                self.on_best(
                    self.best_score,
                    source,
                    selected,
                    self.completed_candidates,
                )
        return metrics


def _run_evox_worker(
    problems: tuple[PlacementProblem, ...],
    capacities: dict[str, int],
    feedback: dict[object, Mapping[str, object] | None],
    messages: Any,
    output_dir: Path,
    model: str,
    iterations: int,
    initial_program: str,
    api_base: str | None,
    system_prompt: str,
    tick_number: int,
    search_seconds: float,
) -> None:
    """Run SkyDiscover in a process the parent can stop without thread cleanup."""
    from threading import Event

    from skydiscover.search import default_discovery_controller

    # SkyDiscover uses this only for in-process shutdown checks. A threading
    # event avoids leaking POSIX semaphores when this worker is hard-killed.
    default_discovery_controller.mp = SimpleNamespace(Event=Event)

    def publish_best(
        score: float,
        source: str,
        selected: Mapping[str, PlacementCandidate],
        completed_candidates: int,
    ) -> None:
        messages.send(
            (
                "best",
                score,
                source,
                {
                    job_id: candidate.candidate_id
                    for job_id, candidate in selected.items()
                },
                completed_candidates,
            )
        )

    evaluator = OnlineEvoxEvaluator(
        problems,
        capacities,
        feedback,
        on_best=publish_best,
        on_progress=lambda count: messages.send(("progress", count)),
    )
    try:
        run_evox_discovery(
            evaluator.evaluate,
            output_dir,
            model=model,
            iterations=iterations,
            initial_program=initial_program,
            api_base=api_base,
            system_prompt=system_prompt,
            tick_number=tick_number,
            time_budget_seconds=search_seconds,
            usage_callback=lambda usage: messages.send(("usage", usage)),
        )
        messages.send(("completed", evaluator.completed_candidates))
    except EvoxDeadlineExceeded:
        messages.send(("timeout", evaluator.completed_candidates))
    except BaseException as error:
        messages.send(("error", repr(error), evaluator.completed_candidates))
    finally:
        messages.close()


def _consume_worker_message(
    message: tuple[object, ...],
    best: tuple[float, str, dict[str, str]] | None,
    llm_usage: list[LlmUsage],
    client: OnlineSkyDiscoverEvoxClient,
) -> tuple[float, str, dict[str, str]] | None:
    """Retain one worker's usage and latest fully validated best result."""
    kind = message[0]
    if kind == "best":
        score, source, selected_ids, completed = message[1:]
        if (
            isinstance(score, (int, float))
            and isinstance(source, str)
            and isinstance(selected_ids, dict)
            and isinstance(completed, int)
        ):
            client.completed_candidates = max(client.completed_candidates, completed)
            candidate = (float(score), source, selected_ids)
            if best is None or candidate[0] > best[0]:
                return candidate
    elif kind == "usage" and isinstance(message[1], LlmUsage):
        llm_usage.append(message[1])
    elif kind == "completed" and isinstance(message[1], int):
        client.worker_completed = True
        client.completed_candidates = max(client.completed_candidates, message[1])
    elif kind == "progress" and isinstance(message[1], int):
        client.completed_candidates = max(client.completed_candidates, message[1])
    elif kind == "timeout":
        client.timed_out = True
        if isinstance(message[1], int):
            client.completed_candidates = max(client.completed_candidates, message[1])
    elif kind == "error":
        client.error = str(message[1])
        if isinstance(message[2], int):
            client.completed_candidates = max(client.completed_candidates, message[2])
    return best


def _selected_candidates(
    problems: Sequence[PlacementProblem],
    selected_ids: Mapping[str, str],
) -> dict[str, PlacementCandidate]:
    """Resolve worker-selected IDs without executing generated source again."""
    problem_by_id = {problem.job_id: problem for problem in problems}
    selected = {}
    for job_id, candidate_id in selected_ids.items():
        try:
            problem = problem_by_id[job_id]
            selected[job_id] = next(
                candidate
                for candidate in problem.candidates
                if candidate.candidate_id == candidate_id
            )
        except (KeyError, StopIteration) as error:
            raise ValueError("EvoX worker returned an invalid candidate") from error
    return selected


class EvoxPlannerDriver:
    """Create plans with online LLM program evolution on every tick."""

    method_name = "evox"

    def __init__(
        self,
        user_id: str,
        scenario: Scenario,
        job_store: JobStoreClient,
        resource_map_store: ResourceMapStoreClient,
        model_catalog_store: ModelCatalogStoreClient,
        plan_store: PlanStoreClient,
        backend: str,
        config_policy: ConfigPolicy,
        client: EvoxClient,
    ) -> None:
        """Keep Store clients and the persistent online EvoX client."""
        self.user_id = user_id
        self.benchmark_name = scenario.benchmark_name
        jobs = (*scenario.online_jobs, *scenario.batch_jobs)
        self.benchmark_job_ids = {job.store_job_id: job.name for job in jobs}
        self._job_is_batch = {
            job.store_job_id: isinstance(job, BatchJob) for job in jobs
        }
        self.job_store = job_store
        self.resource_map_store = resource_map_store
        self.model_catalog_store = model_catalog_store
        self.plan_store = plan_store
        self.backend = backend
        self.config_policy = config_policy
        self.client = client
        self.feedback = OutcomeFeedback(user_id)

    @classmethod
    def build(
        cls,
        user_id: str,
        scenario: Scenario,
        settings: Mapping[str, object],
        postgres_client: object | None = None,
        client: EvoxClient | None = None,
    ) -> EvoxPlannerDriver:
        """Build online EvoX with Store clients and SkyDiscover settings."""
        from tandemn_system_data.clients import (
            GpuMetricStore,
            JobStore,
            ModelCatalogStore,
            PlanStore,
            PostgresClient,
            ResourceMapStore,
        )
        from tandemn_system_data.clients.event_log import PostgresEventLog

        model = _required_text(settings.get("model"), "model")
        iterations = _positive_int(settings.get("iterations", 5), "iterations")
        output_dir = Path(
            _optional_text(settings.get("output_dir")) or "outputs/evox-online"
        )
        api_base = _optional_text(
            settings.get("api_base") or os.getenv("OPENAI_BASE_URL")
        )
        postgres = postgres_client or PostgresClient()
        driver = cls(
            user_id=user_id,
            scenario=scenario,
            job_store=JobStore(postgres),
            resource_map_store=ResourceMapStore(postgres, user_id=user_id),
            model_catalog_store=ModelCatalogStore(postgres),
            plan_store=PlanStore(postgres),
            backend=_required_text(settings.get("backend"), "backend"),
            config_policy=load_config_policy(),
            client=client
            or OnlineSkyDiscoverEvoxClient(
                model=model,
                iterations=iterations,
                output_dir=output_dir,
                api_base=api_base,
            ),
        )
        driver.feedback.bind(GpuMetricStore(postgres), PostgresEventLog(postgres))
        return driver

    def create_plans(
        self,
        tick_number: int,
        logical_time: datetime,
    ) -> PlannerResult:
        """Read one Store snapshot, persist one plan, and return its identity."""
        self._tick_number = tick_number
        self._llm_usage: tuple[LlmUsage, ...] = ()
        self._deadline = monotonic() + _EVOX_TICK_BUDGET_SECONDS
        selection_source = None
        timed_out = False
        completed_candidates = 0
        try:
            self.feedback.observe(logical_time)
            self._feedback_window_start = logical_time
            snapshot = self._read_store_snapshot()
            actions = self._build_actions(snapshot)
            selection_source = getattr(self.client, "selection_source", None)
            timed_out = bool(getattr(self.client, "timed_out", False))
            completed_candidates = int(
                getattr(self.client, "completed_candidates", 0)
            )
            client_error = getattr(self.client, "error", None)
            if not actions:
                return PlannerResult(
                    tick_number=tick_number,
                    plans=(),
                    llm_usage=self._llm_usage,
                    timed_out=timed_out,
                    selection_source=selection_source,
                    completed_candidates=completed_candidates,
                )

            from tandemn_system_data.models import Plan

            plan = Plan(
                user_id=self.user_id,
                koi_version=self.method_name,
                tick_rationale=_tick_rationale(
                    self.method_name,
                    tick_number,
                    actions,
                    selection_source,
                ),
                actions=actions,
            )
            self.plan_store.create(plan)
            self._record_action_feedback(actions, logical_time)
            return PlannerResult(
                tick_number=tick_number,
                plans=(CreatedPlan(plan_id=plan.plan_id),),
                degraded=(
                    timed_out
                    or selection_source == "safe_fallback"
                    or client_error is not None
                ),
                error=client_error,
                llm_usage=self._llm_usage,
                timed_out=timed_out,
                selection_source=selection_source,
                completed_candidates=completed_candidates,
            )
        except Exception as error:
            return PlannerResult(
                tick_number=tick_number,
                plans=(),
                degraded=True,
                error=repr(error),
                llm_usage=self._llm_usage,
                timed_out=timed_out,
                selection_source=selection_source,
                completed_candidates=completed_candidates,
            )

    def _read_store_snapshot(self) -> StoreSnapshot:
        """Freeze resources and canonical lifecycle groups for one tick."""
        resource_map = self.resource_map_store.get()
        running = tuple(self.job_store.running_jobs(self.user_id))
        waiting = tuple(self.job_store.waiting_jobs(self.user_id))
        running_jobs = tuple(_running_job_payload(job) for job in running)
        waiting_jobs = tuple(_model_payload(job) for job in waiting)
        model_ids = {
            _required_text(_known_job_fields(job).get("model_id"), "model_id")
            for job in (*running_jobs, *waiting_jobs)
        }
        model_catalogs = {}
        for model_id in sorted(model_ids):
            catalog = self.model_catalog_store.get(model_id)
            if catalog is None:
                raise ValueError(f"model catalog missing {model_id!r}")
            model_catalogs[model_id] = _model_payload(catalog)
        return StoreSnapshot(
            resources=_resources_with_free_capacity(resource_map, running),
            running_jobs=running_jobs,
            waiting_jobs=waiting_jobs,
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=model_catalogs,
        )

    def _build_actions(self, snapshot: StoreSnapshot) -> list[dict[str, object]]:
        """Select waiting placements and eligible running-job swaps."""
        resources = deepcopy(snapshot.resources)
        feedback = getattr(self, "feedback", None)
        eligible_running_jobs = tuple(
            running
            for running in snapshot.running_jobs
            if feedback is None
            or not feedback.has_pending_batch(
                _required_text(running.get("job_id"), "running Store job_id")
            )
        )
        problems = build_placement_problems(
            snapshot.waiting_jobs,
            resources,
            self.benchmark_name,
            self.benchmark_job_ids,
            self.config_policy,
            snapshot.model_catalogs,
            running_jobs=eligible_running_jobs,
        )
        capacities = capacity_limits(resources)
        selected = dict(self._select_placements(problems, capacities))
        validate_selection(problems, selected, capacities)
        problems_by_id = {problem.job_id: problem for problem in problems}

        def selected_plan(problem: PlacementProblem) -> dict[str, object] | None:
            candidate = selected.get(problem.job_id)
            if candidate is None:
                return None
            plan = self._build_place_plan(problem, candidate)
            verify_place_plan(
                plan,
                candidate,
                self.config_policy,
                action_type=problem.action_type,
                rank_name=problem.rank_id,
            )
            reserve_vram_placement(
                candidate.target,
                candidate.gpu_count,
                candidate.dp,
            )
            return plan

        actions: list[dict[str, object]] = []
        for running in snapshot.running_jobs:
            job_id = _required_text(running.get("job_id"), "running Store job_id")
            problem = problems_by_id.get(job_id)
            plan = None if problem is None else selected_plan(problem)
            actions.append(plan or {"job_id": job_id, "type": "keep"})
        for waiting in snapshot.waiting_jobs:
            job_id = _required_text(waiting.get("job_id"), "waiting Store job_id")
            plan = selected_plan(problems_by_id[job_id])
            actions.append(plan or {"job_id": job_id, "type": "defer"})
        _validate_complete_job_actions(snapshot, actions)
        return actions

    def _select_placements(
        self,
        problems: Sequence[PlacementProblem],
        capacities: Mapping[str, int],
    ) -> Mapping[str, PlacementCandidate]:
        """Evolve and validate the current tick's placement response."""
        if not problems:
            if isinstance(self.client, OnlineSkyDiscoverEvoxClient):
                self.client.timed_out = False
                self.client.selection_source = "no_active_jobs"
                self.client.completed_candidates = 0
                self.client.error = None
            return {}
        feedback = self._candidate_feedback(problems)
        selected, self._llm_usage = self.client.select(
            self._tick_number,
            problems,
            capacities,
            feedback,
            deadline=getattr(self, "_deadline", None),
        )
        return selected

    def _candidate_feedback(
        self,
        problems: Sequence[PlacementProblem],
    ) -> dict[PlacementFingerprint, dict[str, object]]:
        """Freeze known outcomes for every candidate before online selection."""
        feedback = getattr(self, "feedback", None)
        if feedback is None:
            return {}
        outcomes = {}
        for problem in problems:
            for candidate in problem.candidates:
                placement = _placement_fingerprint(problem, candidate)
                outcome = feedback.for_fingerprint(placement)
                if outcome is not None:
                    outcomes[placement] = outcome
        return outcomes

    def _record_action_feedback(
        self,
        actions: Sequence[Mapping[str, object]],
        logical_time: datetime,
    ) -> None:
        """Track persisted canonical place and swap ladders for later outcomes."""
        feedback = getattr(self, "feedback", None)
        if feedback is None:
            return
        for action in actions:
            if action.get("type") not in {"place", "swap"}:
                continue
            job_id = _required_text(action.get("job_id"), "plan job_id")
            ladder = action.get("ladder")
            if not isinstance(ladder, list) or len(ladder) != 1:
                raise ValueError("placement actions require one ladder rank")
            rank = ladder[0]
            if not isinstance(rank, Mapping):
                raise ValueError("placement ladder rank must be an object")
            config = rank.get("config")
            if not isinstance(config, Mapping):
                raise ValueError("placement ladder config must be an object")
            environment = rank.get("env")
            if not isinstance(environment, (list, tuple, str)):
                raise ValueError("placement ladder is missing environment")
            feedback.record(
                fingerprint=fingerprint(
                    job_id=job_id,
                    model_id=_required_text(config.get("model_id"), "model_id"),
                    environment=_env_key(environment),
                    instance_type=_optional_text(config.get("instance_type")),
                    gpu_type=_required_text(config.get("gpu_type"), "gpu_type"),
                    tp=_positive_int(config.get("tp"), "tp"),
                    pp=_positive_int(config.get("pp"), "pp"),
                    dp=_positive_int(config.get("dp"), "dp"),
                ),
                rank_id=_required_text(rank.get("rank_id"), "placement rank_id"),
                window_start=logical_time,
                is_batch=getattr(self, "_job_is_batch", {}).get(job_id, False),
                required_tps=_number(action.get("target_tps")),
                ttft_target_ms=_number(action.get("target_p99_ttft_ms")),
                tpot_target_ms=_number(action.get("target_p99_tpot_ms")),
            )

    def _build_place_plan(
        self,
        problem: PlacementProblem,
        candidate: PlacementCandidate,
    ) -> dict[str, object]:
        """Build EvoX's complete per-job placement or swap plan."""
        return build_evox_plan(
            problem.store_job,
            candidate,
            self.backend,
            self.config_policy,
            action_type=problem.action_type,
            rank_name=problem.rank_id,
        )


def build_evox_plan(
    store_job: Mapping[str, object],
    candidate: PlacementCandidate,
    backend: str,
    config_policy: ConfigPolicy,
    *,
    action_type: str = "place",
    rank_name: str | None = None,
) -> dict[str, object]:
    """Return one complete online-EvoX placement or swap plan."""
    if action_type not in {"place", "swap"}:
        raise ValueError("EvoX action type must be place or swap")
    if action_type == "swap" and rank_name is None:
        raise ValueError("EvoX swaps require the active rank ID")
    known = _known_job_fields(store_job)
    model_id = _required_text(known.get("model_id"), "model_id")
    rule = config_policy.rule_for(candidate.target.gpu_type, model_id)
    if rule is None or not rule.allows_parallelism(candidate.tp, candidate.pp):
        raise ValueError("EvoX selected a policy-invalid placement")
    per_replica_gpu_count = candidate.gpu_count
    return {
        "job_id": candidate.job_id,
        "type": action_type,
        "target_tps": _target_tps(known),
        "target_p99_ttft_ms": _number(known.get("target_p99_ttft_ms")),
        "target_p99_tpot_ms": _number(known.get("target_p99_tpot_ms")),
        "ladder": [
            {
                "role": known.get("role") or "aggregate",
                "rank_id": rank_name if rank_name is not None else rank_id(known),
                "env": candidate.target.environment_key.split("|"),
                "config": {
                    **candidate.launch_config,
                    "model_id": model_id,
                    "precision": rule.precision,
                    "instance_type": candidate.target.instance_type,
                    "gpu_type": candidate.target.gpu_type,
                    "gpu_count": per_replica_gpu_count,
                    "count": per_replica_gpu_count,
                    "tp": candidate.tp,
                    "pp": candidate.pp,
                    "dp": candidate.dp,
                    "ep": known.get("ep") or 1,
                    "rank_traffic_share": 1.0,
                },
                "n_replicas": candidate.dp,
                "mechanism_id": known.get("mechanism_id"),
            }
        ],
    }


def _placement_fingerprint(
    problem: PlacementProblem,
    candidate: PlacementCandidate,
) -> PlacementFingerprint:
    """Build the stable feedback identity for one placement candidate."""
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


def _candidate_feedback(
    feedback: Mapping[object, Mapping[str, object] | None] | None,
    problem: PlacementProblem,
    candidate: PlacementCandidate,
) -> dict[str, object] | None:
    """Return the frozen observed outcome for one payload candidate."""
    if feedback is None:
        return None
    outcome = feedback.get(_placement_fingerprint(problem, candidate))
    if outcome is None:
        outcome = feedback.get((problem.job_id, candidate.candidate_id))
    if outcome is None:
        outcome = feedback.get(candidate.candidate_id)
    return dict(outcome) if isinstance(outcome, Mapping) else None


def _observed_reward(outcome: Mapping[str, object] | None) -> float | None:
    """Read a bounded reward from one public feedback outcome."""
    if outcome is None:
        return None
    reward = _finite_number(outcome.get("reward"))
    if reward is None:
        return None
    return max(-1.0, min(1.0, reward))


def build_placement_problems(
    waiting_jobs: Sequence[Mapping[str, Any]],
    resources: dict[str, dict[str, Any]],
    benchmark_name: str,
    benchmark_job_ids: Mapping[str, str],
    config_policy: ConfigPolicy,
    model_catalogs: Mapping[str, Mapping[str, Any]],
    *,
    running_jobs: Sequence[Mapping[str, Any]] = (),
) -> tuple[PlacementProblem, ...]:
    """Build feasible EvoX candidates for waiting jobs and running swaps."""
    requirements = VRAM_REQUIREMENTS_GIB.get(benchmark_name, {})
    problems = []

    def add_problem(
        job: Mapping[str, Any],
        *,
        action_type: str,
        rank_name: str | None = None,
        active_rank: Mapping[str, Any] | None = None,
    ) -> None:
        store_job = dict(job)
        job_id = _required_text(store_job.get("job_id"), "Store job_id")
        known = _known_job_fields(store_job)
        benchmark_job_id = benchmark_job_ids.get(job_id, job_id)
        model_id = _required_text(known.get("model_id"), "model_id")
        try:
            model_catalog = model_catalogs[model_id]
        except KeyError as error:
            raise ValueError(f"model catalog missing {model_id!r}") from error
        required_vram_gib = _required_vram_gib(
            known,
            requirements.get(benchmark_job_id),
        )
        candidates = ()
        if required_vram_gib is not None:
            candidates = enumerate_candidates(
                job_id,
                model_id,
                known,
                required_vram_gib,
                resources,
                config_policy,
                model_catalog,
            )
            if active_rank is not None:
                candidates = tuple(
                    candidate
                    for candidate in candidates
                    if not _matches_active_shape(candidate, active_rank)
                )
        priority_class, priority = _priority(known)
        problems.append(
            PlacementProblem(
                job_id=job_id,
                model_id=model_id,
                priority=priority,
                priority_class=priority_class,
                store_job=store_job,
                candidates=candidates,
                action_type=action_type,
                rank_id=rank_name,
            )
        )

    for waiting in waiting_jobs:
        add_problem(waiting, action_type="place")
    for running in running_jobs:
        rank_name = active_rank_id(running)
        if rank_name is None:
            continue
        add_problem(
            running,
            action_type="swap",
            rank_name=rank_name,
            active_rank=_active_rank(running, rank_name),
        )
    return tuple(problems)


def _active_rank(
    running_job: Mapping[str, Any],
    rank_name: str,
) -> Mapping[str, Any] | None:
    """Return the sole active rank payload selected for a running-job swap."""
    active_ranks = running_job.get("active_ranks")
    if not isinstance(active_ranks, list):
        return None
    for active_rank in active_ranks:
        if isinstance(active_rank, Mapping) and active_rank.get("rank_id") == rank_name:
            return active_rank
    return None


def _matches_active_shape(
    candidate: PlacementCandidate,
    active_rank: Mapping[str, Any],
) -> bool:
    """Return whether a candidate would leave active physical capacity unchanged."""
    shape = active_rank.get("shape_json")
    if not isinstance(shape, Mapping):
        return False
    environment = shape.get("env") or shape.get("pool_id")
    if isinstance(environment, str):
        environment = environment.split("|")
    if not isinstance(environment, (list, tuple)):
        return False
    return (
        list(environment) == candidate.target.environment_key.split("|")
        and shape.get("instance_type") == candidate.target.instance_type
        and shape.get("count", shape.get("gpu_count")) == candidate.gpu_count
        and shape.get("tp") == candidate.tp
        and shape.get("pp") == candidate.pp
        and active_rank.get("n_replicas") == candidate.dp
    )


def _current_deployment(problem: PlacementProblem) -> dict[str, object] | None:
    """Expose the active rank's physical shape for a running EvoX problem."""
    if problem.rank_id is None:
        return None
    active_rank = _active_rank(problem.store_job, problem.rank_id)
    if active_rank is None:
        return None
    shape = active_rank.get("shape_json")
    if not isinstance(shape, Mapping):
        return None
    environment = shape.get("env") or shape.get("pool_id")
    if isinstance(environment, (list, tuple)):
        environment = "|".join(str(value) for value in environment)
    if not isinstance(environment, str):
        return None
    return {
        "rank_id": problem.rank_id,
        "environment": environment,
        "instance_type": shape.get("instance_type"),
        "gpu_type": shape.get("gpu_type") or environment.split("|")[-1],
        "gpu_count": shape.get("gpu_count", shape.get("count")),
        "count": shape.get("count"),
        "tp": shape.get("tp"),
        "pp": shape.get("pp"),
        "dp": shape.get("dp"),
        "n_replicas": active_rank.get("n_replicas"),
    }


def _env_key(value: object) -> str:
    """Return one canonical pipe-delimited environment key."""
    if isinstance(value, (list, tuple)):
        return "|".join(str(part) for part in value)
    return str(value)


def enumerate_candidates(
    job_id: str,
    model_id: str,
    known: Mapping[str, object],
    required_vram_gib: float,
    resources: dict[str, dict[str, Any]],
    config_policy: ConfigPolicy,
    model_catalog: Mapping[str, Any],
) -> tuple[PlacementCandidate, ...]:
    """Enumerate policy-compliant static shapes that fit current capacity."""
    candidates = []
    for target in _resource_targets(resources):
        memory_gib = _target_memory_gib(target)
        if memory_gib is None:
            continue
        rule = config_policy.rule_for(target.gpu_type, model_id)
        if rule is None:
            continue
        launch_config = _materialize_launch_config(model_catalog, target.gpu_type)
        minimum_gpus = math.ceil(required_vram_gib / memory_gib)
        for tp in sorted(rule.allowed_tp):
            pp = math.ceil(minimum_gpus / tp)
            if not rule.allows_parallelism(tp, pp):
                continue
            per_replica_gpu_count = tp * pp
            dp = 1
            while True:
                footprint = _target_footprint(
                    target,
                    per_replica_gpu_count,
                    dp,
                )
                if footprint is None or footprint > _target_available(target):
                    break
                candidate_id = (
                    f"{target.environment_key}::{target.instance_type or 'aggregate'}::"
                    f"tp{tp}-pp{pp}-dp{dp}"
                )
                candidates.append(
                    PlacementCandidate(
                        candidate_id=candidate_id,
                        job_id=job_id,
                        target=target,
                        tp=tp,
                        pp=pp,
                        dp=dp,
                        footprint=footprint,
                        memory_gib=memory_gib,
                        required_vram_gib=required_vram_gib,
                        model_weight_gib=_model_weight_gib(
                            model_id,
                            rule.precision,
                            required_vram_gib,
                        ),
                        kv_demand=_kv_demand(model_id, rule.precision, known),
                        price_per_hour=_placement_price(target, footprint),
                        launch_config=launch_config,
                    )
                )
                dp += 1
    return tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id))


def _materialize_launch_config(
    catalog: Mapping[str, Any],
    gpu_type: str,
) -> dict[str, object]:
    """Resolve one Store model catalog against an exact target GPU."""
    profile_fields = {
        "max_num_seq",
        "max_num_batched_tokens",
        "block_size",
        "kvcache_dtype",
    }
    launch_config = {}
    for field in LAUNCH_FIELDS:
        if field not in profile_fields:
            value = catalog.get(field)
            if value is None:
                raise ValueError(f"model catalog is missing {field}")
            launch_config[field] = value
            continue

        entries = catalog.get(field)
        if not isinstance(entries, list):
            raise ValueError(f"model catalog {field} must be a GPU profile list")
        value = None
        for raw_entry in entries:
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"model catalog {field} entry must be an object")
            if raw_entry.get("gpu_type") == gpu_type:
                value = raw_entry.get("value")
                break
        if value is None:
            raise ValueError(f"model catalog {field} has no profile for {gpu_type}")
        launch_config[field] = value
    return launch_config


def capacity_limits(
    resources: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Expose environment and nested pool capacities as linear constraints."""
    capacities = {
        f"environment::{environment_key}": max(0, int(environment.get("free", 0)))
        for environment_key, environment in resources.items()
    }
    for target in _resource_targets(resources):
        if target.pool is not None:
            capacities[f"pool::{_target_id(target)}"] = max(
                0,
                int(target.pool.get("free", 0)),
            )
    return capacities


def validate_selection(
    problems: Sequence[PlacementProblem],
    selected: Mapping[str, PlacementCandidate],
    capacities: Mapping[str, int],
) -> None:
    """Require known, non-overcommitted candidates from one selector."""
    problem_by_id = {problem.job_id: problem for problem in problems}
    unknown = set(selected) - set(problem_by_id)
    if unknown:
        raise ValueError(f"selector returned unknown jobs: {sorted(unknown)}")

    used = {key: 0 for key in capacities}
    for job_id, candidate in selected.items():
        expected = {
            value.candidate_id: value for value in problem_by_id[job_id].candidates
        }
        if candidate.candidate_id not in expected or candidate.job_id != job_id:
            raise ValueError(f"selector returned an invalid candidate for {job_id}")
        for key in candidate.capacity_keys:
            if key not in used:
                raise ValueError(f"candidate references unknown capacity {key}")
            used[key] += candidate.footprint
            if used[key] > capacities[key]:
                raise ValueError(f"selector overcommitted {key}")


def verify_place_plan(
    plan: Mapping[str, object],
    candidate: PlacementCandidate,
    config_policy: ConfigPolicy,
    *,
    action_type: str | None = None,
    rank_name: str | None = None,
) -> None:
    """Verify one complete per-job placement or swap plan before returning it."""
    required_plan_keys = {
        "job_id",
        "type",
        "target_tps",
        "target_p99_ttft_ms",
        "target_p99_tpot_ms",
        "ladder",
    }
    if set(plan) != required_plan_keys:
        raise ValueError("placement plan has incomplete top-level fields")
    plan_action_type = plan.get("type")
    if (
        plan.get("job_id") != candidate.job_id
        or plan_action_type not in {"place", "swap"}
        or (action_type is not None and plan_action_type != action_type)
    ):
        raise ValueError("placement plan has the wrong job or action type")
    ladder = plan.get("ladder")
    if not isinstance(ladder, list) or len(ladder) != 1:
        raise ValueError("placement plan must contain exactly one ladder rank")
    rank = ladder[0]
    if not isinstance(rank, Mapping):
        raise ValueError("placement ladder rank must be an object")
    required_rank_keys = {
        "role",
        "rank_id",
        "env",
        "config",
        "n_replicas",
        "mechanism_id",
    }
    if set(rank) != required_rank_keys:
        raise ValueError("placement ladder rank has incomplete fields")
    if rank.get("env") != candidate.target.environment_key.split("|"):
        raise ValueError("placement plan environment does not match its candidate")
    if rank_name is not None and rank.get("rank_id") != rank_name:
        raise ValueError("swap plan does not retain the active rank ID")
    if rank.get("n_replicas") != candidate.dp:
        raise ValueError("placement plan has the wrong replica count")
    config = rank.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("placement ladder config must be an object")
    required_config_keys = {
        *LAUNCH_FIELDS,
        "model_id",
        "precision",
        "instance_type",
        "gpu_type",
        "gpu_count",
        "count",
        "tp",
        "pp",
        "dp",
        "ep",
        "rank_traffic_share",
    }
    if set(config) != required_config_keys:
        raise ValueError("placement plan config has incomplete fields")
    expected = {
        "gpu_type": candidate.target.gpu_type,
        "instance_type": candidate.target.instance_type,
        "gpu_count": candidate.gpu_count,
        "count": candidate.gpu_count,
        "tp": candidate.tp,
        "pp": candidate.pp,
        "dp": candidate.dp,
        "rank_traffic_share": 1.0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"placement plan config has invalid {key}")
    for key, value in candidate.launch_config.items():
        if config.get(key) != value:
            raise ValueError(f"placement plan config has invalid catalog field {key}")
    model_id = config.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("placement plan config requires model_id")
    rule = config_policy.rule_for(candidate.target.gpu_type, model_id)
    if rule is None or not rule.allows_parallelism(candidate.tp, candidate.pp):
        raise ValueError("placement plan violates the configuration policy")
    if config.get("precision") != rule.precision:
        raise ValueError("placement plan precision violates the configuration policy")


def build_evox_payload(
    problems: Sequence[PlacementProblem],
    capacities: Mapping[str, int],
    feedback: Mapping[object, Mapping[str, object] | None] | None = None,
) -> dict[str, object]:
    """Serialize public placement facts for current-tick evolution."""
    jobs = []
    catalog_profiles: dict[tuple[str, str], dict[str, object]] = {}
    for problem in problems:
        candidates = []
        for candidate in problem.candidates:
            catalog_profiles[(problem.model_id, candidate.target.gpu_type)] = {
                "model_id": problem.model_id,
                "gpu_type": candidate.target.gpu_type,
                **candidate.launch_config,
            }
            candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "environment": candidate.target.environment_key,
                    "instance_type": candidate.target.instance_type,
                    "gpu_type": candidate.target.gpu_type,
                    "tp": candidate.tp,
                    "pp": candidate.pp,
                    "dp": candidate.dp,
                    "gpu_count": candidate.gpu_count,
                    "count": candidate.gpu_count,
                    "n_replicas": candidate.dp,
                    "footprint": candidate.footprint,
                    "memory_gib": candidate.memory_gib,
                    "required_vram_gib": candidate.required_vram_gib,
                    "model_weight_gib": candidate.model_weight_gib,
                    "kv_demand": candidate.kv_demand,
                    "price_per_hour": candidate.price_per_hour,
                    "feedback": _candidate_feedback(feedback, problem, candidate),
                    "consumption": {
                        key: candidate.footprint for key in candidate.capacity_keys
                    },
                }
            )
        jobs.append(
            {
                "job_id": problem.job_id,
                "model_id": problem.model_id,
                "lifecycle": problem.lifecycle,
                "active_rank_id": problem.active_rank_id,
                "current_deployment": _current_deployment(problem),
                "priority": problem.priority,
                "priority_class": problem.priority_class,
                "candidates": candidates,
            }
        )
    return {
        "schema_version": "tandemn-evox-placement/v3",
        "capacities": [
            {"capacity_id": key, "available": value}
            for key, value in sorted(capacities.items())
        ],
        "model_catalog": [catalog_profiles[key] for key in sorted(catalog_profiles)],
        "jobs": jobs,
    }


def select_evox_placements(
    policy: EvoxPolicy,
    problems: Sequence[PlacementProblem],
    capacities: Mapping[str, int],
    feedback: Mapping[object, Mapping[str, object] | None] | None = None,
) -> dict[str, PlacementCandidate]:
    """Run and strictly validate one evolved program's output."""
    result = policy(build_evox_payload(problems, capacities, feedback))
    if not isinstance(result, Mapping) or set(result) != {"placements"}:
        raise ValueError("EvoX policy must return only placements")
    values = result.get("placements")
    if not isinstance(values, list):
        raise ValueError("EvoX placements must be a list")

    problem_by_id = {problem.job_id: problem for problem in problems}
    decisions: dict[str, str | None] = {}
    for index, value in enumerate(values):
        if not isinstance(value, Mapping) or set(value) != {
            "job_id",
            "candidate_id",
        }:
            raise ValueError(f"EvoX placements[{index}] has an invalid shape")
        job_id = _required_text(value.get("job_id"), "job_id")
        if job_id not in problem_by_id:
            raise ValueError(f"EvoX policy returned unknown job {job_id}")
        if job_id in decisions:
            raise ValueError(f"EvoX policy repeated job {job_id}")
        candidate_id = value.get("candidate_id")
        if candidate_id is not None and not isinstance(candidate_id, str):
            raise ValueError("EvoX candidate_id must be text or null")
        decisions[job_id] = candidate_id
    if set(decisions) != set(problem_by_id):
        raise ValueError("EvoX policy must decide every waiting job exactly once")

    selected = {}
    for job_id, candidate_id in decisions.items():
        if candidate_id is None:
            continue
        candidates = {
            candidate.candidate_id: candidate
            for candidate in problem_by_id[job_id].candidates
        }
        try:
            selected[job_id] = candidates[candidate_id]
        except KeyError as error:
            raise ValueError(
                f"EvoX policy returned an invalid candidate for {job_id}"
            ) from error
    validate_selection(problems, selected, capacities)
    return selected


def load_policy_file(path: Path) -> EvoxPolicy:
    """Load one EvoX candidate program for online evaluation."""
    if not path.is_file():
        raise ValueError(f"EvoX policy does not exist: {path}")
    spec = importlib.util.spec_from_file_location("tandemn_online_evox_file", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load EvoX policy: {path}")
    module = importlib.util.module_from_spec(spec)
    _execute_module(spec.loader, module)
    policy = getattr(module, "compute_placement", None)
    if not callable(policy):
        raise ValueError("EvoX policy must define compute_placement(payload)")
    return cast(EvoxPolicy, policy)


def load_policy_source(source: str) -> EvoxPolicy:
    """Load the online winner directly from SkyDiscover source."""
    module = ModuleType("tandemn_online_evox")
    exec(compile(source, "<online-evox>", "exec"), module.__dict__)
    policy = getattr(module, "compute_placement", None)
    if not callable(policy):
        raise ValueError("EvoX policy must define compute_placement(payload)")
    return cast(EvoxPolicy, policy)


class _UsageBuffer(list[LlmUsage]):
    """Retain completed calls locally and publish them before worker termination."""

    def __init__(self, callback: Callable[[LlmUsage], None] | None) -> None:
        super().__init__()
        self.callback = callback

    def append(self, usage: LlmUsage) -> None:
        super().append(usage)
        if self.callback is not None:
            self.callback(usage)


def run_evox_discovery(
    evaluator: str | Path | Callable[[str], Mapping[str, object]],
    output_dir: str | Path,
    *,
    model: str = "gpt-5",
    iterations: int = 100,
    initial_program: str | Path | None = None,
    api_base: str | None = None,
    system_prompt: str = EVOX_SYSTEM_PROMPT,
    tick_number: int = 0,
    time_budget_seconds: float = 300,
    usage_callback: Callable[[LlmUsage], None] | None = None,
) -> object:
    """Run SkyDiscover EvoX online for one frozen control-tick snapshot."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    try:
        from skydiscover import run_discovery
        from skydiscover.config import Config
        from skydiscover.llm.llm_pool import LLMPool
        from skydiscover.llm.openai import OpenAILLM
    except ImportError as error:
        raise RuntimeError(
            "EvoX discovery requires the sibling SkyDiscover project"
        ) from error

    model_config: dict[str, object] = {"name": model, "weight": 1.0}
    if api_base is not None:
        model_config["api_base"] = api_base
    if api_key := os.getenv("OPENAI_API_KEY"):
        model_config["api_key"] = api_key
    config = Config.from_dict(
        {
            "language": "python",
            "max_iterations": iterations,
            "checkpoint_interval": max(1, min(10, iterations)),
            "diff_based_generation": True,
            "max_solution_length": 60_000,
            "llm": {
                "models": [model_config],
                "max_tokens": 32_000,
                "timeout": 250,
            },
            "search": {
                "type": "evox",
                "num_context_programs": 4,
                "share_llm": True,
                "database": {"auto_generate_variation_operators": True},
            },
            "prompt": {"system_message": system_prompt},
            "evaluator": {
                "timeout": 360,
                "max_retries": 3,
                "cascade_evaluation": False,
            },
        }
    )
    llm_usage: list[LlmUsage] = _UsageBuffer(usage_callback)

    class TrackedOpenAILLM(OpenAILLM):
        def __init__(self, model_config: object) -> None:
            super().__init__(model_config)
            self.client = OpenAIUsageClient(
                self.client,
                tick_number,
                model,
                llm_usage,
            )

    for configured_model in config.llm.models:
        configured_model.init_client = TrackedOpenAILLM
    original_availability_check = LLMPool.check_availability

    async def check_availability(pool: Any, timeout: float = 15.0) -> bool:
        try:
            model = pool.models[0]
            await asyncio.wait_for(
                model.generate(
                    "",
                    [{"role": "user", "content": "ping"}],
                    max_tokens=128,
                ),
                timeout=timeout,
            )
            return True
        except Exception:
            return False

    LLMPool.check_availability = check_availability
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def deadline_exceeded(_signum: int, _frame: object) -> None:
        raise _EvoxDeadlineSignal

    signal.signal(signal.SIGALRM, deadline_exceeded)
    signal.setitimer(signal.ITIMER_REAL, time_budget_seconds)
    try:
        try:
            result = run_discovery(
                evaluator=evaluator,
                initial_program=initial_program or EVOX_INITIAL_PROGRAM,
                iterations=iterations,
                config=config,
                output_dir=str(output_dir),
                cleanup=True,
            )
        except _EvoxDeadlineSignal:
            raise EvoxDeadlineExceeded(tuple(llm_usage)) from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        LLMPool.check_availability = original_availability_check
    return SimpleNamespace(
        best_solution=getattr(result, "best_solution", None),
        llm_usage=tuple(llm_usage),
    )


def _execute_module(loader: object, module: ModuleType) -> None:
    """Execute one importlib loader after its capability was checked."""
    execute = getattr(loader, "exec_module", None)
    if not callable(execute):
        raise RuntimeError("EvoX policy loader cannot execute modules")
    execute(module)


def _required_text(value: object, name: str) -> str:
    """Require one nonempty text value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: object) -> str | None:
    """Return one optional nonempty text value."""
    return None if value is None else _required_text(value, "optional text")


def _positive_int(value: object, name: str) -> int:
    """Require one positive integer value."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _target_id(target: ResourceTarget) -> str:
    """Return a stable identity for one concrete Store resource pool."""
    return f"{target.environment_key}::{target.instance_type or 'aggregate'}"


def _required_vram_gib(
    known: Mapping[str, object],
    benchmark_value: float | None,
) -> float | None:
    """Read an explicit public requirement or the checked-in benchmark fact."""
    for key in ("required_vram_gib", "max_vram_gib", "vram_gib"):
        value = _positive_number(known.get(key))
        if value is not None:
            return value
    return benchmark_value


def _priority(known: Mapping[str, object]) -> tuple[str, int]:
    """Normalize public priority classes to four ordered admission levels."""
    explicit = _finite_number(known.get("priority"))
    if explicit is not None:
        level = min(4, max(1, round(explicit)))
        return f"LEVEL_{level}", level
    raw = getattr(known.get("priority_class"), "value", known.get("priority_class"))
    priority_class = str(raw or "STANDARD").upper()
    return priority_class, _PRIORITY_LEVELS.get(priority_class, 2)


def _target_memory_gib(target: ResourceTarget) -> float | None:
    """Read accelerator memory from Store facts or the static hardware table."""
    sources = (target.pool or {}, target.environment)
    for source in sources:
        for key in ("memory_gib", "memory_gb", "gpu_memory_gib"):
            value = _positive_number(source.get(key))
            if value is not None:
                return value
    return _HARDWARE_MEMORY_GIB.get(_canonical_hardware(target.gpu_type))


def _model_weight_gib(
    model_id: str,
    precision: str,
    fallback: float,
) -> float:
    """Estimate resident model weights independently from KV-cache demand."""
    architecture = MODEL_ARCHITECTURES.get(model_id)
    if architecture is None:
        return fallback
    return architecture.parameter_count * _precision_bytes(precision) / 2**30


def _kv_demand(
    model_id: str,
    precision: str,
    known: Mapping[str, object],
) -> float:
    """Estimate Prism's static token-growth-weighted KV demand."""
    architecture = MODEL_ARCHITECTURES.get(model_id)
    if architecture is None or not architecture.causal:
        return 0.0
    tpot_ms = _positive_number(known.get("target_p99_tpot_ms"))
    if tpot_ms is None:
        return 0.0
    token_rate = _positive_number(known.get("token_growth_rate"))
    if token_rate is None:
        token_rate = _token_growth_rate(known, tpot_ms)
    if token_rate is None:
        return 0.0
    kv_gib_per_token = (
        _cache_elements_per_token(architecture) * _precision_bytes(precision) / 2**30
    )
    return token_rate * kv_gib_per_token / (tpot_ms / 1_000)


def _token_growth_rate(
    known: Mapping[str, object],
    tpot_ms: float,
) -> float | None:
    """Derive a static token rate from Store workload facts."""
    kind = getattr(known.get("kind"), "value", known.get("kind"))
    if kind == "batch":
        token_budget = _positive_number(known.get("total_token_budget"))
        deadline_hours = _positive_number(known.get("deadline_hrs"))
        if token_budget is None or deadline_hours is None:
            return None
        return token_budget / (deadline_hours * 3_600)

    input_tokens = _positive_number(known.get("isl_token_avg"))
    output_tokens = _positive_number(known.get("osl_token_avg"))
    if input_tokens is None or output_tokens is None:
        return None
    request_rate = _positive_number(known.get("request_arrival_rate"))
    if request_rate is None:
        concurrency = _positive_number(known.get("max_concurrent_streaming"))
        ttft_ms = _positive_number(known.get("target_p99_ttft_ms"))
        if concurrency is None or ttft_ms is None:
            return None
        service_seconds = (ttft_ms + max(0.0, output_tokens - 1) * tpot_ms) / 1_000
        if service_seconds <= 0:
            return None
        request_rate = concurrency / service_seconds
    return request_rate * (input_tokens + output_tokens)


def _placement_price(target: ResourceTarget, footprint: int) -> float:
    """Estimate hourly reservation cost from public resource-map prices."""
    source = target.pool or target.environment
    per_gpu = _positive_number(source.get("price_per_gpu_hour"))
    if per_gpu is not None:
        return per_gpu * footprint
    per_instance = _positive_number(source.get("price_per_instance_hour"))
    if per_instance is None:
        return 0.0
    gpus_per_instance = _positive_number(source.get("gpus_per_instance")) or 1.0
    return per_instance * footprint / gpus_per_instance


def _precision_bytes(precision: str) -> int:
    """Return the static storage width implied by a policy precision."""
    normalized = precision.lower()
    if "fp8" in normalized or "int8" in normalized:
        return 1
    if "fp32" in normalized:
        return 4
    return 2


def _tick_rationale(
    method_name: str,
    tick_number: int,
    actions: Sequence[Mapping[str, object]],
    selection_source: str | None = None,
) -> str:
    """Summarize one EvoX action set deterministically."""
    counts = {"place": 0, "defer": 0, "keep": 0, "preempt": 0}
    for action in actions:
        action_type = action.get("type")
        if action_type in counts:
            counts[action_type] += 1
    rationale = (
        f"{method_name} cluster simulator tick {tick_number}: "
        f"place={counts['place']}, defer={counts['defer']}, "
        f"keep={counts['keep']}, preempt={counts['preempt']}."
    )
    if selection_source is not None:
        rationale += f" selection_source={selection_source}."
    return rationale


def _positive_number(value: object) -> float | None:
    """Return one optional finite positive numeric value."""
    number = _finite_number(value)
    return number if number is not None and number > 0 else None


def _finite_number(value: object) -> float | None:
    """Return one optional finite numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _validate_complete_job_actions(
    snapshot: StoreSnapshot,
    actions: Sequence[Mapping[str, object]],
) -> None:
    """Require exactly one canonical action for every Store job."""
    store_ids: list[str] = []
    for jobs in (
        snapshot.running_jobs,
        snapshot.waiting_jobs,
    ):
        store_ids.extend(
            _required_text(job.get("job_id"), "Store job_id") for job in jobs
        )
    action_ids = [
        _required_text(action.get("job_id"), "plan job_id") for action in actions
    ]
    if len(store_ids) != len(set(store_ids)):
        raise ValueError("Store snapshot contains duplicate job_id values")
    if len(action_ids) != len(set(action_ids)) or set(action_ids) != set(store_ids):
        raise ValueError("plan must contain exactly one action for every Store job")


__all__ = [
    "EVOX_INITIAL_PROGRAM",
    "EVOX_SYSTEM_PROMPT",
    "EvoxClient",
    "EvoxPolicy",
    "JobStoreClient",
    "ModelCatalogStoreClient",
    "OnlineEvoxEvaluator",
    "OnlineSkyDiscoverEvoxClient",
    "EvoxPlannerDriver",
    "PlacementCandidate",
    "PlacementProblem",
    "PlanStoreClient",
    "ResourceMapStoreClient",
    "StoreSnapshot",
    "build_evox_plan",
    "build_evox_payload",
    "build_placement_problems",
    "capacity_limits",
    "enumerate_candidates",
    "load_policy_file",
    "load_policy_source",
    "run_evox_discovery",
    "select_evox_placements",
    "validate_selection",
    "verify_place_plan",
]
