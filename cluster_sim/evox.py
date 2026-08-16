"""Evolve placement programs online with EvoX for each control tick."""

from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

from models import CreatedPlan, PlannerResult, Scenario
from policy import ConfigPolicy, load_config_policy
from vram import (
    MODEL_ARCHITECTURES,
    VRAM_REQUIREMENTS_GIB,
    ResourceTarget,
    _HARDWARE_MEMORY_GIB,
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

EVOX_SYSTEM_PROMPT = """You are evolving a deterministic GPU placement program
online for the current simulator control tick.

Improve only compute_placement(payload). The input uses schema
tandemn-evox-placement/v1 and contains public static facts: jobs, each job's
priority and feasible policy-compliant candidates, and nested resource
capacities. Each candidate includes its exact capacity consumption, GPU shape,
VRAM demand, model-weight memory, estimated KV demand, and hourly price.

Return {"placements": [{"job_id": str, "candidate_id": str | null}, ...]} with
exactly one entry for every input job. A non-null candidate must belong to that
job. Across selected candidates, consumption for every capacity key must not
exceed its available value. The program must be deterministic, self-contained,
side-effect free, and use no third-party packages or hidden runtime state.

The evaluator rejects malformed or infeasible output and exposes a
larger-is-better combined_score. Preserve high-priority admission before
improving lower-priority admission, then seek lower cost and capacity usage.
Explore better global assignment, packing, ordering, and tie-breaking rather
than adding prose or external calls.
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


@dataclass(frozen=True)
class PlacementCandidate:
    """Represent one policy-compliant placement for a waiting EvoX job."""

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

    @property
    def gpu_count(self) -> int:
        """Return the logical accelerator count in the serving shape."""
        return self.tp * self.pp * self.dp

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
    """Hold public facts and feasible placements for one waiting EvoX job."""

    job_id: str
    priority: int
    priority_class: str
    store_job: dict[str, Any]
    candidates: tuple[PlacementCandidate, ...]


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
        problems: Sequence[PlacementProblem],
        capacities: Mapping[str, int],
    ) -> Mapping[str, PlacementCandidate]: ...


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

    def select(
        self,
        problems: Sequence[PlacementProblem],
        capacities: Mapping[str, int],
    ) -> Mapping[str, PlacementCandidate]:
        """Evolve, verify, and execute the current tick's winning program."""
        evaluator = OnlineEvoxEvaluator(problems, capacities)
        result = run_evox_discovery(
            evaluator.evaluate,
            self.output_dir / f"tick-{self.tick_number:06d}",
            model=self.model,
            iterations=self.iterations,
            initial_program=self.initial_program,
            api_base=self.api_base,
        )
        best_solution = getattr(result, "best_solution", None)
        if not isinstance(best_solution, str) or not best_solution:
            raise RuntimeError("online EvoX returned no placement program")
        policy = load_policy_source(best_solution)
        selected = select_evox_placements(policy, problems, capacities)
        self.initial_program = best_solution
        self.tick_number += 1
        return selected


class OnlineEvoxEvaluator:
    """Score online candidate programs using verified static placement facts."""

    def __init__(
        self,
        problems: Sequence[PlacementProblem],
        capacities: Mapping[str, int],
    ) -> None:
        """Freeze the current tick throughout one online evolution run."""
        self.problems = tuple(problems)
        self.capacities = dict(capacities)

    def evaluate(self, program_path: str) -> dict[str, object]:
        """Return a larger-is-better score after strict plan verification."""
        try:
            policy = load_policy_file(Path(program_path))
            selected = select_evox_placements(
                policy,
                self.problems,
                self.capacities,
            )
        except Exception:
            return {
                "combined_score": 0.0,
                "feasible": 0.0,
                "invalid_program": 1.0,
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
        tie_break = 0.0
        if selected:
            tie_break = 1 / (1 + price) + 1 / (1_000 * (1 + footprint))
        return {
            "combined_score": float(2 * priority_score + tie_break),
            "feasible": 1.0,
            "admitted_jobs": float(len(selected)),
            "priority_score": float(priority_score),
            "hourly_price": float(price),
            "gpu_footprint": float(footprint),
        }


class EvoxPlannerDriver:
    """Create plans with online LLM program evolution on every tick."""

    method_name = "evox"

    def __init__(
        self,
        user_id: str,
        scenario: Scenario,
        job_store: JobStoreClient,
        resource_map_store: ResourceMapStoreClient,
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
        self.job_store = job_store
        self.resource_map_store = resource_map_store
        self.plan_store = plan_store
        self.backend = backend
        self.config_policy = config_policy
        self.client = client

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
            JobStore,
            PlanStore,
            PostgresClient,
            ResourceMapStore,
        )

        model = _required_text(settings.get("model"), "model")
        iterations = _positive_int(settings.get("iterations", 8), "iterations")
        output_dir = Path(
            _optional_text(settings.get("output_dir")) or "outputs/evox-online"
        )
        api_base = _optional_text(settings.get("api_base"))
        postgres = postgres_client or PostgresClient()
        return cls(
            user_id=user_id,
            scenario=scenario,
            job_store=JobStore(postgres),
            resource_map_store=ResourceMapStore(postgres, user_id=user_id),
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

    def create_plans(
        self,
        tick_number: int,
        logical_time: datetime,
    ) -> PlannerResult:
        """Read one Store snapshot, persist one plan, and return its identity."""
        del logical_time
        try:
            snapshot = self._read_store_snapshot()
            actions = self._build_actions(snapshot)
            if not actions:
                return PlannerResult(tick_number=tick_number, plans=())

            from tandemn_system_data.models import Plan

            plan = Plan(
                user_id=self.user_id,
                koi_version=self.method_name,
                tick_rationale=_tick_rationale(
                    self.method_name,
                    tick_number,
                    actions,
                ),
                actions=actions,
            )
            self.plan_store.create(plan)
            return PlannerResult(
                tick_number=tick_number,
                plans=(CreatedPlan(plan_id=plan.plan_id),),
            )
        except Exception as error:
            return PlannerResult(
                tick_number=tick_number,
                plans=(),
                degraded=True,
                error=repr(error),
            )

    def _read_store_snapshot(self) -> StoreSnapshot:
        """Freeze resources and canonical lifecycle groups for one tick."""
        resource_map = self.resource_map_store.get()
        running = tuple(self.job_store.running_jobs(self.user_id))
        waiting = tuple(self.job_store.waiting_jobs(self.user_id))
        paused = tuple(self.job_store.paused_jobs(self.user_id))
        return StoreSnapshot(
            resources=_resources_with_free_capacity(resource_map, running),
            running_jobs=tuple(_running_job_payload(job) for job in running),
            waiting_jobs=tuple(_model_payload(job) for job in waiting),
            paused_jobs=tuple(_model_payload(job) for job in paused),
            finished_jobs=(),
        )

    def _build_actions(self, snapshot: StoreSnapshot) -> list[dict[str, object]]:
        """Select waiting placements and include every lifecycle action."""
        resources = deepcopy(snapshot.resources)
        problems = build_placement_problems(
            snapshot.waiting_jobs,
            resources,
            self.benchmark_name,
            self.benchmark_job_ids,
            self.config_policy,
        )
        capacities = capacity_limits(resources)
        selected = dict(self._select_placements(problems, capacities))
        validate_selection(problems, selected, capacities)

        actions: list[dict[str, object]] = []
        for running in snapshot.running_jobs:
            actions.append({"job_id": running["job_id"], "type": "keep"})
        for problem in problems:
            candidate = selected.get(problem.job_id)
            if candidate is None:
                actions.append({"job_id": problem.job_id, "type": "defer"})
                continue
            plan = self._build_place_plan(problem, candidate)
            verify_place_plan(plan, candidate, self.config_policy)
            reserve_vram_placement(candidate.target, candidate.gpu_count)
            actions.append(plan)
        for paused in snapshot.paused_jobs:
            actions.append({"job_id": paused["job_id"], "type": "defer"})
        _validate_complete_job_actions(snapshot, actions)
        return actions

    def _select_placements(
        self,
        problems: Sequence[PlacementProblem],
        capacities: Mapping[str, int],
    ) -> Mapping[str, PlacementCandidate]:
        """Evolve and validate the current tick's placement response."""
        if not problems:
            return {}
        return self.client.select(problems, capacities)

    def _build_place_plan(
        self,
        problem: PlacementProblem,
        candidate: PlacementCandidate,
    ) -> dict[str, object]:
        """Build EvoX's complete per-job placement plan."""
        return build_evox_plan(
            problem.store_job,
            candidate,
            self.backend,
            self.config_policy,
        )


def build_evox_plan(
    store_job: Mapping[str, object],
    candidate: PlacementCandidate,
    backend: str,
    config_policy: ConfigPolicy,
) -> dict[str, object]:
    """Return one complete online-EvoX placement plan."""
    known = _known_job_fields(store_job)
    model_id = _required_text(known.get("model_id"), "model_id")
    rule = config_policy.rule_for(candidate.target.gpu_type, model_id)
    if rule is None or candidate.tp not in rule.allowed_tp:
        raise ValueError("EvoX selected a policy-invalid placement")
    return {
        "job_id": candidate.job_id,
        "type": "place",
        "tenant_id": known.get("tenant_id") or store_job.get("user_id"),
        "target_tps": _target_tps(known),
        "target_p99_ttft_ms": _number(known.get("target_p99_ttft_ms")),
        "target_p99_tpot_ms": _number(known.get("target_p99_tpot_ms")),
        "mechanism_id": known.get("mechanism_id"),
        "budget_ref": known.get("budget_ref"),
        "rationale": known.get("rationale"),
        "ladder": [
            {
                "role": known.get("role") or "aggregate",
                "rank_id": known.get("rank_id") or "rank_0",
                "env": candidate.target.environment_key.split("|"),
                "config": {
                    "model_id": model_id,
                    "precision": rule.precision,
                    "engine_name": known.get("engine_name") or backend,
                    "instance_type": candidate.target.instance_type,
                    "gpu_type": candidate.target.gpu_type,
                    "gpu_count": candidate.gpu_count,
                    "count": known.get("count"),
                    "tp": candidate.tp,
                    "pp": candidate.pp,
                    "dp": candidate.dp,
                    "ep": known.get("ep"),
                    "max_num_seq": known.get("max_num_seq"),
                    "max_num_batched_tokens": known.get("max_num_batched_tokens"),
                    "gpu_mem_util": known.get("gpu_mem_util"),
                    "prefix_cache_enabled": known.get("prefix_cache_enabled"),
                    "rank_traffic_share": 1.0,
                },
                "n_replicas": 1,
                "mechanism_id": known.get("mechanism_id"),
            }
        ],
    }


def build_placement_problems(
    waiting_jobs: Sequence[Mapping[str, Any]],
    resources: dict[str, dict[str, Any]],
    benchmark_name: str,
    benchmark_job_ids: Mapping[str, str],
    config_policy: ConfigPolicy,
) -> tuple[PlacementProblem, ...]:
    """Build feasible EvoX candidates for every waiting Store job."""
    requirements = VRAM_REQUIREMENTS_GIB.get(benchmark_name, {})
    problems = []
    for waiting in waiting_jobs:
        store_job = dict(waiting)
        job_id = _required_text(store_job.get("job_id"), "waiting Store job_id")
        known = _known_job_fields(store_job)
        benchmark_job_id = benchmark_job_ids.get(job_id, job_id)
        required_vram_gib = _required_vram_gib(
            known,
            requirements.get(benchmark_job_id),
        )
        candidates = ()
        if required_vram_gib is not None:
            model_id = _required_text(known.get("model_id"), "model_id")
            candidates = enumerate_candidates(
                job_id,
                model_id,
                known,
                required_vram_gib,
                resources,
                config_policy,
            )
        priority_class, priority = _priority(known)
        problems.append(
            PlacementProblem(
                job_id=job_id,
                priority=priority,
                priority_class=priority_class,
                store_job=store_job,
                candidates=candidates,
            )
        )
    return tuple(problems)


def enumerate_candidates(
    job_id: str,
    model_id: str,
    known: Mapping[str, object],
    required_vram_gib: float,
    resources: dict[str, dict[str, Any]],
    config_policy: ConfigPolicy,
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
        minimum_gpus = math.ceil(required_vram_gib / memory_gib)
        for tp in sorted(rule.allowed_tp):
            pp = math.ceil(minimum_gpus / tp)
            dp = 1
            gpu_count = tp * pp * dp
            footprint = _target_footprint(target, gpu_count)
            if footprint is None or footprint > _target_available(target):
                continue
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
                )
            )
    return tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id))


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
) -> None:
    """Verify one complete per-job placement plan before it is returned."""
    required_plan_keys = {
        "job_id",
        "type",
        "tenant_id",
        "target_tps",
        "target_p99_ttft_ms",
        "target_p99_tpot_ms",
        "mechanism_id",
        "budget_ref",
        "rationale",
        "ladder",
    }
    if set(plan) != required_plan_keys:
        raise ValueError("placement plan has incomplete top-level fields")
    if plan.get("job_id") != candidate.job_id or plan.get("type") != "place":
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
    if rank.get("n_replicas") != 1:
        raise ValueError("placement plan must use one replica group")
    config = rank.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("placement ladder config must be an object")
    required_config_keys = {
        "model_id",
        "precision",
        "engine_name",
        "instance_type",
        "gpu_type",
        "gpu_count",
        "count",
        "tp",
        "pp",
        "dp",
        "ep",
        "max_num_seq",
        "max_num_batched_tokens",
        "gpu_mem_util",
        "prefix_cache_enabled",
        "rank_traffic_share",
    }
    if set(config) != required_config_keys:
        raise ValueError("placement plan config has incomplete fields")
    expected = {
        "gpu_type": candidate.target.gpu_type,
        "instance_type": candidate.target.instance_type,
        "gpu_count": candidate.gpu_count,
        "tp": candidate.tp,
        "pp": candidate.pp,
        "dp": candidate.dp,
        "rank_traffic_share": 1.0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"placement plan config has invalid {key}")
    model_id = config.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("placement plan config requires model_id")
    rule = config_policy.rule_for(candidate.target.gpu_type, model_id)
    if rule is None or candidate.tp not in rule.allowed_tp:
        raise ValueError("placement plan violates the configuration policy")
    if config.get("precision") != rule.precision:
        raise ValueError("placement plan precision violates the configuration policy")


def build_evox_payload(
    problems: Sequence[PlacementProblem],
    capacities: Mapping[str, int],
) -> dict[str, object]:
    """Serialize public placement facts for current-tick evolution."""
    jobs = []
    for problem in problems:
        candidates = []
        for candidate in problem.candidates:
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
                    "footprint": candidate.footprint,
                    "memory_gib": candidate.memory_gib,
                    "required_vram_gib": candidate.required_vram_gib,
                    "model_weight_gib": candidate.model_weight_gib,
                    "kv_demand": candidate.kv_demand,
                    "price_per_hour": candidate.price_per_hour,
                    "consumption": {
                        key: candidate.footprint for key in candidate.capacity_keys
                    },
                }
            )
        jobs.append(
            {
                "job_id": problem.job_id,
                "priority": problem.priority,
                "priority_class": problem.priority_class,
                "candidates": candidates,
            }
        )
    return {
        "schema_version": "tandemn-evox-placement/v1",
        "capacities": [
            {"capacity_id": key, "available": value}
            for key, value in sorted(capacities.items())
        ],
        "jobs": jobs,
    }


def select_evox_placements(
    policy: EvoxPolicy,
    problems: Sequence[PlacementProblem],
    capacities: Mapping[str, int],
) -> dict[str, PlacementCandidate]:
    """Run and strictly validate one evolved program's output."""
    result = policy(build_evox_payload(problems, capacities))
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


def run_evox_discovery(
    evaluator: str | Path | Callable[[str], Mapping[str, object]],
    output_dir: str | Path,
    *,
    model: str = "gpt-5",
    iterations: int = 100,
    initial_program: str | Path | None = None,
    api_base: str | None = None,
) -> object:
    """Run SkyDiscover EvoX online for one frozen control-tick snapshot."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    sibling_root = Path(__file__).resolve().parents[2] / "skydiscover"
    if sibling_root.is_dir() and str(sibling_root) not in sys.path:
        sys.path.insert(0, str(sibling_root))
    try:
        from skydiscover import run_discovery
        from skydiscover.config import Config
    except ImportError as error:
        raise RuntimeError(
            "EvoX discovery requires the sibling SkyDiscover project"
        ) from error

    model_config: dict[str, object] = {"name": model, "weight": 1.0}
    if api_base is not None:
        model_config["api_base"] = api_base
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
                "timeout": 600,
            },
            "search": {
                "type": "evox",
                "num_context_programs": 4,
                "share_llm": True,
                "database": {"auto_generate_variation_operators": True},
            },
            "prompt": {"system_message": EVOX_SYSTEM_PROMPT},
            "evaluator": {
                "timeout": 360,
                "max_retries": 3,
                "cascade_evaluation": False,
            },
        }
    )
    return run_discovery(
        evaluator=evaluator,
        initial_program=initial_program or EVOX_INITIAL_PROGRAM,
        iterations=iterations,
        config=config,
        output_dir=str(output_dir),
        cleanup=True,
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
) -> str:
    """Summarize one EvoX action set deterministically."""
    counts = {"place": 0, "defer": 0, "keep": 0, "preempt": 0}
    for action in actions:
        action_type = action.get("type")
        if action_type in counts:
            counts[action_type] += 1
    return (
        f"{method_name} cluster simulator tick {tick_number}: "
        f"place={counts['place']}, defer={counts['defer']}, "
        f"keep={counts['keep']}, preempt={counts['preempt']}."
    )


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
        snapshot.paused_jobs,
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
