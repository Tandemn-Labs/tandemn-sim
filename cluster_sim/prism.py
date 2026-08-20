"""Place models by Prism's static KV-cache pressure heuristic."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from models import CreatedPlan, PlannerResult, Scenario
from policy import ConfigPolicy, load_config_policy
from rank_ids import rank_id
from vram import (
    _HARDWARE_MEMORY_GIB,
    LAUNCH_FIELDS,
    MODEL_ARCHITECTURES,
    VRAM_REQUIREMENTS_GIB,
    ModelCatalogStoreClient,
    ResourceTarget,
    StoreSnapshot,
    _cache_elements_per_token,
    _canonical_hardware,
    _known_job_fields,
    _materialize_launch_config,
    _model_payload,
    _resource_targets,
    _resources_with_free_capacity,
    _running_job_payload,
    _target_available,
    _target_footprint,
    _target_tps,
    _validate_complete_job_actions,
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


class PrismPlacementError(RuntimeError):
    """Report that no GPU can hold one Prism model part."""


@dataclass(frozen=True)
class PrismGpu:
    """One physical GPU exposed to the pure Prism placement algorithm."""

    gpu_id: str
    memory_gib: float


@dataclass(frozen=True)
class PrismModel:
    """One model or tensor-parallel model part placed by Prism."""

    model_id: str
    weight_gib: float
    kv_demand: float
    current_gpu_id: str | None = None
    parallel_group: str | None = None


@dataclass(frozen=True)
class PlacementCandidate:
    """One policy-compliant Prism placement for one waiting job."""

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
        """Return a stable resource-pool identity for pressure balancing."""
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
    """Public static facts and feasible Prism candidates for one waiting job."""

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
    """Expose the Store resource-map fields used by Prism."""

    market: object

    def scheduling_summary(self) -> Mapping[str, Mapping[str, Any]]: ...


class ResourceMapStoreClient(Protocol):
    """Read one current Store resource map."""

    def get(self) -> ResourceMap: ...


class PlanStoreClient(Protocol):
    """Persist one canonical Store plan."""

    def create(self, plan: object) -> object: ...


@dataclass
class _GpuState:
    residual_gib: float
    kv_demand: float = 0.0
    model_ids: list[str] = field(default_factory=list)
    parallel_groups: set[str] = field(default_factory=set)


@dataclass
class _PoolState:
    residual_gib: float = 0.0
    kv_demand: float = 0.0

    @property
    def pressure(self) -> float:
        """Return current aggregate KV pressure for one Store target."""
        if self.residual_gib <= 0:
            return 0.0 if self.kv_demand == 0 else math.inf
        return self.kv_demand / self.residual_gib


class PrismPlannerDriver:
    """Adapt Prism's pressure ordering to aggregate Store resource pools."""

    method_name = "prism"

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
    ) -> None:
        """Keep Store clients and immutable benchmark-to-Store identity."""
        self.user_id = user_id
        self.benchmark_name = scenario.benchmark_name
        jobs = (*scenario.online_jobs, *scenario.batch_jobs)
        self.benchmark_job_ids = {job.store_job_id: job.name for job in jobs}
        self.job_store = job_store
        self.resource_map_store = resource_map_store
        self.model_catalog_store = model_catalog_store
        self.plan_store = plan_store
        self.backend = backend
        self.config_policy = config_policy

    @classmethod
    def build(
        cls,
        user_id: str,
        scenario: Scenario,
        settings: Mapping[str, object],
        postgres_client: object | None = None,
    ) -> PrismPlannerDriver:
        """Build Prism with Tandemn Store public clients."""
        from tandemn_system_data.clients import (
            JobStore,
            ModelCatalogStore,
            PlanStore,
            PostgresClient,
            ResourceMapStore,
        )

        postgres = postgres_client or PostgresClient()
        return cls(
            user_id=user_id,
            scenario=scenario,
            job_store=JobStore(postgres),
            resource_map_store=ResourceMapStore(postgres, user_id=user_id),
            model_catalog_store=ModelCatalogStore(postgres),
            plan_store=PlanStore(postgres),
            backend=_required_text(settings.get("backend"), "backend"),
            config_policy=load_config_policy(),
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
                tick_rationale=_tick_rationale(self.method_name, tick_number, actions),
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
        waiting_jobs = tuple(_model_payload(job) for job in waiting)
        model_ids = {
            _required_text(_known_job_fields(job).get("model_id"), "model_id")
            for job in waiting_jobs
        }
        model_catalogs = {}
        for model_id in sorted(model_ids):
            catalog = self.model_catalog_store.get(model_id)
            if catalog is None:
                raise ValueError(f"model catalog missing {model_id!r}")
            model_catalogs[model_id] = _model_payload(catalog)
        return StoreSnapshot(
            resources=_resources_with_free_capacity(resource_map, running),
            running_jobs=tuple(_running_job_payload(job) for job in running),
            waiting_jobs=waiting_jobs,
            paused_jobs=tuple(_model_payload(job) for job in paused),
            finished_jobs=(),
            model_catalogs=model_catalogs,
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
            plan = self._build_place_plan(
                problem,
                candidate,
                snapshot.model_catalogs,
            )
            verify_place_plan(
                plan,
                candidate,
                snapshot.model_catalogs,
                self.config_policy,
            )
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
        """Place high KV-demand jobs onto the least-pressured target."""
        return select_prism_placements(problems, capacities)

    def _build_place_plan(
        self,
        problem: PlacementProblem,
        candidate: PlacementCandidate,
        model_catalogs: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, object]:
        """Build Prism's complete per-job placement plan."""
        return build_prism_plan(
            problem.store_job,
            candidate,
            model_catalogs,
            self.config_policy,
        )


def build_placement_problems(
    waiting_jobs: Sequence[Mapping[str, Any]],
    resources: dict[str, dict[str, Any]],
    benchmark_name: str,
    benchmark_job_ids: Mapping[str, str],
    config_policy: ConfigPolicy,
) -> tuple[PlacementProblem, ...]:
    """Build feasible Prism candidates for every waiting Store job."""
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
    """Enumerate policy-compliant Prism shapes that fit current capacity."""
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
    """Require known, non-overcommitted candidates from Prism selection."""
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
    model_catalogs: Mapping[str, Mapping[str, Any]],
    config_policy: ConfigPolicy,
) -> None:
    """Verify one complete Prism placement plan before it is returned."""
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
    required_config_keys = set(LAUNCH_FIELDS) | {
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
    model_id = config.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("placement plan config requires model_id")
    rule = config_policy.rule_for(candidate.target.gpu_type, model_id)
    if rule is None or candidate.tp not in rule.allowed_tp:
        raise ValueError("placement plan violates the configuration policy")
    if config.get("precision") != rule.precision:
        raise ValueError("placement plan precision violates the configuration policy")
    try:
        catalog = model_catalogs[model_id]
    except KeyError as error:
        raise ValueError(f"model catalog missing {model_id!r}") from error
    launch_config = _materialize_launch_config(
        catalog,
        candidate.target.gpu_type,
    )
    for launch_field, launch_value in launch_config.items():
        if config.get(launch_field) != launch_value:
            raise ValueError(f"placement plan config has invalid {launch_field}")


def build_prism_plan(
    store_job: Mapping[str, object],
    candidate: PlacementCandidate,
    model_catalogs: Mapping[str, Mapping[str, Any]],
    config_policy: ConfigPolicy,
) -> dict[str, object]:
    """Return one complete Prism placement plan."""
    known = _known_job_fields(store_job)
    model_id = _required_text(known.get("model_id"), "model_id")
    try:
        catalog = model_catalogs[model_id]
    except KeyError as error:
        raise ValueError(f"model catalog missing {model_id!r}") from error
    launch_config = _materialize_launch_config(
        catalog,
        candidate.target.gpu_type,
    )
    rule = config_policy.rule_for(candidate.target.gpu_type, model_id)
    if rule is None or candidate.tp not in rule.allowed_tp:
        raise ValueError("Prism selected a policy-invalid placement")
    return {
        "job_id": candidate.job_id,
        "type": "place",
        "target_tps": _target_tps(known),
        "target_p99_ttft_ms": _number(known.get("target_p99_ttft_ms")),
        "target_p99_tpot_ms": _number(known.get("target_p99_tpot_ms")),
        "ladder": [
            {
                "role": known.get("role") or "aggregate",
                "rank_id": rank_id(known),
                "env": candidate.target.environment_key.split("|"),
                "config": {
                    **launch_config,
                    "model_id": model_id,
                    "precision": rule.precision,
                    "instance_type": candidate.target.instance_type,
                    "gpu_type": candidate.target.gpu_type,
                    "gpu_count": candidate.gpu_count,
                    "count": candidate.gpu_count,
                    "tp": candidate.tp,
                    "pp": candidate.pp,
                    "dp": candidate.dp,
                    "ep": known.get("ep") or 1,
                    "rank_traffic_share": 1.0,
                },
                "n_replicas": 1,
                "mechanism_id": known.get("mechanism_id"),
            }
        ],
    }


def compute_model_placement(
    gpus: Sequence[PrismGpu],
    models: Sequence[PrismModel],
    migration_threshold: float = 0.0,
) -> dict[str, tuple[str, ...]]:
    """Implement Prism Algorithm 1 with retention and TP anti-affinity."""
    if migration_threshold < 0:
        raise ValueError("migration_threshold cannot be negative")
    if len({gpu.gpu_id for gpu in gpus}) != len(gpus):
        raise ValueError("Prism GPU IDs must be unique")
    if len({model.model_id for model in models}) != len(models):
        raise ValueError("Prism model IDs must be unique")

    states = {}
    for gpu in gpus:
        if not math.isfinite(gpu.memory_gib) or gpu.memory_gib <= 0:
            raise ValueError("Prism GPU memory must be positive")
        states[gpu.gpu_id] = _GpuState(residual_gib=gpu.memory_gib)

    ordered = sorted(models, key=lambda model: (-model.kv_demand, model.model_id))
    for model in ordered:
        if not math.isfinite(model.weight_gib) or model.weight_gib <= 0:
            raise ValueError("Prism model weight must be positive")
        if not math.isfinite(model.kv_demand) or model.kv_demand < 0:
            raise ValueError("Prism KV demand cannot be negative")

        eligible = [
            gpu_id
            for gpu_id, state in states.items()
            if model.weight_gib < state.residual_gib
            and (
                model.parallel_group is None
                or model.parallel_group not in state.parallel_groups
            )
        ]
        if not eligible:
            raise PrismPlacementError(f"no GPU can fit {model.model_id}")
        best_gpu = min(
            eligible,
            key=lambda gpu_id: (_pressure(states[gpu_id]), gpu_id),
        )
        current_gpu = model.current_gpu_id
        if current_gpu is not None and current_gpu in eligible:
            current_pressure = _pressure(states[current_gpu])
            best_pressure = _pressure(states[best_gpu])
            if current_pressure - best_pressure <= migration_threshold:
                best_gpu = current_gpu

        state = states[best_gpu]
        state.residual_gib -= model.weight_gib
        state.kv_demand += model.kv_demand
        state.model_ids.append(model.model_id)
        if model.parallel_group is not None:
            state.parallel_groups.add(model.parallel_group)

    return {gpu_id: tuple(state.model_ids) for gpu_id, state in sorted(states.items())}


def select_prism_placements(
    problems: Sequence[PlacementProblem],
    capacities: Mapping[str, int],
) -> dict[str, PlacementCandidate]:
    """Apply Prism's pressure rule to Store-expressible exclusive shapes."""
    remaining = dict(capacities)
    states: dict[str, _PoolState] = {}
    selected = {}
    ordered = sorted(
        problems,
        key=lambda problem: (
            -max(
                (candidate.kv_demand for candidate in problem.candidates),
                default=0.0,
            ),
            -problem.priority,
            problem.job_id,
        ),
    )
    for problem in ordered:
        feasible = [
            candidate
            for candidate in problem.candidates
            if candidate.model_weight_gib < candidate.footprint * candidate.memory_gib
            and all(
                remaining.get(key, 0) >= candidate.footprint
                for key in candidate.capacity_keys
            )
        ]
        if not feasible:
            continue
        candidate = min(
            feasible,
            key=lambda value: (
                states.get(value.target_id, _PoolState()).pressure,
                value.price_per_hour,
                value.footprint,
                value.candidate_id,
            ),
        )
        selected[problem.job_id] = candidate
        for key in candidate.capacity_keys:
            remaining[key] -= candidate.footprint
        state = states.setdefault(candidate.target_id, _PoolState())
        state.residual_gib += (
            candidate.footprint * candidate.memory_gib - candidate.model_weight_gib
        )
        state.kv_demand += candidate.kv_demand
    return selected


def _target_id(target: ResourceTarget) -> str:
    """Return the same target identity used by PlacementCandidate."""
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
    explicit = _number(known.get("priority"))
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
    """Summarize one baseline's deterministic action set."""
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
    """Return one optional positive numeric value."""
    number = _number(value)
    return number if number is not None and number > 0 else None


def _number(value: object) -> float | None:
    """Return one optional finite numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _pressure(state: _GpuState) -> float:
    """Return one physical GPU's current Prism pressure."""
    if state.residual_gib <= 0:
        return math.inf
    return state.kv_demand / state.residual_gib


def _required_text(value: object, name: str) -> str:
    """Require one nonempty text value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


__all__ = [
    "JobStoreClient",
    "PlacementCandidate",
    "PlacementProblem",
    "PlanStoreClient",
    "PrismGpu",
    "PrismModel",
    "PrismPlacementError",
    "PrismPlannerDriver",
    "ResourceMap",
    "ResourceMapStoreClient",
    "StoreSnapshot",
    "build_placement_problems",
    "build_prism_plan",
    "capacity_limits",
    "compute_model_placement",
    "enumerate_candidates",
    "select_prism_placements",
    "validate_selection",
    "verify_place_plan",
]
