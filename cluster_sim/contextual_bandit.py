"""Place jobs with a seeded contextual Thompson-sampling baseline.

The baseline sees only scheduler-visible Store fields, policy-compliant candidate
placements, and completed simulated requests. It never reads Koi state or uses
Koi's mechanism graph.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from random import Random
from typing import Any, Protocol

from models import CreatedPlan, PlannerResult, Scenario
from policy import ConfigPolicy, load_config_policy
from rank_ids import active_rank_id, rank_id, reconcile_swap_rank_ids
from vram import (
    _HARDWARE_MEMORY_GIB,
    LAUNCH_FIELDS,
    MODEL_ARCHITECTURES,
    VRAM_REQUIREMENTS_GIB,
    ModelCatalogStoreClient,
    ResourceTarget,
    _cache_elements_per_token,
    _canonical_hardware,
    _known_job_fields,
    _materialize_launch_config,
    _model_payload,
    _number,
    _resource_targets,
    _resources_with_free_capacity,
    _running_job_payload,
    _target_available,
    _target_footprint,
    _target_tps,
    _text,
    _validate_complete_job_actions,
    reserve_vram_placement,
)

_FEATURE_COUNT = 11
_PRIORITY_LEVELS = {
    "BEST_EFFORT": 1,
    "LOW": 1,
    "NORMAL": 2,
    "STANDARD": 2,
    "HIGH": 3,
    "CRITICAL": 4,
    "P0": 4,
}


@dataclass(frozen=True)
class StoreSnapshot:
    """One frozen Store view used throughout a contextual-bandit tick."""

    resources: dict[str, dict[str, Any]]
    running_jobs: tuple[dict[str, Any], ...]
    waiting_jobs: tuple[dict[str, Any], ...]
    paused_jobs: tuple[dict[str, Any], ...]
    finished_jobs: tuple[dict[str, Any], ...]
    model_catalogs: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class PlacementCandidate:
    """One policy-compliant placement for one Store job."""

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
        """Return the accelerator count for each serving replica."""
        return self.tp * self.pp

    @property
    def target_id(self) -> str:
        """Return a stable resource-pool identity for selection."""
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
    """Public static facts and feasible candidates for one Store job."""

    job_id: str
    priority: int
    priority_class: str
    store_job: dict[str, Any]
    candidates: tuple[PlacementCandidate, ...]
    action_type: str = "place"
    rank_id: str | None = None


class JobStoreClient(Protocol):
    """Read canonical lifecycle groups from Tandemn Store."""

    def running_jobs(self, user_id: str) -> Sequence[object]: ...

    def waiting_jobs(self, user_id: str) -> Sequence[object]: ...

    def paused_jobs(self, user_id: str) -> Sequence[object]: ...


class ResourceMap(Protocol):
    """Expose the Store resource-map fields used by this adapter."""

    market: object

    def scheduling_summary(self) -> Mapping[str, Mapping[str, Any]]: ...


class ResourceMapStoreClient(Protocol):
    """Read one current Store resource map."""

    def get(self) -> ResourceMap: ...


class PlanStoreClient(Protocol):
    """Persist one canonical Store plan."""

    def create(self, plan: object) -> object: ...


@dataclass(frozen=True)
class _PendingObservation:
    """Keep a placement context until Store exposes its outcome."""

    features: tuple[float, ...]
    rank_id: str
    window_start: datetime
    required_tps: float | None
    is_batch: bool
    ttft_target_ms: float | None
    tpot_target_ms: float | None


class ContextualBanditPlannerDriver:
    """Choose safe placements with an online linear Thompson sampler."""

    method_name = "contextual_bandit"

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
        exploration_scale: float = 0.25,
        seed: int = 0,
    ) -> None:
        """Initialize a reproducible posterior over public action features."""
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
        if not math.isfinite(exploration_scale) or exploration_scale <= 0:
            raise ValueError("exploration_scale must be positive and finite")
        self.exploration_scale = exploration_scale
        self._precision = [
            [1.0 if row == column else 0.0 for column in range(_FEATURE_COUNT)]
            for row in range(_FEATURE_COUNT)
        ]
        self._reward_weight = [0.0] * _FEATURE_COUNT
        self._pending: dict[str, _PendingObservation] = {}
        self._random = Random(seed)
        self.observation_count = 0
        self.last_reward: float | None = None

    @classmethod
    def build(
        cls,
        user_id: str,
        scenario: Scenario,
        postgres_client: object | None = None,
    ) -> ContextualBanditPlannerDriver:
        """Build the baseline with the same Store clients as static planners."""
        from tandemn_system_data.clients import (
            GpuMetricStore,
            JobStore,
            ModelCatalogStore,
            PlanStore,
            PostgresClient,
            ResourceMapStore,
        )
        from tandemn_system_data.clients.event_log import PostgresEventLog

        postgres = postgres_client or PostgresClient()
        driver = cls(
            user_id=user_id,
            scenario=scenario,
            job_store=JobStore(postgres),
            resource_map_store=ResourceMapStore(postgres, user_id=user_id),
            model_catalog_store=ModelCatalogStore(postgres),
            plan_store=PlanStore(postgres),
            backend="vllm",
            config_policy=load_config_policy(),
        )
        driver.metric_store = GpuMetricStore(postgres)
        driver.event_log = PostgresEventLog(postgres)
        driver._event_cursor: str | None = None
        return driver

    def create_plans(
        self,
        tick_number: int,
        logical_time: datetime,
    ) -> PlannerResult:
        """Read one Store snapshot, persist one plan, and return its identity."""
        try:
            self._observe_store_results(logical_time)
            self._feedback_start = logical_time
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

    def _observe_store_results(self, logical_time: datetime) -> None:
        """Learn only from this baseline's prior public Store outcomes."""
        for event in self.event_log.read_after(
            self._event_cursor,
            types={"batch.outcome"},
        ):
            self._event_cursor = event.event_id
            if event.user_id != self.user_id or event.job_id is None:
                continue
            observation = self._pending.get(event.job_id)
            if observation is None or not observation.is_batch:
                continue
            self._observe_metrics(
                event.job_id,
                observation,
                logical_time,
                batch_deadline_met=event.payload_json.get("deadline_met") is True,
            )

        for job_id, observation in tuple(self._pending.items()):
            if not observation.is_batch:
                self._observe_metrics(job_id, observation, logical_time)

    def _observe_metrics(
        self,
        job_id: str,
        observation: _PendingObservation,
        logical_time: datetime,
        *,
        batch_deadline_met: bool | None = None,
    ) -> None:
        """Aggregate one action's persisted rank telemetry into a reward."""
        metrics = self.metric_store.rows_for_rank_window(
            self.user_id,
            job_id,
            observation.rank_id,
            observation.window_start,
            logical_time,
        )
        if not metrics and batch_deadline_met is None:
            return
        self.observe_results(
            job_id,
            p99_ttft_ms=_worst_metric(metrics, "p99_ttft_ms"),
            p99_tpot_ms=_worst_metric(metrics, "p99_tpot_ms"),
            throughput_token_per_sec=_mean_rank_throughput(metrics),
            batch_deadline_met=batch_deadline_met,
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
        """Select waiting placements and running-job swaps for one tick."""
        resources = deepcopy(snapshot.resources)
        problems = build_placement_problems(
            snapshot.waiting_jobs,
            resources,
            self.benchmark_name,
            self.benchmark_job_ids,
            self.config_policy,
            running_jobs=snapshot.running_jobs,
        )
        capacities = capacity_limits(resources)
        selected = dict(self._select_placements(problems, capacities))
        validate_selection(problems, selected, capacities)
        problems_by_id = {problem.job_id: problem for problem in problems}

        def selected_plan(problem: PlacementProblem) -> dict[str, object] | None:
            candidate = selected.get(problem.job_id)
            if candidate is None:
                return None
            plan = self._build_place_plan(
                problem,
                candidate,
                snapshot.model_catalogs,
            )
            if problem.action_type == "swap":
                if reconcile_swap_rank_ids(problem.store_job, plan):
                    self._pending.pop(problem.job_id, None)
                    return None
                emitted_rank_id = str(plan["ladder"][0]["rank_id"])
                self._pending[problem.job_id] = replace(
                    self._pending[problem.job_id],
                    rank_id=emitted_rank_id,
                )
            verify_place_plan(
                plan,
                candidate,
                snapshot.model_catalogs,
                self.config_policy,
                action_type=problem.action_type,
                rank_name=self._pending[problem.job_id].rank_id,
            )
            reserve_vram_placement(
                candidate.target,
                candidate.gpu_count,
                candidate.dp,
            )
            return plan

        actions: list[dict[str, object]] = []
        for running in snapshot.running_jobs:
            job_id = _text(running.get("job_id"), "running Store job_id")
            problem = problems_by_id.get(job_id)
            plan = None if problem is None else selected_plan(problem)
            actions.append(plan or {"job_id": job_id, "type": "keep"})
        for waiting in snapshot.waiting_jobs:
            job_id = _text(waiting.get("job_id"), "waiting Store job_id")
            problem = problems_by_id[job_id]
            plan = selected_plan(problem)
            actions.append(plan or {"job_id": job_id, "type": "defer"})
        _validate_complete_job_actions(snapshot, actions)
        return actions

    def _select_placements(
        self,
        problems: Sequence[PlacementProblem],
        capacities: Mapping[str, int],
    ) -> Mapping[str, PlacementCandidate]:
        """Sample a value model and greedily select feasible placements."""
        remaining = dict(capacities)
        selected: dict[str, PlacementCandidate] = {}
        for problem in sorted(
            problems,
            key=lambda value: (-value.priority, value.job_id),
        ):
            feasible = [
                candidate
                for candidate in problem.candidates
                if all(
                    remaining.get(key, 0) >= candidate.footprint
                    for key in candidate.capacity_keys
                )
            ]
            if not feasible:
                continue
            sampled_weight = self._sample_weight()
            ranked = []
            for candidate in feasible:
                features = _features(
                    problem,
                    candidate,
                    remaining,
                    len(problems),
                )
                ranked.append(
                    (
                        -_dot(sampled_weight, features),
                        candidate.price_per_hour,
                        candidate.footprint,
                        candidate.candidate_id,
                        candidate,
                        features,
                    )
                )
            _, _, _, _, candidate, features = min(
                ranked,
                key=lambda value: value[:4],
            )
            for key in candidate.capacity_keys:
                remaining[key] -= candidate.footprint
            selected[problem.job_id] = candidate
            known = _known_job_fields(problem.store_job)
            kind = getattr(known.get("kind"), "value", known.get("kind"))
            planned_rank_id = problem.rank_id or rank_id(known)
            self._pending[problem.job_id] = _PendingObservation(
                features=features,
                rank_id=planned_rank_id,
                window_start=self._feedback_start,
                required_tps=_target_tps(known),
                is_batch=kind == "batch",
                ttft_target_ms=_number(known.get("target_p99_ttft_ms")),
                tpot_target_ms=_number(known.get("target_p99_tpot_ms")),
            )
        return selected

    def _build_place_plan(
        self,
        problem: PlacementProblem,
        candidate: PlacementCandidate,
        model_catalogs: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, object]:
        """Build a policy-compliant placement or swap plan."""
        pending = self._pending.get(candidate.job_id)
        return build_contextual_bandit_plan(
            problem.store_job,
            candidate,
            model_catalogs,
            self.config_policy,
            rank_name=(None if pending is None else pending.rank_id),
            action_type=getattr(problem, "action_type", "place"),
        )

    def observe_results(
        self,
        job_id: str,
        *,
        p99_ttft_ms: float | None,
        p99_tpot_ms: float | None,
        throughput_token_per_sec: float | None,
        batch_deadline_met: bool | None = None,
    ) -> None:
        """Update from public Store latency, throughput, and deadline outcomes."""
        observation = self._pending.get(job_id)
        if observation is None:
            return
        if observation.is_batch and batch_deadline_met is None:
            return
        self._pending.pop(job_id)
        throughput_score = _throughput_score(
            throughput_token_per_sec,
            observation.required_tps,
        )
        if observation.is_batch:
            deadline_score = 1.0 if batch_deadline_met else -1.0
            reward = 0.4 * throughput_score + 0.6 * deadline_score
        else:
            reward = (
                0.4 * _latency_score(p99_ttft_ms, observation.ttft_target_ms)
                + 0.4 * _latency_score(p99_tpot_ms, observation.tpot_target_ms)
                + 0.2 * throughput_score
            )
        _rank_one_update(
            self._precision,
            self._reward_weight,
            observation.features,
            reward,
        )
        self.observation_count += 1
        self.last_reward = reward

    def _sample_weight(self) -> tuple[float, ...]:
        """Draw one linear reward model from the current Gaussian posterior."""
        lower = _cholesky(self._precision)
        mean = _solve_cholesky(lower, self._reward_weight)
        noise = [self._random.gauss(0.0, 1.0) for _ in range(_FEATURE_COUNT)]
        uncertainty = _solve_transposed_lower(lower, noise)
        return tuple(
            value + self.exploration_scale * offset
            for value, offset in zip(mean, uncertainty, strict=True)
        )


def build_placement_problems(
    waiting_jobs: Sequence[Mapping[str, Any]],
    resources: dict[str, dict[str, Any]],
    benchmark_name: str,
    benchmark_job_ids: Mapping[str, str],
    config_policy: ConfigPolicy,
    *,
    running_jobs: Sequence[Mapping[str, Any]] = (),
) -> tuple[PlacementProblem, ...]:
    """Build feasible static candidates for waiting jobs and running swaps."""
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
        job_id = _text(store_job.get("job_id"), "Store job_id")
        known = _known_job_fields(store_job)
        benchmark_job_id = benchmark_job_ids.get(job_id, job_id)
        required_vram_gib = _required_vram_gib(
            known,
            requirements.get(benchmark_job_id),
        )
        candidates = ()
        if required_vram_gib is not None:
            model_id = _text(known.get("model_id"), "model_id")
            candidates = enumerate_candidates(
                job_id,
                model_id,
                known,
                required_vram_gib,
                resources,
                config_policy,
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
        active_rank = _active_rank(running, rank_name)
        add_problem(
            running,
            action_type="swap",
            rank_name=rank_name,
            active_rank=active_rank,
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
    """Return whether a candidate leaves the active physical shape unchanged."""
    shape = active_rank.get("shape_json")
    if not isinstance(shape, Mapping):
        return False
    environment = shape.get("env") or shape.get("pool_id")
    if isinstance(environment, str):
        environment = environment.split("|")
    if not isinstance(environment, (list, tuple)):
        return False
    expected = {
        "instance_type": candidate.target.instance_type,
        "gpu_type": candidate.target.gpu_type,
        "gpu_count": candidate.gpu_count,
        "count": candidate.gpu_count,
        "tp": candidate.tp,
        "pp": candidate.pp,
    }
    return (
        list(environment) == candidate.target.environment_key.split("|")
        and active_rank.get("n_replicas") == candidate.dp
        and all(shape.get(key) == value for key, value in expected.items())
    )


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
                    )
                )
                dp += 1
    return tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id))


def capacity_limits(resources: dict[str, dict[str, Any]]) -> dict[str, int]:
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


def build_contextual_bandit_plan(
    store_job: Mapping[str, object],
    candidate: PlacementCandidate,
    model_catalogs: Mapping[str, Mapping[str, Any]],
    config_policy: ConfigPolicy,
    *,
    rank_name: str | None = None,
    action_type: str = "place",
) -> dict[str, object]:
    """Return one complete contextual-bandit placement or swap plan."""
    if action_type not in {"place", "swap"}:
        raise ValueError("contextual bandit action type must be place or swap")
    if action_type == "swap" and rank_name is None:
        raise ValueError("contextual bandit swaps require a rank ID")
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
    if rule is None or not rule.allows_parallelism(candidate.tp, candidate.pp):
        raise ValueError("contextual bandit selected a policy-invalid placement")
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
                    **launch_config,
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


def verify_place_plan(
    plan: Mapping[str, object],
    candidate: PlacementCandidate,
    model_catalogs: Mapping[str, Mapping[str, Any]],
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
    if rank_name is not None and rank.get("rank_id") != rank_name:
        raise ValueError("placement plan does not use the selected rank ID")
    if rank.get("env") != candidate.target.environment_key.split("|"):
        raise ValueError("placement plan environment does not match its candidate")
    if rank.get("n_replicas") != candidate.dp:
        raise ValueError("placement plan has the wrong replica count")
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
    if rule is None or not rule.allows_parallelism(candidate.tp, candidate.pp):
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
    for field, launch_value in launch_config.items():
        if config.get(field) != launch_value:
            raise ValueError(f"placement plan config has invalid {field}")


def _features(
    problem: PlacementProblem,
    candidate: PlacementCandidate,
    remaining: Mapping[str, int],
    queue_depth: int,
) -> tuple[float, ...]:
    """Encode only public job state and a proposed safe placement."""
    known = _known_job_fields(problem.store_job)
    available = min(remaining[key] for key in candidate.capacity_keys)
    capacity_share = candidate.footprint / max(1, available)
    memory_capacity = max(
        1.0,
        candidate.memory_gib * candidate.gpu_count * candidate.dp,
    )
    latency_tightness = max(
        _tightness(known.get("target_p99_ttft_ms")),
        _tightness(known.get("target_p99_tpot_ms")),
    )
    return (
        1.0,
        problem.priority / 4,
        min(1.0, queue_depth / 32),
        _scaled(known.get("isl_token_avg"), 4_096),
        _scaled(known.get("osl_token_avg"), 2_048),
        latency_tightness,
        min(1.0, candidate.gpu_count / 64),
        min(1.0, candidate.memory_gib / 192),
        min(1.0, candidate.kv_demand / memory_capacity),
        min(1.0, candidate.price_per_hour / 100),
        min(1.0, capacity_share),
    )


def _scaled(value: object, scale: float) -> float:
    """Clamp one nonnegative public workload feature to a stable range."""
    number = _finite_nonnegative(value)
    return 0.0 if number is None else min(1.0, number / scale)


def _tightness(value: object) -> float:
    """Return a normalized preference for stricter latency targets."""
    target = _finite_nonnegative(value)
    return 0.0 if target is None or target == 0 else min(1.0, 1_000 / target)


def _latency_score(observed: float | None, target: float | None) -> float:
    """Score lower latency against one required p99 target."""
    if observed is None or target is None or target <= 0:
        return -1.0
    return max(-1.0, min(1.0, 1 - observed / target))


def _throughput_score(observed: float | None, required: float | None) -> float:
    """Score delivered output throughput against one required rate."""
    if observed is None or required is None or required <= 0:
        return -1.0
    return max(-1.0, min(1.0, observed / required - 1))


def _worst_metric(metrics: Sequence[object], field: str) -> float | None:
    """Return the worst finite rank value across one Store telemetry window."""
    values = [_finite_nonnegative(getattr(metric, field, None)) for metric in metrics]
    present = [value for value in values if value is not None]
    return None if not present else max(present)


def _mean_rank_throughput(metrics: Sequence[object]) -> float | None:
    """Average one rank's per-timestamp throughput without GPU-row double counting."""
    by_timestamp: dict[object, float] = {}
    for metric in metrics:
        value = _finite_nonnegative(getattr(metric, "throughput_token_per_sec", None))
        if value is None:
            continue
        timestamp = getattr(metric, "ts", None)
        by_timestamp[timestamp] = max(by_timestamp.get(timestamp, 0.0), value)
    return None if not by_timestamp else sum(by_timestamp.values()) / len(by_timestamp)


def _finite_nonnegative(value: object) -> float | None:
    """Read one finite nonnegative value without accepting boolean values."""
    number = _number(value)
    if number is None or not math.isfinite(number) or number < 0:
        return None
    return number


def _rank_one_update(
    precision: list[list[float]],
    reward_weight: list[float],
    features: Sequence[float],
    reward: float,
) -> None:
    """Apply the conjugate linear-Gaussian posterior update."""
    for row, left in enumerate(features):
        reward_weight[row] += left * reward
        for column, right in enumerate(features):
            precision[row][column] += left * right


def _cholesky(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Factor one positive-definite precision matrix without NumPy."""
    lower = [[0.0] * _FEATURE_COUNT for _ in range(_FEATURE_COUNT)]
    for row in range(_FEATURE_COUNT):
        for column in range(row + 1):
            value = matrix[row][column] - sum(
                lower[row][index] * lower[column][index] for index in range(column)
            )
            if row == column:
                if value <= 0:
                    raise RuntimeError(
                        "contextual bandit precision is not positive definite"
                    )
                lower[row][column] = math.sqrt(value)
            else:
                lower[row][column] = value / lower[column][column]
    return lower


def _solve_cholesky(
    lower: Sequence[Sequence[float]],
    value: Sequence[float],
) -> list[float]:
    """Solve L L^T x = value for a Cholesky factor L."""
    intermediate = [0.0] * _FEATURE_COUNT
    for row in range(_FEATURE_COUNT):
        intermediate[row] = (
            value[row]
            - sum(lower[row][column] * intermediate[column] for column in range(row))
        ) / lower[row][row]
    return _solve_transposed_lower(lower, intermediate)


def _solve_transposed_lower(
    lower: Sequence[Sequence[float]],
    value: Sequence[float],
) -> list[float]:
    """Solve L^T x = value for a lower-triangular matrix L."""
    result = [0.0] * _FEATURE_COUNT
    for row in range(_FEATURE_COUNT - 1, -1, -1):
        result[row] = (
            value[row]
            - sum(
                lower[column][row] * result[column]
                for column in range(row + 1, _FEATURE_COUNT)
            )
        ) / lower[row][row]
    return result


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return one finite linear action score."""
    return sum(first * second for first, second in zip(left, right, strict=True))


def _target_id(target: ResourceTarget) -> str:
    """Return the target identity used by placement candidates."""
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
    """Estimate static token-growth-weighted KV demand."""
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
    counts = {"place": 0, "swap": 0, "defer": 0, "keep": 0, "preempt": 0}
    for action in actions:
        action_type = action.get("type")
        if action_type in counts:
            counts[action_type] += 1
    return (
        f"{method_name} cluster simulator tick {tick_number}: "
        f"place={counts['place']}, swap={counts['swap']}, defer={counts['defer']}, "
        f"keep={counts['keep']}, preempt={counts['preempt']}."
    )


def _positive_number(value: object) -> float | None:
    """Return one optional positive numeric value."""
    number = _number(value)
    return number if number is not None and number > 0 else None


def _required_text(value: object, name: str) -> str:
    """Require one nonempty text value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


__all__ = [
    "ContextualBanditPlannerDriver",
    "PlacementCandidate",
    "PlacementProblem",
    "StoreSnapshot",
    "build_contextual_bandit_plan",
    "verify_place_plan",
]
