"""Run NVIDIA AIC placement through the cluster simulator planner boundary.

Store owns job identity and lifecycle. AIC evaluates each homogeneous hardware
system independently and owns only hardware, replica, and TP/PP/DP selection.
All scheduler-visible job fields come from Store; unavailable fields are null.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from io import StringIO
from typing import Any, Protocol

from models import CreatedPlan, PlannerResult, Scenario
from policy import VALID_PP_DEGREES
from rank_ids import rank_id

_AIC_CANDIDATE_LIMIT = 1000
_AIC_FALLBACK_MODES = ("HYBRID", "EMPIRICAL", "SOL")
_AIC_PROFILE_BACKENDS = (("vllm", "0.22.0"), ("sglang", "0.5.14"))
_A10G_PROFILE_FALLBACK_SYSTEMS = ("l40s", "a100_sxm", "h100_sxm", "h200_sxm")
_AIC_SYSTEM_ALIASES = {
    "h100": "h100_sxm",
    "h100-sxm": "h100_sxm",
    "h100-sxm-80gb": "h100_sxm",
    "h100_sxm": "h100_sxm",
    "a100": "a100_sxm",
    "a100-40gb": "a100_sxm",
    "a100-80gb": "a100_sxm",
    "a100-sxm": "a100_sxm",
    "a100-sxm-80gb": "a100_sxm",
    "a100_sxm": "a100_sxm",
    "l40s": "l40s",
    "l40s-48gb": "l40s",
}


class AicEvaluationError(RuntimeError):
    """Report an AIC evaluation failure for one hardware candidate."""


class AicUnavailableError(RuntimeError):
    """Report that the NVIDIA AIC package is unavailable."""


@dataclass(frozen=True)
class AicSettings:
    """Configuration for the embedded AIC client and hardware mapping."""

    backend: str
    backend_version: str | None
    backend_versions: Mapping[str, str]
    database_mode: str
    serving_mode: str
    system_mappings: Mapping[str, str]


@dataclass(frozen=True)
class AicRequest:
    """Inputs for one homogeneous AIC recommendation."""

    model: str
    system: str
    available_devices: int
    backend: str
    backend_version: str | None
    database_mode: str
    serving_mode: str
    input_tokens: int
    output_tokens: int
    target_ttft_ms: float
    target_tpot_ms: float
    optimization_goal: str
    target_request_rate: float | None
    target_concurrency: float | None


@dataclass(frozen=True)
class AicResult:
    """Normalized performance and configuration returned by AIC."""

    devices: int
    replicas: int
    ttft_ms: float
    tpot_ms: float
    request_latency_ms: float
    throughput_tokens_per_second: float
    configuration: Mapping[str, object]


@dataclass(frozen=True)
class StoreSnapshot:
    """One frozen Store view used throughout an AIC control tick."""

    resources: dict[str, dict[str, Any]]
    running_jobs: tuple[dict[str, Any], ...]
    waiting_jobs: tuple[dict[str, Any], ...]
    paused_jobs: tuple[dict[str, Any], ...]
    finished_jobs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ResourceTarget:
    """One concrete Store environment that can receive an AIC placement."""

    environment_key: str
    gpu_type: str
    instance_type: str | None
    environment: dict[str, Any]
    pool: dict[str, Any] | None


@dataclass(frozen=True)
class HardwareCandidate:
    """Concrete Store targets normalized to one AIC hardware system."""

    system: str
    targets: tuple[ResourceTarget, ...]
    available_devices: int


@dataclass(frozen=True)
class AicPlacement:
    """One feasible AIC result bound to a concrete Store target."""

    target: ResourceTarget
    replicas: int
    tp: int
    pp: int
    dp: int
    max_num_seq: int
    devices: int
    ttft_ms: float
    tpot_ms: float
    request_latency_ms: float
    throughput_tokens_per_second: float


class AicClient(Protocol):
    """Evaluate one job against one homogeneous AIC hardware system."""

    def evaluate(self, request: AicRequest) -> AicResult:
        """Return AIC's best aggregate-serving configuration."""


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


class NvidiaAicClient:
    """Call NVIDIA AIC's public recommendation API."""

    def evaluate(self, request: AicRequest) -> AicResult:
        """Run load matching and normalize AIC's selected result row."""
        if request.serving_mode != "agg":
            raise AicEvaluationError("AIC requires aggregate serving")
        try:
            from aiconfigurator.cli.api import cli_recommend
            from aiconfigurator.sdk.picking import pick_optimization_type
        except ImportError as error:
            raise AicUnavailableError(
                "AIC is not installed; run `uv sync --group dynamo`"
            ) from error

        previous_logging_threshold = logging.root.manager.disable
        try:
            logging.disable(logging.CRITICAL)
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                recommendation = cli_recommend(
                    model_path=request.model,
                    system=request.system,
                    target_request_rate=request.target_request_rate,
                    target_concurrency=request.target_concurrency,
                    backend=request.backend,
                    backend_version=request.backend_version,
                    database_mode=request.database_mode,
                    isl=request.input_tokens,
                    osl=request.output_tokens,
                    ttft=request.target_ttft_ms,
                    tpot=request.target_tpot_ms,
                    strict_sla=True,
                    top_n=_AIC_CANDIDATE_LIMIT,
                )
        except SystemExit as error:
            raise AicEvaluationError(str(error)) from error
        except Exception as error:
            raise AicEvaluationError(str(error)) from error
        finally:
            logging.disable(previous_logging_threshold)

        frame = recommendation.best_configs.get("agg")
        if frame is None or frame.empty:
            raise AicEvaluationError("AIC returned no aggregate configuration")
        try:
            frame = frame[frame["total_gpus_needed"] <= request.available_devices]
            frame = frame[
                frame.apply(
                    lambda row: _parallelism(row.to_dict())[1] in VALID_PP_DEGREES,
                    axis=1,
                )
            ]
            if frame.empty:
                raise AicEvaluationError(
                    "AIC returned no configuration within capacity and PP limits"
                )
            if request.optimization_goal == "latency":
                selected = frame.sort_values(
                    by=["ttft", "tpot", "total_gpus_needed"],
                    ascending=[True, True, True],
                ).head(1)
            else:
                picked = pick_optimization_type(
                    pareto_df=frame,
                    optimization_type="throughput",
                    total_gpus=request.available_devices,
                    serving_mode=request.serving_mode,
                    top_n=1,
                )
                selected = picked["best_config_df"]
        except AicEvaluationError:
            raise
        except Exception as error:
            raise AicEvaluationError(str(error)) from error
        if selected is None or selected.empty:
            raise AicEvaluationError("AIC returned no selected configuration")

        row = selected.iloc[0]
        configuration = _json_value(row.to_dict())
        if not isinstance(configuration, Mapping):
            raise AicEvaluationError("AIC returned an invalid configuration")
        replicas = _aic_positive_int(row.get("replicas_needed", 1), "replicas")
        devices = _aic_positive_int(
            row.get("total_gpus_needed", row.get("num_total_gpus")),
            "total_gpus_needed",
        )
        per_replica_throughput = _aic_positive_float(
            row.get("tokens/s"),
            "tokens/s",
        )
        ttft_ms = _aic_positive_float(row.get("ttft"), "ttft")
        tpot_ms = _aic_positive_float(row.get("tpot"), "tpot")
        request_latency_ms = _aic_positive_float(
            row.get(
                "request_latency",
                ttft_ms + tpot_ms * max(request.output_tokens - 1, 0),
            ),
            "request_latency",
        )
        return AicResult(
            devices=devices,
            replicas=replicas,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            request_latency_ms=request_latency_ms,
            throughput_tokens_per_second=per_replica_throughput * replicas,
            configuration={str(key): value for key, value in configuration.items()},
        )


class AicPlannerDriver:
    """Create canonical Store plans from greedy AIC placement decisions."""

    method_name = "dynamo_aic"

    def __init__(
        self,
        user_id: str,
        scenario: Scenario,
        job_store: JobStoreClient,
        resource_map_store: ResourceMapStoreClient,
        plan_store: PlanStoreClient,
        settings: AicSettings,
        client: AicClient,
    ) -> None:
        """Keep Store clients and AIC configuration."""
        self.user_id = user_id
        self.job_store = job_store
        self.resource_map_store = resource_map_store
        self.plan_store = plan_store
        self.settings = settings
        self.client = client

    @classmethod
    def build(
        cls,
        user_id: str,
        scenario: Scenario,
        settings: Mapping[str, object],
        postgres_client: object | None = None,
        client: AicClient | None = None,
    ) -> AicPlannerDriver:
        """Build AIC with Tandemn Store public clients."""
        from tandemn_system_data.clients import (
            JobStore,
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
            plan_store=PlanStore(postgres),
            settings=_load_settings(settings),
            client=client or NvidiaAicClient(),
        )

    def create_plans(self, tick_number: int, logical_time: datetime) -> PlannerResult:
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
                tick_rationale=_tick_rationale(tick_number, actions),
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
        """Evaluate waiting jobs greedily and build every lifecycle action."""
        resources = deepcopy(snapshot.resources)
        actions: list[dict[str, object]] = []
        for running in snapshot.running_jobs:
            actions.append({"job_id": running["job_id"], "type": "keep"})
        for waiting in snapshot.waiting_jobs:
            job_id = waiting.get("job_id")
            if not isinstance(job_id, str):
                raise ValueError("waiting Store job is missing job_id")
            placement = _evaluate_job(
                waiting,
                resources,
                self.settings,
                self.client,
            )
            if placement is None:
                actions.append({"job_id": job_id, "type": "defer"})
                continue
            _reserve(placement)
            actions.append(
                _place_action(
                    job_id,
                    waiting,
                    placement,
                    self.settings.backend,
                )
            )
        for paused in snapshot.paused_jobs:
            actions.append({"job_id": paused["job_id"], "type": "defer"})
        _validate_complete_job_actions(snapshot, actions)
        return actions


def _evaluate_job(
    store_job: Mapping[str, object],
    resources: dict[str, dict[str, Any]],
    settings: AicSettings,
    client: AicClient,
) -> AicPlacement | None:
    """Evaluate every available hardware system and return the best result."""
    known = _known_job_fields(store_job)
    try:
        goal = _optimization_goal(known)
    except ValueError:
        return None
    placements = []
    for hardware in _hardware_candidates(resources, settings.system_mappings):
        try:
            request = _build_aic_request(known, hardware, settings, goal)
        except ValueError:
            return None
        for attempt in _aic_requests(request):
            try:
                result = client.evaluate(attempt)
            except AicUnavailableError:
                raise
            except (AicEvaluationError, ValueError):
                continue
            try:
                placement = _normalize_result(result, attempt, hardware)
            except (AicEvaluationError, ValueError):
                break
            placements.append(placement)
            break
    if not placements:
        return None
    return min(placements, key=lambda value: _placement_score(value, goal))


def _aic_requests(request: AicRequest) -> tuple[AicRequest, ...]:
    """Return the normal AIC request followed by estimate-only fallbacks."""
    attempts = [request]
    for database_mode in _AIC_FALLBACK_MODES:
        for backend, backend_version in _AIC_PROFILE_BACKENDS:
            attempt = replace(
                request,
                database_mode=database_mode,
                backend=backend,
                backend_version=backend_version,
            )
            if attempt not in attempts:
                attempts.append(attempt)
    return tuple(attempts)


def _build_aic_request(
    known: Mapping[str, object],
    hardware: HardwareCandidate,
    settings: AicSettings,
    goal: str,
) -> AicRequest:
    """Map one waiting job and normalized hardware system into AIC inputs."""
    input_tokens = _positive_rounded(known.get("isl_token_avg"), "isl_token_avg")
    output_tokens = _positive_rounded(known.get("osl_token_avg"), "osl_token_avg")
    target_ttft_ms = _positive_number(
        known.get("target_p99_ttft_ms"),
        "target_p99_ttft_ms",
    )
    target_tpot_ms = _positive_number(
        known.get("target_p99_tpot_ms"),
        "target_p99_tpot_ms",
    )
    target_request_rate, target_concurrency = _traffic_targets(known)
    return AicRequest(
        model=_text(known.get("model_id"), "model_id"),
        system=hardware.system,
        available_devices=hardware.available_devices,
        backend=settings.backend,
        backend_version=(
            settings.backend_versions.get(hardware.system)
            or settings.backend_versions.get(_hardware_key(hardware.system))
            or settings.backend_version
        ),
        database_mode=settings.database_mode,
        serving_mode=settings.serving_mode,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        target_ttft_ms=target_ttft_ms,
        target_tpot_ms=target_tpot_ms,
        optimization_goal=goal,
        target_request_rate=target_request_rate,
        target_concurrency=target_concurrency,
    )


def _normalize_result(
    result: AicResult,
    request: AicRequest,
    hardware: HardwareCandidate,
) -> AicPlacement:
    """Validate AIC parallelism and bind it to one concrete Store target."""
    replicas = _positive_int(result.replicas, "replicas")
    tp, pp, dp = _parallelism(result.configuration)
    if pp not in VALID_PP_DEGREES:
        raise AicEvaluationError(f"AIC PP must be one of {sorted(VALID_PP_DEGREES)}")
    devices = replicas * tp * pp * dp
    if result.devices != devices:
        raise AicEvaluationError(
            "AIC device count does not match replicas multiplied by TP/PP/DP"
        )
    if result.ttft_ms > request.target_ttft_ms:
        raise AicEvaluationError("AIC TTFT exceeds the job target")
    if result.tpot_ms > request.target_tpot_ms:
        raise AicEvaluationError("AIC TPOT exceeds the job target")
    target = _select_target(hardware.targets, replicas, tp * pp * dp)
    if target is None:
        raise AicEvaluationError("AIC configuration is unavailable in Store")
    return AicPlacement(
        target=target,
        replicas=replicas,
        tp=tp,
        pp=pp,
        dp=dp,
        max_num_seq=_aic_positive_int(result.configuration.get("bs"), "bs"),
        devices=devices,
        ttft_ms=result.ttft_ms,
        tpot_ms=result.tpot_ms,
        request_latency_ms=result.request_latency_ms,
        throughput_tokens_per_second=result.throughput_tokens_per_second,
    )


def _placement_score(placement: AicPlacement, goal: str) -> tuple[object, ...]:
    """Rank feasible hardware deterministically for one optimization goal."""
    target_key = (
        placement.target.environment_key,
        placement.target.instance_type or "",
    )
    if goal == "latency":
        return (
            placement.ttft_ms,
            placement.tpot_ms,
            placement.devices,
            *target_key,
        )
    return (
        -placement.throughput_tokens_per_second,
        placement.request_latency_ms,
        placement.devices,
        *target_key,
    )


def _place_action(
    job_id: str,
    store_job: Mapping[str, object],
    placement: AicPlacement,
    backend: str,
) -> dict[str, object]:
    """Combine deterministic job fields with one AIC placement."""
    known = _known_job_fields(store_job)
    per_replica_gpu_count = placement.tp * placement.pp
    rank_config: dict[str, object] = {
        "model_id": known.get("model_id"),
        "engine_name": known.get("engine_name") or backend,
        "instance_type": placement.target.instance_type,
        "gpu_type": placement.target.gpu_type,
        "gpu_count": per_replica_gpu_count,
        "count": per_replica_gpu_count,
        "tp": placement.tp,
        "pp": placement.pp,
        "dp": placement.dp,
        "ep": known.get("ep") or 1,
        "max_num_seq": placement.max_num_seq,
        "max_num_batched_tokens": known.get("max_num_batched_tokens") or 2048,
        "gpu_mem_util": known.get("gpu_mem_util"),
        "prefix_cache_enabled": known.get("prefix_cache_enabled"),
        "rank_traffic_share": 1.0,
    }
    return {
        "job_id": job_id,
        "type": "place",
        "target_tps": _target_tps(known),
        "target_p99_ttft_ms": _number(known.get("target_p99_ttft_ms")),
        "target_p99_tpot_ms": _number(known.get("target_p99_tpot_ms")),
        "ladder": [
            {
                "role": known.get("role") or "aggregate",
                "rank_id": rank_id(known),
                "env": placement.target.environment_key.split("|"),
                "config": rank_config,
                "n_replicas": placement.replicas * placement.dp,
                "mechanism_id": known.get("mechanism_id"),
            }
        ],
    }


def _hardware_candidates(
    resources: dict[str, dict[str, Any]],
    system_mappings: Mapping[str, str],
) -> tuple[HardwareCandidate, ...]:
    """Collapse concrete Store resources into homogeneous AIC systems."""
    grouped: dict[str, list[ResourceTarget]] = {}
    for environment_key, environment in sorted(resources.items()):
        gpu_type_value = environment.get("gpu_type")
        if gpu_type_value is None:
            gpu_type_value = environment_key.split("|")[-1]
        gpu_type = str(gpu_type_value)
        pools = environment.get("pools", [])
        if not isinstance(pools, list):
            raise ValueError("resource pools must be a list")
        target_pools: Sequence[dict[str, Any] | None] = pools or (None,)
        for pool in target_pools:
            systems = _aic_systems(environment, pool, gpu_type, system_mappings)
            target = ResourceTarget(
                environment_key=environment_key,
                gpu_type=gpu_type,
                instance_type=(
                    None if pool is None else _optional_text(pool.get("instance_type"))
                ),
                environment=environment,
                pool=pool,
            )
            if _target_available(target) > 0:
                for system in systems:
                    grouped.setdefault(system, []).append(target)

    candidates = []
    for system, targets in sorted(grouped.items()):
        ordered_targets = tuple(
            sorted(
                targets,
                key=lambda target: (
                    target.environment_key,
                    target.instance_type or "",
                ),
            )
        )
        candidates.append(
            HardwareCandidate(
                system=system,
                targets=ordered_targets,
                available_devices=max(
                    _target_available(target) for target in ordered_targets
                ),
            )
        )
    return tuple(candidates)


def _aic_system(
    environment: Mapping[str, object],
    pool: Mapping[str, object] | None,
    gpu_type: str,
    system_mappings: Mapping[str, str],
) -> str | None:
    """Resolve a Store GPU description to one AIC system identifier."""
    explicit = None
    if pool is not None:
        explicit = pool.get("aic_system")
    explicit = explicit or environment.get("aic_system")
    if explicit is not None:
        return _text(explicit, "aic_system")
    normalized = _hardware_key(gpu_type)
    return (
        system_mappings.get(gpu_type)
        or system_mappings.get(normalized)
        or _AIC_SYSTEM_ALIASES.get(normalized)
    )


def _aic_systems(
    environment: Mapping[str, object],
    pool: Mapping[str, object] | None,
    gpu_type: str,
    system_mappings: Mapping[str, str],
) -> tuple[str, ...]:
    """Return AIC systems that can estimate one Store GPU target."""
    system = _aic_system(environment, pool, gpu_type, system_mappings)
    if system is not None:
        return (system,)
    if _hardware_key(gpu_type) == "a10g":
        return _A10G_PROFILE_FALLBACK_SYSTEMS
    return ()


def _select_target(
    targets: Sequence[ResourceTarget],
    replicas: int,
    gpu_count: int,
) -> ResourceTarget | None:
    """Choose the tightest concrete target that fits an AIC configuration."""
    feasible = []
    for target in targets:
        footprint = _target_footprint(target, replicas, gpu_count)
        if footprint is None or footprint > _target_available(target):
            continue
        feasible.append(
            (
                _target_available(target) - footprint,
                target.environment_key,
                target.instance_type or "",
                target,
            )
        )
    return None if not feasible else min(feasible, key=lambda value: value[:3])[3]


def _reserve(placement: AicPlacement) -> None:
    """Reserve one concrete target after it wins cross-hardware comparison."""
    gpu_count = placement.tp * placement.pp * placement.dp
    footprint = _target_footprint(
        placement.target,
        placement.replicas,
        gpu_count,
    )
    if footprint is None or footprint > _target_available(placement.target):
        raise RuntimeError("selected AIC placement no longer fits")
    environment = placement.target.environment
    environment["free"] = int(environment["free"]) - footprint
    if placement.target.pool is not None:
        pool = placement.target.pool
        pool["free"] = int(pool["free"]) - footprint


def _target_available(target: ResourceTarget) -> int:
    """Return currently available capacity for one concrete Store target."""
    available = int(target.environment.get("free", 0))
    if target.pool is not None:
        available = min(available, int(target.pool.get("free", 0)))
    return max(0, available)


def _target_footprint(
    target: ResourceTarget,
    replicas: int,
    gpu_count: int,
) -> int | None:
    """Translate an AIC GPU layout into the Store target's allocation unit."""
    if target.pool is None:
        return replicas * gpu_count
    allocation_kind = str(
        target.pool.get("allocation_kind")
        or target.pool.get("allocation_unit")
        or "instance"
    ).lower()
    if allocation_kind == "gpu":
        return replicas * gpu_count
    gpus_per_instance = _positive_int(
        target.pool.get("gpus_per_instance"),
        "gpus_per_instance",
    )
    if gpu_count > gpus_per_instance:
        return None
    return replicas * gpus_per_instance


def _parallelism(configuration: Mapping[str, object]) -> tuple[int, int, int]:
    """Normalize AIC's explicit or compact parallelism representation."""
    parsed: dict[str, int] = {}
    parallel = configuration.get("parallel")
    if isinstance(parallel, str):
        parsed = {
            name: int(value)
            for name, value in re.findall(r"(tp|pp|dp)[=_-]?(\d+)", parallel.lower())
        }
    values = []
    for short_name, aliases in (
        ("tp", ("tp", "tensor_parallel", "tensor_parallel_size")),
        ("pp", ("pp", "pipeline_parallel", "pipeline_parallel_size")),
        ("dp", ("dp", "data_parallel", "data_parallel_size")),
    ):
        value = next(
            (configuration.get(alias) for alias in aliases if alias in configuration),
            parsed.get(short_name, 1),
        )
        values.append(_aic_positive_int(value, short_name))
    return values[0], values[1], values[2]


def _resources_with_free_capacity(
    resource_map: ResourceMap,
    running_jobs: Sequence[object],
) -> dict[str, dict[str, Any]]:
    """Normalize Store totals and subtract active canonical rank allocations."""
    raw = resource_map.scheduling_summary()
    default_market = _default_market(resource_map)
    resources = {
        _normalize_env_key(str(key), default_market): dict(value)
        for key, value in raw.items()
    }
    used_by_env: dict[str, int] = {}
    used_by_pool: dict[tuple[str, str], int] = {}
    for running in running_jobs:
        payload = _model_payload(running)
        for rank_value in payload.get("ranks", []):
            rank = _mapping(rank_value, "active rank")
            shape = _mapping(rank.get("shape_json", {}), "rank shape")
            raw_env = shape.get("env") or shape.get("pool_id")
            if raw_env is None:
                continue
            env_key = _normalize_env_key(_env_key(raw_env), default_market)
            count = _positive_int(shape.get("count"), "active rank count")
            replicas = _positive_int(rank.get("n_replicas"), "active rank replicas")
            environment = resources.get(env_key)
            if environment is None:
                continue
            pool = _select_pool(
                environment,
                _optional_text(shape.get("instance_type")),
            )
            footprint = count
            if pool is not None:
                allocation_kind = str(
                    pool.get("allocation_kind")
                    or pool.get("allocation_unit")
                    or "instance"
                ).lower()
                if allocation_kind != "gpu":
                    footprint = replicas * _positive_int(
                        pool.get("gpus_per_instance"),
                        "gpus_per_instance",
                    )
                pool_key = (env_key, str(pool.get("instance_type")))
                used_by_pool[pool_key] = used_by_pool.get(pool_key, 0) + footprint
            used_by_env[env_key] = used_by_env.get(env_key, 0) + footprint

    for env_key, environment in resources.items():
        environment["free"] = max(
            0,
            int(environment.get("total", 0)) - used_by_env.get(env_key, 0),
        )
        pools = [dict(pool) for pool in environment.get("pools", [])]
        for pool in pools:
            pool_key = (env_key, str(pool.get("instance_type")))
            pool["free"] = max(
                0,
                int(pool.get("total", 0)) - used_by_pool.get(pool_key, 0),
            )
        environment["pools"] = pools
    return resources


def _running_job_payload(running_job: object) -> dict[str, Any]:
    """Flatten a Store running-job aggregate for the frozen snapshot."""
    payload = _model_payload(running_job)
    job = dict(_mapping(payload.get("job"), "running job"))
    job["active_ranks"] = payload.get("ranks", [])
    return job


def _model_payload(value: object) -> dict[str, Any]:
    """Serialize one Store model or aggregate."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="json"))
    if isinstance(value, Mapping):
        return dict(value)
    job = getattr(value, "job", None)
    ranks = getattr(value, "ranks", None)
    if job is not None and ranks is not None:
        return {
            "job": _model_payload(job),
            "ranks": [_model_payload(rank) for rank in ranks],
        }
    raise TypeError(f"Store value {type(value).__name__} is not serializable")


def _known_job_fields(
    store_job: Mapping[str, object],
) -> dict[str, object]:
    """Flatten scheduler-visible fields from one Store job."""
    spec = store_job.get("spec_json")
    store_spec = spec if isinstance(spec, Mapping) else {}
    store_features_value = store_spec.get("job_features")
    store_features = (
        store_features_value if isinstance(store_features_value, Mapping) else {}
    )
    known: dict[str, object] = {}
    for source in (
        store_job,
        store_spec,
        store_features,
    ):
        for key, value in source.items():
            if value is not None:
                known.setdefault(str(key), value)
    return known


def _optimization_goal(known: Mapping[str, object]) -> str:
    """Choose the AIC objective from the canonical Store job kind."""
    value = known.get("kind")
    kind = _text(getattr(value, "value", value), "kind")
    if kind == "online":
        return "latency"
    if kind == "batch":
        return "throughput"
    raise ValueError(f"unsupported job kind {kind!r}")


def _traffic_targets(
    known: Mapping[str, object],
) -> tuple[float | None, float | None]:
    """Map canonical Store traffic intent into AIC's exclusive inputs."""
    kind_value = known.get("kind")
    kind = _text(getattr(kind_value, "value", kind_value), "kind")
    if kind == "batch":
        return _batch_request_rate(known), None
    if kind != "online":
        raise ValueError(f"unsupported job kind {kind!r}")
    if known.get("_traffic_mode") == "concurrency":
        return None, _positive_number(
            known.get("max_concurrent_streaming"),
            "max_concurrent_streaming",
        )
    return (
        _positive_number(
            known.get("request_arrival_rate"),
            "request_arrival_rate",
        ),
        None,
    )


def _batch_request_rate(known: Mapping[str, object]) -> float:
    """Derive batch requests per second from Store budget and deadline."""
    token_budget = _positive_number(
        known.get("total_token_budget"),
        "total_token_budget",
    )
    deadline_hours = _positive_number(known.get("deadline_hrs"), "deadline_hrs")
    input_tokens = _positive_number(known.get("isl_token_avg"), "isl_token_avg")
    output_tokens = _positive_number(known.get("osl_token_avg"), "osl_token_avg")
    return token_budget / ((input_tokens + output_tokens) * deadline_hours * 3600)


def _target_tps(known: Mapping[str, object]) -> float | None:
    """Derive output-token throughput from canonical Store job fields."""
    output_tokens = _number(known.get("osl_token_avg"))
    if output_tokens is None:
        return None
    kind_value = known.get("kind")
    kind = getattr(kind_value, "value", kind_value)
    try:
        request_rate = (
            _batch_request_rate(known)
            if kind == "batch"
            else _positive_number(
                known.get("request_arrival_rate"),
                "request_arrival_rate",
            )
        )
    except ValueError:
        return None
    return round(request_rate * output_tokens, 3)


def _validate_complete_job_actions(
    snapshot: StoreSnapshot,
    actions: Sequence[Mapping[str, object]],
) -> None:
    """Require exactly one canonical action for every Store job in the snapshot."""
    store_ids = []
    for jobs in (
        snapshot.running_jobs,
        snapshot.waiting_jobs,
        snapshot.paused_jobs,
    ):
        for job in jobs:
            store_ids.append(_text(job.get("job_id"), "Store job_id"))
    action_ids = [_text(action.get("job_id"), "plan job_id") for action in actions]
    if len(store_ids) != len(set(store_ids)):
        raise ValueError("Store snapshot contains duplicate job_id values")
    if len(action_ids) != len(set(action_ids)) or set(action_ids) != set(store_ids):
        raise ValueError("plan must contain exactly one action for every Store job")


def _tick_rationale(
    tick_number: int,
    actions: Sequence[Mapping[str, object]],
) -> str:
    """Summarize the deterministic action set persisted for one tick."""
    counts = {"place": 0, "defer": 0, "keep": 0, "preempt": 0}
    for action in actions:
        action_type = action.get("type")
        if action_type in counts:
            counts[action_type] += 1
    return (
        f"AIC cluster simulator tick {tick_number}: "
        f"place={counts['place']}, defer={counts['defer']}, "
        f"keep={counts['keep']}, preempt={counts['preempt']}."
    )


def _load_settings(values: Mapping[str, object]) -> AicSettings:
    """Validate AIC adapter settings."""
    serving_mode = _text(values.get("serving_mode", "agg"), "serving_mode")
    if serving_mode != "agg":
        raise ValueError("AIC requires aggregate serving")
    version_value = values.get("backend_version")
    versions = _mapping(values.get("backend_versions", {}), "backend_versions")
    mappings = _mapping(values.get("system_mappings", {}), "system_mappings")
    return AicSettings(
        backend=_text(values.get("backend"), "backend"),
        backend_version=(
            None if version_value is None else _text(version_value, "backend_version")
        ),
        backend_versions={
            _aic_lookup_key(_text(key, "backend_versions key")): _text(
                value,
                f"backend_versions.{key}",
            )
            for key, value in versions.items()
        },
        database_mode=_text(
            values.get("database_mode", "HYBRID"),
            "database_mode",
        ),
        serving_mode=serving_mode,
        system_mappings={
            _hardware_key(_text(key, "system_mappings key")): _text(
                value,
                f"system_mappings.{key}",
            )
            for key, value in mappings.items()
        },
    )


def _select_pool(
    environment: Mapping[str, Any],
    instance_type: str | None,
) -> dict[str, Any] | None:
    """Resolve one environment pool from an active chain shape."""
    pools = environment.get("pools", [])
    if not isinstance(pools, list):
        raise ValueError("resource pools must be a list")
    typed_pools = []
    for value in pools:
        if not isinstance(value, dict):
            raise ValueError("resource pool must be an object")
        typed_pools.append(value)
    if not typed_pools:
        return None
    if instance_type is not None:
        for pool in typed_pools:
            if pool.get("instance_type") == instance_type:
                return pool
        return None
    return typed_pools[0] if len(typed_pools) == 1 else None


def _default_market(resource_map: ResourceMap) -> str:
    """Read the resource map's default market."""
    markets = getattr(resource_map, "market", None)
    if isinstance(markets, (list, tuple)) and markets:
        return str(markets[0])
    return str(markets or "reserved")


def _normalize_env_key(env_key: str, default_market: str) -> str:
    """Normalize legacy four-part environment keys."""
    parts = env_key.split("|")
    return "|".join((default_market, *parts)) if len(parts) == 4 else env_key


def _env_key(value: object) -> str:
    """Return one canonical pipe-delimited environment key."""
    if isinstance(value, (list, tuple)):
        return "|".join(str(part) for part in value)
    return str(value)


def _hardware_key(value: str) -> str:
    """Normalize one hardware label for AIC-system lookup."""
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _aic_lookup_key(value: str) -> str:
    """Normalize a hardware label or AIC system to one settings key."""
    hardware_key = _hardware_key(value)
    system = _AIC_SYSTEM_ALIASES.get(hardware_key, hardware_key)
    return _hardware_key(system)


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Require one mapping value."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


def _text(value: object, name: str) -> str:
    """Require one nonempty text value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: object) -> str | None:
    """Return optional nonempty text."""
    return None if value is None else _text(value, "optional text")


def _positive_int(value: object, name: str) -> int:
    """Require one positive integer value."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_rounded(value: object, name: str) -> int:
    """Round one positive numeric workload value to an integer."""
    return max(1, round(_positive_number(value, name)))


def _positive_number(value: object, name: str) -> float:
    """Require one positive numeric value."""
    number = _number(value)
    if number is None or number <= 0:
        raise ValueError(f"{name} must be a positive number")
    return number


def _number(value: object) -> float | None:
    """Return one optional numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _json_value(value: object) -> object:
    """Convert AIC and dataframe scalar values into Python JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    item_method = getattr(value, "item", None)
    return item_method() if callable(item_method) else value


def _aic_positive_int(value: object, name: str) -> int:
    """Normalize one positive integer returned by AIC."""
    converted = _json_value(value)
    if isinstance(converted, bool) or not isinstance(converted, (int, float)):
        raise AicEvaluationError(f"AIC returned invalid {name}")
    number = int(converted)
    if number <= 0 or float(converted) != number:
        raise AicEvaluationError(f"AIC returned invalid {name}")
    return number


def _aic_positive_float(value: object, name: str) -> float:
    """Normalize one positive float returned by AIC."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise AicEvaluationError(f"AIC returned invalid {name}") from error
    if number <= 0:
        raise AicEvaluationError(f"AIC returned invalid {name}")
    return number
