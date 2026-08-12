"""Place models by Prism's static KV-cache pressure heuristic."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from policy import ConfigPolicy
from static_placement import (
    PlacementCandidate,
    PlacementProblem,
    StaticPlacementPlannerDriver,
    StoreSnapshot,
)
from vram import _known_job_fields, _number, _target_tps


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


class PrismPlannerDriver(StaticPlacementPlannerDriver):
    """Adapt Prism's pressure ordering to aggregate Store resource pools."""

    method_name = "prism"

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
    ) -> dict[str, object]:
        """Build Prism's complete per-job placement plan."""
        return build_prism_plan(
            problem.store_job,
            candidate,
            self.backend,
            self.config_policy,
        )


def build_prism_plan(
    store_job: Mapping[str, object],
    candidate: PlacementCandidate,
    backend: str,
    config_policy: ConfigPolicy,
) -> dict[str, object]:
    """Return one complete Prism placement plan."""
    known = _known_job_fields(store_job)
    model_id = _required_text(known.get("model_id"), "model_id")
    rule = config_policy.rule_for(candidate.target.gpu_type, model_id)
    if rule is None or candidate.tp not in rule.allowed_tp:
        raise ValueError("Prism selected a policy-invalid placement")
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
    "PrismGpu",
    "PrismModel",
    "PrismPlacementError",
    "PrismPlannerDriver",
    "StoreSnapshot",
    "build_prism_plan",
    "compute_model_placement",
    "select_prism_placements",
]
