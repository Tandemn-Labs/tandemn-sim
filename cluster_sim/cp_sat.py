"""Optimize public-facts GPU placement globally with OR-Tools CP-SAT."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from models import Scenario
from policy import ConfigPolicy, load_config_policy
from static_placement import (
    JobStoreClient,
    PlacementCandidate,
    PlacementProblem,
    PlanStoreClient,
    ResourceMapStoreClient,
    StaticPlacementPlannerDriver,
    StoreSnapshot,
)
from vram import _known_job_fields, _number, _target_tps


class CpSatUnavailableError(RuntimeError):
    """Report that the optional OR-Tools solver is unavailable."""


class _SolverResult(Protocol):
    """Expose the CP-SAT result method used to read Boolean variables."""

    def Value(self, variable: object) -> int: ...


class CpSatPlannerDriver(StaticPlacementPlannerDriver):
    """Create globally optimized Store plans from static public facts."""

    method_name = "cp_sat"

    def __init__(
        self,
        user_id: str,
        scenario: Scenario,
        job_store: JobStoreClient,
        resource_map_store: ResourceMapStoreClient,
        plan_store: PlanStoreClient,
        backend: str,
        config_policy: ConfigPolicy,
        max_time_seconds: float = 30.0,
    ) -> None:
        """Keep the common Store adapter and a deterministic solver limit."""
        super().__init__(
            user_id,
            scenario,
            job_store,
            resource_map_store,
            plan_store,
            backend,
            config_policy,
        )
        if max_time_seconds <= 0:
            raise ValueError("max_time_seconds must be positive")
        self.max_time_seconds = max_time_seconds

    @classmethod
    def build(
        cls,
        user_id: str,
        scenario: Scenario,
        settings: Mapping[str, object],
        postgres_client: object | None = None,
    ) -> CpSatPlannerDriver:
        """Build CP-SAT with Tandemn Store public clients."""
        from tandemn_system_data.clients import (
            JobStore,
            PlanStore,
            PostgresClient,
            ResourceMapStore,
        )

        raw_limit = settings.get("max_time_seconds", 30.0)
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, (int, float)):
            raise ValueError("max_time_seconds must be numeric")
        postgres = postgres_client or PostgresClient()
        return cls(
            user_id=user_id,
            scenario=scenario,
            job_store=JobStore(postgres),
            resource_map_store=ResourceMapStore(postgres, user_id=user_id),
            plan_store=PlanStore(postgres),
            backend=_required_text(settings.get("backend"), "backend"),
            config_policy=load_config_policy(),
            max_time_seconds=float(raw_limit),
        )

    def _select_placements(
        self,
        problems: Sequence[PlacementProblem],
        capacities: Mapping[str, int],
    ) -> Mapping[str, PlacementCandidate]:
        """Solve priority admission first, then cost and capacity usage."""
        return solve_cp_sat(problems, capacities, self.max_time_seconds)

    def _build_place_plan(
        self,
        problem: PlacementProblem,
        candidate: PlacementCandidate,
    ) -> dict[str, object]:
        """Build CP-SAT's complete per-job placement plan."""
        return build_cp_sat_plan(
            problem.store_job,
            candidate,
            self.backend,
            self.config_policy,
        )


def build_cp_sat_plan(
    store_job: Mapping[str, object],
    candidate: PlacementCandidate,
    backend: str,
    config_policy: ConfigPolicy,
) -> dict[str, object]:
    """Return one complete CP-SAT placement plan."""
    known = _known_job_fields(store_job)
    model_id = _required_text(known.get("model_id"), "model_id")
    rule = config_policy.rule_for(candidate.target.gpu_type, model_id)
    if rule is None or candidate.tp not in rule.allowed_tp:
        raise ValueError("CP-SAT selected a policy-invalid placement")
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


def solve_cp_sat(
    problems: Sequence[PlacementProblem],
    capacities: Mapping[str, int],
    max_time_seconds: float = 30.0,
) -> dict[str, PlacementCandidate]:
    """Return a lexicographically optimized feasible placement assignment."""
    if max_time_seconds <= 0:
        raise ValueError("max_time_seconds must be positive")
    candidates = tuple(
        candidate for problem in problems for candidate in problem.candidates
    )
    if not candidates:
        return {}
    try:
        from ortools.sat.python import cp_model
    except (ImportError, OSError) as error:
        raise CpSatUnavailableError(
            "CP-SAT requires a working OR-Tools installation"
        ) from error

    model = cp_model.CpModel()
    variables = {
        (candidate.job_id, candidate.candidate_id): model.NewBoolVar(f"x_{index}")
        for index, candidate in enumerate(candidates)
    }
    for problem in problems:
        model.Add(
            sum(
                variables[(candidate.job_id, candidate.candidate_id)]
                for candidate in problem.candidates
            )
            <= 1
        )
    for capacity_key, available in capacities.items():
        terms = [
            candidate.footprint * variables[(candidate.job_id, candidate.candidate_id)]
            for candidate in candidates
            if capacity_key in candidate.capacity_keys
        ]
        if terms:
            model.Add(sum(terms) <= available)

    priority_base = len(problems) + 1
    priorities = {problem.job_id: problem.priority for problem in problems}
    priority_terms = [
        priority_base ** (priorities[candidate.job_id] - 1)
        * variables[(candidate.job_id, candidate.candidate_id)]
        for candidate in candidates
    ]
    priority_objective = sum(priority_terms)
    model.Maximize(priority_objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        status_name = solver.StatusName(status)
        raise RuntimeError(f"CP-SAT admission solve failed: {status_name}")

    first_selection = _read_selection(candidates, variables, solver)
    best_priority = sum(
        priority_base ** (priorities[job_id] - 1) for job_id in first_selection
    )
    model.Add(priority_objective == best_priority)

    capacity_base = sum(capacities.values()) + 1
    tie_base = len(candidates) * (len(problems) + 1) + 1
    secondary_terms = []
    for ordinal, candidate in enumerate(candidates, start=1):
        price_milli = round(candidate.price_per_hour * 1_000)
        coefficient = (
            price_milli * capacity_base + candidate.footprint
        ) * tie_base + ordinal
        secondary_terms.append(
            coefficient * variables[(candidate.job_id, candidate.candidate_id)]
        )
    model.Minimize(sum(secondary_terms))
    second_status = solver.Solve(model)
    if second_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return first_selection
    return _read_selection(candidates, variables, solver)


def _read_selection(
    candidates: Sequence[PlacementCandidate],
    variables: Mapping[tuple[str, str], object],
    solver: _SolverResult,
) -> dict[str, PlacementCandidate]:
    """Read selected candidate variables from one CP-SAT solution."""
    return {
        candidate.job_id: candidate
        for candidate in candidates
        if solver.Value(variables[(candidate.job_id, candidate.candidate_id)]) == 1
    }


def _required_text(value: object, name: str) -> str:
    """Require one nonempty text setting."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


__all__ = [
    "CpSatPlannerDriver",
    "CpSatUnavailableError",
    "StoreSnapshot",
    "build_cp_sat_plan",
    "solve_cp_sat",
]
