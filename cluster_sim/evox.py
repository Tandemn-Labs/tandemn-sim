"""Evolve placement programs online with EvoX for each control tick."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

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
    validate_selection,
)
from vram import _known_job_fields, _number, _target_tps

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


class EvoxPlannerDriver(StaticPlacementPlannerDriver):
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
        """Keep the common Store adapter and persistent online EvoX client."""
        super().__init__(
            user_id,
            scenario,
            job_store,
            resource_map_store,
            plan_store,
            backend,
            config_policy,
        )
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


__all__ = [
    "EVOX_INITIAL_PROGRAM",
    "EVOX_SYSTEM_PROMPT",
    "EvoxClient",
    "EvoxPolicy",
    "OnlineEvoxEvaluator",
    "OnlineSkyDiscoverEvoxClient",
    "EvoxPlannerDriver",
    "StoreSnapshot",
    "build_evox_plan",
    "build_evox_payload",
    "load_policy_file",
    "load_policy_source",
    "run_evox_discovery",
    "select_evox_placements",
]
