"""Own the complete OpenEvolve planner path for each cluster control tick."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any, Protocol, cast

from llm_usage import LlmUsage
from models import BatchJob, CreatedPlan, OnlineJob, PlannerResult, Scenario
from outcome_feedback import OutcomeFeedback, fingerprint
from policy import (
    VALID_PP_DEGREES,
    ConfigPolicy,
    load_config_policy,
)
from rank_ids import active_rank_id, rank_id
from vram import _HARDWARE_MEMORY_GIB, VRAM_REQUIREMENTS_GIB, _canonical_hardware

_PRIORITY_LEVELS = {
    "BEST_EFFORT": 1,
    "LOW": 1,
    "NORMAL": 2,
    "STANDARD": 2,
    "HIGH": 3,
    "CRITICAL": 4,
    "P0": 4,
}

OPENEOLVE_SYSTEM_PROMPT = """You improve a deterministic placement policy for one
frozen Tandemn scheduler tick. Modify only compute_placement(payload). The payload
contains each job's required_vram_gib, available GPU resources with
gpu_memory_gib, allowed configurations, resolved Store model_catalog profiles,
and JSON-only outcome_feedback evidence from prior canonical placements. Use the
model catalog as supplementary workload context. Outcome evidence is keyed by
job_id, model_id, target, and tp/pp/dp; unknown configurations are neutral.
Return one ordered decision for every job in payload['jobs_for_tick'], preserving
each job's lifecycle. Waiting jobs require place or defer decisions; running jobs
require keep or swap decisions.

Never use network, filesystem, subprocesses, hidden state, or third-party packages.
Do not construct Tandemn Store actions. Choose only from the supplied environments
and configurations. Preserve the function signature and return only
{'placements': [...]}.

Keep placement configuration limited to tp, pp, and dp. Tandemn materializes
authoritative model catalog fields after the policy chooses a target GPU; never
copy those fields into the policy response. PP must never exceed 4. A swap must
select a physically different target from the running job's active_placement.

Prioritize valid plans, then admission of urgent work, then measured outcome
feedback where available, then lower cost and GPU use. Defer a job when no
policy-compliant placement fits.
VRAM for one runnable copy is gpu_memory_gib * tp * pp. DP creates runtime
replicas and does not increase the VRAM available to one copy. Existing active
allocations remain reserved while a swap is evaluated, so a swap needs enough
free capacity for its complete replacement ladder.
"""

OPENEOLVE_INITIAL_PROGRAM = '''\
# EVOLVE-BLOCK-START
def compute_placement(payload):
    """Keep running jobs and place waiting jobs on free compliant resources."""
    remaining_by_environment = {
        environment: int(resource.get("free", 0))
        for environment, resource in payload["tandemn_state"]["resources"].items()
    }
    remaining_by_pool = {
        (environment, pool["instance_type"]): int(pool.get("free", 0))
        for environment, resource in payload["tandemn_state"]["resources"].items()
        for pool in resource.get("pools", [])
    }
    policy = {
        (item["model_id"], item["gpu_type"]): item
        for item in payload["config_policy"]
    }
    placements = []
    for job in payload["jobs_for_tick"]:
        decision = {
            "lifecycle": job["lifecycle"],
            "decision": "keep" if job["lifecycle"] == "running" else "defer",
            "environment": None,
            "instance_type": None,
            "configuration": None,
        }
        if job["lifecycle"] == "running":
            placements.append(decision)
            continue
        for environment, resource in payload["tandemn_state"]["resources"].items():
            gpu_type = str(resource.get("gpu_type") or environment.split("|")[-1])
            rule = policy.get((job["model_name"], gpu_type))
            if rule is None:
                continue
            required_vram_gib = float(job["required_vram_gib"])
            gpu_memory_gib = float(resource["gpu_memory_gib"])
            minimum_gpus = max(1, int(required_vram_gib / gpu_memory_gib))
            if minimum_gpus * gpu_memory_gib < required_vram_gib:
                minimum_gpus += 1
            allowed_tp = sorted(int(value) for value in rule["allowed_tp"])
            larger_tp = [value for value in allowed_tp if value >= minimum_gpus]
            tp = min(larger_tp) if larger_tp else max(allowed_tp)
            pp = max(1, (minimum_gpus + tp - 1) // tp)
            if "allowed_pp" in rule and pp not in rule["allowed_pp"]:
                continue
            pools = resource.get("pools", [])
            targets = pools if pools else [None]
            for pool in targets:
                instance_type = None
                gpu_count = tp * pp
                dp = 1
                footprint = dp * gpu_count
                available = remaining_by_environment.get(environment, 0)
                if pool is not None:
                    instance_type = pool["instance_type"]
                    allocation_kind = str(
                        pool.get("allocation_kind")
                        or pool.get("allocation_unit")
                        or "instance"
                    ).lower()
                    if allocation_kind != "gpu":
                        gpus_per_instance = int(pool["gpus_per_instance"])
                        if gpu_count > gpus_per_instance:
                            continue
                        footprint = dp * gpus_per_instance
                    available = min(
                        available,
                        remaining_by_pool.get((environment, instance_type), 0),
                    )
                if available < footprint:
                    continue
                decision = {
                    "lifecycle": job["lifecycle"],
                    "decision": "place",
                    "environment": environment,
                    "instance_type": instance_type,
                    "configuration": {
                        "tp": tp,
                        "pp": pp,
                        "dp": dp,
                    },
                }
                remaining_by_environment[environment] -= footprint
                if instance_type is not None:
                    remaining_by_pool[(environment, instance_type)] -= footprint
                break
            if decision["decision"] == "place":
                break
        placements.append(decision)
    return {"placements": placements}
# EVOLVE-BLOCK-END
'''


@dataclass(frozen=True)
class StoreSnapshot:
    """One frozen Store view used throughout an OpenEvolve control tick."""

    resources: dict[str, dict[str, Any]]
    running_jobs: tuple[dict[str, Any], ...]
    waiting_jobs: tuple[dict[str, Any], ...]
    paused_jobs: tuple[dict[str, Any], ...]
    finished_jobs: tuple[dict[str, Any], ...]
    model_catalogs: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class OpenEvolvePlacement:
    """One narrow placement decision returned by an evolved policy."""

    lifecycle: str
    decision: str
    environment: str | None
    instance_type: str | None
    configuration: Mapping[str, object] | None


class OpenEvolvePolicy(Protocol):
    """Return constrained ordered placement decisions for one tick payload."""

    def __call__(self, payload: Mapping[str, object]) -> Mapping[str, object]: ...


class OpenEvolveRunner(Protocol):
    """Expose OpenEvolve's high-level synchronous library entry point."""

    def __call__(
        self,
        initial_program: str,
        evaluator: str,
        config: Mapping[str, object],
        iterations: int,
        output_dir: str,
        cleanup: bool,
    ) -> object: ...


class OpenEvolveClient(Protocol):
    """Evolve and execute one placement policy for a frozen tick."""

    def place(
        self,
        tick_number: int,
        model: str,
        max_tokens: int,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[tuple[OpenEvolvePlacement, ...], tuple[LlmUsage, ...]]: ...


class JobStoreClient(Protocol):
    """Read canonical lifecycle groups from Tandemn Store."""

    def running_jobs(self, user_id: str) -> Sequence[object]: ...

    def waiting_jobs(self, user_id: str) -> Sequence[object]: ...

    def paused_jobs(self, user_id: str) -> Sequence[object]: ...


class ResourceMap(Protocol):
    """Expose the Store resource-map fields used by this planner."""

    market: object

    def scheduling_summary(self) -> Mapping[str, Mapping[str, Any]]: ...


class ResourceMapStoreClient(Protocol):
    """Read one current Store resource map."""

    def get(self) -> ResourceMap: ...


class ModelCatalogStoreClient(Protocol):
    """Read one model catalog by its canonical model ID."""

    def get(self, model_id: str) -> object | None: ...


class PlanStoreClient(Protocol):
    """Persist one canonical Store plan."""

    def create(self, plan: object) -> object: ...


class OnlineOpenEvolveClient:
    """Warm-start one bounded OpenEvolve search for each frozen tick."""

    def __init__(
        self,
        iterations: int,
        output_dir: Path,
        api_base: str | None = None,
        runner: OpenEvolveRunner | None = None,
    ) -> None:
        """Keep tick-local search settings and the prior winner as a seed."""
        if iterations < 1:
            raise ValueError("iterations must be positive")
        self.iterations = iterations
        self.output_dir = output_dir
        self.api_base = api_base
        self.runner = runner
        self.initial_program = OPENEOLVE_INITIAL_PROGRAM
        self.tick_number = 0

    def place(
        self,
        tick_number: int,
        model: str,
        max_tokens: int,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[tuple[OpenEvolvePlacement, ...], tuple[LlmUsage, ...]]:
        """Evolve and execute one policy against the current tick only."""
        try:
            payload = _mapping(json.loads(user_prompt), "scheduler payload")
        except json.JSONDecodeError as error:
            raise ValueError("scheduler payload is not valid JSON") from error
        model_catalog = payload.get("model_catalog")
        if not isinstance(model_catalog, list):
            raise ValueError("scheduler payload model_catalog must be an array")
        tick_system_prompt = (
            f"{system_prompt}\n\nFrozen resolved model catalog for this tick:\n"
            f"{json.dumps(model_catalog, sort_keys=True, separators=(',', ':'))}"
        )
        result = (self.runner or run_openevolve)(
            self.initial_program,
            _evaluator_source(payload),
            _openevolve_config(
                model,
                max_tokens,
                tick_system_prompt,
                self.api_base,
            ),
            self.iterations,
            str(self.output_dir / f"tick-{self.tick_number:06d}"),
            False,
        )
        source = getattr(result, "best_code", None)
        if not isinstance(source, str) or not source:
            raise RuntimeError("OpenEvolve returned no placement policy")
        policy = load_policy_source(source)
        response = policy(payload)
        document = _mapping(response, "OpenEvolve policy response")
        placements = _parse_placements(document)
        _validate_policy_response(payload, document)
        self.initial_program = source
        self.tick_number += 1
        usage = tuple(
            LlmUsage(
                tick_number=tick_number,
                input_tokens=_nonnegative_int(row.get("input_tokens"), "input_tokens"),
                output_tokens=_nonnegative_int(
                    row.get("output_tokens"), "output_tokens"
                ),
                model=_required_text(row.get("model"), "model"),
            )
            for raw_row in getattr(result, "llm_usage", ())
            for row in (_mapping(raw_row, "OpenEvolve LLM usage"),)
        )
        return placements, usage


class OpenEvolvePlannerDriver:
    """Evolve placements and create canonical Store plans for each tick."""

    method_name = "openevolve"

    def __init__(
        self,
        user_id: str,
        scenario: Scenario,
        job_store: JobStoreClient,
        resource_map_store: ResourceMapStoreClient,
        model_catalog_store: ModelCatalogStoreClient,
        plan_store: PlanStoreClient,
        model: str,
        max_tokens: int,
        backend: str,
        client: OpenEvolveClient,
    ) -> None:
        """Keep Store clients, evolution settings, and simulator jobs."""
        self.user_id = user_id
        self.jobs: dict[str, OnlineJob | BatchJob] = {}
        for online_job in scenario.online_jobs:
            self.jobs[online_job.store_job_id] = online_job
        for batch_job in scenario.batch_jobs:
            self.jobs[batch_job.store_job_id] = batch_job
        self.job_store = job_store
        self.resource_map_store = resource_map_store
        self.model_catalog_store = model_catalog_store
        self.plan_store = plan_store
        self.model = model
        self.max_tokens = max_tokens
        self.backend = backend
        self.client = client
        self.feedback = OutcomeFeedback(user_id)

    @classmethod
    def build(
        cls,
        user_id: str,
        scenario: Scenario,
        settings: Mapping[str, object],
        postgres_client: object | None = None,
        client: OpenEvolveClient | None = None,
    ) -> OpenEvolvePlannerDriver:
        """Build the standalone Store planner and online OpenEvolve client."""
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
            model=_required_text(settings.get("model"), "model"),
            max_tokens=_positive_int(settings.get("max_tokens"), "max_tokens"),
            backend=_required_text(settings.get("backend"), "backend"),
            client=client
            or OnlineOpenEvolveClient(
                _positive_int(settings.get("iterations", 8), "iterations"),
                Path(
                    _optional_text(settings.get("output_dir")) or "outputs/openevolve"
                ),
                _optional_text(settings.get("api_base")),
            ),
        )
        driver.feedback.bind(GpuMetricStore(postgres), PostgresEventLog(postgres))
        return driver

    def create_plans(self, tick_number: int, logical_time: datetime) -> PlannerResult:
        """Read one snapshot, evolve placements, and persist one Store plan."""
        try:
            self.feedback.observe(logical_time)
            snapshot = self._read_store_snapshot()
            decision_jobs = self._actionable_job_ids(snapshot)
            placements: tuple[OpenEvolvePlacement, ...] = ()
            llm_usage: tuple[LlmUsage, ...] = ()
            if decision_jobs:
                jobs = tuple(self.jobs[job_id] for _, job_id in decision_jobs)
                config_policy = load_config_policy()
                prompt = _build_user_prompt(
                    tick_number,
                    logical_time,
                    snapshot,
                    jobs,
                    config_policy,
                    _outcome_feedback_payload(
                        self.feedback,
                        snapshot,
                        jobs,
                        config_policy,
                    ),
                )
                placements, llm_usage = self.client.place(
                    tick_number,
                    self.model,
                    self.max_tokens,
                    OPENEOLVE_SYSTEM_PROMPT,
                    prompt,
                )

            actions = self._build_actions(snapshot, placements)
            if not actions:
                return PlannerResult(
                    tick_number=tick_number,
                    plans=(),
                    llm_usage=llm_usage,
                )

            from tandemn_system_data.models import Plan

            plan = Plan(
                user_id=self.user_id,
                koi_version=self.method_name,
                tick_rationale=_tick_rationale(tick_number, actions),
                actions=actions,
            )
            self.plan_store.create(plan)
            self._record_action_feedback(actions, logical_time)
            return PlannerResult(
                tick_number=tick_number,
                plans=(CreatedPlan(plan_id=plan.plan_id),),
                llm_usage=llm_usage,
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
        running_jobs = tuple(_running_job_payload(job) for job in running)
        waiting_jobs = tuple(_model_payload(job) for job in waiting)
        model_ids = {
            _required_text(
                _mapping(job.get("spec_json"), "job spec_json").get("model_id"),
                "model_id",
            )
            for job in (*running_jobs, *waiting_jobs)
            if job.get("job_id") in self.jobs
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

    def _actionable_job_ids(
        self,
        snapshot: StoreSnapshot,
    ) -> tuple[tuple[str, str], ...]:
        """Return eligible Store jobs in the policy response order."""
        actionable = []
        for lifecycle, jobs in (
            ("running", snapshot.running_jobs),
            ("waiting", snapshot.waiting_jobs),
        ):
            for job in jobs:
                job_id = _required_text(job.get("job_id"), f"{lifecycle} Store job_id")
                if job_id not in self.jobs:
                    continue
                if lifecycle == "running":
                    if active_rank_id(job) is None:
                        continue
                    if (
                        isinstance(self.jobs[job_id], BatchJob)
                        and self.feedback.has_pending_batch(job_id)
                    ):
                        continue
                actionable.append((lifecycle, job_id))
        return tuple(actionable)

    def _build_actions(
        self,
        snapshot: StoreSnapshot,
        placements: Sequence[OpenEvolvePlacement],
    ) -> list[dict[str, object]]:
        """Convert evolved decisions into complete canonical Store actions."""
        decision_jobs = self._actionable_job_ids(snapshot)
        _validate_response_count(decision_jobs, placements)
        placement_by_job = {
            job_id: placement
            for (_, job_id), placement in zip(decision_jobs, placements, strict=True)
        }
        resources = deepcopy(snapshot.resources)
        actions: list[dict[str, object]] = []
        for running in snapshot.running_jobs:
            job_id = _required_text(running.get("job_id"), "running Store job_id")
            if job_id not in self.jobs:
                actions.append({"job_id": job_id, "type": "keep"})
                continue
            if (
                isinstance(self.jobs[job_id], BatchJob)
                and self.feedback.has_pending_batch(job_id)
            ):
                actions.append({"job_id": job_id, "type": "keep"})
                continue
            placement = placement_by_job.get(job_id)
            rank_name = active_rank_id(running)
            if (
                placement is None
                or placement.decision == "keep"
                or rank_name is None
                or _is_noop_swap(
                    running,
                    placement,
                    resources,
                )
            ):
                actions.append({"job_id": job_id, "type": "keep"})
                continue
            if placement.decision != "swap":
                raise ValueError("running jobs require keep or swap decisions")
            actions.append(
                _place_action(
                    job_id,
                    running,
                    self.jobs[job_id],
                    placement,
                    resources,
                    snapshot.model_catalogs,
                    action_type="swap",
                    existing_rank_id=rank_name,
                )
            )
        for waiting in snapshot.waiting_jobs:
            job_id = waiting.get("job_id")
            if not isinstance(job_id, str):
                raise ValueError("waiting Store job is missing job_id")
            if job_id not in self.jobs:
                actions.append({"job_id": job_id, "type": "defer"})
                continue
            placement = placement_by_job[job_id]
            if placement.decision == "defer":
                actions.append({"job_id": job_id, "type": "defer"})
                continue
            if placement.decision != "place":
                raise ValueError("waiting jobs require place or defer decisions")
            actions.append(
                _place_action(
                    job_id,
                    waiting,
                    self.jobs[job_id],
                    placement,
                    resources,
                    snapshot.model_catalogs,
                )
            )
        _validate_complete_job_actions(snapshot, actions)
        return actions

    def _record_action_feedback(
        self,
        actions: Sequence[Mapping[str, object]],
        logical_time: datetime,
    ) -> None:
        """Track persisted canonical place and swap ladders for later outcomes."""
        for action in actions:
            if action.get("type") not in {"place", "swap"}:
                continue
            job_id = _required_text(action.get("job_id"), "plan job_id")
            scenario_job = self.jobs.get(job_id)
            if scenario_job is None:
                continue
            ladders = action.get("ladder")
            if not isinstance(ladders, list) or len(ladders) != 1:
                raise ValueError("OpenEvolve placement actions require one ladder")
            ladder = _mapping(ladders[0], "placement ladder")
            config = _mapping(ladder.get("config"), "placement ladder config")
            environment = ladder.get("env")
            if not isinstance(environment, (list, tuple, str)):
                raise ValueError("placement ladder is missing environment")
            self.feedback.record(
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
                rank_id=_required_text(ladder.get("rank_id"), "rank_id"),
                window_start=logical_time,
                is_batch=isinstance(scenario_job, BatchJob),
                required_tps=_number(action.get("target_tps")),
                ttft_target_ms=_number(action.get("target_p99_ttft_ms")),
                tpot_target_ms=_number(action.get("target_p99_tpot_ms")),
            )


def _build_user_prompt(
    tick_number: int,
    logical_time: datetime,
    snapshot: StoreSnapshot,
    jobs: tuple[OnlineJob | BatchJob, ...],
    config_policy: ConfigPolicy,
    outcome_feedback: Sequence[Mapping[str, object]] = (),
) -> str:
    """Serialize the frozen snapshot, JSON-only outcome evidence, and decisions."""
    running_by_id = {
        _required_text(job.get("job_id"), "running Store job_id"): job
        for job in snapshot.running_jobs
    }
    store_jobs_by_id = {
        _required_text(job.get("job_id"), "Store job_id"): job
        for jobs_in_lifecycle in (snapshot.running_jobs, snapshot.waiting_jobs)
        for job in jobs_in_lifecycle
    }
    jobs_for_tick = []
    for job in jobs:
        active_job = running_by_id.get(job.store_job_id)
        payload = _job_prompt_payload(job)
        store_job = store_jobs_by_id.get(job.store_job_id)
        known = (
            _known_job_fields(store_job, job)
            if store_job is not None
            else {"model_id": job.model_name}
        )
        payload["job_id"] = job.store_job_id
        payload["model_id"] = str(known.get("model_id") or job.model_name)
        payload["priority"] = _priority(known)
        payload["lifecycle"] = "running" if active_job is not None else "waiting"
        if active_job is not None:
            payload["active_placement"] = _active_placement(active_job)
        jobs_for_tick.append(payload)
    return json.dumps(
        {
            "tick": tick_number,
            "logical_time": logical_time.isoformat(),
            "tandemn_state": {
                "resources": _resource_prompt_payload(snapshot.resources),
                "active_jobs": snapshot.running_jobs,
                "pending_jobs": snapshot.waiting_jobs,
            },
            "config_policy": _config_policy_payload(
                config_policy,
                snapshot.resources,
                jobs,
            ),
            "model_catalog": _model_catalog_payload(
                config_policy,
                snapshot.model_catalogs,
                snapshot.resources,
                jobs,
            ),
            "outcome_feedback": [
                {str(key): value for key, value in evidence.items()}
                for evidence in outcome_feedback
            ],
            "jobs_for_tick": jobs_for_tick,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _outcome_feedback_payload(
    feedback: OutcomeFeedback,
    snapshot: StoreSnapshot,
    jobs: Sequence[OnlineJob | BatchJob],
    config_policy: ConfigPolicy,
) -> list[dict[str, object]]:
    """Expose prior outcomes for policy-compliant placements in this tick."""
    del snapshot, config_policy
    return list(feedback.outcomes_for_job_ids([job.store_job_id for job in jobs]))


def _config_policy_payload(
    config_policy: ConfigPolicy,
    resources: Mapping[str, Mapping[str, Any]],
    jobs: Sequence[OnlineJob | BatchJob],
) -> list[dict[str, object]]:
    """Return policy entries relevant to this tick's models and hardware."""
    hardware = {
        str(resource.get("gpu_type") or environment_key.split("|")[-1])
        for environment_key, resource in resources.items()
    }
    entries: list[dict[str, object]] = []
    for model_id in sorted({job.model_name for job in jobs}):
        for gpu_type in sorted(hardware):
            rule = config_policy.rule_for(gpu_type, model_id)
            if rule is not None:
                entry: dict[str, object] = {
                    "model_id": model_id,
                    "gpu_type": gpu_type,
                    "precision": rule.precision,
                    "allowed_tp": list(rule.allowed_tp),
                    "allowed_pp": list(
                        rule.allowed_pp
                        if rule.allowed_pp is not None
                        else sorted(VALID_PP_DEGREES)
                    ),
                }
                entries.append(entry)
    return entries


def _model_catalog_payload(
    config_policy: ConfigPolicy,
    catalogs: Mapping[str, Mapping[str, Any]],
    resources: Mapping[str, Mapping[str, Any]],
    jobs: Sequence[OnlineJob | BatchJob],
) -> list[dict[str, object]]:
    """Resolve Store launch profiles for this tick's feasible model/GPU pairs."""
    hardware = {
        str(resource.get("gpu_type") or environment_key.split("|")[-1])
        for environment_key, resource in resources.items()
    }
    profiles = []
    for model_id in sorted({job.model_name for job in jobs}):
        try:
            catalog = catalogs[model_id]
        except KeyError as error:
            raise ValueError(f"model catalog missing {model_id!r}") from error
        for gpu_type in sorted(hardware):
            if config_policy.rule_for(gpu_type, model_id) is None:
                continue
            profiles.append(
                {
                    "model_id": model_id,
                    "gpu_type": gpu_type,
                    **_materialize_launch_config(catalog, gpu_type),
                }
            )
    return profiles


def _place_action(
    job_id: str,
    store_job: Mapping[str, object],
    scenario_job: OnlineJob | BatchJob,
    placement: OpenEvolvePlacement,
    resources: dict[str, dict[str, Any]],
    catalogs: Mapping[str, Mapping[str, Any]],
    action_type: str = "place",
    existing_rank_id: str | None = None,
) -> dict[str, object]:
    """Combine deterministic job fields with one place or swap decision."""
    if action_type not in {"place", "swap"}:
        raise ValueError("OpenEvolve action type must be place or swap")
    if action_type == "swap" and existing_rank_id is None:
        raise ValueError("OpenEvolve swaps require the active rank ID")
    known = _known_job_fields(store_job, scenario_job)
    target_tps = _number(known.get("target_tps"))
    return {
        "job_id": job_id,
        "type": action_type,
        "target_tps": (
            target_tps if target_tps is not None else _target_tps(scenario_job)
        ),
        "target_p99_ttft_ms": _number(known.get("target_p99_ttft_ms")),
        "target_p99_tpot_ms": _number(known.get("target_p99_tpot_ms")),
        "ladder": [
            _placement_ladder(
                scenario_job,
                known,
                placement,
                resources,
                catalogs,
                existing_rank_id,
            )
        ],
    }


def _placement_ladder(
    job: OnlineJob | BatchJob,
    known: Mapping[str, object],
    placement: OpenEvolvePlacement,
    resources: dict[str, dict[str, Any]],
    catalogs: Mapping[str, Mapping[str, Any]],
    existing_rank_id: str | None = None,
) -> dict[str, object]:
    """Validate one placement and reserve its Store resource footprint."""
    if placement.environment is None or placement.configuration is None:
        raise ValueError(f"place decision for {job.store_job_id} is incomplete")
    environment = resources.get(placement.environment)
    if environment is None:
        raise ValueError(f"unknown environment {placement.environment!r}")
    config = placement.configuration
    expected_config = {"tp", "pp", "dp"}
    if set(config) != expected_config:
        raise ValueError("OpenEvolve configuration must contain only tp, pp, and dp")
    tp = _positive_int(config.get("tp"), "tp")
    pp = _positive_int(config.get("pp"), "pp")
    dp = _positive_int(config.get("dp"), "dp")
    model_id = str(known.get("model_id") or job.model_name)
    gpu_type = str(environment.get("gpu_type") or placement.environment.split("|")[-1])
    try:
        catalog = catalogs[model_id]
    except KeyError as error:
        raise ValueError(f"model catalog missing {model_id!r}") from error
    launch_config = _materialize_launch_config(catalog, gpu_type)
    policy_rule = load_config_policy().rule_for(gpu_type, model_id)
    if policy_rule is None:
        raise ValueError(f"{model_id} is unsupported on {gpu_type}")
    if not policy_rule.allows_parallelism(tp, pp):
        raise ValueError(
            f"tp={tp}, pp={pp} is not allowed for {model_id} on {gpu_type}"
        )
    gpu_count = tp * pp
    pool = _select_pool(environment, placement.instance_type)
    required_vram_gib = _job_required_vram_gib(job)
    gpu_memory_gib = _gpu_memory_gib(environment, gpu_type, pool)
    _validate_vram_fit(
        job.store_job_id,
        required_vram_gib,
        gpu_memory_gib,
        tp,
        pp,
    )
    capacity_per_replica = gpu_count
    instance_type = None
    if pool is not None:
        instance_type = _required_text(pool.get("instance_type"), "instance_type")
        allocation_kind = str(
            pool.get("allocation_kind") or pool.get("allocation_unit") or "instance"
        ).lower()
        if allocation_kind != "gpu":
            capacity_per_replica = _positive_int(
                pool.get("gpus_per_instance"),
                "gpus_per_instance",
            )
            if gpu_count > capacity_per_replica:
                raise ValueError(
                    f"{job.store_job_id} needs {gpu_count} GPUs per replica; "
                    f"{instance_type} has {capacity_per_replica}"
                )
    footprint = dp * capacity_per_replica
    available = int(environment["free"])
    if pool is not None:
        available = min(available, int(pool["free"]))
    if footprint > available:
        raise ValueError(
            f"{placement.environment} has {available} free GPUs, not {footprint}"
        )
    environment["free"] = int(environment["free"]) - footprint
    if pool is not None:
        pool["free"] = int(pool["free"]) - footprint

    rank_config: dict[str, object] = {
        **launch_config,
        "model_id": model_id,
        "precision": policy_rule.precision,
        "instance_type": instance_type,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "count": gpu_count,
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "ep": known.get("ep") or 1,
        "rank_traffic_share": 1.0,
    }
    return {
        "role": known.get("role") or "aggregate",
        "rank_id": existing_rank_id or rank_id(known),
        "env": placement.environment.split("|"),
        "config": rank_config,
        "n_replicas": dp,
        "mechanism_id": known.get("mechanism_id"),
    }


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
    launch_fields = (
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
    launch_config = {}
    for field in launch_fields:
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
            entry = _mapping(raw_entry, field)
            if entry.get("gpu_type") == gpu_type:
                value = entry.get("value")
                break
        if value is None:
            raise ValueError(f"model catalog {field} has no profile for {gpu_type}")
        launch_config[field] = value
    return launch_config


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
            pool = _select_pool(environment, _optional_text(shape.get("instance_type")))
            footprint = count * replicas
            if pool is not None:
                kind = str(
                    pool.get("allocation_kind")
                    or pool.get("allocation_unit")
                    or "instance"
                ).lower()
                if kind != "gpu":
                    footprint = replicas * _positive_int(
                        pool.get("gpus_per_instance"),
                        "gpus_per_instance",
                    )
                pool_key = (env_key, str(pool.get("instance_type")))
                used_by_pool[pool_key] = used_by_pool.get(pool_key, 0) + footprint
            used_by_env[env_key] = used_by_env.get(env_key, 0) + footprint

    for env_key, environment in resources.items():
        total = int(environment.get("total", 0))
        environment["free"] = max(0, total - used_by_env.get(env_key, 0))
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


def _active_placement(running_job: Mapping[str, object]) -> dict[str, object]:
    """Expose the active ladder's physical shape for a running swap decision."""
    active_ranks = running_job.get("active_ranks")
    if not isinstance(active_ranks, list) or len(active_ranks) != 1:
        raise ValueError("running job must have exactly one active rank")
    rank = _mapping(active_ranks[0], "active rank")
    shape = _mapping(rank.get("shape_json", {}), "active rank shape")
    active_environment = shape.get("env") or shape.get("pool_id")
    if active_environment is None:
        raise ValueError("active rank is missing environment")
    return {
        "environment": _env_key(active_environment),
        "instance_type": _optional_text(shape.get("instance_type")),
        "gpu_count": _positive_int(shape.get("count"), "active rank count"),
        "tp": _positive_int(shape.get("tp", 1), "active rank tp"),
        "pp": _positive_int(shape.get("pp", 1), "active rank pp"),
        "n_replicas": _positive_int(
            rank.get("n_replicas"),
            "active rank replicas",
        ),
    }


def _is_noop_swap(
    running_job: Mapping[str, object],
    placement: OpenEvolvePlacement,
    resources: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return whether a swap selects the running ladder's same physical shape."""
    if placement.decision != "swap" or placement.configuration is None:
        return False
    active = _active_placement(running_job)
    config = placement.configuration
    environment = resources.get(placement.environment)
    if environment is None:
        return False
    pool = _select_pool(environment, placement.instance_type)
    instance_type = (
        None
        if pool is None
        else _required_text(pool.get("instance_type"), "instance_type")
    )
    return (
        placement.environment == active["environment"]
        and instance_type == active["instance_type"]
        and _positive_int(config.get("tp"), "tp")
        * _positive_int(config.get("pp"), "pp")
        == active["gpu_count"]
        and _positive_int(config.get("tp"), "tp") == active["tp"]
        and _positive_int(config.get("pp"), "pp") == active["pp"]
        and _positive_int(config.get("dp"), "dp") == active["n_replicas"]
    )


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


def _parse_placements(
    document: Mapping[str, object],
) -> tuple[OpenEvolvePlacement, ...]:
    """Parse and validate one evolved placement response."""
    values = document.get("placements")
    if not isinstance(values, list):
        raise ValueError("OpenEvolve response is missing placements")
    placements = []
    for index, value in enumerate(values):
        item = _mapping(value, f"placements[{index}]")
        lifecycle = _required_text(item.get("lifecycle"), "lifecycle")
        decision = _required_text(item.get("decision"), "decision")
        allowed_decisions = {
            "waiting": {"place", "defer"},
            "running": {"keep", "swap"},
        }
        if lifecycle not in allowed_decisions:
            raise ValueError("lifecycle must be waiting or running")
        if decision not in allowed_decisions[lifecycle]:
            raise ValueError(f"{lifecycle} jobs cannot use {decision} decisions")
        if decision in {"defer", "keep"}:
            if any(
                item.get(key) is not None
                for key in ("environment", "instance_type", "configuration")
            ):
                raise ValueError(f"{decision} decisions must use null placement fields")
            placements.append(
                OpenEvolvePlacement(lifecycle, decision, None, None, None)
            )
            continue
        configuration = _mapping(item.get("configuration"), "configuration")
        if set(configuration) != {"tp", "pp", "dp"}:
            raise ValueError("configuration must contain only tp, pp, and dp")
        placements.append(
            OpenEvolvePlacement(
                lifecycle=lifecycle,
                decision=decision,
                environment=_required_text(item.get("environment"), "environment"),
                instance_type=_optional_text(item.get("instance_type")),
                configuration=configuration,
            )
        )
    return tuple(placements)


def _validate_response_count(
    expected_jobs: Sequence[tuple[str, str]],
    placements: Sequence[OpenEvolvePlacement],
) -> None:
    """Require one lifecycle-valid ordered decision per actionable Store job."""
    if len(placements) != len(expected_jobs):
        raise ValueError("OpenEvolve must decide every actionable job exactly once")
    for (lifecycle, _), placement in zip(expected_jobs, placements, strict=True):
        if placement.lifecycle != lifecycle:
            raise ValueError("OpenEvolve placement lifecycle does not match Store job")
        allowed_decisions = {
            "waiting": {"place", "defer"},
            "running": {"keep", "swap"},
        }
        if placement.decision not in allowed_decisions[lifecycle]:
            raise ValueError(
                f"{lifecycle} jobs cannot use {placement.decision} decisions"
            )


def _validate_complete_job_actions(
    snapshot: StoreSnapshot,
    actions: Sequence[Mapping[str, object]],
) -> None:
    """Require exactly one canonical action for every Store job."""
    store_ids = []
    for jobs in (
        snapshot.running_jobs,
        snapshot.waiting_jobs,
    ):
        for job in jobs:
            store_ids.append(_required_text(job.get("job_id"), "Store job_id"))
    action_ids = [
        _required_text(action.get("job_id"), "plan job_id") for action in actions
    ]
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
        f"OpenEvolve cluster simulator tick {tick_number}: "
        f"place={counts['place']}, defer={counts['defer']}, "
        f"keep={counts['keep']}, preempt={counts['preempt']}."
    )


def _target_tps(job: OnlineJob | BatchJob) -> float | None:
    """Derive required output-token throughput from the scenario workload."""
    features = job.workload.get("features", {})
    explicit = _number(features.get("target_tps"))
    if explicit is not None:
        return explicit
    if isinstance(job, OnlineJob):
        phases = job.workload.get("phases", [])
        rates = [
            float(phase["rate"])
            for phase in phases
            if isinstance(phase, Mapping) and "rate" in phase
        ]
        output_tokens = _number(features.get("osl_token_avg"))
        if not rates or output_tokens is None:
            return None
        return round(max(rates) * output_tokens, 3)

    output = job.workload.get("output")
    average = _number(getattr(output, "average", None))
    duration_seconds = (job.deadline_at_ns - job.releases_at_ns) / 1_000_000_000
    if average is None or duration_seconds <= 0:
        return None
    record_count = _positive_int(job.workload.get("record_count"), "record_count")
    return float(round(record_count * average / duration_seconds, 3))


def _job_prompt_payload(job: OnlineJob | BatchJob) -> dict[str, object]:
    """Expose workload requirements without transferring Store job identity."""
    required_vram_gib = _job_required_vram_gib(job)
    if required_vram_gib is None:
        raise ValueError(f"{job.store_job_id} is missing required_vram_gib")
    payload = asdict(job)
    payload.pop("name", None)
    payload.pop("store_job_id", None)
    payload["kind"] = "online" if isinstance(job, OnlineJob) else "batch"
    payload["required_vram_gib"] = required_vram_gib
    return payload


def _job_required_vram_gib(job: OnlineJob | BatchJob) -> float | None:
    """Read one explicit or precomputed maximum workload VRAM requirement."""
    features = job.workload.get("features", {})
    sources = (features, job.workload)
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in ("required_vram_gib", "max_vram_gib", "vram_gib"):
            value = _number(source.get(key))
            if value is not None and value > 0:
                return value
    for requirements in VRAM_REQUIREMENTS_GIB.values():
        for job_id in (job.store_job_id, job.name):
            value = requirements.get(job_id)
            if value is not None:
                return float(value)
    return None


def _resource_prompt_payload(
    resources: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Expose normalized per-GPU memory alongside each schedulable resource."""
    payload = {
        environment_key: deepcopy(dict(resource))
        for environment_key, resource in resources.items()
    }
    for environment_key, resource in payload.items():
        gpu_type = str(resource.get("gpu_type") or environment_key.split("|")[-1])
        memory_gib = _gpu_memory_gib(resource, gpu_type)
        if memory_gib is None:
            raise ValueError(f"{environment_key} is missing gpu_memory_gib")
        resource["gpu_memory_gib"] = memory_gib
        pools = resource.get("pools", [])
        if not isinstance(pools, list):
            raise ValueError("resource pools must be a list")
        for pool in pools:
            if not isinstance(pool, dict):
                raise ValueError("resource pool must be an object")
            pool_memory_gib = _gpu_memory_gib(resource, gpu_type, pool)
            if pool_memory_gib is None:
                raise ValueError(f"{environment_key} pool is missing gpu_memory_gib")
            pool["gpu_memory_gib"] = pool_memory_gib
    return payload


def _gpu_memory_gib(
    resource: Mapping[str, object],
    gpu_type: str,
    pool: Mapping[str, object] | None = None,
) -> float | None:
    """Read per-GPU memory from resource facts or the shared hardware table."""
    for source in (pool or {}, resource):
        for key in (
            "gpu_memory_gib",
            "gpu_memory_gb",
            "memory_gib",
            "memory_gb",
        ):
            value = _number(source.get(key))
            if value is not None and value > 0:
                return value
    memory_gib = _HARDWARE_MEMORY_GIB.get(_canonical_hardware(gpu_type))
    return None if memory_gib is None else float(memory_gib)


def _validate_vram_fit(
    job_id: str,
    required_vram_gib: float | None,
    gpu_memory_gib: float | None,
    tp: int,
    pp: int,
) -> None:
    """Require one TP/PP copy to hold the workload's maximum VRAM demand."""
    if required_vram_gib is None:
        raise ValueError(f"{job_id} is missing required_vram_gib")
    if gpu_memory_gib is None:
        raise ValueError(f"{job_id} target is missing gpu_memory_gib")
    available_vram_gib = gpu_memory_gib * tp * pp
    if available_vram_gib < required_vram_gib:
        raise ValueError(
            f"{job_id} needs {required_vram_gib} GiB but tp={tp}, pp={pp} on "
            f"{gpu_memory_gib} GiB GPUs provides {available_vram_gib} GiB"
        )


def _known_job_fields(
    store_job: Mapping[str, object],
    job: OnlineJob | BatchJob,
) -> dict[str, object]:
    """Merge Store and scenario fields once using deterministic precedence."""
    spec = store_job.get("spec_json")
    store_spec = spec if isinstance(spec, Mapping) else {}
    store_features_value = store_spec.get("job_features")
    store_features = (
        store_features_value if isinstance(store_features_value, Mapping) else {}
    )
    scenario_features_value = job.workload.get("features", {})
    scenario_features = (
        scenario_features_value if isinstance(scenario_features_value, Mapping) else {}
    )
    known: dict[str, object] = {}
    for source in (
        store_job,
        store_spec,
        store_features,
        job.workload,
        scenario_features,
    ):
        for key, value in source.items():
            if value is not None:
                known.setdefault(str(key), value)
    return known


def _select_pool(
    environment: Mapping[str, Any],
    instance_type: str | None,
) -> dict[str, Any] | None:
    """Resolve one concrete resource pool."""
    pools = environment.get("pools", [])
    if not isinstance(pools, list):
        raise ValueError("resource pools must be a list")
    if not pools:
        if instance_type is not None:
            raise ValueError("environment has no instance pools")
        return None
    typed_pools = []
    for pool in pools:
        if not isinstance(pool, dict):
            raise ValueError("resource pool must be an object")
        typed_pools.append(pool)
    if instance_type is not None:
        for pool in typed_pools:
            if pool.get("instance_type") == instance_type:
                return pool
        raise ValueError(f"unknown instance_type {instance_type!r}")
    if len(typed_pools) == 1:
        return typed_pools[0]
    raise ValueError("instance_type is required when an environment has multiple pools")


def _default_market(resource_map: object) -> str:
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


def run_openevolve(
    initial_program: str,
    evaluator: str,
    config: Mapping[str, object],
    iterations: int,
    output_dir: str,
    cleanup: bool,
) -> object:
    """Run OpenEvolve in a child process to avoid this adapter's name collision."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="tandemn-openevolve-") as directory:
        root = Path(directory)
        program_path = root / "initial_program.py"
        evaluator_path = root / "evaluator.py"
        result_path = root / "result.json"
        program_path.write_text(initial_program, encoding="utf-8")
        evaluator_path.write_text(evaluator, encoding="utf-8")
        script = """\
import asyncio
import json
import sys
from pathlib import Path
from openevolve import run_evolution
from openevolve.config import Config
from openevolve.llm.openai import OpenAILLM

llm_usage = []

class TrackedOpenAILLM(OpenAILLM):
    async def _call_api(self, params):
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.chat.completions.create(**params),
        )
        usage = response.usage
        llm_usage.append({
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "model": response.model or self.model,
        })
        return response.choices[0].message.content

(
    program_path,
    evaluator_path,
    output_dir,
    result_path,
    config_json,
    iterations,
) = sys.argv[1:]
config = Config.from_dict(json.loads(config_json))
for model_config in config.llm.models:
    model_config.init_client = TrackedOpenAILLM
result = run_evolution(
    initial_program=program_path,
    evaluator=evaluator_path,
    config=config,
    iterations=int(iterations),
    output_dir=output_dir,
    cleanup=False,
)
Path(result_path).write_text(
    json.dumps({"best_code": result.best_code, "llm_usage": llm_usage}),
    encoding="utf-8",
)
"""
        environment = dict(os.environ)
        local_module_root = str(Path(__file__).resolve().parent)
        inherited_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            entry
            for entry in inherited_path.split(os.pathsep)
            if entry and Path(entry).resolve() != Path(local_module_root)
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(program_path),
                str(evaluator_path),
                str(target),
                str(result_path),
                json.dumps(config),
                str(iterations),
            ],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"OpenEvolve failed: {message}")
        try:
            document = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("OpenEvolve returned no readable result") from error
    return SimpleNamespace(
        best_code=document.get("best_code"),
        llm_usage=document.get("llm_usage", []),
    )


def load_policy_source(source: str) -> OpenEvolvePolicy:
    """Load one OpenEvolve winner without granting it Store access."""
    module = ModuleType("tandemn_openevolve_policy")
    exec(compile(source, "<openevolve-policy>", "exec"), module.__dict__)
    policy = getattr(module, "compute_placement", None)
    if not callable(policy):
        raise ValueError("OpenEvolve policy must define compute_placement(payload)")
    return cast(OpenEvolvePolicy, policy)


def _openevolve_config(
    model: str,
    max_tokens: int,
    system_prompt: str,
    api_base: str | None,
) -> dict[str, object]:
    """Build a serializable OpenEvolve configuration for a single tick."""
    llm: dict[str, object] = {
        "models": [{"name": model, "max_tokens": max_tokens, "weight": 1.0}],
        "max_tokens": max_tokens,
    }
    if api_base is not None:
        llm["api_base"] = api_base
    return {
        "max_iterations": 1,
        "random_seed": 42,
        "diff_based_evolution": True,
        "max_code_length": 20_000,
        "llm": llm,
        "prompt": {
            "system_message": system_prompt,
            "num_top_programs": 2,
            "num_diverse_programs": 1,
        },
        "database": {
            "population_size": 32,
            "archive_size": 8,
            "num_islands": 1,
        },
        "evaluator": {
            "timeout": 30,
            "max_retries": 0,
            "cascade_evaluation": False,
            "use_llm_feedback": False,
        },
    }


def _evaluator_source(payload: Mapping[str, object]) -> str:
    """Create OpenEvolve's self-contained evaluator for one frozen snapshot."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"""\
from copy import deepcopy
import importlib.util
import math

PAYLOAD = {serialized!r}

def evaluate(program_path):
    spec = importlib.util.spec_from_file_location("candidate_policy", program_path)
    if spec is None or spec.loader is None:
        return {{"combined_score": 0.0, "valid": 0.0}}
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        payload = __import__("json").loads(PAYLOAD)
        result = module.compute_placement(payload)
        validation = _validate(payload, result)
    except Exception:
        return {{"combined_score": 0.0, "valid": 0.0}}
    if not validation.get("valid"):
        return {{"combined_score": 0.0, "valid": 0.0}}
    placements = result["placements"]
    selected = [
        (job, placement)
        for job, placement in zip(payload["jobs_for_tick"], placements)
        if placement["decision"] in ("place", "swap")
    ]
    priority_base = len(payload["jobs_for_tick"]) + 1
    priority_score = sum(
        priority_base ** (int(job.get("priority", 2)) - 1)
        for job, _placement in selected
    )
    gpu_footprint = validation["gpu_footprint"]
    hourly_price = validation["hourly_price"]
    observed_reward = _observed_reward(
        payload,
        placements,
    )
    return {{
        "combined_score": float(
            4 * priority_score
            + observed_reward
            + (0.0 if not selected else 1 / (1 + hourly_price)
               + 1 / (1_000 * (1 + gpu_footprint)))
        ),
        "valid": 1.0,
        "admitted_jobs": float(len(selected)),
        "priority_score": float(priority_score),
        "mean_observed_reward": float(observed_reward),
        "hourly_price": float(hourly_price),
        "gpu_footprint": float(gpu_footprint),
    }}

def _observed_reward(payload, placements):
    rewards = {{}}
    evidence = payload.get("outcome_feedback", [])
    if isinstance(evidence, list):
        for outcome in evidence:
            key = _outcome_key(outcome)
            reward = outcome.get("reward") if isinstance(outcome, dict) else None
            if (
                key is not None
                and not isinstance(reward, bool)
                and isinstance(reward, (int, float))
                and math.isfinite(reward)
            ):
                rewards[key] = max(-1.0, min(1.0, float(reward)))
    selected_rewards = []
    for job, placement in zip(payload["jobs_for_tick"], placements):
        if placement.get("decision") not in ("place", "swap"):
            continue
        key = _selected_outcome_key(payload, job, placement)
        reward = rewards.get(key)
        if reward is not None:
            selected_rewards.append(reward)
    return (
        0.0
        if not selected_rewards
        else sum(selected_rewards) / len(selected_rewards)
    )

def _outcome_key(outcome):
    if not isinstance(outcome, dict):
        return None
    job_id = outcome.get("job_id")
    model_id = outcome.get("model_id")
    environment = outcome.get("environment")
    instance_type = outcome.get("instance_type")
    gpu_type = outcome.get("gpu_type")
    parallelism = (outcome.get("tp"), outcome.get("pp"), outcome.get("dp"))
    if (
        not all(isinstance(value, str) and value for value in (
            job_id,
            model_id,
            environment,
            gpu_type,
        ))
        or (instance_type is not None and not isinstance(instance_type, str))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in parallelism
        )
    ):
        return None
    return (
        job_id,
        model_id,
        environment,
        instance_type,
        gpu_type,
        *parallelism,
    )

def _selected_outcome_key(payload, job, placement):
    state = payload.get("tandemn_state")
    resources = state.get("resources", {{}}) if isinstance(state, dict) else {{}}
    environment = placement.get("environment")
    resource = resources.get(environment) if isinstance(resources, dict) else None
    if not isinstance(environment, str) or not isinstance(resource, dict):
        return None
    pools = resource.get("pools", [])
    if not isinstance(pools, list):
        return None
    pool = None
    instance_type = placement.get("instance_type")
    if pools:
        if instance_type is None and len(pools) == 1:
            pool = pools[0]
        elif isinstance(instance_type, str):
            pool = next(
                (
                    candidate
                    for candidate in pools
                    if isinstance(candidate, dict)
                    and candidate.get("instance_type") == instance_type
                ),
                None,
            )
        if not isinstance(pool, dict):
            return None
        instance_type = pool.get("instance_type")
    elif instance_type is not None:
        return None
    configuration = placement.get("configuration")
    if not isinstance(configuration, dict):
        return None
    job_id = job.get("job_id")
    model_id = job.get("model_id", job.get("model_name"))
    gpu_type = str(resource.get("gpu_type") or environment.split("|")[-1])
    return _outcome_key(
        {{
            "job_id": job_id,
            "model_id": model_id,
            "environment": environment,
            "instance_type": instance_type,
            "gpu_type": gpu_type,
            "tp": configuration.get("tp"),
            "pp": configuration.get("pp"),
            "dp": configuration.get("dp"),
        }}
    )

def _validate(payload, result):
    jobs = payload.get("jobs_for_tick")
    placements = result.get("placements") if isinstance(result, dict) else None
    if (
        not isinstance(result, dict)
        or set(result) != {{"placements"}}
        or not isinstance(jobs, list)
        or not isinstance(placements, list)
    ):
        return {{"valid": False}}
    if len(jobs) != len(placements):
        return {{"valid": False}}
    resources = deepcopy(payload["tandemn_state"]["resources"])
    policies = {{
        (value["model_id"], value["gpu_type"]): value
        for value in payload["config_policy"]
    }}
    catalogs = {{
        (value["model_id"], value["gpu_type"]): value
        for value in payload["model_catalog"]
    }}
    total_footprint = 0
    total_price = 0.0
    for job, placement in zip(jobs, placements):
        if (
            not isinstance(placement, dict)
            or set(placement) != {{
                "lifecycle",
                "decision",
                "environment",
                "instance_type",
                "configuration",
            }}
        ):
            return {{"valid": False}}
        lifecycle = job.get("lifecycle")
        if placement.get("lifecycle") != lifecycle:
            return {{"valid": False}}
        decision = placement.get("decision")
        keys = ("environment", "instance_type", "configuration")
        if lifecycle == "waiting":
            if decision == "defer":
                if any(placement.get(key) is not None for key in keys):
                    return {{"valid": False}}
                continue
            if decision != "place":
                return {{"valid": False}}
        elif lifecycle == "running":
            if decision == "keep":
                if any(placement.get(key) is not None for key in keys):
                    return {{"valid": False}}
                continue
            if decision != "swap":
                return {{"valid": False}}
        else:
            return {{"valid": False}}
        environment = placement.get("environment")
        configuration = placement.get("configuration")
        resource = resources.get(environment)
        if (
            not isinstance(environment, str)
            or not isinstance(configuration, dict)
            or not isinstance(resource, dict)
            or set(configuration) != {{"tp", "pp", "dp"}}
        ):
            return {{"valid": False}}
        names = ("tp", "pp", "dp")
        values = [configuration.get(key) for key in names]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values
        ):
            return {{"valid": False}}
        gpu_type = str(resource.get("gpu_type") or environment.split("|")[-1])
        policy = policies.get((job["model_name"], gpu_type))
        catalog = catalogs.get((job["model_name"], gpu_type))
        if (
            policy is None
            or catalog is None
            or configuration["tp"] not in policy["allowed_tp"]
            or (
                "allowed_pp" in policy
                and configuration["pp"] not in policy["allowed_pp"]
            )
        ):
            return {{"valid": False}}

        gpu_count = values[0] * values[1]
        capacity_per_replica = gpu_count
        pools = resource.get("pools", [])
        if not isinstance(pools, list) or any(
            not isinstance(pool, dict) for pool in pools
        ):
            return {{"valid": False}}
        instance_type = placement.get("instance_type")
        pool = None
        if pools:
            if instance_type is None and len(pools) == 1:
                pool = pools[0]
            elif isinstance(instance_type, str):
                pool = next(
                    (
                        candidate
                        for candidate in pools
                        if candidate.get("instance_type") == instance_type
                    ),
                    None,
                )
            if pool is None:
                return {{"valid": False}}
            allocation_kind = str(
                pool.get("allocation_kind")
                or pool.get("allocation_unit")
                or "instance"
            ).lower()
            if allocation_kind != "gpu":
                gpus_per_instance = pool.get("gpus_per_instance")
                if (
                    isinstance(gpus_per_instance, bool)
                    or not isinstance(gpus_per_instance, int)
                    or gpus_per_instance < gpu_count
                ):
                    return {{"valid": False}}
                capacity_per_replica = gpus_per_instance
        elif instance_type is not None:
            return {{"valid": False}}
        if lifecycle == "running":
            active = job.get("active_placement")
            if not isinstance(active, dict):
                return {{"valid": False}}
            target_instance_type = None if pool is None else pool.get("instance_type")
            if (
                environment == active.get("environment")
                and target_instance_type == active.get("instance_type")
                and values[0] * values[1] == active.get("gpu_count")
                and values[0] == active.get("tp")
                and values[1] == active.get("pp")
                and values[2] == active.get("n_replicas")
            ):
                return {{"valid": False}}

        required_vram_gib = job.get("required_vram_gib")
        if (
            isinstance(required_vram_gib, bool)
            or not isinstance(required_vram_gib, (int, float))
            or required_vram_gib <= 0
        ):
            return {{"valid": False}}
        memory_sources = (pool or {{}}, resource)
        gpu_memory_gib = None
        for source in memory_sources:
            for key in (
                "gpu_memory_gib",
                "gpu_memory_gb",
                "memory_gib",
                "memory_gb",
            ):
                value = source.get(key)
                if (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and value > 0
                ):
                    gpu_memory_gib = value
                    break
            if gpu_memory_gib is not None:
                break
        if (
            gpu_memory_gib is None
            or gpu_memory_gib * values[0] * values[1] < required_vram_gib
        ):
            return {{"valid": False}}

        footprint = values[2] * capacity_per_replica
        available = int(resource.get("free", 0))
        if pool is not None:
            available = min(available, int(pool.get("free", 0)))
        if available < footprint:
            return {{"valid": False}}
        resource["free"] = int(resource.get("free", 0)) - footprint
        if pool is not None:
            pool["free"] = int(pool.get("free", 0)) - footprint
        total_footprint += footprint
        price_source = pool or resource
        per_gpu = price_source.get("price_per_gpu_hour")
        if (
            isinstance(per_gpu, (int, float))
            and not isinstance(per_gpu, bool)
            and math.isfinite(per_gpu)
            and per_gpu > 0
        ):
            total_price += per_gpu * footprint
        else:
            per_instance = price_source.get("price_per_instance_hour")
            if (
                isinstance(per_instance, (int, float))
                and not isinstance(per_instance, bool)
                and math.isfinite(per_instance)
                and per_instance > 0
            ):
                gpus_per_instance = price_source.get("gpus_per_instance")
                if (
                    isinstance(gpus_per_instance, bool)
                    or not isinstance(gpus_per_instance, (int, float))
                    or not math.isfinite(gpus_per_instance)
                    or gpus_per_instance <= 0
                ):
                    gpus_per_instance = 1.0
                total_price += per_instance * footprint / gpus_per_instance
    return {{
        "valid": True,
        "gpu_footprint": total_footprint,
        "hourly_price": total_price,
    }}
"""


def _validate_policy_response(
    payload: Mapping[str, object],
    response: Mapping[str, object],
) -> None:
    """Reject a winning policy unless it still validates against this snapshot."""
    if set(response) != {"placements"}:
        raise ValueError("OpenEvolve policy must return only placements")
    jobs = payload.get("jobs_for_tick")
    raw_placements = response.get("placements")
    if not isinstance(jobs, list) or not isinstance(raw_placements, list):
        raise ValueError("OpenEvolve jobs and placements must be arrays")
    placements = _parse_placements(response)
    if len(placements) != len(jobs):
        raise ValueError(
            f"OpenEvolve expected {len(jobs)} placements, received {len(placements)}"
        )

    state = _mapping(payload.get("tandemn_state"), "tandemn_state")
    resources = deepcopy(_mapping(state.get("resources"), "resources"))
    policies = payload.get("config_policy")
    if not isinstance(policies, list):
        raise ValueError("config_policy must be an array")
    catalog_profiles = payload.get("model_catalog")
    if not isinstance(catalog_profiles, list):
        raise ValueError("model_catalog must be an array")
    catalog_pairs = set()
    for raw_profile in catalog_profiles:
        profile = _mapping(raw_profile, "model catalog profile")
        catalog_pairs.add((profile.get("model_id"), profile.get("gpu_type")))

    expected_placement = {
        "lifecycle",
        "decision",
        "environment",
        "instance_type",
        "configuration",
    }
    for index, (raw_job, raw_placement, placement) in enumerate(
        zip(jobs, raw_placements, placements, strict=True)
    ):
        try:
            job = _mapping(raw_job, "job")
            item = _mapping(raw_placement, "placement")
            if set(item) != expected_placement:
                raise ValueError("placement contains unexpected fields")
            lifecycle = _required_text(job.get("lifecycle"), "job lifecycle")
            if placement.lifecycle != lifecycle:
                raise ValueError("placement lifecycle does not match job lifecycle")
            if lifecycle == "waiting":
                if placement.decision == "defer":
                    continue
                if placement.decision != "place":
                    raise ValueError("waiting jobs must be placed or deferred")
            elif lifecycle == "running":
                if placement.decision == "keep":
                    continue
                if placement.decision != "swap":
                    raise ValueError("running jobs must be kept or swapped")
            else:
                raise ValueError("job lifecycle must be waiting or running")
            if placement.environment is None or placement.configuration is None:
                raise ValueError("placement decision is incomplete")
            resource = resources.get(placement.environment)
            if not isinstance(resource, dict):
                raise ValueError("environment is unknown")
            config = placement.configuration
            if set(config) != {"tp", "pp", "dp"}:
                raise ValueError("configuration contains unexpected fields")
            tp = _positive_int(config.get("tp"), "tp")
            pp = _positive_int(config.get("pp"), "pp")
            dp = _positive_int(config.get("dp"), "dp")
            gpu_type = str(
                resource.get("gpu_type") or placement.environment.split("|")[-1]
            )
            policy = next(
                (
                    rule
                    for rule in policies
                    if isinstance(rule, Mapping)
                    and rule.get("model_id") == job.get("model_name")
                    and rule.get("gpu_type") == gpu_type
                ),
                None,
            )
            if policy is None:
                raise ValueError(
                    f"{job.get('model_name')} is unsupported on {gpu_type}"
                )
            if (job.get("model_name"), gpu_type) not in catalog_pairs:
                raise ValueError(
                    f"model catalog missing {job.get('model_name')} on {gpu_type}"
                )
            allowed_tp = policy.get("allowed_tp")
            if not isinstance(allowed_tp, list) or tp not in allowed_tp:
                raise ValueError(f"tp={tp} is not allowed on {gpu_type}")
            allowed_pp = policy.get("allowed_pp")
            if allowed_pp is not None and (
                not isinstance(allowed_pp, list) or pp not in allowed_pp
            ):
                raise ValueError(f"pp={pp} is not allowed on {gpu_type}")

            gpu_count = tp * pp
            pool = _select_pool(resource, placement.instance_type)
            if lifecycle == "running":
                active = _mapping(job.get("active_placement"), "active placement")
                target_instance_type = (
                    None
                    if pool is None
                    else _required_text(pool.get("instance_type"), "instance_type")
                )
                if (
                    placement.environment == active.get("environment")
                    and target_instance_type == active.get("instance_type")
                    and tp * pp == active.get("gpu_count")
                    and tp == active.get("tp")
                    and pp == active.get("pp")
                    and dp == active.get("n_replicas")
                ):
                    raise ValueError("swap must select a different physical placement")
            capacity_per_replica = gpu_count
            if pool is not None:
                allocation_kind = str(
                    pool.get("allocation_kind")
                    or pool.get("allocation_unit")
                    or "instance"
                ).lower()
                if allocation_kind != "gpu":
                    capacity_per_replica = _positive_int(
                        pool.get("gpus_per_instance"),
                        "gpus_per_instance",
                    )
                    if gpu_count > capacity_per_replica:
                        raise ValueError(
                            f"configuration needs {gpu_count} GPUs per replica but "
                            f"the instance has {capacity_per_replica}"
                        )
            required_vram_gib = _number(job.get("required_vram_gib"))
            _validate_vram_fit(
                str(job.get("model_name") or f"placement {index}"),
                (
                    required_vram_gib
                    if required_vram_gib is not None and required_vram_gib > 0
                    else None
                ),
                _gpu_memory_gib(resource, gpu_type, pool),
                tp,
                pp,
            )
            footprint = dp * capacity_per_replica
            available = int(resource.get("free", 0))
            if pool is not None:
                available = min(available, int(pool.get("free", 0)))
            if footprint > available:
                raise ValueError(
                    f"placement needs {footprint} GPUs but only {available} are free"
                )
            resource["free"] = int(resource.get("free", 0)) - footprint
            if pool is not None:
                pool["free"] = int(pool.get("free", 0)) - footprint
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"OpenEvolve policy returned an invalid placement {index}: {error}"
            ) from error


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Require one mapping value."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return {str(key): item for key, item in value.items()}


def _required_text(value: object, name: str) -> str:
    """Require one nonempty text setting."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: object) -> str | None:
    """Return one optional nonempty text setting."""
    return None if value is None else _required_text(value, "optional text")


def _positive_int(value: object, name: str) -> int:
    """Require one positive integer setting."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    """Return one nonnegative integer token count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _number(value: object) -> float | None:
    """Return one optional numeric target."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _priority(known: Mapping[str, object]) -> int:
    """Normalize Store priority fields to the shared four-level scale."""
    explicit = _number(known.get("priority"))
    if explicit is not None and math.isfinite(explicit):
        return min(4, max(1, round(explicit)))
    raw = getattr(known.get("priority_class"), "value", known.get("priority_class"))
    return _PRIORITY_LEVELS.get(str(raw or "STANDARD").upper(), 2)


__all__ = [
    "OPENEOLVE_INITIAL_PROGRAM",
    "OPENEOLVE_SYSTEM_PROMPT",
    "OnlineOpenEvolveClient",
    "OpenEvolveClient",
    "OpenEvolvePlacement",
    "OpenEvolvePlannerDriver",
    "OpenEvolvePolicy",
    "StoreSnapshot",
    "load_policy_source",
    "run_openevolve",
]
