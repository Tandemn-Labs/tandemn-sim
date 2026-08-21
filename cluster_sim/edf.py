"""Place waiting jobs by earliest end time using static VRAM requirements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from typing import Protocol

from models import BatchJob, CreatedPlan, OnlineJob, PlannerResult, Scenario
from policy import ConfigPolicy, load_config_policy
from vram import (
    VRAM_REQUIREMENTS_GIB,
    ModelCatalogStoreClient,
    StoreSnapshot,
    _known_job_fields,
    _model_payload,
    _place_action,
    _resources_with_free_capacity,
    _running_job_payload,
    _text,
    _validate_complete_job_actions,
    reserve_vram_placement,
    select_vram_placement,
)


class JobStoreClient(Protocol):
    """Read canonical lifecycle groups from Tandemn Store."""

    def running_jobs(self, user_id: str) -> Sequence[object]: ...

    def waiting_jobs(self, user_id: str) -> Sequence[object]: ...

    def paused_jobs(self, user_id: str) -> Sequence[object]: ...


class ResourceMap(Protocol):
    """Expose the Store resource-map fields used by EDF."""

    market: object

    def scheduling_summary(self) -> Mapping[str, Mapping[str, object]]: ...


class ResourceMapStoreClient(Protocol):
    """Read one current Store resource map."""

    def get(self) -> ResourceMap: ...


class PlanStoreClient(Protocol):
    """Persist one canonical Store plan."""

    def create(self, plan: object) -> object: ...


class EdfPlannerDriver:
    """Create Store plans by placing the earliest-ending waiting jobs first."""

    method_name = "edf"

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
        """Keep Store clients and benchmark scheduling facts."""
        self.user_id = user_id
        self.benchmark_name = scenario.benchmark_name
        jobs = (*scenario.online_jobs, *scenario.batch_jobs)
        self.benchmark_job_ids = {job.store_job_id: job.name for job in jobs}
        self.end_times_ns = {job.store_job_id: _end_time_ns(job) for job in jobs}
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
    ) -> EdfPlannerDriver:
        """Build EDF with Tandemn Store public clients."""
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
            backend=_text(settings.get("backend"), "backend"),
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
        waiting_jobs = tuple(_model_payload(job) for job in waiting)
        model_ids = {
            _text(_known_job_fields(job).get("model_id"), "model_id")
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
        """Place the full waiting queue by earliest end time."""
        benchmark = VRAM_REQUIREMENTS_GIB.get(self.benchmark_name, {})
        resources = deepcopy(snapshot.resources)
        actions: list[dict[str, object]] = []
        for running in snapshot.running_jobs:
            actions.append({"job_id": running["job_id"], "type": "keep"})
        for waiting in sorted(snapshot.waiting_jobs, key=self._waiting_order):
            job_id = _text(waiting.get("job_id"), "waiting Store job_id")
            known = _known_job_fields(waiting)
            model_id = _text(known.get("model_id"), "model_id")
            benchmark_job_id = self.benchmark_job_ids.get(job_id, job_id)
            required_vram_gib = benchmark.get(benchmark_job_id)
            placement = (
                None
                if required_vram_gib is None
                else select_vram_placement(
                    required_vram_gib,
                    resources,
                    model_id=model_id,
                    config_policy=self.config_policy,
                )
            )
            if placement is None:
                actions.append({"job_id": job_id, "type": "defer"})
                continue
            target, tp, pp, dp = placement
            reserve_vram_placement(target, tp * pp * dp)
            actions.append(
                _place_action(
                    job_id,
                    waiting,
                    target,
                    tp,
                    pp,
                    dp,
                    snapshot.model_catalogs,
                    self.config_policy,
                )
            )
        for paused in snapshot.paused_jobs:
            actions.append({"job_id": paused["job_id"], "type": "defer"})
        _validate_complete_job_actions(snapshot, actions)
        return actions

    def _waiting_order(self, job: Mapping[str, object]) -> tuple[int, str]:
        """Return deterministic EDF order for one waiting Store job."""
        job_id = _text(job.get("job_id"), "waiting Store job_id")
        try:
            return self.end_times_ns[job_id], job_id
        except KeyError as error:
            raise ValueError(f"scenario end time missing for {job_id!r}") from error


def _end_time_ns(job: OnlineJob | BatchJob) -> int:
    """Return the benchmark end used for EDF priority."""
    return job.stops_at_ns if isinstance(job, OnlineJob) else job.deadline_at_ns


def _tick_rationale(
    tick_number: int,
    actions: Sequence[Mapping[str, object]],
) -> str:
    """Summarize the deterministic EDF action set persisted for one tick."""
    counts = {"place": 0, "defer": 0, "keep": 0, "preempt": 0}
    for action in actions:
        action_type = action.get("type")
        if action_type in counts:
            counts[action_type] += 1
    return (
        f"EDF cluster simulator tick {tick_number}: "
        f"place={counts['place']}, defer={counts['defer']}, "
        f"keep={counts['keep']}, preempt={counts['preempt']}."
    )
