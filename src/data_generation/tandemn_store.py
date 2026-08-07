"""Method-neutral adapter for reading current state from Tandemn Store."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol


class TandemnStoreError(RuntimeError):
    """Report an unavailable dependency or invalid Store state."""


@dataclass(frozen=True)
class ClusterStateSnapshot:
    """Method-neutral resource and job state for one user."""

    user_id: str
    resource_map_version: int
    resource_map_updated_at: str
    resources: Mapping[str, Mapping[str, object]]
    running_jobs: tuple[Mapping[str, object], ...]
    waiting_jobs: tuple[Mapping[str, object], ...]
    paused_jobs: tuple[Mapping[str, object], ...]
    finished_jobs: tuple[Mapping[str, object], ...]

    def jobs_by_id(self) -> dict[str, Mapping[str, object]]:
        """Index every Store job by its canonical job_id."""
        indexed: dict[str, Mapping[str, object]] = {}
        for jobs in (
            self.running_jobs,
            self.waiting_jobs,
            self.paused_jobs,
            self.finished_jobs,
        ):
            for job in jobs:
                job_id = job.get("job_id")
                if not isinstance(job_id, str) or not job_id:
                    raise TandemnStoreError("Store job is missing job_id")
                if job_id in indexed:
                    raise TandemnStoreError(f"duplicate Store job_id {job_id!r}")
                indexed[job_id] = job
        return indexed

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation for any scheduling method."""
        return {
            "user_id": self.user_id,
            "resource_map": {
                "version": self.resource_map_version,
                "updated_at": self.resource_map_updated_at,
                "environments": dict(self.resources),
            },
            "jobs": {
                "by_id": self.jobs_by_id(),
                "running": list(self.running_jobs),
                "waiting": list(self.waiting_jobs),
                "paused": list(self.paused_jobs),
                "finished": list(self.finished_jobs),
            },
        }


class ClusterStateProvider(Protocol):
    """Shared state boundary that scheduling methods may depend on."""

    def read_cluster_state(self) -> ClusterStateSnapshot:
        """Return the current resource map and job lifecycle state."""


class PlanPublisher(Protocol):
    """Publish one state-legal plan for Orca to apply."""

    def publish_plan(
        self,
        plan: Mapping[str, object],
        method_name: str,
    ) -> str:
        """Persist a plan and return its canonical plan ID."""


class TandemnStoreAdapter:
    """Read current cluster state through Tandemn Store clients."""

    def __init__(
        self,
        user_id: str,
        postgres_client: object | None = None,
        *,
        job_user_ids: Iterable[str] | None = None,
    ) -> None:
        if not user_id:
            raise ValueError("user_id is required for Tandemn Store state")
        try:
            clients = import_module("tandemn_system_data.clients")
            models = import_module("tandemn_system_data.models")
        except ImportError as error:
            raise TandemnStoreError(
                "tandemn-store is required to read cluster state"
            ) from error

        client = postgres_client or clients.PostgresClient()
        self.user_id = user_id
        requested_user_ids = tuple(dict.fromkeys((user_id, *(job_user_ids or ()))))
        self._job_user_ids = requested_user_ids
        self._job_store = clients.JobStore(client)
        self._resource_map_store = clients.ResourceMapStore(
            client,
            user_id=user_id,
        )
        self._finished_status = models.JobStatus.FINISHED

    def read_cluster_state(self) -> ClusterStateSnapshot:
        """Read resources and every canonical job lifecycle group."""
        resource_map = self._resource_map_store.get()
        running_jobs = tuple(
            job
            for user_id in self._job_user_ids
            for job in self._job_store.running_jobs(user_id)
        )
        waiting_jobs = tuple(
            job
            for user_id in self._job_user_ids
            for job in self._job_store.waiting_jobs(user_id)
        )
        paused_jobs = tuple(
            job
            for user_id in self._job_user_ids
            for job in self._job_store.paused_jobs(user_id)
        )
        finished_jobs = tuple(
            job
            for user_id in self._job_user_ids
            for job in self._finished_jobs(user_id)
        )
        resources = _capacity_summary(resource_map, running_jobs)

        return ClusterStateSnapshot(
            user_id=self.user_id,
            resource_map_version=int(resource_map.version),
            resource_map_updated_at=str(resource_map.updated_at),
            resources=resources,
            running_jobs=tuple(_running_job_payload(job) for job in running_jobs),
            waiting_jobs=tuple(_model_payload(job) for job in waiting_jobs),
            paused_jobs=tuple(_model_payload(job) for job in paused_jobs),
            finished_jobs=tuple(_model_payload(job) for job in finished_jobs),
        )

    def _finished_jobs(self, user_id: str) -> list[object]:
        """Read terminal jobs while Store lacks a public finished_jobs method."""
        finished_jobs = getattr(self._job_store, "finished_jobs", None)
        if callable(finished_jobs):
            return list(finished_jobs(user_id))

        status_reader = getattr(self._job_store, "_jobs_with_status", None)
        if not callable(status_reader):
            raise TandemnStoreError(
                "Tandemn JobStore does not expose a finished-job read"
            )
        return list(status_reader(user_id, self._finished_status))


class TandemnPlanPublisher:
    """Write validated controller plans through Tandemn PlanStore."""

    def __init__(self, user_id: str, postgres_client: object | None = None) -> None:
        if not user_id:
            raise ValueError("user_id is required for Tandemn plan publication")
        try:
            clients = import_module("tandemn_system_data.clients")
            models = import_module("tandemn_system_data.models")
        except ImportError as error:
            raise TandemnStoreError(
                "tandemn-store is required to publish cluster plans"
            ) from error

        client = postgres_client or clients.PostgresClient()
        self.user_id = user_id
        self._models = models
        self._plan_store = clients.PlanStore(client)

    def publish_plan(
        self,
        plan: Mapping[str, object],
        method_name: str,
    ) -> str:
        """Persist one canonical Store plan for Orca."""
        actions_value = plan.get("actions")
        if not isinstance(actions_value, list):
            raise TandemnStoreError("plan actions must be a list")
        actions = []
        for value in actions_value:
            action = _mapping(value, "plan action")
            ladder = action.get("ladder")
            actions.append(
                self._models.PlanAction(
                    job_id=action.get("job_id"),
                    type=action.get("type"),
                    ladder=ladder or None,
                    target_tps=action.get("target_tps"),
                    target_p99_ttft_ms=action.get("target_p99_ttft_ms"),
                    target_p99_tpot_ms=action.get("target_p99_tpot_ms"),
                )
            )
        store_plan = self._models.Plan(
            user_id=self.user_id,
            koi_version=method_name,
            tick_rationale=str(plan.get("tick_rationale") or ""),
            actions=actions,
        )
        self._plan_store.create(store_plan)
        return str(store_plan.plan_id)


def _capacity_summary(
    resource_map: object,
    running_jobs: tuple[object, ...],
) -> dict[str, Mapping[str, object]]:
    """Return total, used, and free GPU capacity for every environment."""
    scheduling_summary = getattr(resource_map, "scheduling_summary", None)
    if not callable(scheduling_summary):
        raise TandemnStoreError("ResourceMap has no scheduling_summary method")
    raw_summary = scheduling_summary()
    if not isinstance(raw_summary, Mapping):
        raise TandemnStoreError("ResourceMap scheduling summary must be a mapping")

    default_market = _default_market(resource_map)
    used_gpus: dict[str, int] = {}
    used_instances: dict[tuple[str, str], int] = {}
    used_pool_gpus: dict[tuple[str, str], int] = {}
    normalized = {
        _normalize_env_key(str(env_key), default_market): _mapping(
            info,
            "resource",
        )
        for env_key, info in raw_summary.items()
    }

    for running_job in running_jobs:
        raw_job = _model_payload(running_job)
        for chain_value in _sequence(raw_job.get("chains"), "running job chains"):
            chain = _mapping(chain_value, "active chain")
            shape = _mapping(chain.get("shape_json", {}), "chain shape_json")
            raw_env = (
                chain.get("target_node") or shape.get("env") or shape.get("pool_id")
            )
            if raw_env is None:
                continue
            env_key = _normalize_env_key(_env_key(raw_env), default_market)
            count = shape.get("count")
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise TandemnStoreError(
                    f"chain {chain.get('chain_id')} has invalid GPU count {count!r}"
                )
            environment = normalized.get(env_key)
            if environment is None:
                continue
            pool = _select_pool(environment, shape)
            if pool is None:
                used_gpus[env_key] = used_gpus.get(env_key, 0) + count
                continue

            instance_type = str(pool.get("instance_type"))
            allocation_kind = str(
                pool.get("allocation_kind") or pool.get("allocation_unit") or "instance"
            ).lower()
            key = (env_key, instance_type)
            if allocation_kind == "gpu":
                used_pool_gpus[key] = used_pool_gpus.get(key, 0) + count
                used_gpus[env_key] = used_gpus.get(env_key, 0) + count
            else:
                used_instances[key] = used_instances.get(key, 0) + 1
                footprint = _integer(pool.get("gpus_per_instance"), default=1)
                used_gpus[env_key] = used_gpus.get(env_key, 0) + footprint

    result: dict[str, Mapping[str, object]] = {}
    for env_key, raw_environment in normalized.items():
        environment = dict(raw_environment)
        pools: list[dict[str, object]] = []
        for pool_value in _sequence(environment.get("pools", []), "resource pools"):
            pool = dict(_mapping(pool_value, "resource pool"))
            instance_type = str(pool.get("instance_type"))
            allocation_kind = str(
                pool.get("allocation_kind") or pool.get("allocation_unit") or "instance"
            ).lower()
            total_gpus = _integer(pool.get("total"))
            if allocation_kind == "gpu":
                used = used_pool_gpus.get((env_key, instance_type), 0)
                free_instances = _integer(pool.get("total_instances"))
            else:
                total_instances = _integer(pool.get("total_instances"))
                used_count = used_instances.get((env_key, instance_type), 0)
                free_instances = max(0, total_instances - used_count)
                used = min(
                    total_gpus,
                    used_count * _integer(pool.get("gpus_per_instance"), default=1),
                )
            pool["used"] = used
            pool["free"] = max(0, total_gpus - used)
            pool["free_instances"] = free_instances
            pools.append(pool)

        total = _integer(environment.get("total"))
        used = min(total, used_gpus.get(env_key, 0))
        environment["used"] = used
        environment["free"] = max(0, total - used)
        environment["pools"] = pools
        result[env_key] = environment
    return result


def _running_job_payload(running_job: object) -> Mapping[str, object]:
    raw = _model_payload(running_job)
    job = dict(_mapping(raw.get("job"), "running job"))
    job["active_chains"] = list(_sequence(raw.get("chains", []), "running job chains"))
    return job


def _model_payload(value: object) -> dict[str, object]:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        raw = dump(mode="json")
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise TandemnStoreError(
            f"Store value {type(value).__name__} is not serializable"
        )
    return _mapping(raw, "Store model")


def _select_pool(
    environment: Mapping[str, object],
    shape: Mapping[str, object],
) -> Mapping[str, object] | None:
    pools = [
        _mapping(pool, "resource pool")
        for pool in _sequence(environment.get("pools", []), "resource pools")
    ]
    if not pools:
        return None
    instance_type = shape.get("instance_type")
    if instance_type is not None:
        selected = next(
            (pool for pool in pools if pool.get("instance_type") == instance_type),
            None,
        )
        if selected is None:
            raise TandemnStoreError(
                f"active chain references unknown instance type {instance_type!r}"
            )
        return selected
    if len(pools) == 1:
        return pools[0]
    raise TandemnStoreError(
        "active chain must identify its instance type when an environment has "
        "multiple pools"
    )


def _default_market(resource_map: object) -> str:
    markets = getattr(resource_map, "market", None)
    if isinstance(markets, (list, tuple)) and markets:
        return str(markets[0])
    if markets:
        return str(markets)
    return "reserved"


def _normalize_env_key(env_key: str, default_market: str) -> str:
    parts = env_key.split("|")
    if len(parts) == 4:
        return "|".join((default_market, *parts))
    return env_key


def _env_key(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "|".join(str(part) for part in value)
    return str(value)


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TandemnStoreError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise TandemnStoreError(f"{name} must be a list")
    return tuple(value)


def _integer(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise TandemnStoreError(f"resource capacity must be an integer: {value!r}")
    return value
