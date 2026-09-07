"""Run preferred best-fit placement with a VRAM fallback."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from models import CreatedPlan, PlannerResult, Scenario
from policy import ConfigPolicy, load_config_policy
from rank_ids import rank_id
from vram import (
    VRAM_REQUIREMENTS_GIB,
    ResourceTarget,
    reserve_vram_placement,
    select_vram_placement,
)

BEST_FIT_CONFIGURATIONS: dict[
    str,
    dict[str, dict[str, object]],
] = {
    "15min_real": {
        "job-benchmark_timing_15min_real-0-1": {
            "hardware": "RTXPRO6000",
            "configuration": {"tp": 4, "pp": 2, "dp": 1},
        },
        "job-benchmark_timing_15min_real-0-2": {
            "hardware": "L4",
            "configuration": {"tp": 2, "pp": 2, "dp": 1},
        },
    },
    "simulation_12h_homogenous": {
        "job-benchmark_timing_12hr_homogenous-0-1": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr_homogenous-0-2": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr_homogenous-0-3": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr_homogenous-0-4": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr_homogenous-0-5": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr_homogenous-0-6": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr_homogenous-10-1": {
            "hardware": "H100",
            "configuration": {"tp": 8, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr_homogenous-10-2": {
            "hardware": "H100",
            "configuration": {"tp": 4, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_12hr_homogenous-10-3": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 2, "dp": 4},
        },
        "job-benchmark_timing_12hr_homogenous-25-1": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr_homogenous-25-2": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr_homogenous-25-3": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr_homogenous-25-4": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr_homogenous-45-2": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 2, "dp": 8},
        },
        "job-benchmark_timing_12hr_homogenous-90-1": {
            "hardware": "H100",
            "configuration": {"tp": 8, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr_homogenous-90-2": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 3, "dp": 2},
        },
        "job-benchmark_timing_12hr_homogenous-135-1": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr_homogenous-135-2": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr_homogenous-180-1": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_12hr_homogenous-180-2": {
            "hardware": "H100",
            "configuration": {"tp": 4, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr_homogenous-240-1": {
            "hardware": "H100",
            "configuration": {"tp": 4, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_12hr_homogenous-240-2": {
            "hardware": "H100",
            "configuration": {"tp": 4, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_12hr_homogenous-300-1": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 2, "dp": 8},
        },
        "job-benchmark_timing_12hr_homogenous-330-1": {
            "hardware": "H100",
            "configuration": {"tp": 4, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr_homogenous-375-1": {
            "hardware": "H100",
            "configuration": {"tp": 8, "pp": 2, "dp": 4},
        },
        "job-benchmark_timing_12hr_homogenous-420-1": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 40},
        },
        "job-benchmark_timing_12hr_homogenous-420-2": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_12hr_homogenous-465-1": {
            "hardware": "H100",
            "configuration": {"tp": 4, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_12hr_homogenous-480-1": {
            "hardware": "H100",
            "configuration": {"tp": 8, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr_homogenous-510-1": {
            "hardware": "H100",
            "configuration": {"tp": 8, "pp": 2, "dp": 3},
        },
        "job-benchmark_timing_12hr_homogenous-510-2": {
            "hardware": "H100",
            "configuration": {"tp": 4, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr_homogenous-570-1": {
            "hardware": "H100",
            "configuration": {"tp": 4, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr_homogenous-570-2": {
            "hardware": "H100",
            "configuration": {"tp": 4, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr_homogenous-630-1": {
            "hardware": "H100",
            "configuration": {"tp": 4, "pp": 1, "dp": 12},
        },
        "job-benchmark_timing_12hr_homogenous-630-2": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_12hr_homogenous-690-1": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_12hr_homogenous-690-2": {
            "hardware": "H100",
            "configuration": {"tp": 8, "pp": 2, "dp": 4},
        },
    },
    "real_12h": {
        "job-benchmark_timing_real_12hr_192gpu-0-1": {
            "hardware": "L4",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-0-2": {
            "hardware": "L4",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-0-3": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_12hr_192gpu-0-4": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-0-5": {
            "hardware": "L4",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-0-6": {
            "hardware": "L4",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_12hr_192gpu-45-3": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_12hr_192gpu-150-2": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-210-1": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_12hr_192gpu-210-2": {
            "hardware": "H100",
            "configuration": {"tp": 4, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_12hr_192gpu-360-1": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_real_12hr_192gpu-360-2": {
            "hardware": "L4",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_real_12hr_192gpu-420-1": {
            "hardware": "RTXPRO6000",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-480-1": {
            "hardware": "RTXPRO6000",
            "configuration": {"tp": 4, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-480-2": {
            "hardware": "RTXPRO6000",
            "configuration": {"tp": 4, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_real_12hr_192gpu-690-1": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
    },
    "simulation_12h": {
        "job-benchmark_timing_12hr-0-1": {
            "hardware": "L40S",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-0-2": {
            "hardware": "L40S",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-0-3": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-0-4": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-0-5": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-0-6": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-10-1": {
            "hardware": "MI300",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-10-2": {
            "hardware": "MI300",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr-10-3": {
            "hardware": "L40S",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-25-1": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-25-2": {
            "hardware": "L40S",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-25-3": {
            "hardware": "L40S",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr-25-4": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr-45-2": {
            "hardware": "L40S",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr-90-1": {
            "hardware": "MI300",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-90-2": {
            "hardware": "L40S",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr-135-1": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr-135-2": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_12hr-180-1": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_12hr-180-2": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-240-1": {
            "hardware": "MI300",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-240-2": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_12hr-300-1": {
            "hardware": "L40S",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr-330-1": {
            "hardware": "MI300",
            "configuration": {"tp": 2, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr-375-1": {
            "hardware": "MI300",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr-420-1": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 40},
        },
        "job-benchmark_timing_12hr-420-2": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_12hr-465-1": {
            "hardware": "MI300",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-480-1": {
            "hardware": "MI300",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-510-1": {
            "hardware": "MI300",
            "configuration": {"tp": 8, "pp": 2, "dp": 3},
        },
        "job-benchmark_timing_12hr-510-2": {
            "hardware": "MI300",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_12hr-570-1": {
            "hardware": "MI300",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_12hr-570-2": {
            "hardware": "MI300",
            "configuration": {"tp": 2, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_12hr-630-1": {
            "hardware": "MI300",
            "configuration": {"tp": 2, "pp": 1, "dp": 24},
        },
        "job-benchmark_timing_12hr-630-2": {
            "hardware": "MI300",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_12hr-690-1": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_12hr-690-2": {
            "hardware": "MI300",
            "configuration": {"tp": 8, "pp": 1, "dp": 6},
        },
    },
    "simulation_3h": {
        "job-benchmark_timing_3hr-0-1": {
            "hardware": "A10G",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_3hr-0-2": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_3hr-0-3": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_3hr-0-4": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_3hr-0-5": {
            "hardware": "L40S",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_3hr-7-1": {
            "hardware": "A100-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_3hr-7-2": {
            "hardware": "L40S",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_3hr-7-3": {
            "hardware": "A100-80GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_3hr-15-1": {
            "hardware": "A100-40GB",
            "configuration": {"tp": 4, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_3hr-15-3": {
            "hardware": "A100-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_3hr-32-1": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 16},
        },
        "job-benchmark_timing_3hr-32-2": {
            "hardware": "A100-80GB",
            "configuration": {"tp": 1, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_3hr-49-1": {
            "hardware": "L40S",
            "configuration": {"tp": 2, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_3hr-49-2": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 8},
        },
        "job-benchmark_timing_3hr-67-1": {
            "hardware": "MI300",
            "configuration": {"tp": 8, "pp": 2, "dp": 1},
        },
        "job-benchmark_timing_3hr-67-2": {
            "hardware": "MI300",
            "configuration": {"tp": 2, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_3hr-67-3": {
            "hardware": "A100-80GB",
            "configuration": {"tp": 4, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_3hr-85-1": {
            "hardware": "A100-40GB",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_3hr-100-1": {
            "hardware": "MI300",
            "configuration": {"tp": 1, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_3hr-100-2": {
            "hardware": "H100",
            "configuration": {"tp": 1, "pp": 1, "dp": 2},
        },
        "job-benchmark_timing_3hr-125-1": {
            "hardware": "H100",
            "configuration": {"tp": 4, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_3hr-125-2": {
            "hardware": "L40S",
            "configuration": {"tp": 4, "pp": 1, "dp": 4},
        },
        "job-benchmark_timing_3hr-145-1": {
            "hardware": "H100",
            "configuration": {"tp": 2, "pp": 1, "dp": 1},
        },
        "job-benchmark_timing_3hr-145-2": {
            "hardware": "H100",
            "configuration": {"tp": 8, "pp": 2, "dp": 1},
        },
        "job-benchmark_timing_3hr-160-1": {
            "hardware": "MI300",
            "configuration": {"tp": 2, "pp": 1, "dp": 8},
        },
    },
}


_HARDWARE_ALIASES = {
    "h100": "h100",
    "h100-sxm": "h100",
    "h100-sxm-80gb": "h100",
    "a100": "a100-80gb",
    "a100-80gb": "a100-80gb",
    "a100-sxm": "a100-80gb",
    "a100-sxm-80gb": "a100-80gb",
    "a100-40gb": "a100-40gb",
    "a100-pcie": "a100-40gb",
    "a100-pcie-40gb": "a100-40gb",
    "a10g": "a10g",
    "a10g-24gb": "a10g",
    "l4": "l4",
    "l4-24gb": "l4",
    "l40s": "l40s",
    "l40s-48gb": "l40s",
    # ponytail: local GCP catalog alias; centralize GPU identity if aliases expand.
    "rtxpro6000": "rtxpro6000",
    "rtx-pro-6000": "rtxpro6000",
    "rtx-pro-6000-bse": "rtxpro6000",
    "mi300": "mi300",
    "mi300x": "mi300",
    "mi300x-192gb": "mi300",
}

LAUNCH_FIELDS = (
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


@dataclass(frozen=True)
class StoreSnapshot:
    """One frozen Store view used throughout a best-fit control tick."""

    resources: dict[str, dict[str, Any]]
    running_jobs: tuple[dict[str, Any], ...]
    waiting_jobs: tuple[dict[str, Any], ...]
    paused_jobs: tuple[dict[str, Any], ...]
    finished_jobs: tuple[dict[str, Any], ...]
    model_catalogs: dict[str, dict[str, Any]]


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


class ModelCatalogStoreClient(Protocol):
    """Read one model catalog by canonical model ID."""

    def get(self, model_id: str) -> object | None: ...


class PlanStoreClient(Protocol):
    """Persist one canonical Store plan."""

    def create(self, plan: object) -> object: ...


class BestFitPlannerDriver:
    """Prefer benchmark placements and fall back to memory best-fit."""

    method_name = "bestfit"

    def __init__(
        self,
        user_id: str,
        scenario: Scenario,
        job_store: JobStoreClient,
        resource_map_store: ResourceMapStoreClient,
        model_catalog_store: ModelCatalogStoreClient,
        plan_store: PlanStoreClient,
        backend: str,
    ) -> None:
        """Keep Store clients and benchmark-to-Store job identity."""
        self.user_id = user_id
        self.benchmark_name = scenario.benchmark_name
        jobs = (*scenario.online_jobs, *scenario.batch_jobs)
        self.benchmark_job_ids = {job.store_job_id: job.name for job in jobs}
        self.job_store = job_store
        self.resource_map_store = resource_map_store
        self.model_catalog_store = model_catalog_store
        self.plan_store = plan_store
        self.backend = backend

    @classmethod
    def build(
        cls,
        user_id: str,
        scenario: Scenario,
        settings: Mapping[str, object],
        postgres_client: object | None = None,
    ) -> BestFitPlannerDriver:
        """Build best-fit with Tandemn Store public clients."""
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
            paused_jobs=(),
            finished_jobs=(),
            model_catalogs=model_catalogs,
        )

    def _build_actions(self, snapshot: StoreSnapshot) -> list[dict[str, object]]:
        """Place greedily using preferred mappings, then VRAM fallbacks."""
        benchmark = BEST_FIT_CONFIGURATIONS.get(self.benchmark_name, {})
        vram_requirements = VRAM_REQUIREMENTS_GIB.get(self.benchmark_name, {})
        config_policy = load_config_policy()
        resources = deepcopy(snapshot.resources)
        actions: list[dict[str, object]] = []
        for running in snapshot.running_jobs:
            actions.append({"job_id": running["job_id"], "type": "keep"})
        for waiting in snapshot.waiting_jobs:
            job_id = _text(waiting.get("job_id"), "waiting Store job_id")
            benchmark_job_id = self.benchmark_job_ids.get(job_id, job_id)
            mapped = benchmark.get(benchmark_job_id)
            placement = None if mapped is None else _mapped_placement(mapped, resources)
            used_vram_fallback = False
            if placement is None:
                required_vram_gib = vram_requirements.get(benchmark_job_id)
                if required_vram_gib is not None:
                    placement = select_vram_placement(
                        required_vram_gib,
                        resources,
                        model_id=_text(
                            _known_job_fields(waiting).get("model_id"), "model_id"
                        ),
                        config_policy=config_policy,
                    )
                    used_vram_fallback = placement is not None
            if placement is None:
                actions.append({"job_id": job_id, "type": "defer"})
                continue
            target, tp, pp, dp = placement
            if used_vram_fallback:
                reserve_vram_placement(target, tp * pp * dp)
            else:
                _reserve(target, tp * pp, dp)
            actions.append(
                _place_action(
                    job_id,
                    waiting,
                    target,
                    tp,
                    pp,
                    dp,
                    snapshot.model_catalogs,
                    config_policy,
                )
            )
        _validate_complete_job_actions(snapshot, actions)
        return actions


def _mapped_placement(
    mapped: Mapping[str, object],
    resources: dict[str, dict[str, Any]],
) -> tuple[ResourceTarget, int, int, int] | None:
    """Resolve one mapped hardware layout against current Store capacity."""
    hardware = _text(mapped.get("hardware"), "mapped hardware")
    configuration = _mapping(mapped.get("configuration"), "mapped configuration")
    tp = _positive_int(configuration.get("tp"), "tp")
    pp = _positive_int(configuration.get("pp"), "pp")
    dp = _positive_int(configuration.get("dp"), "dp")
    targets = _matching_targets(resources, hardware)
    feasible = []
    for target in targets:
        footprint = _target_footprint(target, tp * pp, dp)
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
    if not feasible:
        return None
    return min(feasible, key=lambda value: value[:3])[3], tp, pp, dp


def _matching_targets(
    resources: dict[str, dict[str, Any]],
    hardware: str,
) -> tuple[ResourceTarget, ...]:
    """Return deterministic concrete targets for one mapped accelerator."""
    expected = _canonical_hardware(hardware)
    targets = []
    for environment_key, environment in sorted(resources.items()):
        gpu_type = str(environment.get("gpu_type") or environment_key.split("|")[-1])
        if _canonical_hardware(gpu_type) != expected:
            continue
        pools = environment.get("pools", [])
        if not isinstance(pools, list):
            raise ValueError("resource pools must be a list")
        for pool in pools or [None]:
            if pool is not None and not isinstance(pool, dict):
                raise ValueError("resource pool must be an object")
            targets.append(
                ResourceTarget(
                    environment_key=environment_key,
                    environment=environment,
                    pool=pool,
                    gpu_type=gpu_type,
                    instance_type=(
                        None
                        if pool is None
                        else _optional_text(pool.get("instance_type"))
                    ),
                )
            )
    return tuple(targets)


def _place_action(
    job_id: str,
    store_job: Mapping[str, object],
    target: ResourceTarget,
    tp: int,
    pp: int,
    dp: int,
    model_catalogs: Mapping[str, Mapping[str, Any]],
    config_policy: ConfigPolicy,
) -> dict[str, object]:
    """Combine deterministic Store fields with one mapped placement."""
    known = _known_job_fields(store_job)
    model_id = _text(known.get("model_id"), "model_id")
    try:
        catalog = model_catalogs[model_id]
    except KeyError as error:
        raise ValueError(f"model catalog missing {model_id!r}") from error
    launch_config = _materialize_launch_config(catalog, target.gpu_type)
    rule = config_policy.rule_for(target.gpu_type, model_id)
    if rule is None or not rule.allows_parallelism(tp, pp):
        raise ValueError("BestFit selected a policy-invalid placement")
    per_replica_gpu_count = tp * pp
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
                "env": target.environment_key.split("|"),
                "config": {
                    **launch_config,
                    "model_id": model_id,
                    "precision": rule.precision,
                    "instance_type": target.instance_type,
                    "gpu_type": target.gpu_type,
                    "gpu_count": per_replica_gpu_count,
                    "count": per_replica_gpu_count,
                    "tp": tp,
                    "pp": pp,
                    "dp": dp,
                    "ep": known.get("ep") or 1,
                    "rank_traffic_share": 1.0,
                },
                "n_replicas": dp,
                "mechanism_id": known.get("mechanism_id"),
            }
        ],
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
    launch_config = {}
    for field in LAUNCH_FIELDS:
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


def _target_available(target: ResourceTarget) -> int:
    """Return current capacity for one concrete target."""
    available = int(target.environment.get("free", 0))
    if target.pool is not None:
        available = min(available, int(target.pool.get("free", 0)))
    return max(0, available)


def _target_footprint(
    target: ResourceTarget,
    gpu_count: int,
    replicas: int = 1,
) -> int | None:
    """Translate mapped GPUs into the target's allocation unit."""
    if target.pool is None:
        return gpu_count * replicas
    allocation_kind = str(
        target.pool.get("allocation_kind")
        or target.pool.get("allocation_unit")
        or "instance"
    ).lower()
    if allocation_kind == "gpu":
        return gpu_count * replicas
    gpus_per_instance = _positive_int(
        target.pool.get("gpus_per_instance"),
        "gpus_per_instance",
    )
    if gpu_count > gpus_per_instance:
        return None
    return gpus_per_instance * replicas


def _reserve(target: ResourceTarget, gpu_count: int, replicas: int = 1) -> None:
    """Reserve a mapped placement before evaluating the next waiting job."""
    footprint = _target_footprint(target, gpu_count, replicas)
    if footprint is None or footprint > _target_available(target):
        raise RuntimeError("selected best-fit placement no longer fits")
    target.environment["free"] = int(target.environment["free"]) - footprint
    if target.pool is not None:
        target.pool["free"] = int(target.pool["free"]) - footprint


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
            footprint = count * replicas
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


def _known_job_fields(store_job: Mapping[str, object]) -> dict[str, object]:
    """Flatten scheduler-visible fields from one Store job."""
    spec_value = store_job.get("spec_json")
    spec = spec_value if isinstance(spec_value, Mapping) else {}
    features_value = spec.get("job_features")
    features = features_value if isinstance(features_value, Mapping) else {}
    known: dict[str, object] = {}
    for source in (store_job, spec, features):
        for key, value in source.items():
            if value is not None:
                known.setdefault(str(key), value)
    return known


def _target_tps(known: Mapping[str, object]) -> float | None:
    """Derive output-token throughput from canonical Store job fields."""
    output_tokens = _number(known.get("osl_token_avg"))
    if output_tokens is None:
        return None
    kind = getattr(known.get("kind"), "value", known.get("kind"))
    try:
        if kind == "batch":
            token_budget = _positive_number(
                known.get("total_token_budget"),
                "total_token_budget",
            )
            deadline_hours = _positive_number(
                known.get("deadline_hrs"),
                "deadline_hrs",
            )
            input_tokens = _positive_number(
                known.get("isl_token_avg"),
                "isl_token_avg",
            )
            request_rate = token_budget / (
                (input_tokens + output_tokens) * deadline_hours * 3600
            )
        else:
            request_rate = _positive_number(
                known.get("request_arrival_rate"),
                "request_arrival_rate",
            )
    except ValueError:
        return None
    return round(request_rate * output_tokens, 3)


def _validate_complete_job_actions(
    snapshot: StoreSnapshot,
    actions: Sequence[Mapping[str, object]],
) -> None:
    """Require exactly one canonical action for every Store job."""
    store_ids: list[str] = []
    for jobs in (
        snapshot.running_jobs,
        snapshot.waiting_jobs,
    ):
        store_ids.extend(_text(job.get("job_id"), "Store job_id") for job in jobs)
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
        f"Best-fit cluster simulator tick {tick_number}: "
        f"place={counts['place']}, defer={counts['defer']}, "
        f"keep={counts['keep']}, preempt={counts['preempt']}."
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
        return next(
            (
                pool
                for pool in typed_pools
                if pool.get("instance_type") == instance_type
            ),
            None,
        )
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


def _canonical_hardware(value: str) -> str:
    """Normalize benchmark and Store accelerator labels."""
    key = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return _HARDWARE_ALIASES.get(key, key)


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
