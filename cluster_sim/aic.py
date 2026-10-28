"""Run NVIDIA AIC placement through the cluster simulator planner boundary.

Store owns job identity and lifecycle. AIC evaluates each homogeneous hardware
system independently and owns hardware, replica, and parallelism selection.
All scheduler-visible job fields come from Store; unavailable fields are null.
"""

from __future__ import annotations

import logging
import multiprocessing
import re
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from io import StringIO
from pathlib import Path
from queue import Empty
from time import monotonic
from typing import Any, Protocol

from models import CreatedPlan, PlannerResult, Scenario
from policy import VALID_PP_DEGREES, load_config_policy, simulation_quantization
from rank_ids import rank_id

LOGGER = logging.getLogger(__name__)
_AIC_CALL_TIMEOUT_SECONDS = 60.0
_AIC_TICK_TIMEOUT_SECONDS = 280.0
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
    "a10g": "l4",
    "a10g-24gb": "l4",
    "l40s": "l40s",
    "l40s-48gb": "l40s",
    # ponytail: local GCP catalog alias; centralize GPU identity if aliases expand.
    "rtxpro6000": "rtx_pro_6000_server",
    "rtx-pro-6000": "rtx_pro_6000_server",
    "rtx-pro-6000-bse": "rtx_pro_6000_server",
}


class AicEvaluationError(RuntimeError):
    """Report an AIC evaluation failure for one hardware candidate."""


class AicUnavailableError(AicEvaluationError):
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
    call_timeout_seconds: float
    tick_timeout_seconds: float
    model_configs_root: Path | None = None


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
    parallelism_candidates: tuple[tuple[int, int], ...]
    model_configs_root: Path | None = None
    precision: str | None = None
    chunked_prefill_enable: bool = True


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
    num_experts: int = 0


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
    ep: int
    max_num_seq: int
    devices: int
    ttft_ms: float
    tpot_ms: float
    request_latency_ms: float
    throughput_tokens_per_second: float
    prediction: Mapping[str, object] | None = None


class AicClient(Protocol):
    """Evaluate one job against one homogeneous AIC hardware system."""

    def evaluate(
        self, request: AicRequest, timeout_seconds: float | None = None
    ) -> AicResult:
        """Return AIC's best aggregate-serving configuration."""


class JobStoreClient(Protocol):
    """Read canonical lifecycle groups from Tandemn Store."""

    def running_jobs(self, user_id: str) -> Sequence[object]: ...

    def waiting_jobs(self, user_id: str) -> Sequence[object]: ...

    def paused_jobs(self, user_id: str) -> Sequence[object]: ...


class ModelCatalogStoreClient(Protocol):
    """Read seeded model launch defaults from Tandemn Store."""

    def get(self, model_id: str) -> object | None: ...


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

    def evaluate(
        self, request: AicRequest, timeout_seconds: float | None = None
    ) -> AicResult:
        """Run one AIC recommendation in a process with a hard deadline."""
        timeout = (
            _AIC_CALL_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        if timeout <= 0:
            raise AicEvaluationError("AIC tick deadline expired")
        context = multiprocessing.get_context("spawn")
        results = context.Queue(maxsize=1)
        process = context.Process(
            target=_run_aic_request,
            args=(request, results, monotonic() + timeout),
        )
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            raise AicEvaluationError(f"AIC timed out after {timeout:.1f}s")
        try:
            status, value = results.get(timeout=1)
        except Empty as error:
            raise AicEvaluationError("AIC exited without a result") from error
        finally:
            results.close()
        if status == "ok":
            return value
        if status == "unavailable":
            raise AicUnavailableError(value)
        raise AicEvaluationError(value)


def _run_aic_request(
    request: AicRequest, results: Any, deadline: float | None = None
) -> None:
    """Evaluate one AIC request in a killable child process."""
    try:
        results.put(("ok", _evaluate_aic_request(request, deadline)))
    except AicUnavailableError as error:
        results.put(("unavailable", str(error)))
    except AicEvaluationError as error:
        results.put(("error", str(error)))
    except Exception as error:
        results.put(("error", str(error)))


def _evaluate_aic_request(
    request: AicRequest, deadline: float | None = None
) -> AicResult:
    """Prefer native AIC coverage, then use corrected same-system model proxies."""
    deadline = (
        deadline if deadline is not None else monotonic() + _AIC_CALL_TIMEOUT_SECONDS
    )
    if monotonic() >= deadline:
        raise AicEvaluationError("AIC tick deadline expired")
    try:
        frame, num_experts = _sweep_aic_request(request)
    except AicUnavailableError:
        raise
    except AicEvaluationError as error:
        return _evaluate_proxy_request(request, str(error), deadline)
    result = _select_aic_result(request, frame, num_experts)
    return replace(
        result,
        configuration={
            **result.configuration,
            "aic_prediction": {
                "mode": "native",
                "requested_model_id": request.model,
                "prediction_model_id": request.model,
                "backend": request.backend,
                "backend_version": result.configuration.get(
                    "version", request.backend_version
                ),
                "system": request.system,
                "database_mode": request.database_mode,
                "prefill_speed_ratio": 1.0,
                "decode_speed_ratio": 1.0,
            },
        },
    )


def _sweep_aic_request(request: AicRequest) -> tuple[Any, int]:
    """Return constrained per-worker estimates before load matching."""
    if request.serving_mode != "agg":
        raise AicEvaluationError("AIC requires aggregate serving")
    try:
        from aiconfigurator.sdk.common import ARCHITECTURE_TO_MODEL_FAMILY
        from aiconfigurator.sdk.task_v2 import Task
        from aiconfigurator.sdk.utils import get_model_config_from_model_path
    except ImportError as error:
        raise AicUnavailableError(
            "AIC is not installed; run `uv sync --group dynamo`"
        ) from error

    previous_logging_threshold = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            if request.model_configs_root is not None:
                from aic_proxy import load_model_profile

                profile = load_model_profile(request.model, request.model_configs_root)
                if profile.architecture not in ARCHITECTURE_TO_MODEL_FAMILY:
                    raise AicEvaluationError(
                        f"AIC architecture {profile.architecture} is not supported"
                    )
            model = get_model_config_from_model_path(request.model)
            num_experts = int(model.get("num_experts") or 0)
            pairs = [
                (tp, pp)
                for tp, pp in request.parallelism_candidates
                if tp * pp <= request.available_devices
                and pp <= model["layers"]
                and (not num_experts or num_experts % tp == 0)
            ]
            if not pairs:
                raise AicEvaluationError("AIC has no runtime-compatible layouts")
            tp_candidates = sorted({tp for tp, _ in pairs})
            task = Task(
                serving_mode=request.serving_mode,
                model_path=request.model,
                system_name=request.system,
                backend_name=request.backend,
                backend_version=request.backend_version,
                total_gpus=request.available_devices,
                database_mode=request.database_mode,
                isl=request.input_tokens,
                osl=request.output_tokens,
                ttft=request.target_ttft_ms,
                tpot=request.target_tpot_ms,
                enable_chunked_prefill=request.chunked_prefill_enable,
                agg_num_gpu_candidates=sorted({tp * pp for tp, pp in pairs}),
                agg_tp_candidates=tp_candidates,
                agg_pp_candidates=sorted({pp for _, pp in pairs}),
                agg_dp_candidates=[1],
                agg_cp_candidates=[1],
                agg_moe_tp_candidates=[1],
                agg_moe_ep_candidates=tp_candidates if num_experts else [1],
            )
            frame = task.run()
    except SystemExit as error:
        raise AicEvaluationError(str(error)) from error
    except Exception as error:
        raise AicEvaluationError(str(error)) from error
    finally:
        logging.disable(previous_logging_threshold)

    if frame is None or frame.empty:
        raise AicEvaluationError("AIC returned no aggregate configuration")
    return frame, num_experts


def _select_aic_result(request: AicRequest, frame: Any, num_experts: int) -> AicResult:
    """Apply target SLOs, load matching, capacity, and the baseline objective."""
    from aiconfigurator.sdk.picking import pick_load_match, pick_optimization_type

    try:
        supported = []
        for index, row in frame.iterrows():
            try:
                _runtime_parallelism(row.to_dict(), request, num_experts)
            except AicEvaluationError:
                continue
            supported.append(index)
        frame = frame.loc[supported]
        frame = frame[
            (frame["ttft"] <= request.target_ttft_ms)
            & (frame["tpot"] <= request.target_tpot_ms)
        ]
        if frame.empty:
            raise AicEvaluationError(
                "AIC returned no compatible configuration under SLA"
            )
        matched = pick_load_match(
            pareto_df=frame,
            serving_mode=request.serving_mode,
            target_tpot=request.target_tpot_ms,
            target_request_rate=request.target_request_rate,
            target_concurrency=request.target_concurrency,
            top_n=len(frame),
        )
        frame = matched["best_config_df"]
        frame = frame[frame["total_gpus_needed"] <= request.available_devices]
        if frame.empty:
            raise AicEvaluationError("AIC returned no configuration within capacity")
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
        num_experts=num_experts,
    )


def _evaluate_proxy_request(
    request: AicRequest, native_error: str, deadline: float
) -> AicResult:
    """Correct the first usable ranked proxy per layout before choosing a plan."""
    import pandas as pd
    from aic_proxy import (
        MAX_PROXY_SPEED_RATIO,
        PROXY_ALGORITHM_VERSION,
        correct_proxy_metrics,
        load_model_profile,
        load_supported_profiles,
        phase_correction_factors,
        profile_for_aic_row,
        select_proxy_profiles,
    )

    try:
        target = load_model_profile(request.model, request.model_configs_root)
        if request.precision is not None:
            runtime_dtype = "fp16" if request.precision == "fp16" else "bf16"
            target = replace(
                target,
                weight_dtype="fp8"
                if request.precision == "bf16-fp8"
                else runtime_dtype,
                activation_dtype=runtime_dtype,
                kv_dtype=runtime_dtype,
            )
        profiles = load_supported_profiles(request.system)
    except Exception as error:
        raise AicEvaluationError(f"AIC proxy metadata unavailable: {error}") from error
    rows = []
    failures = []
    for tp, pp in request.parallelism_candidates:
        matches = select_proxy_profiles(
            target,
            profiles,
            system=request.system,
            backend=request.backend,
            tp=tp,
            pp=pp,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            batch_size=1,
        )
        found = False
        for proxy_rank, match in enumerate(matches, start=1):
            profile = match.profile
            for mode in ("SILICON", "HYBRID", "EMPIRICAL", "SOL"):
                if monotonic() >= deadline:
                    break
                proxy_request = replace(
                    request,
                    model=profile.model.model_id,
                    backend=profile.backend,
                    backend_version=profile.backend_version,
                    database_mode=mode,
                    parallelism_candidates=((tp, pp),),
                    target_ttft_ms=request.target_ttft_ms * MAX_PROXY_SPEED_RATIO,
                    target_tpot_ms=request.target_tpot_ms * MAX_PROXY_SPEED_RATIO,
                )
                try:
                    frame, proxy_experts = _sweep_aic_request(proxy_request)
                except AicEvaluationError as error:
                    failures.append(f"{profile.profile_id}/{mode}: {str(error)[:200]}")
                    continue
                corrected_rows = []
                for _, row in frame.iterrows():
                    raw = row.to_dict()
                    try:
                        _runtime_parallelism(raw, proxy_request, proxy_experts)
                        _runtime_parallelism(raw, request, target.num_experts)
                        batch = _aic_positive_int(raw.get("bs"), "bs")
                        prediction_model = profile_for_aic_row(profile.model, raw)
                        prefill_ratio, decode_ratio = phase_correction_factors(
                            target,
                            prediction_model,
                            system=request.system,
                            tp=tp,
                            pp=pp,
                            input_tokens=request.input_tokens,
                            output_tokens=request.output_tokens,
                            batch_size=batch,
                        )
                        corrected = correct_proxy_metrics(
                            raw, prefill_ratio, decode_ratio, request.output_tokens
                        )
                    except (AicEvaluationError, ValueError) as error:
                        failures.append(f"{profile.profile_id}: {str(error)[:200]}")
                        continue
                    corrected["aic_prediction"] = {
                        "mode": "proxy",
                        "algorithm_version": PROXY_ALGORITHM_VERSION,
                        "requested_model_id": request.model,
                        "prediction_model_id": profile.model.model_id,
                        "backend": profile.backend,
                        "backend_version": profile.backend_version,
                        "system": request.system,
                        "database_mode": mode,
                        "proxy_rank": proxy_rank,
                        "selection_distance": match.distance,
                        "selection_batch_size": 1,
                        "reasons": match.reasons,
                        "native_error": native_error[:300],
                        "prefill_speed_ratio": prefill_ratio,
                        "decode_speed_ratio": decode_ratio,
                        "raw_metrics": {
                            name: raw.get(name)
                            for name in (
                                "ttft",
                                "tpot",
                                "tokens/s",
                                "seq/s",
                                "bs",
                                "gemm",
                                "moe",
                                "kvcache",
                            )
                        },
                        "corrected_metrics": {
                            name: corrected[name]
                            for name in ("ttft", "tpot", "tokens/s", "seq/s")
                        },
                    }
                    corrected_rows.append(corrected)
                if corrected_rows:
                    rows.extend(corrected_rows)
                    found = True
                    break
            if found or monotonic() >= deadline:
                break
        if monotonic() >= deadline:
            break
    if not rows:
        detail = "; ".join(failures[-3:]) or "no compatible supported profiles"
        raise AicEvaluationError(f"AIC proxy estimates unavailable: {detail}")
    return _select_aic_result(request, pd.DataFrame(rows), target.num_experts)


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
        model_catalog_store: ModelCatalogStoreClient | None = None,
    ) -> None:
        """Keep Store clients and AIC configuration."""
        self.user_id = user_id
        self.job_store = job_store
        self.resource_map_store = resource_map_store
        self.plan_store = plan_store
        self.settings = settings
        self.client = client
        self.model_catalog_store = model_catalog_store

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
            plan_store=PlanStore(postgres),
            settings=_load_settings(settings),
            client=client or NvidiaAicClient(),
            model_catalog_store=ModelCatalogStore(postgres),
        )

    def create_plans(self, tick_number: int, logical_time: datetime) -> PlannerResult:
        """Read one Store snapshot, persist one plan, and return its identity."""
        del logical_time
        try:
            snapshot = self._read_store_snapshot()
            LOGGER.info(
                "AIC tick %d: running=%d waiting=%d paused=%d",
                tick_number,
                len(snapshot.running_jobs),
                len(snapshot.waiting_jobs),
                len(snapshot.paused_jobs),
            )
            actions = self._build_actions(
                snapshot,
                monotonic() + self.settings.tick_timeout_seconds,
            )
            if not actions:
                LOGGER.info("AIC tick %d: no actions", tick_number)
                return PlannerResult(tick_number=tick_number, plans=())

            from tandemn_system_data.models import Plan

            plan = Plan(
                user_id=self.user_id,
                koi_version=self.method_name,
                tick_rationale=_tick_rationale(tick_number, actions),
                actions=actions,
            )
            self.plan_store.create(plan)
            LOGGER.info(
                "AIC tick %d: persisted plan %s with %d actions",
                tick_number,
                plan.plan_id,
                len(actions),
            )
            return PlannerResult(
                tick_number=tick_number,
                plans=(CreatedPlan(plan_id=plan.plan_id),),
            )
        except Exception as error:
            LOGGER.exception("AIC tick %d failed", tick_number)
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
        waiting_payloads = []
        for job in waiting:
            payload = _model_payload(job)
            model_id = _known_job_fields(payload).get("model_id")
            if self.model_catalog_store is not None and isinstance(model_id, str):
                catalog = self.model_catalog_store.get(model_id)
                if catalog is not None:
                    payload["_model_catalog"] = _model_payload(catalog)
            waiting_payloads.append(payload)
        return StoreSnapshot(
            resources=_resources_with_free_capacity(resource_map, running),
            running_jobs=tuple(_running_job_payload(job) for job in running),
            waiting_jobs=tuple(waiting_payloads),
            paused_jobs=(),
            finished_jobs=(),
        )

    def _build_actions(
        self,
        snapshot: StoreSnapshot,
        deadline: float | None = None,
    ) -> list[dict[str, object]]:
        """Evaluate waiting jobs greedily and build every lifecycle action."""
        deadline = deadline or monotonic() + self.settings.tick_timeout_seconds
        resources = deepcopy(snapshot.resources)
        actions: list[dict[str, object]] = []
        for running in snapshot.running_jobs:
            actions.append({"job_id": running["job_id"], "type": "keep"})
            LOGGER.debug("AIC keeping job %s", running["job_id"])
        for waiting in snapshot.waiting_jobs:
            job_id = waiting.get("job_id")
            if not isinstance(job_id, str):
                raise ValueError("waiting Store job is missing job_id")
            placement = _evaluate_job(
                waiting,
                resources,
                self.settings,
                self.client,
                deadline,
            )
            if placement is None:
                actions.append({"job_id": job_id, "type": "defer"})
                LOGGER.info("AIC deferring job %s: no feasible placement", job_id)
                continue
            _reserve(placement)
            LOGGER.info(
                "AIC placing job %s: target=%s gpus=%d tp=%d pp=%d dp=%d replicas=%d",
                job_id,
                placement.target.environment_key,
                placement.tp * placement.pp * placement.dp,
                placement.tp,
                placement.pp,
                placement.dp,
                placement.replicas,
            )
            actions.append(
                _place_action(
                    job_id,
                    waiting,
                    placement,
                    self.settings.backend,
                )
            )
        _validate_complete_job_actions(snapshot, actions)
        return actions


def _evaluate_job(
    store_job: Mapping[str, object],
    resources: dict[str, dict[str, Any]],
    settings: AicSettings,
    client: AicClient,
    deadline: float,
) -> AicPlacement | None:
    """Evaluate exact systems, then one bounded SOL/EMPIRICAL fallback."""
    known = _known_job_fields(store_job)
    try:
        goal = _optimization_goal(known)
    except ValueError:
        return None
    model_id = _text(known.get("model_id"), "model_id")
    hardware_candidates = _policy_hardware_candidates(
        _hardware_candidates(resources, settings.system_mappings),
        model_id,
    )
    placements = []
    for hardware in hardware_candidates:
        try:
            request = _build_aic_request(known, hardware, settings, goal)
        except ValueError:
            return None
        placement = _evaluate_request(client, request, hardware, settings, deadline)
        if placement is not None:
            placements.append(placement)
    if placements:
        return min(placements, key=lambda value: _placement_score(value, goal))

    fallback = _fallback_hardware(hardware_candidates)
    if fallback is None:
        return None
    request = _build_aic_request(known, fallback, settings, goal)
    for database_mode in ("SOL", "EMPIRICAL"):
        placement = _evaluate_request(
            client,
            replace(request, database_mode=database_mode),
            fallback,
            settings,
            deadline,
        )
        if placement is not None:
            return placement
    return None


def _evaluate_request(
    client: AicClient,
    request: AicRequest,
    hardware: HardwareCandidate,
    settings: AicSettings,
    deadline: float,
) -> AicPlacement | None:
    """Run one bounded AIC request and validate its concrete placement."""
    remaining = deadline - monotonic()
    if remaining <= 0:
        return None
    try:
        LOGGER.info(
            "AIC request system=%s mode=%s timeout=%.1fs",
            request.system,
            request.database_mode,
            min(settings.call_timeout_seconds, remaining),
        )
        result = client.evaluate(
            request,
            timeout_seconds=min(settings.call_timeout_seconds, remaining),
        )
        return _normalize_result(result, request, hardware)
    except AicUnavailableError:
        raise
    except (AicEvaluationError, ValueError) as error:
        LOGGER.info(
            "AIC request failed system=%s mode=%s: %s",
            request.system,
            request.database_mode,
            error,
        )
        return None


def _fallback_hardware(
    candidates: Sequence[HardwareCandidate],
) -> HardwareCandidate | None:
    """Choose one low-cost exact system for the only fallback round."""
    if not candidates:
        return None

    def key(candidate: HardwareCandidate) -> tuple[float, int, str]:
        prices = [
            _number(target.pool.get("price_per_instance_hour"))
            for target in candidate.targets
            if target.pool is not None
        ]
        price = min(
            (value for value in prices if value is not None),
            default=float("inf"),
        )
        return price, -candidate.available_devices, candidate.system

    return min(candidates, key=key)


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
    model_id = _text(known.get("model_id"), "model_id")
    catalog = _catalog_for_job(known)
    policy = load_config_policy()
    pairs = set()
    precision = None
    for target in hardware.targets:
        rule = policy.rule_for(target.gpu_type, model_id)
        if rule is None:
            continue
        precision = rule.precision
        for tp in rule.allowed_tp:
            for pp in sorted(VALID_PP_DEGREES):
                footprint = _target_footprint(target, 1, tp * pp)
                if (
                    rule.allows_parallelism(tp, pp)
                    and footprint is not None
                    and footprint <= _target_available(target)
                ):
                    pairs.add((tp, pp))
    return AicRequest(
        model=model_id,
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
        parallelism_candidates=tuple(sorted(pairs)),
        model_configs_root=settings.model_configs_root,
        precision=precision,
        chunked_prefill_enable=bool(
            _known_or_catalog(known, catalog, "chunked_prefill_enable", True)
        ),
    )


def _normalize_result(
    result: AicResult,
    request: AicRequest,
    hardware: HardwareCandidate,
) -> AicPlacement:
    """Validate AIC parallelism and bind it to one concrete Store target."""
    replicas = _positive_int(result.replicas, "replicas")
    tp, pp, dp, ep = _runtime_parallelism(
        result.configuration, request, result.num_experts
    )
    devices = replicas * tp * pp * dp
    if result.devices != devices:
        raise AicEvaluationError(
            "AIC device count does not match replicas multiplied by TP/PP/DP"
        )
    if result.ttft_ms > request.target_ttft_ms:
        raise AicEvaluationError("AIC TTFT exceeds the job target")
    if result.tpot_ms > request.target_tpot_ms:
        raise AicEvaluationError("AIC TPOT exceeds the job target")
    policy = load_config_policy()
    targets = tuple(
        target
        for target in hardware.targets
        if (rule := policy.rule_for(target.gpu_type, request.model)) is not None
        and rule.allows_parallelism(tp, pp)
    )
    target = _select_target(targets, replicas * dp, tp * pp)
    if target is None:
        raise AicEvaluationError("AIC configuration is unavailable in Store")
    prediction = result.configuration.get("aic_prediction")
    if prediction is not None and not isinstance(prediction, Mapping):
        raise AicEvaluationError("AIC prediction metadata must be an object")
    return AicPlacement(
        target=target,
        replicas=replicas,
        tp=tp,
        pp=pp,
        dp=dp,
        ep=ep,
        max_num_seq=_aic_positive_int(result.configuration.get("bs"), "bs"),
        devices=devices,
        ttft_ms=result.ttft_ms,
        tpot_ms=result.tpot_ms,
        request_latency_ms=result.request_latency_ms,
        throughput_tokens_per_second=result.throughput_tokens_per_second,
        prediction=prediction,
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
    model_id = _text(known.get("model_id"), "model_id")
    rule = load_config_policy().rule_for(placement.target.gpu_type, model_id)
    if rule is None:
        raise ValueError(
            f"{model_id} has no simulation profile on {placement.target.gpu_type}"
        )
    per_replica_gpu_count = placement.tp * placement.pp
    # Keep Store's DP metadata equal to its independently launched replicas.
    effective_dp = placement.replicas * placement.dp
    catalog = _catalog_for_job(known)
    catalog_max_num_seq = _catalog_gpu_value(
        catalog, "max_num_seq", placement.target.gpu_type
    )
    max_num_seq = placement.max_num_seq
    if catalog_max_num_seq is not None:
        max_num_seq = min(
            max_num_seq,
            _positive_int(catalog_max_num_seq, "model catalog max_num_seq"),
        )
    rank_config: dict[str, object] = {
        **_catalog_launch_config(catalog, placement.target.gpu_type),
        **simulation_quantization(rule.precision),
        "model_id": model_id,
        "engine_name": (
            known.get("engine_name") or catalog.get("engine_name") or backend
        ),
        "instance_type": placement.target.instance_type,
        "gpu_type": placement.target.gpu_type,
        "gpu_count": per_replica_gpu_count,
        "count": per_replica_gpu_count,
        "tp": placement.tp,
        "pp": placement.pp,
        "dp": effective_dp,
        "ep": placement.ep,
        "max_num_seq": max_num_seq,
        "max_num_batched_tokens": _known_or_catalog(
            known,
            catalog,
            "max_num_batched_tokens",
            2048,
            gpu_type=placement.target.gpu_type,
        ),
        "chunked_prefill_enable": (
            bool(
                _known_or_catalog(
                    known,
                    catalog,
                    "chunked_prefill_enable",
                    True,
                )
            )
        ),
        "block_size": _known_or_catalog(
            known, catalog, "block_size", 16, gpu_type=placement.target.gpu_type
        ),
        "kvcache_dtype": _known_or_catalog(
            known,
            catalog,
            "kvcache_dtype",
            "auto",
            gpu_type=placement.target.gpu_type,
        ),
        "gpu_mem_util": _known_or_catalog(known, catalog, "gpu_mem_util", 0.9),
        "prefix_cache_enabled": _known_or_catalog(
            known, catalog, "prefix_cache_enabled", False
        ),
        "rank_traffic_share": 1.0,
    }
    if placement.prediction is not None:
        rank_config["aic_prediction"] = dict(placement.prediction)
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
                "n_replicas": effective_dp,
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


def _policy_hardware_candidates(
    candidates: tuple[HardwareCandidate, ...],
    model_id: str,
) -> tuple[HardwareCandidate, ...]:
    """Keep AIC targets with a profiled simulation quantization for the model."""
    policy = load_config_policy()
    filtered = []
    for candidate in candidates:
        by_precision: dict[str, list[ResourceTarget]] = {}
        for target in candidate.targets:
            rule = policy.rule_for(target.gpu_type, model_id)
            if rule is not None:
                by_precision.setdefault(rule.precision, []).append(target)
        for targets in by_precision.values():
            filtered.append(
                replace(
                    candidate,
                    targets=tuple(targets),
                    available_devices=max(
                        _target_available(target) for target in targets
                    ),
                )
            )
    return tuple(filtered)


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
    return () if system is None else (system,)


def _select_target(
    targets: Sequence[ResourceTarget],
    replicas: int,
    per_replica_gpu_count: int,
) -> ResourceTarget | None:
    """Choose the tightest concrete target that fits an AIC configuration."""
    feasible = []
    for target in targets:
        footprint = _target_footprint(target, replicas, per_replica_gpu_count)
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
    per_replica_gpu_count = placement.tp * placement.pp
    effective_replicas = placement.replicas * placement.dp
    footprint = _target_footprint(
        placement.target,
        effective_replicas,
        per_replica_gpu_count,
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
    per_replica_gpu_count: int,
) -> int | None:
    """Translate an AIC GPU layout into the Store target's allocation unit."""
    if target.pool is None:
        return replicas * per_replica_gpu_count
    allocation_kind = str(
        target.pool.get("allocation_kind")
        or target.pool.get("allocation_unit")
        or "instance"
    ).lower()
    if allocation_kind == "gpu":
        return replicas * per_replica_gpu_count
    gpus_per_instance = _positive_int(
        target.pool.get("gpus_per_instance"),
        "gpus_per_instance",
    )
    if per_replica_gpu_count > gpus_per_instance:
        return None
    return replicas * gpus_per_instance


def _parallelism(configuration: Mapping[str, object]) -> tuple[int, int, int]:
    """Normalize AIC's explicit or compact parallelism representation."""
    parsed: dict[str, int] = {}
    parallel = configuration.get("parallel")
    if isinstance(parallel, str):
        parsed = {
            name: int(value)
            for name, value in re.findall(
                r"(?<![a-z])(tp|pp|dp)[=_-]?(\d+)", parallel.lower()
            )
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


def _runtime_parallelism(
    configuration: Mapping[str, object],
    request: AicRequest,
    num_experts: int,
) -> tuple[int, int, int, int]:
    """Require the expert layout that ServingSim will actually execute."""
    tp, pp, dp = _parallelism(configuration)
    if pp not in VALID_PP_DEGREES or (tp, pp) not in request.parallelism_candidates:
        raise AicEvaluationError("AIC TP/PP is unavailable on the target")
    parsed = {}
    parallel = configuration.get("parallel")
    if isinstance(parallel, str):
        parsed = {
            name: int(value)
            for name, value in re.findall(
                r"(?<![a-z])(etp|ep|cp)[=_-]?(\d+)", parallel.lower()
            )
        }
    cp = _aic_positive_int(configuration.get("cp", parsed.get("cp", 1)), "cp")
    if cp != 1:
        raise AicEvaluationError("AIC context parallelism requires CP=1")
    moe_tp = _aic_positive_int(
        configuration.get("moe_tp", parsed.get("etp", None if num_experts else 1)),
        "moe_tp",
    )
    ep = _aic_positive_int(
        configuration.get("moe_ep", parsed.get("ep", None if num_experts else 1)),
        "moe_ep",
    )
    if num_experts:
        # shortcut: local whole-expert layouts only; expand with DP-group support.
        if moe_tp != 1 or dp != 1 or ep != tp or num_experts % ep:
            raise AicEvaluationError(
                "AIC MoE requires MOE_TP=1, attention_DP=1, "
                "and MOE_EP=TP dividing the expert count"
            )
    elif moe_tp != 1 or ep != 1:
        raise AicEvaluationError("AIC dense models require MOE_TP=MOE_EP=1")
    worker_devices = configuration.get("num_total_gpus")
    if (
        worker_devices is not None
        and _aic_positive_int(worker_devices, "num_total_gpus") != tp * pp * dp
    ):
        raise AicEvaluationError("AIC worker GPU count does not match TP/PP/DP")
    return tp, pp, dp, ep


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


def _catalog_for_job(known: Mapping[str, object]) -> dict[str, Any]:
    """Return the optional model-catalog payload attached to a Store job."""
    value = known.get("_model_catalog")
    return _mapping(value, "model catalog") if isinstance(value, Mapping) else {}


def _catalog_gpu_value(
    catalog: Mapping[str, object], field: str, gpu_type: str
) -> object | None:
    """Resolve a model-catalog profile for the actual target GPU."""
    entries = catalog.get(field)
    if not isinstance(entries, list):
        return None
    target = _hardware_key(gpu_type)
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if _hardware_key(str(entry.get("gpu_type", ""))) == target:
            return entry.get("value")
    return None


def _known_or_catalog(
    known: Mapping[str, object],
    catalog: Mapping[str, object],
    field: str,
    default: object,
    *,
    gpu_type: str | None = None,
) -> object:
    """Prefer a per-job override, then the seeded model catalog, then a default."""
    value = known.get(field)
    if value is not None:
        return value
    value = (
        _catalog_gpu_value(catalog, field, gpu_type)
        if gpu_type is not None
        else catalog.get(field)
    )
    return default if value is None else value


def _catalog_launch_config(
    catalog: Mapping[str, object], gpu_type: str
) -> dict[str, object]:
    """Copy model launch defaults, resolving per-GPU fields without substitution."""
    result: dict[str, object] = {}
    for field in (
        "engine_name",
        "engine_version",
        "max_model_len",
        "chunk_size",
        "scheduling_policy",
        "preemption_policy",
        "router_policy",
    ):
        if catalog.get(field) is not None:
            result[field] = catalog[field]
    for field in (
        "max_num_seq",
        "max_num_batched_tokens",
        "block_size",
        "kvcache_dtype",
    ):
        value = _catalog_gpu_value(catalog, field, gpu_type)
        if value is not None:
            result[field] = value
    return result


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
    model_configs_root = values.get("model_configs_root")
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
        call_timeout_seconds=_positive_number(
            values.get("call_timeout_seconds", _AIC_CALL_TIMEOUT_SECONDS),
            "call_timeout_seconds",
        ),
        tick_timeout_seconds=_positive_number(
            values.get("tick_timeout_seconds", _AIC_TICK_TIMEOUT_SECONDS),
            "tick_timeout_seconds",
        ),
        model_configs_root=(
            None
            if model_configs_root is None
            else Path(_text(model_configs_root, "model_configs_root"))
        ),
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
