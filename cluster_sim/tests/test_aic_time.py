"""Time real AIC planning across every simulation_3h control tick.

Run this opt-in integration test on a supported Linux x86-64 environment with
AIConfigurator installed:

    RUN_AIC_TIMING=1 .venv/bin/python -m unittest \
        cluster_sim.tests.test_aic_time -v

The test maintains queued, pending, and active jobs plus resource capacity in
memory. Store, Postgres, plan persistence, and serving simulation are kept
outside the timed section so the result measures AIC plan generation only.
"""

# ruff: noqa: E402

from __future__ import annotations

import importlib.util
import json
import logging
import os
import statistics
import sys
import time
import unittest
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from aic import (
    AicClient,
    AicEvaluationError,
    AicPlannerDriver,
    AicRequest,
    AicResult,
    AicSettings,
    NvidiaAicClient,
    StoreSnapshot,
    _load_settings,
    _parallelism,
)
from models import BatchJob, OnlineJob, Scenario
from policy import ConfigPolicy, load_config_policy
from scenario import load_scenario

LOGGER = logging.getLogger(__name__)
NANOSECONDS_PER_MINUTE = 60_000_000_000
TIMING_ENABLED = os.environ.get("RUN_AIC_TIMING") == "1"
AIC_INSTALLED = importlib.util.find_spec("aiconfigurator") is not None
AIC_SYSTEM_POLICY_HARDWARE = {
    "h100_sxm": "H100-SXM-80GB",
    "a100_sxm": "A100-SXM-80GB",
    "l40s": "L40S-48GB",
}
AIC_ARCHITECTURE_FAMILY_OVERRIDES = {
    "Phi3ForCausalLM": "LLAMA",
    "Glm4MoeForCausalLM": "MOE",
}


@unittest.skipUnless(
    TIMING_ENABLED and AIC_INSTALLED,
    "set RUN_AIC_TIMING=1 and install aiconfigurator to run this timing test",
)
class AicTimingTest(unittest.TestCase):
    """Measure full-plan AIC latency for the three-hour benchmark."""

    def test_simulation_3h_every_tick(self) -> None:
        """Log every tick plan and report mean AIC planning time."""
        output = logging.StreamHandler(sys.stdout)
        output.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(output)
        LOGGER.setLevel(logging.INFO)
        LOGGER.propagate = False
        self.addCleanup(LOGGER.removeHandler, output)

        benchmark_root = ROOT / "benchmarks" / "simulation_3h"
        benchmark_path = benchmark_root / "benchmark.yaml"
        benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))[
            "benchmark"
        ]
        original_mappings = _install_aic_architecture_mappings()
        self.addCleanup(_restore_aic_architecture_mappings, original_mappings)
        LOGGER.info(
            "aic_architecture_family_overrides=%s",
            json.dumps(AIC_ARCHITECTURE_FAMILY_OVERRIDES, sort_keys=True),
        )
        scenario = load_scenario(benchmark_path)
        driver = _driver(scenario)
        state = LocalSchedulerState(
            scenario,
            _resources(benchmark_root / benchmark["resources"]),
        )
        tick_minutes = scenario.tick_duration_ns // NANOSECONDS_PER_MINUTE
        duration_minutes = int(benchmark["duration_minutes"])
        elapsed_times = []

        for tick_number, minute in enumerate(
            range(0, duration_minutes + 1, tick_minutes)
        ):
            started_at = time.perf_counter()
            state.prepare_tick(minute)
            state_before = state.counts()
            actions = driver._build_actions(state.snapshot())
            _apply_config_policy(actions)
            elapsed_seconds = time.perf_counter() - started_at
            state.apply(actions)
            elapsed_times.append(elapsed_seconds)

            LOGGER.info(
                "%s",
                json.dumps(
                    {
                        "tick": tick_number,
                        "minute": minute,
                        "elapsed_seconds": elapsed_seconds,
                        "state_before": state_before,
                        "plan": actions,
                        "state_after": state.counts(),
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )

        average_seconds = statistics.fmean(elapsed_times)
        LOGGER.info(
            "ticks=%d average_elapsed_seconds=%.6f",
            len(elapsed_times),
            average_seconds,
        )

        self.assertEqual(len(elapsed_times), 37)
        self.assertTrue(all(elapsed >= 0 for elapsed in elapsed_times))


@dataclass
class LocalJob:
    """Track one benchmark job without Store lifecycle state."""

    job: OnlineJob | BatchJob
    status: str = "queued"
    allocations: list[tuple[str, int]] = field(default_factory=list)


class LocalSchedulerState:
    """Apply AIC actions to an in-memory job queue and resource map."""

    def __init__(
        self,
        scenario: Scenario,
        resources: dict[str, dict[str, object]],
    ) -> None:
        """Initialize all benchmark jobs as queued with full capacity."""
        jobs = (*scenario.online_jobs, *scenario.batch_jobs)
        self.jobs = {job.store_job_id: LocalJob(job) for job in jobs}
        self.resources = deepcopy(resources)

    def prepare_tick(self, minute: int) -> None:
        """Expire ended jobs and release newly eligible queued jobs."""
        offset_ns = minute * NANOSECONDS_PER_MINUTE
        for local_job in self.jobs.values():
            if (
                local_job.status in {"pending", "active"}
                and _end_ns(local_job.job) <= offset_ns
            ):
                self._finish(local_job)
            if local_job.status == "queued" and _release_ns(
                local_job.job
            ) <= offset_ns < _end_ns(local_job.job):
                local_job.status = "pending"

    def snapshot(self) -> StoreSnapshot:
        """Build the AIC snapshot for current active and pending jobs."""
        active = tuple(
            {"job_id": job_id}
            for job_id, local_job in self.jobs.items()
            if local_job.status == "active"
        )
        pending = tuple(
            _store_job(local_job.job)
            for local_job in self.jobs.values()
            if local_job.status == "pending"
        )
        return StoreSnapshot(
            resources=self.resources,
            running_jobs=active,
            waiting_jobs=pending,
            paused_jobs=(),
            finished_jobs=(),
        )

    def apply(self, actions: list[dict[str, object]]) -> None:
        """Apply one AIC plan to local lifecycle and capacity state."""
        for action in actions:
            job_id = str(action["job_id"])
            local_job = self.jobs[job_id]
            action_type = action["type"]
            if action_type == "keep":
                if local_job.status != "active":
                    raise ValueError(f"cannot keep non-active job {job_id}")
                continue
            if action_type == "defer":
                if local_job.status != "pending":
                    raise ValueError(f"cannot defer non-pending job {job_id}")
                continue
            if action_type != "place" or local_job.status != "pending":
                raise ValueError(f"invalid local action {action_type!r} for {job_id}")
            local_job.allocations = self._reserve(action)
            local_job.status = "active"

    def counts(self) -> dict[str, int]:
        """Summarize local lifecycle state for timing output."""
        return {
            status: sum(job.status == status for job in self.jobs.values())
            for status in ("queued", "pending", "active", "finished")
        }

    def _reserve(self, action: dict[str, object]) -> list[tuple[str, int]]:
        """Subtract every placed rank footprint from local free capacity."""
        allocations = []
        ladder = action.get("ladder")
        if not isinstance(ladder, list):
            raise ValueError("place action is missing its ladder")
        for rank in ladder:
            environment_key = "|".join(str(part) for part in rank["env"])
            config = rank["config"]
            footprint = int(rank["n_replicas"]) * int(config["gpu_count"])
            resource = self.resources[environment_key]
            free = int(resource["free"])
            if footprint > free:
                raise ValueError(f"placement exceeds capacity for {environment_key}")
            resource["free"] = free - footprint
            allocations.append((environment_key, footprint))
        return allocations

    def _finish(self, local_job: LocalJob) -> None:
        """Return an active job's allocations and mark it finished."""
        for environment_key, footprint in local_job.allocations:
            resource = self.resources[environment_key]
            resource["free"] = int(resource["free"]) + footprint
        local_job.allocations.clear()
        local_job.status = "finished"


class PolicyAicClient:
    """Restrict real AIC evaluations to the repository configuration policy."""

    def __init__(self, delegate: AicClient, policy: ConfigPolicy) -> None:
        """Keep the real client and shared policy for each evaluation."""
        self.delegate = delegate
        self.policy = policy

    def evaluate(
        self, request: AicRequest, timeout_seconds: float | None = None
    ) -> AicResult:
        """Reject unsupported model/hardware pairs and disallowed TP sizes."""
        try:
            hardware = AIC_SYSTEM_POLICY_HARDWARE[request.system]
        except KeyError as error:
            raise AicEvaluationError(
                f"no policy hardware mapping for AIC system {request.system}"
            ) from error
        rule = self.policy.rule_for(hardware, request.model)
        if rule is None:
            raise AicEvaluationError(
                f"{request.model} is unsupported on {hardware} by config policy"
            )

        result = self.delegate.evaluate(request, timeout_seconds=timeout_seconds)
        tp, _, _ = _parallelism(result.configuration)
        if tp not in rule.allowed_tp:
            raise AicEvaluationError(
                f"tp={tp} is not allowed for {request.model} on {hardware}; "
                f"expected one of {rule.allowed_tp}"
            )
        return result


def _driver(scenario: Scenario) -> AicPlannerDriver:
    """Build the real AIC planner without external Store clients."""
    return AicPlannerDriver(
        user_id="aic-timing",
        scenario=scenario,
        job_store=object(),
        resource_map_store=object(),
        plan_store=object(),
        settings=_aic_settings(),
        client=PolicyAicClient(NvidiaAicClient(), load_config_policy()),
    )


def _install_aic_architecture_mappings() -> dict[str, str | None]:
    """Temporarily map unsupported architectures to compatible AIC families."""
    from aiconfigurator.sdk.common import ARCHITECTURE_TO_MODEL_FAMILY

    original = {
        architecture: ARCHITECTURE_TO_MODEL_FAMILY.get(architecture)
        for architecture in AIC_ARCHITECTURE_FAMILY_OVERRIDES
    }
    ARCHITECTURE_TO_MODEL_FAMILY.update(AIC_ARCHITECTURE_FAMILY_OVERRIDES)
    return original


def _restore_aic_architecture_mappings(
    original: dict[str, str | None],
) -> None:
    """Restore AIC's process-global architecture mapping after the test."""
    from aiconfigurator.sdk.common import ARCHITECTURE_TO_MODEL_FAMILY

    for architecture, family in original.items():
        if family is None:
            ARCHITECTURE_TO_MODEL_FAMILY.pop(architecture, None)
        else:
            ARCHITECTURE_TO_MODEL_FAMILY[architecture] = family


def _aic_settings() -> AicSettings:
    """Load the implemented AIC method settings from the method catalog."""
    document = yaml.safe_load(
        (ROOT / "configs" / "methods.yaml").read_text(encoding="utf-8")
    )
    settings = document["methods"]["dynamo_aic"]["settings"]
    return _load_settings(settings)


def _resources(resource_path: Path) -> dict[str, dict[str, object]]:
    """Build one full-capacity Store-style target per AIC-supported GPU."""
    document = yaml.safe_load(resource_path.read_text(encoding="utf-8"))
    resources = {}
    for resource in document["resources"]:
        system = resource.get("aic_system")
        if system is None:
            continue
        accelerator = str(resource["accelerator"])
        total = int(resource["total_devices"])
        key = f"reserved|simulation|global|global|{accelerator}"
        resources[key] = {
            "gpu_type": accelerator,
            "aic_system": str(system),
            "total": total,
            "free": total,
            "pools": [],
        }
    return resources


def _store_job(job: OnlineJob | BatchJob) -> dict[str, object]:
    """Convert one scenario job into the scheduler-visible Store payload."""
    features = dict(job.workload["features"])
    kind = "online" if isinstance(job, OnlineJob) else "batch"
    if kind == "online":
        features["_traffic_mode"] = "concurrency"
    return {
        "job_id": job.store_job_id,
        "user_id": "aic-timing",
        "kind": kind,
        "spec_json": {
            "model_id": job.model_name,
            "job_features": features,
        },
    }


def _apply_config_policy(actions: list[dict[str, object]]) -> None:
    """Validate placed ranks and attach the policy-required precision."""
    policy = load_config_policy()
    for action in actions:
        if action["type"] != "place":
            continue
        for rank in action["ladder"]:
            config = rank["config"]
            model_id = str(config["model_id"])
            gpu_type = str(config["gpu_type"])
            tp = int(config["tp"])
            rule = policy.rule_for(gpu_type, model_id)
            if rule is None or tp not in rule.allowed_tp:
                raise ValueError(
                    f"AIC placement violates config policy for {model_id} on "
                    f"{gpu_type} with tp={tp}"
                )
            config["precision"] = rule.precision


def _release_ns(job: OnlineJob | BatchJob) -> int:
    """Return one benchmark job's release offset."""
    return job.starts_at_ns if isinstance(job, OnlineJob) else job.releases_at_ns


def _end_ns(job: OnlineJob | BatchJob) -> int:
    """Return one benchmark job's local lifecycle end offset."""
    return job.stops_at_ns if isinstance(job, OnlineJob) else job.hard_stop_at_ns


if __name__ == "__main__":
    unittest.main()
