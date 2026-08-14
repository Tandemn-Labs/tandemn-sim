"""Define replaceable cluster-planner boundaries for the simulator."""

from __future__ import annotations

import sys
from os import getenv
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from models import CreatedPlan, PlannerResult, Scenario


@dataclass(frozen=True)
class PlannerContext:
    """Dependencies supplied to any benchmark planner by the simulator CLI."""

    user_id: str
    postgres: object
    logical_clock: object
    scenario: Scenario
    koi_root: Path | None
    koi_args: list[str]


class PlannerDriver(Protocol):
    """Create Store plans for one logical cluster control tick."""

    method_name: str

    def create_plans(self, tick_number: int, logical_time: datetime) -> PlannerResult:
        """Read current state and create plans for this control tick."""
        pass


class ScriptedPlannerDriver:
    """Create predeclared Store plans for lifecycle smoke tests.

    This is intentionally not a placement baseline. It proves the simulator's
    Store and Orca path before Koi or another planner owns plan creation.
    """

    method_name = "scripted"

    def __init__(
        self,
        user_id: str,
        plan_store: object,
        actions_by_tick: dict[int, tuple[object, ...]],
    ) -> None:
        """Keep one user's predeclared Store actions keyed by logical tick."""
        self.user_id = user_id
        self.plan_store = plan_store
        self.actions_by_tick = dict(actions_by_tick)

    def create_plans(self, tick_number: int, logical_time: datetime) -> PlannerResult:
        """Create this tick's plan once and return its Store identity."""
        actions = self.actions_by_tick.pop(tick_number, ())
        if not actions:
            return PlannerResult(tick_number=tick_number, plans=())

        from tandemn_system_data.models import Plan

        plan = Plan(
            user_id=self.user_id,
            actions=list(actions),
            tick_rationale=f"scripted cluster simulator tick {tick_number}",
        )
        self.plan_store.create(plan)
        return PlannerResult(
            tick_number=tick_number,
            plans=(CreatedPlan(plan_id=plan.plan_id),),
        )


class KoiPlannerDriver:
    """Adapt one persistent Koi TickRunner to the simulator planner boundary."""

    method_name = "koi"

    def __init__(
        self,
        user_id: str,
        runner: object,
        agent: object,
        llm: object,
        debug_logger: object | None = None,
    ) -> None:
        """Keep one already-built Koi runner and its persistent learning state."""
        self.user_id = user_id
        self.runner = runner
        self.agent = agent
        self.llm = llm
        self.debug_logger = debug_logger

    @classmethod
    def build(
        cls,
        user_id: str,
        logical_clock: object,
        koi_root: Path,
        koi_args: list[str],
    ) -> "KoiPlannerDriver":
        """Build Koi once using its existing Store-backed runner bootstrap."""
        if str(koi_root) not in sys.path:
            sys.path.insert(0, str(koi_root))
        from src.orchestrator.debug_logging import DebugLogger
        from src.orchestrator.runner import build_runner, configure_logging, parse_args

        args = parse_args(["--user-id", user_id, "--tick-interval-sec", "0", *koi_args])
        runner, _, agent, llm = build_runner(args)
        debug_logger = DebugLogger(args.log_dir, trace=args.trace, run_id=args.run_id)
        configure_logging(args.log_level, debug_logger.runner_log_path)
        runner.trace = debug_logger
        runner.tick_interval_sec = 0
        runner.telemetry.now_fn = logical_clock.now
        runner.executor = SimulationPlanExecutor(runner.executor)
        return cls(user_id, runner, agent, llm, debug_logger)

    def create_plans(self, tick_number: int, logical_time: datetime) -> PlannerResult:
        """Run one Koi tick and return the Store plans it created."""
        context = self.runner.run_tick(tick_number)
        if self.debug_logger is not None:
            self.debug_logger.persist_runner_tick(context, self.agent, self.llm)
        events = getattr(getattr(self.agent, "trace", None), "events", None)
        if isinstance(events, list):
            events.clear()
        calls = getattr(self.llm, "calls", None)
        if isinstance(calls, list):
            calls.clear()
        return PlannerResult(
            tick_number=tick_number,
            plans=tuple(CreatedPlan(plan_id=ack["plan_id"]) for ack in context.deploy_acks or []),
            degraded=context.error is not None,
            error=None if context.error is None else repr(context.error),
        )


def build_planner(name: str, context: PlannerContext) -> PlannerDriver:
    """Build one registered benchmark planner from common simulator context."""
    try:
        return PLANNER_BUILDERS[name](context)
    except KeyError as error:
        raise ValueError(f"unknown planner {name!r}: {', '.join(PLANNER_BUILDERS)}") from error


def _build_koi(context: PlannerContext) -> KoiPlannerDriver:
    """Build the real Koi adapter from CLI-provided Koi settings."""
    if context.koi_root is None:
        raise ValueError("planner 'koi' requires --koi-root")
    return KoiPlannerDriver.build(
        context.user_id,
        context.logical_clock,
        context.koi_root,
        context.koi_args,
    )


def _build_aic(context: PlannerContext) -> PlannerDriver:
    """Build the existing Dynamo AIC baseline with its standard backend."""
    from aic import AicPlannerDriver

    return AicPlannerDriver.build(
        context.user_id,
        context.scenario,
        {"backend": "vllm"},
        context.postgres,
    )


def _build_bestfit(context: PlannerContext) -> PlannerDriver:
    """Build the existing memory best-fit baseline with its standard backend."""
    from bestfit import BestFitPlannerDriver

    return BestFitPlannerDriver.build(
        context.user_id,
        context.scenario,
        {"backend": "vllm"},
        context.postgres,
    )


def _build_vram(context: PlannerContext) -> PlannerDriver:
    """Build the existing static VRAM baseline with its standard backend."""
    from vram import VramPlannerDriver

    return VramPlannerDriver.build(
        context.user_id,
        context.scenario,
        {"backend": "vllm"},
        context.postgres,
    )


def _build_gpt(context: PlannerContext) -> PlannerDriver:
    """Build the existing GPT-only baseline from the configured model name."""
    from gpt import GptPlannerDriver

    model = getenv("OPENAI_MODEL")
    if not model:
        raise ValueError("planner 'gpt_only' requires OPENAI_MODEL")
    return GptPlannerDriver.build(
        context.user_id,
        context.scenario,
        {"model": model, "max_tokens": 20_000, "backend": "vllm"},
        context.postgres,
    )


def _build_openevolve(context: PlannerContext) -> PlannerDriver:
    """Build the OpenEvolve online placement-policy baseline."""
    from openevolve import OpenEvolvePlannerDriver

    return OpenEvolvePlannerDriver.build(
        context.user_id,
        context.scenario,
        {
            "model": "gpt-5.6-luna",
            "max_tokens": 4096,
            "iterations": 8,
            "backend": "vllm",
        },
        context.postgres,
    )


PLANNER_BUILDERS: dict[str, Callable[[PlannerContext], PlannerDriver]] = {
    "koi": _build_koi,
    "dynamo_aic": _build_aic,
    "bestfit": _build_bestfit,
    "vram": _build_vram,
    "gpt_only": _build_gpt,
    "openevolve": _build_openevolve,
}


class SimulationPlanExecutor:
    """Normalize simulation plan metadata before writing Store plans."""

    def __init__(self, delegate: object) -> None:
        """Wrap Koi's existing Store executor without changing its behavior."""
        self.delegate = delegate

    def send_to_executor(self, plan: object) -> list[dict[str, str]]:
        """Validate and persist one Koi-validated plan for simulation."""
        return self.delegate.send_to_executor(normalize_plan_for_simulation(plan))


def normalize_plan_for_simulation(plan: object) -> object:
    """Persist Koi's multi-rank arrival split as normalized traffic shares."""
    normalized = deepcopy(plan)
    for action in normalized.actions:
        action_type = getattr(action.type, "value", action.type)
        ranks = action.ladder or []
        if action_type not in {"place", "swap"} or not ranks:
            continue
        if len(ranks) == 1:
            ranks[0].config["rank_traffic_share"] = 1.0
            continue

        shares = [_rank_share(rank.config) for rank in ranks]
        if all(share is None for share in shares):
            for rank in ranks:
                rank.config["rank_traffic_share"] = 1.0 / len(ranks)
            continue
        if any(share is None for share in shares):
            raise ValueError("multi-rank simulation plans cannot mix specified and missing shares")
        numeric_shares = [float(share) for share in shares]
        total = sum(numeric_shares)
        if total <= 0:
            raise ValueError("multi-rank simulation traffic shares must be positive")
        for rank, share in zip(ranks, numeric_shares, strict=True):
            rank.config["rank_traffic_share"] = share / total
    return normalized


def _rank_share(config: dict) -> float | None:
    """Read Koi's explicit share or its sized arrival-rate decision."""
    for key in ("rank_traffic_share", "traffic_share", "_arrival_share_rps"):
        if key in config:
            share = float(config[key])
            if share < 0:
                raise ValueError(f"{key} cannot be negative")
            return share
    return None
