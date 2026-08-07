"""Define replaceable cluster-planner boundaries for the simulator."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Protocol

from models import CreatedPlan, PlannerResult


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

    def __init__(self, user_id: str, runner: object, agent: object, llm: object) -> None:
        """Keep one already-built Koi runner and its persistent learning state."""
        self.user_id = user_id
        self.runner = runner
        self.agent = agent
        self.llm = llm

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
        from src.orchestrator.runner import build_runner, parse_args

        args = parse_args(["--user-id", user_id, "--tick-interval-sec", "0", *koi_args])
        runner, _, agent, llm = build_runner(args)
        runner.tick_interval_sec = 0
        runner.telemetry.now_fn = logical_clock.now
        return cls(user_id, runner, agent, llm)

    def create_plans(self, tick_number: int, logical_time: datetime) -> PlannerResult:
        """Run one Koi tick and return the Store plans it created."""
        context = self.runner.run_tick(tick_number)
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


class SimulationPlanExecutor:
    """Normalize simulation plan metadata before writing Store plans."""

    def __init__(self) -> None:
        """Create the executor around Koi's existing Store plan executor."""
        pass

    def send_to_executor(self, plan: object) -> list[dict[str, str]]:
        """Validate and persist one Koi-validated plan for simulation."""
        pass


def normalize_plan_for_simulation(plan: object) -> object:
    """Validate rank shapes and add equal initial rank traffic shares."""
    pass
