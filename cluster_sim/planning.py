"""Define replaceable cluster-planner boundaries for the simulator."""

from __future__ import annotations

from datetime import datetime
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

    def __init__(self) -> None:
        """Build Koi once after simulator configuration is available."""
        pass

    def create_plans(self, tick_number: int, logical_time: datetime) -> object:
        """Run one Koi tick and return the Store plans it created."""
        pass


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
