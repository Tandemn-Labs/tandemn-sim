"""Test planner-driver adapters without external planner services."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace


from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from planning import KoiPlannerDriver


class KoiPlannerDriverTest(unittest.TestCase):
    """Verify Koi plan acknowledgements cross the simulator boundary."""

    def test_tick_returns_plan_ids_and_clears_debug_buffers(self) -> None:
        """Translate Koi deploy acknowledgements into one PlannerResult."""
        runner = SimpleNamespace(
            run_tick=lambda tick: SimpleNamespace(
                deploy_acks=[{"plan_id": "plan_1"}, {"plan_id": "plan_2"}],
                error=None,
            )
        )
        agent = SimpleNamespace(trace=SimpleNamespace(events=["debug"]))
        llm = SimpleNamespace(calls=["debug"])

        result = KoiPlannerDriver("user_1", runner, agent, llm).create_plans(3, None)

        self.assertEqual(result.tick_number, 3)
        self.assertEqual([plan.plan_id for plan in result.plans], ["plan_1", "plan_2"])
        self.assertFalse(result.degraded)
        self.assertEqual(agent.trace.events, [])
        self.assertEqual(llm.calls, [])

    def test_tick_marks_koi_errors_as_degraded(self) -> None:
        """Return Koi fallback plan acknowledgements with an error diagnostic."""
        runner = SimpleNamespace(
            run_tick=lambda tick: SimpleNamespace(
                deploy_acks=[{"plan_id": "plan_fallback"}],
                error=RuntimeError("planner failed"),
            )
        )

        result = KoiPlannerDriver("user_1", runner, SimpleNamespace(), SimpleNamespace()).create_plans(
            4,
            None,
        )

        self.assertTrue(result.degraded)
        self.assertIn("planner failed", result.error)
        self.assertEqual(result.plans[0].plan_id, "plan_fallback")


if __name__ == "__main__":
    unittest.main()
