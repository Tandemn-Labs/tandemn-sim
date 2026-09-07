"""Test planner-driver adapters without external planner services."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from planning import (
    PLANNER_BUILDERS,
    KoiPlannerDriver,
    SimulationPlanExecutor,
    normalize_plan_for_simulation,
)


class KoiPlannerDriverTest(unittest.TestCase):
    """Verify Koi plan acknowledgements cross the simulator boundary."""

    def test_tick_returns_provider_token_usage(self) -> None:
        """Capture Koi's OpenAI-compatible response at its API boundary."""
        response = SimpleNamespace(
            model="koi-model-used",
            usage=SimpleNamespace(prompt_tokens=70, completion_tokens=15),
            choices=[SimpleNamespace(message=SimpleNamespace(content="done"))],
        )
        resource = SimpleNamespace(create=lambda **_kwargs: response)
        sdk_client = SimpleNamespace(chat=SimpleNamespace(completions=resource))
        client = SimpleNamespace(_client=sdk_client, model="koi-model")
        llm = SimpleNamespace(inner=client, calls=[])

        def run_tick(tick: int) -> SimpleNamespace:
            client._client.chat.completions.create(model=client.model)
            return SimpleNamespace(deploy_acks=[], error=None)

        result = KoiPlannerDriver(
            "user_1",
            SimpleNamespace(run_tick=run_tick),
            SimpleNamespace(),
            llm,
        ).create_plans(6, None)

        self.assertIs(client._client, sdk_client)
        self.assertEqual(result.llm_usage[0].tick_number, 6)
        self.assertEqual(result.llm_usage[0].input_tokens, 70)
        self.assertEqual(result.llm_usage[0].output_tokens, 15)
        self.assertEqual(result.llm_usage[0].model, "koi-model-used")

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

    def test_tick_persists_debug_data_before_clearing_buffers(self) -> None:
        """Keep Koi's trace-all records available to the simulator run."""
        context = SimpleNamespace(deploy_acks=[], error=None)
        runner = SimpleNamespace(run_tick=lambda tick: context)
        agent = SimpleNamespace(trace=SimpleNamespace(events=["debug"]))
        llm = SimpleNamespace(calls=["debug"])
        debug_logger = SimpleNamespace(persist_runner_tick=lambda *args: calls.append(args))
        calls = []

        KoiPlannerDriver("user_1", runner, agent, llm, debug_logger).create_plans(3, None)

        self.assertEqual(calls[0], (context, agent, llm))
        self.assertEqual(agent.trace.events, [])
        self.assertEqual(llm.calls, [])

    def test_registers_existing_baseline_drivers(self) -> None:
        """Expose the implemented baseline drivers through the simulator CLI."""
        self.assertEqual(
            set(PLANNER_BUILDERS),
            {
                "koi",
                "dynamo_aic",
                "bestfit",
                "vram",
                "edf",
                "prism",
                "cp_sat",
                "gpt_only",
                "openevolve",
                "contextual_bandit",
                "evox",
                "agentic",
            },
        )

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

    def test_multi_rank_arrival_rates_become_normalized_shares(self) -> None:
        """Preserve Koi's 60/40 rank sizing decision for next-tick deployment X."""
        first = SimpleNamespace(config={"_arrival_share_rps": 60.0})
        second = SimpleNamespace(config={"_arrival_share_rps": 40.0})
        plan = SimpleNamespace(actions=[SimpleNamespace(type="place", ladder=[first, second])])

        normalized = normalize_plan_for_simulation(plan)

        ranks = normalized.actions[0].ladder
        self.assertEqual(ranks[0].config["rank_traffic_share"], 0.6)
        self.assertEqual(ranks[1].config["rank_traffic_share"], 0.4)
        self.assertNotIn("rank_traffic_share", first.config)

    def test_single_rank_share_is_one(self) -> None:
        """Declare complete job traffic for a one-rank placement."""
        plan = SimpleNamespace(
            actions=[SimpleNamespace(type="place", ladder=[SimpleNamespace(config={})])]
        )

        normalized = normalize_plan_for_simulation(plan)

        self.assertEqual(normalized.actions[0].ladder[0].config["rank_traffic_share"], 1.0)

    def test_multi_rank_without_share_uses_uniform_traffic(self) -> None:
        """Give baseline planners equal traffic when they supply no split."""
        plan = SimpleNamespace(
            actions=[
                SimpleNamespace(
                    type="place",
                    ladder=[SimpleNamespace(config={}), SimpleNamespace(config={})],
                )
            ]
        )

        normalized = normalize_plan_for_simulation(plan)

        self.assertEqual(normalized.actions[0].ladder[0].config["rank_traffic_share"], 0.5)
        self.assertEqual(normalized.actions[0].ladder[1].config["rank_traffic_share"], 0.5)

    def test_executor_delegates_normalized_plan(self) -> None:
        """Write only the normalized simulation plan through Koi's Store executor."""
        captured = []
        delegate = SimpleNamespace(send_to_executor=lambda plan: captured.append(plan) or [{"plan_id": "p"}])
        plan = SimpleNamespace(
            actions=[
                SimpleNamespace(
                    type="place",
                    ladder=[
                        SimpleNamespace(config={"_arrival_share_rps": 1}),
                        SimpleNamespace(config={"_arrival_share_rps": 3}),
                    ],
                )
            ]
        )

        result = SimulationPlanExecutor(delegate).send_to_executor(plan)

        self.assertEqual(result, [{"plan_id": "p"}])
        self.assertEqual(captured[0].actions[0].ladder[0].config["rank_traffic_share"], 0.25)


if __name__ == "__main__":
    unittest.main()
