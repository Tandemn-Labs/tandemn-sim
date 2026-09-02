"""Test simulation and real-hardware policy selection."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

import policy
from models import Scenario
from planning import PLANNER_BUILDERS, build_planner


class ConfigPolicyModeTest(unittest.TestCase):
    """Keep policy selection inside the policy module."""

    def tearDown(self) -> None:
        """Return later tests to the default simulation policy."""
        policy.configure_config_policy("simulation")

    def test_real_hardware_policy(self) -> None:
        """Expose the requested real-hardware TP and PP limits."""
        policy.configure_config_policy("real_hardware")
        config_policy = policy.load_config_policy()

        phi = config_policy.rule_for("RTXPRO6000", "microsoft/phi-4")
        llama = config_policy.rule_for(
            "A100_80GB",
            "meta-llama/Llama-3.3-70B-Instruct",
        )

        self.assertIsNotNone(phi)
        self.assertIsNotNone(llama)
        assert phi is not None and llama is not None
        self.assertEqual(phi.allowed_tp, (1, 2))
        self.assertEqual(llama.allowed_tp, (1, 2, 4, 8))
        self.assertEqual(phi.allowed_pp, (1, 2, 3, 4))
        self.assertFalse(phi.allows_parallelism(1, 5))

    def test_simulation_policy_remains_default(self) -> None:
        """Keep existing simulation rules when no real mode is selected."""
        policy.configure_config_policy("simulation")
        rule = policy.load_config_policy().rule_for(
            "A10G",
            "microsoft/phi-4",
        )

        self.assertIsNotNone(rule)
        assert rule is not None
        self.assertEqual(rule.allowed_tp, (2,))
        self.assertEqual(rule.allowed_pp, (1, 2, 3, 4))

    def test_planner_build_selects_the_scenario_environment(self) -> None:
        """Configure policy mode once before constructing a planner."""
        context = SimpleNamespace(
            scenario=Scenario("real", 1, 1, (), (), environment="real_hardware")
        )

        with patch.dict(
            PLANNER_BUILDERS,
            {"test": lambda _context: SimpleNamespace(method_name="test")},
        ):
            build_planner("test", context)

        rule = policy.load_config_policy().rule_for(
            "RTXPRO6000",
            "microsoft/phi-4",
        )
        self.assertIsNotNone(rule)
        assert rule is not None
        self.assertEqual(rule.allowed_pp, (1, 2, 3, 4))


if __name__ == "__main__":
    unittest.main()
