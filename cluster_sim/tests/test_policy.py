"""Test cluster configuration policy validation."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

import policy


class ConfigPolicyTest(unittest.TestCase):
    """Keep tensor-parallel degrees within global and model constraints."""

    def setUp(self) -> None:
        """Start each policy test without a prior suite's cached policy."""
        policy.load_config_policy.cache_clear()

    def tearDown(self) -> None:
        """Clear policy instances built from patched policy text."""
        policy.load_config_policy.cache_clear()

    def test_embedded_policy_uses_supported_tp_degrees(self) -> None:
        """Load the checked-in policy under the global TP constraint."""
        config_policy = policy.load_config_policy()

        rule = config_policy.rule_for(
            "L40S-48GB",
            "meta-llama/Llama-3.2-1B-Instruct",
        )
        self.assertIsNotNone(rule)
        assert rule is not None
        self.assertTrue(set(rule.allowed_tp) <= policy.VALID_TP_DEGREES)

    def test_mixtral_a10g_excludes_unprofiled_tp_one(self) -> None:
        """Keep Mixtral off A10G TP one until ServingSim has that profile."""
        rule = policy.load_config_policy().rule_for(
            "A10G",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        )

        self.assertIsNotNone(rule)
        assert rule is not None
        self.assertEqual(rule.allowed_tp, (4, 8))

    def test_phi_a10g_excludes_unprofiled_tp_one(self) -> None:
        """Keep Phi-4 off A10G TP one until ServingSim has final_layernorm data."""
        rule = policy.load_config_policy().rule_for("A10G", "microsoft/phi-4")

        self.assertIsNotNone(rule)
        assert rule is not None
        self.assertEqual(rule.allowed_tp, (2,))

    def test_a100_profiles_exclude_incomplete_tp_one(self) -> None:
        """Keep TP1-only MoE profile bundles out of A100 placement choices."""
        config_policy = policy.load_config_policy()

        qwen = config_policy.rule_for(
            "A100_80GB",
            "Qwen/Qwen3-Next-80B-A3B-Thinking",
        )
        glm = config_policy.rule_for("A100_80GB", "zai-org/GLM-4.5-Air")

        self.assertIsNotNone(qwen)
        self.assertIsNotNone(glm)
        assert qwen is not None and glm is not None
        self.assertEqual(qwen.allowed_tp, (4, 8))
        self.assertEqual(glm.allowed_tp, (4, 8))

    def test_accepts_one_and_supported_even_tp_degrees(self) -> None:
        """Allow no sharding or an explicitly supported even degree."""
        policy_text = """\
H100:
- example/model → fp16 TP: 1, 2, 4, 6, 8
        """

        with patch.object(policy, "_POLICY_TEXT", policy_text):
            policy.load_config_policy.cache_clear()
            rule = policy.load_config_policy().rule_for("H100", "example/model")

        self.assertIsNotNone(rule)
        assert rule is not None
        self.assertEqual(rule.allowed_tp, (1, 2, 4, 6, 8))

    def test_rejects_unsupported_tp_degrees(self) -> None:
        """Reject odd sharding and even degrees beyond the supported set."""
        for tp in (3, 10):
            with self.subTest(tp=tp):
                policy.load_config_policy.cache_clear()
                policy_text = f"H100:\n- example/model → fp16 TP: {tp}\n"
                with patch.object(policy, "_POLICY_TEXT", policy_text):
                    policy.load_config_policy.cache_clear()
                    with self.assertRaisesRegex(ValueError, "unsupported TP values"):
                        policy.load_config_policy()

    def test_rejects_phi_tp_above_model_limit(self) -> None:
        """Reject Phi-4 policy entries that exceed TP two."""
        policy_text = """\
H100:
- microsoft/phi-4 → bf16 TP: 1, 2, 4
"""

        with patch.object(policy, "_POLICY_TEXT", policy_text):
            policy.load_config_policy.cache_clear()
            with self.assertRaisesRegex(
                ValueError,
                "microsoft/phi-4 TP values must not exceed 2",
            ):
                policy.load_config_policy()


if __name__ == "__main__":
    unittest.main()
