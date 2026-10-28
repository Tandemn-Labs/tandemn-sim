"""Verify independent AIC proxy matching without importing Koi."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from aic_proxy import (
    ModelProfile,
    SupportedProfile,
    correct_proxy_metrics,
    load_model_profile,
    load_supported_profiles,
    model_profile_from_config,
    phase_correction_factors,
    profile_for_aic_row,
    select_proxy_profiles,
)

NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Thinking"
NEXT_CONFIG = {
    "architectures": ["Qwen3NextForCausalLM"],
    "model_type": "qwen3_next",
    "num_hidden_layers": 48,
    "hidden_size": 2048,
    "intermediate_size": 5120,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "vocab_size": 151936,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 512,
    "full_attention_interval": 4,
    "max_position_embeddings": 262144,
    "torch_dtype": "bfloat16",
}


class AicProxyTest(unittest.TestCase):
    """Check metadata, reference ordering, and compatible proxy inventory."""

    def setUp(self) -> None:
        """Keep inventory/cache tests independent of other test modules."""
        load_model_profile.cache_clear()
        load_supported_profiles.cache_clear()
        self.addCleanup(load_model_profile.cache_clear)
        self.addCleanup(load_supported_profiles.cache_clear)

    def test_next_metadata_loads_without_aic_architecture_support(self) -> None:
        """Load original Next metadata even when AIC rejects the architecture."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / f"{NEXT_MODEL}.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(NEXT_CONFIG))
            with patch(
                "aiconfigurator.sdk.utils.get_model_config_from_model_path",
                side_effect=AssertionError("Next must use its local config"),
            ):
                profile = load_model_profile(NEXT_MODEL, root)

        self.assertEqual(profile.model_id, NEXT_MODEL)
        self.assertEqual(profile.architecture, "Qwen3NextForCausalLM")
        self.assertEqual(profile.num_experts, 512)
        self.assertEqual(profile.active_experts, 10)
        self.assertEqual(profile.head_dim, 256)
        self.assertEqual(profile.max_context, 262144)
        self.assertGreater(profile.parameter_count, 75e9)

    def test_parsed_aic_and_raw_metadata_normalize_equally(self) -> None:
        """Normalize SDK aliases without losing original structural fields."""
        parsed = {
            "architecture": "Qwen3NextForCausalLM",
            "layers": 48,
            "n": 16,
            "n_kv": 2,
            "d": 256,
            "topk": 10,
            "inter_size": 5120,
            "moe_inter_size": 512,
            "vocab": 151936,
            "context": 262144,
            "raw_config": NEXT_CONFIG,
        }
        self.assertEqual(
            model_profile_from_config(NEXT_MODEL, parsed),
            model_profile_from_config(NEXT_MODEL, NEXT_CONFIG),
        )

    def test_metadata_validation_rejects_incomplete_or_invalid_structure(self) -> None:
        """Do not silently invent missing target geometry or expert counts."""
        for changes in (
            {"hidden_size": None},
            {"num_experts_per_tok": 0},
            {"num_experts_per_tok": 513},
            {"num_hidden_layers": 2.5},
            {"model_params_b": float("nan")},
            {"weight_dtype": "unknown"},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    model_profile_from_config(NEXT_MODEL, {**NEXT_CONFIG, **changes})
        for model_id in ("", "../outside", "/absolute"):
            with self.subTest(model_id=model_id):
                with self.assertRaises(ValueError):
                    load_model_profile(model_id, Path("model-configs"))

    def test_optional_hybrid_geometry_uses_reference_defaults(self) -> None:
        """Keep Koi's defaults for omitted dense-MLP and KV-head metadata."""
        profile = model_profile_from_config(
            NEXT_MODEL,
            {**NEXT_CONFIG, "intermediate_size": None, "num_key_value_heads": 0},
        )
        self.assertEqual(profile.intermediate_size, profile.hidden_size * 4)
        self.assertEqual(profile.kv_heads, profile.attention_heads)
        self.assertEqual(profile.expert_intermediate_size, 512)

    def test_exact_identity_precedes_size_and_backend_preferences(self) -> None:
        """Keep a supported native identity ahead of differently named proxies."""
        target = _model()
        exact = SupportedProfile(target, "h100_sxm", "sglang", "0.5.14")
        other = _supported("other")
        matches = _select(target, [other, exact])
        self.assertEqual(matches[0].profile, exact)
        self.assertEqual(matches[0].distance, 0)
        self.assertEqual(matches[0].requested_model_id, NEXT_MODEL)

    def test_size_precedes_dtype_then_backend_then_structural_distance(self) -> None:
        """Use Koi's final ordering, rather than distance alone."""
        target = _model()
        profiles = [
            _supported("far", parameter_count=40e9),
            _supported("close-fp8", parameter_count=79e9, weight_dtype="fp8"),
            replace(_supported("close-sglang", parameter_count=79e9), backend="sglang"),
            _supported("close-vllm", parameter_count=79e9),
        ]
        matches = _select(target, profiles)
        self.assertEqual(
            [match.profile.model.model_id for match in matches],
            ["close-vllm", "close-sglang", "close-fp8", "far"],
        )

    def test_incompatible_model_kinds_and_topologies_are_filtered(self) -> None:
        """Require same-system MoE proxies with runnable attention/expert widths."""
        valid = _supported("valid")
        profiles = [
            replace(valid, system="a100_sxm"),
            _supported("dense", num_experts=0, active_experts=0),
            _supported("vision", multimodal=True),
            _supported("encoder", generative=False),
            _supported("wrong-heads", attention_heads=3),
            _supported("wrong-experts", num_experts=10),
            _supported("too-few-layers", layers=1),
            valid,
        ]
        matches = _select(_model(), profiles, pp=2)
        self.assertEqual([match.profile for match in matches], [valid])

    def test_top_five_and_ties_are_deterministic(self) -> None:
        """Inventory order cannot alter tied proxy selections."""
        profiles = [_supported(f"proxy-{number}") for number in range(7)]
        forward = _select(_model(), profiles)
        backward = _select(_model(), list(reversed(profiles)))
        self.assertEqual(forward, backward)
        self.assertEqual(len(forward), 5)
        self.assertEqual(forward[0].profile.model.model_id, "proxy-0")

    def test_context_extrapolation_is_scored_and_reported(self) -> None:
        """Use the target workload when assessing a proxy's context coverage."""
        short = _supported("short", max_context=256)
        long = _supported("long")
        matches = _select(_model(), [short, long], input_tokens=4096)
        self.assertEqual(matches[0].profile, long)
        self.assertIn("context_extrapolation", matches[1].reasons)
        self.assertGreater(matches[1].distance, matches[0].distance)

    def test_structural_distances_match_frozen_koi_reference(self) -> None:
        """Reproduce complete-input Koi scores without a runtime Koi dependency."""
        mixtral = _supported(
            "mistralai/Mixtral-proxy",
            architecture="MixtralForCausalLM",
            parameter_count=40e9,
            hidden_size=4096,
            intermediate_size=14336,
            layers=32,
            attention_heads=32,
            kv_heads=8,
            head_dim=128,
            num_experts=8,
            active_experts=2,
            expert_intermediate_size=14336,
        )
        larger = _supported(
            "Qwen/Larger-proxy", parameter_count=120e9, layers=64, weight_dtype="fp8"
        )
        matches = _select(_model(), [mixtral, larger])
        self.assertEqual([match.profile for match in matches], [larger, mixtral])
        # Frozen from Koi bca19fed, same GPU, TP=EP=4, PP=1, ISL=1000/OSL=100/BS=8.
        self.assertAlmostEqual(matches[0].distance, 1.7431028368992258)
        self.assertAlmostEqual(matches[1].distance, 2.90174931693607)

    def test_inventory_uses_measured_profiles_and_reference_backend_versions(
        self,
    ) -> None:
        """Filter support rows before loading models and cache the inventory."""
        valid = {
            "HuggingFaceID": "valid",
            "System": "h100_sxm",
            "Backend": "vllm",
            "Version": "0.22.0",
            "Mode": "agg",
            "Status": "PASS",
            "Source": "silicon",
        }
        rows = [
            valid,
            dict(valid),
            {**valid, "Mode": "disagg"},
            {**valid, "Status": "FAIL"},
            {**valid, "Source": "sol"},
            {**valid, "Version": "0.24.0"},
            {**valid, "System": "a100_sxm"},
            {**valid, "HuggingFaceID": "bad"},
        ]
        with (
            patch(
                "aiconfigurator.sdk.common.get_support_matrix", return_value=rows
            ) as matrix,
            patch(
                "aic_proxy.load_model_profile",
                side_effect=[_model(), ValueError("missing geometry")],
            ) as metadata,
            self.assertLogs("aic_proxy", level="WARNING"),
        ):
            profiles = load_supported_profiles("h100_sxm")
            self.assertIs(load_supported_profiles("h100_sxm"), profiles)
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0].backend_version, "0.22.0")
        self.assertEqual(metadata.call_count, 2)
        matrix.assert_called_once()

    def test_empty_inventory_and_invalid_target_topology(self) -> None:
        """An unavailable proxy stays unavailable without fabricated fallback."""
        self.assertEqual(_select(_model(), []), ())
        self.assertEqual(_select(_model(), [_supported("valid")], tp=3), ())
        with self.assertRaises(ValueError):
            _select(_model(), [], batch_size=0)

    def test_phase_factors_recompute_for_workload_and_keep_identity(self) -> None:
        """Use phase-specific candidate costs rather than one model-size scalar."""
        target = _model()
        proxy = replace(
            target,
            model_id="proxy",
            parameter_count=120e9,
            layers=64,
            weight_dtype="fp8",
        )
        kwargs = {
            "system": "h100_sxm",
            "tp": 4,
            "pp": 1,
            "input_tokens": 1000,
            "output_tokens": 100,
            "batch_size": 8,
        }
        with patch(
            "aic_proxy._system_performance", return_value=(989e12, 3350e9, 900e9)
        ):
            identical = phase_correction_factors(target, target, **kwargs)
            normal = phase_correction_factors(target, proxy, **kwargs)
            large_batch = phase_correction_factors(
                target, proxy, **{**kwargs, "batch_size": 256}
            )
        self.assertEqual(identical, (1.0, 1.0))
        self.assertAlmostEqual(normal[0], 1.2055210130407779)
        self.assertAlmostEqual(normal[1], 0.7529385722940809)
        self.assertNotEqual(normal[0], normal[1])
        self.assertNotEqual(normal, large_batch)
        self.assertTrue(all(0.05 <= factor <= 20 for factor in (*normal, *large_batch)))

    def test_proxy_correction_matches_example_without_reusing_replica_counts(
        self,
    ) -> None:
        """Correct per-replica rates while preserving candidate concurrency."""
        raw = {
            "ttft": 100,
            "tpot": 10,
            "tokens/s": 2000,
            "seq/s": 20,
            "num_total_gpus": 4,
            "bs": 8,
            "concurrency": 8,
            "replicas_needed": 2,
            "total_gpus_needed": 8,
            "tokens/s/gpu_cluster": 500,
        }
        corrected = correct_proxy_metrics(raw, 0.5, 0.8, 100)
        self.assertEqual(corrected["ttft"], 200)
        self.assertEqual(corrected["tpot"], 12.5)
        self.assertAlmostEqual(corrected["tokens/s"], 1517.2413793103449)
        self.assertAlmostEqual(corrected["seq/s"], 15.17241379310345)
        self.assertAlmostEqual(corrected["tokens/s/gpu"], 379.3103448275862)
        self.assertEqual(corrected["tokens/s/user"], 80)
        self.assertEqual(corrected["request_latency"], 1437.5)
        self.assertEqual(corrected["concurrency"], 8)
        self.assertEqual(corrected["bs"], 8)
        self.assertNotIn("replicas_needed", corrected)
        self.assertNotIn("total_gpus_needed", corrected)
        self.assertNotIn("tokens/s/gpu_cluster", corrected)
        self.assertEqual(raw["ttft"], 100)
        self.assertEqual(raw["replicas_needed"], 2)

    def test_corrections_reject_invalid_metrics_and_ratios(self) -> None:
        """Invalid estimates must not create deployment capacity."""
        row = {
            "ttft": 100,
            "tpot": 10,
            "tokens/s": 1000,
            "seq/s": 10,
            "num_total_gpus": 4,
        }
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    correct_proxy_metrics(row, value, 1, 100)
                with self.assertRaises(ValueError):
                    correct_proxy_metrics({**row, "tokens/s": value}, 1, 1, 100)

    def test_row_quantization_is_used_for_correction_geometry(self) -> None:
        """Checkpoint/profile names must not override the actual AIC row dtype."""
        model = _model()
        quantized = profile_for_aic_row(
            model, {"moe": "w4a16_mxfp4", "gemm": "bfloat16", "kvcache": "fp8"}
        )
        self.assertEqual(quantized.weight_dtype, "fp4")
        self.assertEqual(quantized.kv_dtype, "fp8")
        self.assertEqual(model.weight_dtype, "bf16")


def _model() -> ModelProfile:
    return model_profile_from_config(NEXT_MODEL, {**NEXT_CONFIG, "model_params_b": 80})


def _supported(model_id: str, **changes) -> SupportedProfile:
    return SupportedProfile(
        replace(_model(), model_id=model_id, **changes),
        "h100_sxm",
        "vllm",
        "0.22.0",
    )


def _select(target, profiles, **overrides):
    return select_proxy_profiles(
        target,
        profiles,
        **{
            "system": "h100_sxm",
            "backend": "vllm",
            "tp": 4,
            "pp": 1,
            "input_tokens": 1000,
            "output_tokens": 100,
            "batch_size": 8,
            **overrides,
        },
    )


if __name__ == "__main__":
    unittest.main()
