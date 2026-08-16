"""Test the workload-derived maximum-VRAM cluster simulator adapter."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import OnlineJob, Scenario
from policy import load_config_policy
from vram import (
    MODEL_ARCHITECTURES,
    VRAM_REQUIREMENTS_GIB,
    StoreSnapshot,
    VramPlannerDriver,
    generate_vram_requirements,
    model_architecture_from_huggingface,
)


class VramPlannerDriverTest(unittest.TestCase):
    """Verify generated requirements and capacity-aware placement."""

    def test_policy_normalizes_store_hardware_name(self) -> None:
        """Resolve a Store hardware label to the matching policy section."""
        rule = load_config_policy().rule_for(
            "H100-SXM-80GB",
            "meta-llama/Llama-3.3-70B-Instruct",
        )

        self.assertIsNotNone(rule)
        assert rule is not None
        self.assertEqual(rule.precision, "bf16")
        self.assertEqual(rule.allowed_tp, (2, 4, 8))

    def test_mapping_covers_all_benchmarks(self) -> None:
        """Keep one positive VRAM requirement for every benchmark job."""
        self.assertEqual(
            {name: len(jobs) for name, jobs in VRAM_REQUIREMENTS_GIB.items()},
            {
                "real_life_12h": 32,
                "real_life_24h": 41,
                "real_life_3h": 19,
                "real_3h_homogenous": 25,
                "simulation_12h": 37,
                "simulation_24h": 48,
                "simulation_3h": 25,
            },
        )
        for jobs in VRAM_REQUIREMENTS_GIB.values():
            self.assertTrue(all(required_gib > 0 for required_gib in jobs.values()))

    def test_simulation_3h_keeps_checked_in_vram_baseline(self) -> None:
        """Keep the approved VRAM baseline independent from benchmark drift."""
        requirements = VRAM_REQUIREMENTS_GIB["simulation_3h"]
        expected = {
            "job-benchmark_timing_3hr-0-1": 1064.314,
            "job-benchmark_timing_3hr-0-2": 1640.938,
            "job-benchmark_timing_3hr-0-3": 6591.789,
            "job-benchmark_timing_3hr-0-4": 807.966,
            "job-benchmark_timing_3hr-7-1": 16243.584,
            "job-benchmark_timing_3hr-7-3": 3932.206,
            "job-benchmark_timing_3hr-32-1": 19992.668,
            "job-benchmark_timing_3hr-32-2": 2071.983,
            "job-benchmark_timing_3hr-49-1": 11219.803,
            "job-benchmark_timing_3hr-49-2": 1935.896,
            "job-benchmark_timing_3hr-67-1": 2206.908,
            "job-benchmark_timing_3hr-67-2": 995.407,
            "job-benchmark_timing_3hr-67-3": 331.271,
            "job-benchmark_timing_3hr-100-2": 1743.919,
            "job-benchmark_timing_3hr-125-1": 702.026,
            "job-benchmark_timing_3hr-145-1": 1171.456,
        }

        self.assertEqual({key: requirements[key] for key in expected}, expected)

    def test_known_vram_results_match_benchmark_inputs(self) -> None:
        """Preserve generated encoder and frontier-model VRAM outcomes."""
        mappings = VRAM_REQUIREMENTS_GIB["simulation_3h"]
        encoder = mappings["job-benchmark_timing_3hr-0-5"]
        kimi = mappings["job-benchmark_timing_3hr-67-1"]

        self.assertEqual(encoder, 1.053)
        self.assertEqual(kimi, 1928.689)
        self.assertEqual(
            MODEL_ARCHITECTURES["moonshotai/Kimi-K2-Instruct"].mla_cache_elements, 576
        )

    def test_huggingface_mla_config_derives_cache_shape(self) -> None:
        """Derive DeepSeek-style MLA cache elements from public config fields."""
        architecture = model_architecture_from_huggingface(
            {
                "model_type": "deepseek_v3",
                "hidden_size": 7168,
                "intermediate_size": 18432,
                "num_hidden_layers": 61,
                "num_attention_heads": 128,
                "num_key_value_heads": 128,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
            },
            684_531_386_000,
        )

        self.assertEqual(architecture.mla_cache_elements, 576)
        self.assertEqual(architecture.head_dim, 56)

    def test_huggingface_hybrid_config_derives_linear_cache(self) -> None:
        """Derive Qwen3-Next full-attention and recurrent-cache dimensions."""
        architecture = model_architecture_from_huggingface(
            {
                "model_type": "qwen3_next",
                "hidden_size": 2048,
                "intermediate_size": 5120,
                "num_hidden_layers": 48,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "full_attention_interval": 4,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
            },
            81_324_862_720,
        )

        self.assertEqual(architecture.attention_layers, 12)
        self.assertEqual(architecture.linear_attention_layers, 36)
        self.assertEqual(architecture.linear_cache_elements, 557_056)

    def test_placement_uses_best_available_memory_fit(self) -> None:
        """Use another fitting GPU when the smallest accelerator is full."""
        driver = _driver()
        job_id = "job-benchmark_timing_3hr-0-5"

        first_tick = driver._build_actions(_snapshot(job_id, a10g_free=0, h100_free=8))
        second_tick = driver._build_actions(_snapshot(job_id, a10g_free=1, h100_free=8))

        self.assertEqual(first_tick[0]["type"], "place")
        self.assertEqual(first_tick[0]["ladder"][0]["config"]["gpu_type"], "H100")
        self.assertEqual(second_tick[0]["type"], "place")
        self.assertEqual(
            second_tick[0]["ladder"][0]["config"]["gpu_type"],
            "A10G",
        )

    def test_placement_minimizes_total_provisioned_vram(self) -> None:
        """Prefer seven 80 GiB GPUs over three 192 GiB GPUs."""
        driver = _driver(benchmark_name="real_life_12h")
        job_id = "job-benchmark_timing_real_12hr_192gpu-0-3"

        actions = driver._build_actions(
            _snapshot(
                job_id,
                a10g_free=0,
                h100_free=8,
                mi300x_free=8,
            )
        )

        config = actions[0]["ladder"][0]["config"]
        self.assertEqual(config["gpu_type"], "H100")
        self.assertEqual(config["gpu_count"], 7)
        self.assertEqual(config["tp"], 1)
        self.assertEqual(config["pp"], 7)
        self.assertEqual(config["dp"], 1)
        self.assertEqual(config["precision"], "fp16")

    def test_fresh_store_id_resolves_to_benchmark_requirement(self) -> None:
        """Look up requirements by benchmark identity after CLI ID replacement."""
        benchmark_job_id = "job-benchmark_timing_3hr-0-5"
        store_job_id = "job-fresh-store-id"
        scenario_job = OnlineJob(
            name=benchmark_job_id,
            store_job_id=store_job_id,
            model_name="intfloat/multilingual-e5-large",
            starts_at_ns=0,
            stops_at_ns=1,
            workload={},
        )
        driver = _driver((scenario_job,))

        actions = driver._build_actions(
            _snapshot(store_job_id, a10g_free=1, h100_free=0)
        )

        self.assertEqual(actions[0]["type"], "place")
        self.assertEqual(actions[0]["ladder"][0]["config"]["gpu_type"], "A10G")

    def test_terminal_jobs_produce_no_actions(self) -> None:
        """Keep terminal jobs out of future placement plans."""
        base = _snapshot(
            "job-benchmark_timing_3hr-0-5",
            a10g_free=1,
            h100_free=0,
        )
        snapshot = StoreSnapshot(
            resources=base.resources,
            running_jobs=(),
            waiting_jobs=(),
            paused_jobs=(),
            finished_jobs=({"job_id": "finished"},),
        )

        self.assertEqual(_driver()._build_actions(snapshot), [])

    def test_policy_skips_unsupported_model_hardware_pair(self) -> None:
        """Defer when capacity exists only on policy-incompatible hardware."""
        driver = _driver()
        job_id = "job-benchmark_timing_3hr-15-3"

        actions = driver._build_actions(
            _snapshot(
                job_id,
                a10g_free=0,
                h100_free=0,
                l40s_free=8,
                model_id="microsoft/phi-4",
            )
        )

        self.assertEqual(actions, [{"job_id": job_id, "type": "defer"}])

    def test_policy_selects_allowed_tp_and_precision(self) -> None:
        """Emit policy-approved TP and precision for the selected hardware."""
        driver = _driver()
        job_id = "job-benchmark_timing_3hr-67-3"

        actions = driver._build_actions(
            _snapshot(
                job_id,
                a10g_free=0,
                h100_free=8,
                model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",
            )
        )

        config = actions[0]["ladder"][0]["config"]
        self.assertEqual(config["gpu_type"], "H100")
        self.assertEqual(config["tp"], 4)
        self.assertEqual(config["pp"], 1)
        self.assertEqual(config["dp"], 1)
        self.assertEqual(config["precision"], "bf16")


def _driver(
    jobs: tuple[OnlineJob, ...] = (),
    benchmark_name: str = "simulation_3h",
) -> VramPlannerDriver:
    """Build a VRAM driver without external Store clients."""
    scenario = Scenario(benchmark_name, 1, 1, jobs, ())
    return VramPlannerDriver(
        user_id="user",
        scenario=scenario,
        job_store=object(),
        resource_map_store=object(),
        plan_store=object(),
        backend="vllm",
        config_policy=load_config_policy(),
    )


def _snapshot(
    job_id: str,
    a10g_free: int,
    h100_free: int,
    mi300x_free: int = 0,
    l40s_free: int = 0,
    model_id: str = "intfloat/multilingual-e5-large",
) -> StoreSnapshot:
    """Create one tick snapshot with independently controlled hardware."""
    return StoreSnapshot(
        resources={
            "reserved|aws|us-east|use1-az1|A10G": {
                "gpu_type": "A10G",
                "total": a10g_free,
                "free": a10g_free,
                "pools": [],
            },
            "reserved|aws|us-east|use1-az1|H100": {
                "gpu_type": "H100",
                "total": h100_free,
                "free": h100_free,
                "pools": [],
            },
            "reserved|aws|us-east|use1-az1|MI300X": {
                "gpu_type": "MI300X",
                "total": mi300x_free,
                "free": mi300x_free,
                "pools": [],
            },
            "reserved|aws|us-east|use1-az1|L40S": {
                "gpu_type": "L40S",
                "total": l40s_free,
                "free": l40s_free,
                "pools": [],
            },
        },
        running_jobs=(),
        waiting_jobs=(
            {
                "job_id": job_id,
                "user_id": "tenant",
                "kind": "online",
                "spec_json": {
                    "model_id": model_id,
                    "job_features": {
                        "target_p99_ttft_ms": 120,
                        "target_p99_tpot_ms": 1,
                    },
                },
            },
        ),
        paused_jobs=(),
        finished_jobs=(),
    )


if __name__ == "__main__":
    unittest.main()
