"""Maximum-VRAM placement method checks."""

import unittest
from dataclasses import replace
from pathlib import Path

from data_generation.controller import load_benchmark, scheduler_job
from data_generation.methods.vram.vram import (
    MODEL_MEMORY,
    build_node_parallel_placement,
    cache_bytes,
    estimate_max_vram,
    load_vram_method,
)
from data_generation.scheduling import DecisionFailureCategory, DecisionStatus

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIRECTORY = ROOT / "benchmarks" / "simulation_3h"
METHOD_SETTINGS: dict[str, object] = {
    "default_optimization_goal": "throughput",
}


class VramMethodTests(unittest.TestCase):
    """Check estimates, best-fit selection, and live capacity mutation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark = load_benchmark(BENCHMARK_DIRECTORY / "benchmark.yaml")
        cls.jobs = {job.job_id: job for job in cls.benchmark.jobs}

    def test_estimate_uses_actual_rate_max_tokens_and_headroom(self) -> None:
        job = scheduler_job(self.jobs["job-benchmark_timing_3hr-0-1"])

        estimate = estimate_max_vram(job)

        self.assertEqual(estimate.required_concurrency, 638)
        self.assertEqual(estimate.maximum_live_tokens, 1_850_200)
        self.assertAlmostEqual(estimate.weight_vram_gb, 2.46)
        self.assertAlmostEqual(estimate.kv_vram_gb, 60.6273536)
        self.assertAlmostEqual(estimate.required_vram_gb, 77.70482432)

    def test_cache_bytes_follow_each_vllm_architecture(self) -> None:
        encoder = MODEL_MEMORY["intfloat/multilingual-e5-large"]
        standard = MODEL_MEMORY["meta-llama/Llama-3.3-70B-Instruct"]
        mla = MODEL_MEMORY["deepseek-ai/DeepSeek-V3"]
        hybrid = MODEL_MEMORY["Qwen/Qwen3-Next-80B-A3B-Thinking"]

        self.assertEqual(cache_bytes(encoder.cache, 2), (0, 0))
        self.assertEqual(cache_bytes(standard.cache, 2), (327_680, 0))
        self.assertEqual(cache_bytes(mla.cache, 2), (70_272, 0))
        self.assertEqual(
            cache_bytes(hybrid.cache, 2),
            (24_576, 39_518_208),
        )

    def test_hybrid_model_adds_recurrent_state_per_request(self) -> None:
        job = scheduler_job(self.jobs["job-benchmark_timing_3hr-67-3"])

        estimate = estimate_max_vram(job)

        self.assertEqual(estimate.required_concurrency, 900)
        self.assertAlmostEqual(
            estimate.recurrent_state_vram_gb,
            35.5663872,
        )

    def test_parallelism_uses_dp_after_filling_one_node(self) -> None:
        devices, configuration = build_node_parallel_placement(15, 8)

        self.assertEqual(devices, 16)
        self.assertEqual(configuration, {"tp": 8, "pp": 1, "dp": 2})

    def test_small_encoder_selects_one_a10g(self) -> None:
        method = load_vram_method(BENCHMARK_DIRECTORY, METHOD_SETTINGS)
        job = scheduler_job(self.jobs["job-benchmark_timing_3hr-0-5"])

        decision = method.schedule(job)

        self.assertEqual(decision.status, DecisionStatus.ACCEPTED)
        self.assertEqual(decision.accelerator, "A10G-24GB")
        self.assertEqual(decision.devices, 1)
        self.assertEqual(decision.configuration["tp"], 1)

    def test_runtime_calculates_from_current_job_information(self) -> None:
        method = load_vram_method(BENCHMARK_DIRECTORY, METHOD_SETTINGS)
        original = scheduler_job(self.jobs["job-benchmark_timing_3hr-0-5"])
        llama = scheduler_job(self.jobs["job-benchmark_timing_3hr-0-2"])
        changed_job = replace(llama, job_id=original.job_id)

        decision = method.schedule(changed_job)

        self.assertEqual(decision.status, DecisionStatus.ACCEPTED)
        self.assertEqual(decision.accelerator, "MI300X-192GB")

    def test_runtime_falls_back_when_preferred_hardware_is_full(self) -> None:
        method = load_vram_method(BENCHMARK_DIRECTORY, METHOD_SETTINGS)
        encoder = scheduler_job(self.jobs["job-benchmark_timing_3hr-0-5"])
        for index in range(48):
            decision = method.schedule(replace(encoder, job_id=f"encoder-{index}"))
            self.assertEqual(decision.status, DecisionStatus.ACCEPTED)
            self.assertEqual(decision.accelerator, "A10G-24GB")

        fallback = method.schedule(replace(encoder, job_id="encoder-fallback"))

        self.assertEqual(fallback.status, DecisionStatus.ACCEPTED)
        self.assertEqual(fallback.accelerator, "A100-PCIe-40GB")

    def test_frontier_model_spills_over_to_data_parallelism(self) -> None:
        method = load_vram_method(BENCHMARK_DIRECTORY, METHOD_SETTINGS)
        job = scheduler_job(self.jobs["job-benchmark_timing_3hr-67-1"])

        decision = method.schedule(job)

        self.assertEqual(decision.status, DecisionStatus.ACCEPTED)
        self.assertEqual(decision.accelerator, "MI300X-192GB")
        self.assertEqual(decision.devices, 16)
        self.assertEqual(decision.market, "on_demand")
        self.assertEqual(decision.configuration["tp"], 8)
        self.assertEqual(decision.configuration["pp"], 1)
        self.assertEqual(decision.configuration["dp"], 2)

    def test_release_restores_reserved_capacity(self) -> None:
        method = load_vram_method(BENCHMARK_DIRECTORY, METHOD_SETTINGS)
        job = scheduler_job(self.jobs["job-benchmark_timing_3hr-0-5"])

        method.schedule(job)
        allocated = next(
            pool
            for pool in method.resource_snapshot()["pools"]
            if pool["pool_id"] == "a10g-24gb"
        )
        self.assertEqual(allocated["available_devices"], 47)

        method.release(job.job_id)
        released = next(
            pool
            for pool in method.resource_snapshot()["pools"]
            if pool["pool_id"] == "a10g-24gb"
        )
        self.assertEqual(released["available_devices"], 48)

    def test_live_capacity_can_reject_a_later_large_job(self) -> None:
        method = load_vram_method(BENCHMARK_DIRECTORY, METHOD_SETTINGS)
        kimi = scheduler_job(self.jobs["job-benchmark_timing_3hr-67-1"])
        deepseek = scheduler_job(self.jobs["job-benchmark_timing_3hr-145-2"])

        method.schedule(kimi)
        decision = method.schedule(deepseek)

        self.assertEqual(decision.status, DecisionStatus.REJECTED)
        self.assertEqual(
            decision.failure_category,
            DecisionFailureCategory.INSUFFICIENT_CAPACITY,
        )


if __name__ == "__main__":
    unittest.main()
