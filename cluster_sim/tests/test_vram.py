"""Test the workload-derived maximum-VRAM cluster simulator adapter."""

# ruff: noqa: E402

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster_sim"))

from models import OnlineJob, Scenario
from policy import load_config_policy
from serving import build_rank_cluster
from tandemn_system_data.ids import is_valid_id
from tandemn_system_data.models import Plan
from vram import (
    MODEL_ARCHITECTURES,
    VRAM_REQUIREMENTS_GIB,
    ModelArchitecture,
    ResourceTarget,
    StoreSnapshot,
    VramPlannerDriver,
    _place_action,
    calculate_max_vram_gib,
    generate_vram_requirements,
    model_architecture_from_huggingface,
)

_SERVINGSIM_HARDWARE = {
    "profile": "H100",
    "memory_gb": 80,
    "memory_bandwidth_gb_s": 3350,
    "link_bandwidth_gb_s": 900,
}


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
                "15min_real": 1,
                "simulation_12h": 37,
                "simulation_24h": 48,
                "simulation_3h": 25,
            },
        )
        for jobs in VRAM_REQUIREMENTS_GIB.values():
            self.assertTrue(all(required_gib > 0 for required_gib in jobs.values()))

    def test_checked_in_vram_map_matches_benchmarks(self) -> None:
        """Keep the checked-in artifact derived from executable benchmark YAML."""
        self.assertEqual(
            generate_vram_requirements(ROOT / "benchmarks"),
            VRAM_REQUIREMENTS_GIB,
        )

    def test_known_vram_results_match_benchmark_inputs(self) -> None:
        """Preserve generated encoder and frontier-model VRAM outcomes."""
        mappings = VRAM_REQUIREMENTS_GIB["simulation_3h"]
        small_model = mappings["job-benchmark_timing_3hr-0-5"]
        kimi = mappings["job-benchmark_timing_3hr-67-1"]
        deepseek = mappings["job-benchmark_timing_3hr-145-2"]

        self.assertEqual(small_model, 2.458)
        self.assertEqual(kimi, 972.772)
        self.assertEqual(deepseek, 794.59)
        self.assertEqual(
            MODEL_ARCHITECTURES["moonshotai/Kimi-K2-Instruct"].mla_cache_elements, 576
        )

    def test_fp8_weights_keep_runtime_memory_at_bf16(self) -> None:
        architecture = ModelArchitecture(
            parameter_count=2**30,
            hidden_size=4,
            intermediate_size=8,
            attention_layers=1,
            kv_heads=1,
            head_dim=1,
        )
        job = {
            "kind": "batch",
            "input_source": {"requests_per_shard": 2},
            "spec_json": {
                "job_features": {"isl_token_max": 3, "osl_token_max": 5},
                "precision": "bf16",
            },
        }
        fp8_job = {
            **job,
            "spec_json": {**job["spec_json"], "precision": "bf16-fp8"},
        }

        bf16 = calculate_max_vram_gib(architecture, job)
        fp8 = calculate_max_vram_gib(architecture, fp8_job)

        self.assertEqual(bf16 - fp8, 1.0)

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
        rank_config = first_tick[0]["ladder"][0]["config"]
        for field, value in _launch_config().items():
            self.assertEqual(rank_config[field], value)
        self.assertTrue(is_valid_id(first_tick[0]["ladder"][0]["rank_id"], "rank"))
        Plan(user_id="tenant", actions=[first_tick[0]])

    def test_snapshot_reads_each_relevant_model_catalog_once(self) -> None:
        """Freeze Store catalog records with the rest of the planning tick."""
        driver = _driver()
        requested = []
        waiting = _snapshot(
            "job-benchmark_timing_3hr-0-5",
            a10g_free=1,
            h100_free=0,
        ).waiting_jobs
        driver.job_store = SimpleNamespace(
            running_jobs=lambda _user_id: (),
            waiting_jobs=lambda _user_id: waiting,
            paused_jobs=lambda _user_id: (),
        )
        driver.resource_map_store = SimpleNamespace(
            get=lambda: SimpleNamespace(
                market="reserved",
                scheduling_summary=lambda: {},
            )
        )
        driver.model_catalog_store = SimpleNamespace(
            get=lambda model_id: (
                requested.append(model_id)
                or SimpleNamespace(
                    model_dump=lambda **_kwargs: _catalogs(model_id)[model_id]
                )
            )
        )

        snapshot = driver._read_store_snapshot()

        self.assertEqual(requested, ["intfloat/multilingual-e5-large"])
        self.assertEqual(
            snapshot.model_catalogs,
            _catalogs("intfloat/multilingual-e5-large"),
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

    def test_data_parallelism_creates_per_replica_rank_shapes(self) -> None:
        """Keep DP in replica count rather than one ServingSim rank shape."""
        store_job = _snapshot(
            "job",
            a10g_free=0,
            h100_free=4,
        ).waiting_jobs[0]
        action = _place_action(
            "job",
            store_job,
            ResourceTarget(
                environment_key="reserved|aws|us-east|use1-az1|H100",
                environment={},
                pool=None,
                gpu_type="H100",
                instance_type=None,
            ),
            1,
            1,
            4,
            _catalogs("intfloat/multilingual-e5-large"),
            load_config_policy(),
        )

        shape = action["ladder"][0]["config"]
        self.assertEqual(shape["count"], 1)
        self.assertEqual(shape["gpu_count"], 1)
        self.assertEqual(action["ladder"][0]["n_replicas"], 4)
        Plan(user_id="tenant", actions=[action])
        cluster = build_rank_cluster(
            SimpleNamespace(
                rank_name=action["ladder"][0]["rank_id"],
                model_name=shape["model_id"],
                shape=shape,
                chains=(object(),) * 4,
            ),
            _SERVINGSIM_HARDWARE,
        )
        self.assertEqual(cluster["nodes"][0]["instances"][0]["num_npus"], 1)
        self.assertEqual(cluster["nodes"][0]["num_instances"], 4)

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
            model_catalogs={},
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
        model_catalog_store=object(),
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
        model_catalogs=_catalogs(model_id),
    )


def _launch_config() -> dict[str, object]:
    """Return one resolved launch profile from Tandemn Store."""
    return {
        "engine_name": "vllm",
        "engine_version": "0.16.0",
        "weight_dtype": "float16",
        "weight_quantization_method": "none",
        "weight_quantization_bits": 16,
        "activation_dtype": "float16",
        "gpu_mem_util": 0.9,
        "max_model_len": 131072,
        "prefix_cache_enabled": False,
        "chunked_prefill_enable": True,
        "chunk_size": 100,
        "scheduling_policy": "fcfs",
        "preemption_policy": "recompute",
        "router_policy": "kv_router",
        "max_num_seq": 256,
        "max_num_batched_tokens": 8192,
        "block_size": 16,
        "kvcache_dtype": "auto",
    }


def _catalogs(model_id: str) -> dict[str, dict[str, object]]:
    """Return one raw Store catalog with exact target-GPU profiles."""
    catalog = _launch_config()
    for field in (
        "max_num_seq",
        "max_num_batched_tokens",
        "block_size",
        "kvcache_dtype",
    ):
        catalog[field] = [
            {"gpu_type": gpu_type, "value": catalog[field]}
            for gpu_type in ("A10G", "H100", "MI300X", "L40S")
        ]
    return {model_id: catalog}


if __name__ == "__main__":
    unittest.main()
