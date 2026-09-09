# Real 100-GPU benchmark

Benchmark ID: `real100gpubenchmark`

This executable real-hardware benchmark runs a 180-minute workload followed by a 30-minute drain on an exact 120-GPU, 16-node reserved fleet. No on-demand capacity is available.

## Fleet

| Accelerator | Node shape | Nodes | Region | Total GPUs |
|---|---:|---:|---|---:|
| `A100-40GB` | 8 x 40 GB | 2 | `us-central1` | 16 |
| `RTXPRO6000` | 4 x 96 GB | 2 | `us-central1` | 8 |
| `H100` | 8 x 80 GB | 4 | `europe-west2` | 32 |
| `A100-80GB` | 8 x 80 GB | 2 | `us-east5` | 16 |
| `RTXPRO6000` | 8 x 96 GB | 6 | `us-east5` | 48 |
| **Total** |  | **16** |  | **120** |

## Workload

The manifest contains 15 online jobs and 5 batch campaigns. The selected batch campaigns are Mini Nightly Eval, Phi-4 Quant Probe, Calibration Probe, Sales Reports, and the Qwen3-235B large batch, totaling 126,200 records across 306 shards. Their definitions are unchanged from `simulation_3h` except for the added Qwen3-235B campaign; the external pipeline owns batch dataset staging.

Existing online traffic retains the original `simulation_3h` phase rates and concurrency caps with fixed rates; the two Qwen3-235B workloads add 40 concurrent streams. The maximum simultaneous online declaration is 432 streams at T+100. Product Launch Chat and Product Launch Moderation run through T+109, while Legal/Finance Review runs from T+135 through T+180, so cold starts do not consume their measurement windows. Online traffic stops at minute 180 and in-flight work drains through the hard stop at minute 210.

## Model Allowlist

- `meta-llama/Llama-3.2-1B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `microsoft/phi-4`
- `Qwen/Qwen2.5-Coder-32B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`
- `Qwen/Qwen3-Next-80B-A3B-Thinking`
- `zai-org/GLM-4.5-Air`
- `QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ`

`QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ` is used by the Qwen3-235B batch and online campaigns.

## Difficulty Contract

All 120 GPUs remain schedulable. The T+32, T+49, T+67, T+100, T+125, and T+135 arrival waves create persistent code, product-launch, high-priority safety, and Qwen3 placement pressure. Completion of every batch campaign before its deadline is not guaranteed. Sales Reports intentionally creates late deadline pressure. A plan-only preflight is required before a live run to confirm admission contention on the exact fleet.

## Workload Overview (JSON)

```json
{
  "benchmark_id": "real100gpubenchmark",
  "workload_duration_minutes": 180,
  "drain_minutes": 30,
  "workloads": [
    {
      "job_id": "job-benchmark_timing_3hr-85-1-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/batch-calibration-probe.yaml",
      "kind": "batch",
      "driver": "external_mock_data",
      "user_id": "tenant_koi_profiling",
      "model": "microsoft/phi-4",
      "precision": "fp16",
      "start_at_offset_minutes": 85,
      "priority_class": "LOW",
      "input_tokens": {
        "avg": 1400,
        "min": 350,
        "max": 4096
      },
      "output_tokens": {
        "avg": 512,
        "min": 128,
        "max": 1600
      },
      "deadline_at_offset_minutes": 170,
      "hard_stop_at_offset_minutes": 210,
      "campaign_id": "campaign-benchmark_timing_3hr-85-1",
      "completion_policy": "stop_when_complete",
      "request_pattern": {
        "type": "Constant",
        "peak_to_mean_ratio": 1
      },
      "workload_features": {
        "peak_to_mean_ratio": 1,
        "prefix_concentration": 0,
        "multi_turn_ratio": 0,
        "shared_prefix_length_avg": 1,
        "session_affinity": false,
        "total_token_budget": 11472000
      },
      "dataset": {
        "id": "calibration_probe",
        "record_count": 6000,
        "seed": 204,
        "shard_count": 12,
        "requests_per_shard": 500,
        "max_in_flight_requests": 16
      }
    },
    {
      "job_id": "job-benchmark_timing_3hr-15-1-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/batch-mini-nightly-eval.yaml",
      "kind": "batch",
      "driver": "external_mock_data",
      "user_id": "tenant_eval_platform",
      "model": "meta-llama/Llama-3.3-70B-Instruct",
      "precision": "fp16",
      "start_at_offset_minutes": 15,
      "priority_class": "STANDARD",
      "input_tokens": {
        "avg": 1800,
        "min": 450,
        "max": 6000
      },
      "output_tokens": {
        "avg": 1200,
        "min": 300,
        "max": 4000
      },
      "deadline_at_offset_minutes": 150,
      "hard_stop_at_offset_minutes": 210,
      "campaign_id": "campaign-benchmark_timing_3hr-15-1",
      "completion_policy": "stop_when_complete",
      "request_pattern": {
        "type": "Constant",
        "peak_to_mean_ratio": 1
      },
      "workload_features": {
        "peak_to_mean_ratio": 1,
        "prefix_concentration": 0,
        "multi_turn_ratio": 0,
        "shared_prefix_length_avg": 1,
        "session_affinity": false,
        "total_token_budget": 21600000
      },
      "dataset": {
        "id": "mini_nightly_eval",
        "record_count": 7200,
        "seed": 201,
        "shard_count": 36,
        "requests_per_shard": 200,
        "max_in_flight_requests": 4
      }
    },
    {
      "job_id": "job-benchmark_timing_3hr-15-3-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/batch-phi4-quant-probe.yaml",
      "kind": "batch",
      "driver": "external_mock_data",
      "user_id": "tenant_koi_profiling",
      "model": "microsoft/phi-4",
      "precision": "fp16",
      "start_at_offset_minutes": 15,
      "priority_class": "LOW",
      "input_tokens": {
        "avg": 1000,
        "min": 250,
        "max": 4096
      },
      "output_tokens": {
        "avg": 512,
        "min": 128,
        "max": 2048
      },
      "deadline_at_offset_minutes": 120,
      "hard_stop_at_offset_minutes": 210,
      "campaign_id": "campaign-benchmark_timing_3hr-15-3",
      "completion_policy": "stop_when_complete",
      "request_pattern": {
        "type": "Constant",
        "peak_to_mean_ratio": 1
      },
      "workload_features": {
        "peak_to_mean_ratio": 1,
        "prefix_concentration": 0,
        "multi_turn_ratio": 0,
        "shared_prefix_length_avg": 1,
        "session_affinity": false,
        "total_token_budget": 13608000
      },
      "dataset": {
        "id": "quant_probe_phi4",
        "record_count": 9000,
        "seed": 203,
        "shard_count": 18,
        "requests_per_shard": 500,
        "max_in_flight_requests": 16
      }
    },
    {
      "job_id": "job-benchmark_timing_3hr-0-5-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/batch-qwen3-235b-large.yaml",
      "kind": "batch",
      "driver": "external_mock_data",
      "user_id": "tenant_large_model_eval",
      "model": "QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ",
      "precision": "fp16",
      "start_at_offset_minutes": 0,
      "priority_class": "LOW",
      "input_tokens": {
        "avg": 4000,
        "min": 1000,
        "max": 12000
      },
      "output_tokens": {
        "avg": 1200,
        "min": 300,
        "max": 4000
      },
      "deadline_at_offset_minutes": 180,
      "hard_stop_at_offset_minutes": 210,
      "campaign_id": "campaign-benchmark_timing_3hr-0-5",
      "completion_policy": "stop_when_complete",
      "request_pattern": {
        "type": "Constant",
        "peak_to_mean_ratio": 1
      },
      "workload_features": {
        "peak_to_mean_ratio": 1,
        "prefix_concentration": 0,
        "multi_turn_ratio": 0,
        "shared_prefix_length_avg": 1,
        "session_affinity": false,
        "total_token_budget": 520000000
      },
      "dataset": {
        "id": "qwen3_235b_large_batch",
        "record_count": 100000,
        "seed": 209,
        "shard_count": 200,
        "requests_per_shard": 500,
        "max_in_flight_requests": 64
      }
    },
    {
      "job_id": "job-benchmark_timing_3hr-125-2-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/batch-sales-reports.yaml",
      "kind": "batch",
      "driver": "external_mock_data",
      "user_id": "tenant_reports",
      "model": "Qwen/Qwen3-Next-80B-A3B-Thinking",
      "precision": "fp16",
      "start_at_offset_minutes": 125,
      "priority_class": "LOW",
      "input_tokens": {
        "avg": 18000,
        "min": 4500,
        "max": 64000
      },
      "output_tokens": {
        "avg": 1400,
        "min": 350,
        "max": 5000
      },
      "deadline_at_offset_minutes": 180,
      "hard_stop_at_offset_minutes": 210,
      "campaign_id": "campaign-benchmark_timing_3hr-125-2",
      "completion_policy": "stop_when_complete",
      "request_pattern": {
        "type": "Constant",
        "peak_to_mean_ratio": 1
      },
      "workload_features": {
        "peak_to_mean_ratio": 1,
        "prefix_concentration": 0,
        "multi_turn_ratio": 0,
        "shared_prefix_length_avg": 1,
        "session_affinity": false,
        "total_token_budget": 77600000
      },
      "dataset": {
        "id": "sales_reports_short",
        "record_count": 4000,
        "seed": 206,
        "shard_count": 40,
        "requests_per_shard": 100,
        "max_in_flight_requests": 64
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-0-1-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/chat-free-core.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_chat_free",
      "model": "meta-llama/Llama-3.2-1B-Instruct",
      "precision": "fp16",
      "start_at_offset_minutes": 0,
      "priority_class": "LOW",
      "input_tokens": {
        "avg": 900,
        "min": 225,
        "max": 2200
      },
      "output_tokens": {
        "avg": 300,
        "min": 75,
        "max": 700
      },
      "stop_at_offset_minutes": 180,
      "operation": "base_service",
      "request_pattern": {
        "type": "Diurnal",
        "peak_to_mean_ratio": 2,
        "rate_requests_per_second": 2.6,
        "phase_type": "gamma",
        "smoothness": 0.894,
        "concurrency": 64,
        "duration_seconds": 10800
      },
      "workload_features": {
        "prefix_concentration": 0.5,
        "multi_turn_ratio": 0.7,
        "shared_prefix_length_avg": 225,
        "session_affinity": true
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-0-2-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/chat-premium-core.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_chat_premium",
      "model": "meta-llama/Llama-3.3-70B-Instruct",
      "precision": "bf16",
      "start_at_offset_minutes": 0,
      "priority_class": "HIGH",
      "input_tokens": {
        "avg": 1200,
        "min": 300,
        "max": 4096
      },
      "output_tokens": {
        "avg": 500,
        "min": 125,
        "max": 1400
      },
      "stop_at_offset_minutes": 180,
      "operation": "base_service",
      "request_pattern": {
        "type": "Diurnal",
        "peak_to_mean_ratio": 2,
        "rate_requests_per_second": 0.85,
        "phase_type": "gamma",
        "smoothness": 0.183,
        "concurrency": 24,
        "duration_seconds": 10800
      },
      "workload_features": {
        "prefix_concentration": 0.5,
        "multi_turn_ratio": 0.7,
        "shared_prefix_length_avg": 300,
        "session_affinity": true
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-0-3-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/code-autocomplete-core.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_code_autocomplete",
      "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
      "precision": "fp16",
      "start_at_offset_minutes": 0,
      "priority_class": "STANDARD",
      "input_tokens": {
        "avg": 1800,
        "min": 450,
        "max": 5000
      },
      "output_tokens": {
        "avg": 120,
        "min": 30,
        "max": 350
      },
      "stop_at_offset_minutes": 180,
      "operation": "base_service",
      "request_pattern": {
        "type": "Bursty",
        "peak_to_mean_ratio": 3.5,
        "rate_requests_per_second": 4.0,
        "phase_type": "gamma",
        "smoothness": 0.974,
        "concurrency": 32,
        "duration_seconds": 10800
      },
      "workload_features": {
        "prefix_concentration": 0,
        "multi_turn_ratio": 0,
        "shared_prefix_length_avg": 1,
        "session_affinity": false
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-32-2-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/enterprise-chat.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_enterprise_chat",
      "model": "microsoft/phi-4",
      "precision": "fp16",
      "start_at_offset_minutes": 32,
      "priority_class": "STANDARD",
      "input_tokens": {
        "avg": 1500,
        "min": 375,
        "max": 5500
      },
      "output_tokens": {
        "avg": 420,
        "min": 105,
        "max": 1200
      },
      "stop_at_offset_minutes": 152,
      "operation": "base_service",
      "request_pattern": {
        "type": "Diurnal",
        "peak_to_mean_ratio": 2,
        "rate_requests_per_second": 0.9,
        "phase_type": "gamma",
        "smoothness": 0.733,
        "concurrency": 24,
        "duration_seconds": 7200
      },
      "workload_features": {
        "prefix_concentration": 0.5,
        "multi_turn_ratio": 0.7,
        "shared_prefix_length_avg": 375,
        "session_affinity": true
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-67-2-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/frontier-reasoning-glm.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_frontier_glm",
      "model": "zai-org/GLM-4.5-Air",
      "precision": "bf16",
      "start_at_offset_minutes": 67,
      "priority_class": "STANDARD",
      "input_tokens": {
        "avg": 1800,
        "min": 450,
        "max": 6000
      },
      "output_tokens": {
        "avg": 2500,
        "min": 625,
        "max": 9000
      },
      "stop_at_offset_minutes": 142,
      "operation": "base_service",
      "request_pattern": {
        "type": "Bursty",
        "peak_to_mean_ratio": 5,
        "rate_requests_per_second": 0.08,
        "phase_type": "gamma",
        "smoothness": 0.527,
        "concurrency": 16,
        "duration_seconds": 4500
      },
      "workload_features": {
        "prefix_concentration": 0,
        "multi_turn_ratio": 0,
        "shared_prefix_length_avg": 1,
        "session_affinity": false
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-135-1-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/legal-finance-review.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_legal_finance",
      "model": "meta-llama/Llama-3.3-70B-Instruct",
      "precision": "bf16",
      "start_at_offset_minutes": 135,
      "priority_class": "HIGH",
      "input_tokens": {
        "avg": 3000,
        "min": 750,
        "max": 12000
      },
      "output_tokens": {
        "avg": 700,
        "min": 175,
        "max": 2200
      },
      "stop_at_offset_minutes": 180,
      "operation": "base_service",
      "request_pattern": {
        "type": "Diurnal",
        "peak_to_mean_ratio": 2,
        "rate_requests_per_second": 0.34,
        "phase_type": "gamma",
        "smoothness": 1.431,
        "concurrency": 16,
        "duration_seconds": 2700
      },
      "workload_features": {
        "prefix_concentration": 0,
        "multi_turn_ratio": 0,
        "shared_prefix_length_avg": 1,
        "session_affinity": false
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-125-1-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/meeting-summary.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_meeting_summary",
      "model": "Qwen/Qwen3-Next-80B-A3B-Thinking",
      "precision": "bf16",
      "start_at_offset_minutes": 125,
      "priority_class": "STANDARD",
      "input_tokens": {
        "avg": 24000,
        "min": 6000,
        "max": 128000
      },
      "output_tokens": {
        "avg": 1200,
        "min": 300,
        "max": 4000
      },
      "stop_at_offset_minutes": 170,
      "operation": "base_service",
      "request_pattern": {
        "type": "Bursty",
        "peak_to_mean_ratio": 3.5,
        "rate_requests_per_second": 0.06,
        "phase_type": "gamma",
        "smoothness": 0.001,
        "concurrency": 8,
        "duration_seconds": 2700
      },
      "workload_features": {
        "prefix_concentration": 0,
        "multi_turn_ratio": 0,
        "shared_prefix_length_avg": 1,
        "session_affinity": false
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-67-3-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/moe-routing-qwen3.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_moe_routing",
      "model": "Qwen/Qwen3-Next-80B-A3B-Thinking",
      "precision": "bf16",
      "start_at_offset_minutes": 67,
      "priority_class": "STANDARD",
      "input_tokens": {
        "avg": 1200,
        "min": 300,
        "max": 4096
      },
      "output_tokens": {
        "avg": 900,
        "min": 225,
        "max": 3000
      },
      "stop_at_offset_minutes": 142,
      "operation": "base_service",
      "request_pattern": {
        "type": "Bursty",
        "peak_to_mean_ratio": 5,
        "rate_requests_per_second": 0.26,
        "phase_type": "gamma",
        "smoothness": 0.917,
        "concurrency": 16,
        "duration_seconds": 4500
      },
      "workload_features": {
        "prefix_concentration": 0,
        "multi_turn_ratio": 0,
        "shared_prefix_length_avg": 1,
        "session_affinity": false
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-7-1-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/phi4-long-context-chat.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_phi4_long_context_chat",
      "model": "microsoft/phi-4",
      "precision": "fp16",
      "start_at_offset_minutes": 7,
      "priority_class": "STANDARD",
      "input_tokens": {
        "avg": 9000,
        "min": 2250,
        "max": 15684
      },
      "output_tokens": {
        "avg": 220,
        "min": 55,
        "max": 700
      },
      "stop_at_offset_minutes": 180,
      "operation": "base_service",
      "request_pattern": {
        "type": "Poisson",
        "peak_to_mean_ratio": 2,
        "rate_requests_per_second": 1.2,
        "phase_type": "poisson",
        "smoothness": null,
        "concurrency": 24,
        "duration_seconds": 10380
      },
      "workload_features": {
        "prefix_concentration": 0.5,
        "multi_turn_ratio": 0.3,
        "shared_prefix_length_avg": 2250,
        "session_affinity": true
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-49-1-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/product-launch-chat.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_chat_free",
      "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
      "precision": "fp16",
      "start_at_offset_minutes": 49,
      "priority_class": "STANDARD",
      "input_tokens": {
        "avg": 850,
        "min": 212,
        "max": 2200
      },
      "output_tokens": {
        "avg": 280,
        "min": 70,
        "max": 650
      },
      "stop_at_offset_minutes": 109,
      "operation": "additive_stream",
      "request_pattern": {
        "type": "Bursty",
        "peak_to_mean_ratio": 3.5,
        "rate_requests_per_second": 3.0,
        "phase_type": "gamma",
        "smoothness": 0.11,
        "concurrency": 64,
        "duration_seconds": 3600
      },
      "workload_features": {
        "prefix_concentration": 0.5,
        "multi_turn_ratio": 0.7,
        "shared_prefix_length_avg": 212,
        "session_affinity": true
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-49-2-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/product-launch-moderation.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_safety_moderation",
      "model": "meta-llama/Llama-3.2-1B-Instruct",
      "precision": "fp16",
      "start_at_offset_minutes": 49,
      "priority_class": "HIGH",
      "input_tokens": {
        "avg": 320,
        "min": 80,
        "max": 1024
      },
      "output_tokens": {
        "avg": 8,
        "min": 2,
        "max": 32
      },
      "stop_at_offset_minutes": 109,
      "operation": "base_service",
      "request_pattern": {
        "type": "Bursty",
        "peak_to_mean_ratio": 3.5,
        "rate_requests_per_second": 28.0,
        "phase_type": "gamma",
        "smoothness": 0.11,
        "concurrency": 32,
        "duration_seconds": 3600
      },
      "workload_features": {
        "prefix_concentration": 0,
        "multi_turn_ratio": 0,
        "shared_prefix_length_avg": 1,
        "session_affinity": false
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-0-7-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/qwen3-235b-online-bursty.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_qwen3_online_bursty",
      "model": "QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ",
      "precision": "fp16",
      "start_at_offset_minutes": 0,
      "priority_class": "STANDARD",
      "input_tokens": {
        "avg": 1800,
        "min": 450,
        "max": 6000
      },
      "output_tokens": {
        "avg": 600,
        "min": 150,
        "max": 2400
      },
      "stop_at_offset_minutes": 180,
      "operation": "base_service",
      "request_pattern": {
        "type": "Bursty",
        "peak_to_mean_ratio": 4,
        "rate_requests_per_second": 0.24,
        "phase_type": "gamma",
        "smoothness": 0.25,
        "concurrency": 16,
        "duration_seconds": 10800
      },
      "workload_features": {
        "prefix_concentration": 0.1,
        "multi_turn_ratio": 0,
        "shared_prefix_length_avg": 1,
        "session_affinity": false
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-0-6-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/qwen3-235b-online-core.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_qwen3_online_core",
      "model": "QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ",
      "precision": "fp16",
      "start_at_offset_minutes": 0,
      "priority_class": "STANDARD",
      "input_tokens": {
        "avg": 3200,
        "min": 800,
        "max": 10000
      },
      "output_tokens": {
        "avg": 800,
        "min": 200,
        "max": 3000
      },
      "stop_at_offset_minutes": 180,
      "operation": "base_service",
      "request_pattern": {
        "type": "Diurnal",
        "peak_to_mean_ratio": 2,
        "rate_requests_per_second": 0.18,
        "phase_type": "gamma",
        "smoothness": 0.5,
        "concurrency": 24,
        "duration_seconds": 10800
      },
      "workload_features": {
        "prefix_concentration": 0.35,
        "multi_turn_ratio": 0.3,
        "shared_prefix_length_avg": 400,
        "session_affinity": true
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-100-2-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/safety-p0-live.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_safety_reasoner",
      "model": "microsoft/phi-4",
      "precision": "fp16",
      "start_at_offset_minutes": 100,
      "priority_class": "HIGH",
      "input_tokens": {
        "avg": 900,
        "min": 225,
        "max": 3000
      },
      "output_tokens": {
        "avg": 160,
        "min": 40,
        "max": 600
      },
      "stop_at_offset_minutes": 150,
      "operation": "base_service",
      "request_pattern": {
        "type": "Bursty",
        "peak_to_mean_ratio": 3,
        "rate_requests_per_second": 3.2,
        "phase_type": "gamma",
        "smoothness": 0.002,
        "concurrency": 32,
        "duration_seconds": 3000
      },
      "workload_features": {
        "prefix_concentration": 0,
        "multi_turn_ratio": 0,
        "shared_prefix_length_avg": 1,
        "session_affinity": false
      }
    },
    {
      "job_id": "job-benchmark_timing_real100gpu-7-3-parallel",
      "file": "benchmarks/real100gpubenchmark/jobs/support-assistant.yaml",
      "kind": "online",
      "driver": "aiperf",
      "user_id": "tenant_support",
      "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
      "precision": "fp16",
      "start_at_offset_minutes": 7,
      "priority_class": "STANDARD",
      "input_tokens": {
        "avg": 1400,
        "min": 350,
        "max": 6000
      },
      "output_tokens": {
        "avg": 350,
        "min": 87,
        "max": 1000
      },
      "stop_at_offset_minutes": 180,
      "operation": "base_service",
      "request_pattern": {
        "type": "Bursty",
        "peak_to_mean_ratio": 3.5,
        "rate_requests_per_second": 2.4,
        "phase_type": "gamma",
        "smoothness": 0.388,
        "concurrency": 64,
        "duration_seconds": 10380
      },
      "workload_features": {
        "prefix_concentration": 0.5,
        "multi_turn_ratio": 0.3,
        "shared_prefix_length_avg": 350,
        "session_affinity": true
      }
    }
  ]
}
```
