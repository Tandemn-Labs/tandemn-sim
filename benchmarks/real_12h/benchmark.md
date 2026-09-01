# Real-life 12-hour benchmark

Benchmark ID: `real_12h`

Benchmark name: `KoiBench-Real-12h-100GPU-Mid`

Purpose: synthetic job replay on a real-cluster mid-length benchmark for the 100 GPU fleet. This run is long enough to show persistent APIs, regional ramps, batch deadlines, frontier arrivals, quantization choices, LMCache behavior, and low-churn adaptation.

Horizon: 720 minutes

Tick: 5 minutes

Engine: `vllm` behind Dynamo-style serving

Markets: `reserved`, `on_demand`

Serving mode: aggregate serving with LMCache, no prefill/decode disaggregation

Real fleet target: 100 GPUs



## Canonical Lifecycle Contract

Schema: `koibench/v1`

Dispatch: releases occur at their exact offset. A job arrival triggers the admission path immediately; the 5-minute tick remains the slow reoptimization loop.

Online lifecycle: `active_start_at_offset_minutes` and `active_end_at_offset_minutes` define a half-open traffic window. `duration_minutes` is retained only as a consistency check. At the end offset the load generator stops new requests, drains in-flight requests, and the executor records `terminal_at`.

Batch lifecycle: `deadline_at_offset_minutes` is an absolute offset from benchmark start. `shard_count` is expanded deterministically because `expansion_policy` is `required`. `requests_per_shard` fixes campaign size. Runtime `started_at`, `completed_at`, and `terminal_at` are produced by the executor, not authored in the job definition.

Overlap: `base_service` remains active, `additive_stream` adds demand to an existing service, and `replace_service` identifies a model/service replacement. Overlap is intentional and additive unless explicitly marked as replacement.

Planner fairness: descriptions, trace names, human-readable tenant labels, and `input_source` are not exposed to planners. Every planner receives the same sanitized job features. Hardware placement and market preferences are deliberately absent from job inputs.

Traffic: `max_concurrent_streaming` is the planner-facing service cap. Client submission rates may be fixed in `input_source`; they remain excluded from the Orca payload so Orca stays in concurrency mode. Patterns are limited to `Poisson`, `Constant`, `Bursty`, and `Diurnal`.

Input modes: every online job uses `api_load_generator` to ping an OpenAI-compatible endpoint under its concurrency profile. Every batch campaign uses a finite `dataset` partitioned into required shards.

Drain: after minute 720, no new work arrives. A 60-minute drain period records batch tardiness and in-flight completion; work unfinished at its deadline is a miss.

Exact manifest counts: **16 online jobs**, **12 batch campaigns**, **678 expanded shards**.

## Real-Run Capacity Gate

The listed concurrency caps are frozen only after an isolated preflight on the exact engine/model build. Apply one global concurrency multiplier per run so normal offered load consumes 55-65% of measured qualified capacity, business peaks consume 80-90%, and the designated overload burst reaches 105-120%. Freeze that multiplier before comparing systems.

Online concurrency is an in-flight request ceiling, not a user-population count. The workload-specific caps preserve headroom for bursty arrivals and long responses without overstating placement demand. At the overlap peak, the benchmark declares at most **560 online streams** and approximately **88.1 mean requests/s**.

## Frozen Real Cluster Resources

Maximum provisioned capacity: **100 GPUs**. Reserved capacity is pre-provisioned; on-demand capacity is a hard launch quota.

| Accelerator | Node/Slice Shape | Nodes/Slices | Reserved | On-Demand Cap | Total |
|---|---:|---:|---:|---:|---:|
| `H100-SXM-80GB` | 8 GPUs, NVSwitch | 4 | 24 | 8 | 32 |
| `A100-SXM-80GB` | 8 GPUs, NVLink | 3 | 16 | 8 | 24 |
| `A100-PCIe-40GB` | 4 GPUs, PCIe | 1 | 4 | 0 | 4 |
| `RTXPRO6000` | 8 GPUs, PCIe | 3 | 16 | 8 | 24 |
| `L4-24GB` | 8 GPUs, PCIe | 2 | 8 | 8 | 16 |
| **Total** |  | **13** | **68** | **32** | **100** |

| Accelerator | `us-east` | `eu-west` | `apac` | Total |
|---|---:|---:|---:|---:|
| H100 | 24 | 0 | 8 | 32 |
| A100-SXM | 0 | 24 | 0 | 24 |
| A100-PCIe | 0 | 4 | 0 | 4 |
| RTXPRO6000 | 8 | 8 | 8 | 24 |
| L4 | 8 | 0 | 8 | 16 |
| **Region total** | **40** | **36** | **24** | **100** |

| Region | Reserved Devices | On-Demand Cap | Total Devices |
|---|---:|---:|---:|
| `us-east` | 32 | 8 | 40 |
| `eu-west` | 28 | 8 | 36 |
| `apac` | 8 | 16 | 24 |

Network caps: 400 Gbps RDMA for H100/A100-SXM nodes and 100 Gbps TCP/IP for A100-PCIe/RTX PRO 6000/L4 nodes. Inter-region caps are 25 Gbps/75 ms RTT (`us-east<->eu-west`), 20 Gbps/145 ms (`us-east<->apac`), and 15 Gbps/220 ms (`eu-west<->apac`). Inferentia and Trainium are not allocated in this real manifest.

## T+00:00 - Always-On APIs

```json
{
  "submit_at_offset_minutes": 0,
  "description": "persistent APIs warm across regions",
  "jobs": [
    {
      "user_id": "tenant_chat_free",
      "kind": "online",
      "spec_json": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "job_features": {
          "isl_token_avg": 900,
          "isl_token_min": 225,
          "isl_token_max": 2200,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 300,
          "osl_token_min": 75,
          "osl_token_max": 700,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Diurnal",
          "peak_to_mean_ratio": 2,
          "workload_prefix_concentration": 0.5,
          "multi_turn_ratio": 0.7,
          "shared_prefix_length_avg": 225,
          "is_session_affinity": true,
          "priority_class": "STANDARD",
          "max_concurrent_streaming": 64,
          "target_p99_ttft_ms": 700,
          "target_p99_tpot_ms": 55
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_free_chat_us",
        "request_arrival_pattern": "Diurnal",
        "max_concurrent_streaming": 64,
        "seed": 101
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-0-1",
      "schedule": {
        "release_at_offset_minutes": 0,
        "active_end_at_offset_minutes": 720,
        "duration_minutes": 720,
        "operation": "base_service",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-0-1"
      }
    },
    {
      "user_id": "tenant_chat_free",
      "kind": "online",
      "spec_json": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "job_features": {
          "isl_token_avg": 900,
          "isl_token_min": 225,
          "isl_token_max": 2200,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 300,
          "osl_token_min": 75,
          "osl_token_max": 700,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Diurnal",
          "peak_to_mean_ratio": 2,
          "workload_prefix_concentration": 0.5,
          "multi_turn_ratio": 0.7,
          "shared_prefix_length_avg": 225,
          "is_session_affinity": true,
          "priority_class": "STANDARD",
          "max_concurrent_streaming": 64,
          "target_p99_ttft_ms": 750,
          "target_p99_tpot_ms": 60
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_free_chat_eu",
        "request_arrival_pattern": "Diurnal",
        "max_concurrent_streaming": 64,
        "seed": 102
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-0-2",
      "schedule": {
        "release_at_offset_minutes": 0,
        "active_end_at_offset_minutes": 720,
        "duration_minutes": 720,
        "operation": "base_service",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-0-2"
      }
    },
    {
      "user_id": "tenant_chat_premium",
      "kind": "online",
      "spec_json": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "job_features": {
          "isl_token_avg": 1200,
          "isl_token_min": 300,
          "isl_token_max": 4096,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 500,
          "osl_token_min": 125,
          "osl_token_max": 1400,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Diurnal",
          "peak_to_mean_ratio": 2,
          "workload_prefix_concentration": 0.5,
          "multi_turn_ratio": 0.7,
          "shared_prefix_length_avg": 300,
          "is_session_affinity": true,
          "priority_class": "HIGH",
          "max_concurrent_streaming": 24,
          "target_p99_ttft_ms": 650,
          "target_p99_tpot_ms": 40
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_premium_chat",
        "request_arrival_pattern": "Diurnal",
        "max_concurrent_streaming": 24,
        "seed": 103
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-0-3",
      "schedule": {
        "release_at_offset_minutes": 0,
        "active_end_at_offset_minutes": 420,
        "duration_minutes": 420,
        "operation": "base_service",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-0-3"
      }
    },
    {
      "user_id": "tenant_code_autocomplete",
      "kind": "online",
      "spec_json": {
        "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "job_features": {
          "isl_token_avg": 1800,
          "isl_token_min": 450,
          "isl_token_max": 5000,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 120,
          "osl_token_min": 30,
          "osl_token_max": 350,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Bursty",
          "peak_to_mean_ratio": 3.5,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "priority_class": "STANDARD",
          "max_concurrent_streaming": 32,
          "target_p99_ttft_ms": 180,
          "target_p99_tpot_ms": 45
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_code_base",
        "request_arrival_pattern": "Bursty",
        "max_concurrent_streaming": 32,
        "seed": 104
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-0-4",
      "schedule": {
        "release_at_offset_minutes": 0,
        "active_end_at_offset_minutes": 720,
        "duration_minutes": 720,
        "operation": "base_service",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-0-4"
      }
    },
    {
      "user_id": "tenant_moderation_p0",
      "kind": "online",
      "spec_json": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "job_features": {
          "isl_token_avg": 256,
          "isl_token_min": 64,
          "isl_token_max": 1024,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 8,
          "osl_token_min": 2,
          "osl_token_max": 32,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Bursty",
          "peak_to_mean_ratio": 3,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "priority_class": "HIGH",
          "max_concurrent_streaming": 32,
          "target_p99_ttft_ms": 100,
          "target_p99_tpot_ms": 30
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_moderation",
        "request_arrival_pattern": "Bursty",
        "max_concurrent_streaming": 32,
        "seed": 105
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-0-5",
      "schedule": {
        "release_at_offset_minutes": 0,
        "active_end_at_offset_minutes": 720,
        "duration_minutes": 720,
        "operation": "base_service",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-0-5"
      }
    },
    {
      "user_id": "tenant_small_model",
      "kind": "online",
      "spec_json": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "precision": "fp16",
        "job_features": {
          "isl_token_avg": 256,
          "isl_token_min": 64,
          "isl_token_max": 512,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 64,
          "osl_token_min": 16,
          "osl_token_max": 128,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Constant",
          "peak_to_mean_ratio": 1,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "priority_class": "STANDARD",
          "max_concurrent_streaming": 8,
          "target_p99_ttft_ms": 300,
          "target_p99_tpot_ms": 40
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "small_model_chat",
        "request_arrival_pattern": "Constant",
        "request_rate_per_second": 2,
        "max_concurrent_streaming": 8,
        "seed": 106
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-0-6",
      "schedule": {
        "release_at_offset_minutes": 0,
        "active_end_at_offset_minutes": 720,
        "duration_minutes": 720,
        "operation": "base_service",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-0-6"
      }
    }
  ],
  "schema_version": "koibench/v1",
  "event_id": "event-benchmark_timing_real_12hr_192gpu-0"
}
```

## T+00:10 - Nightly Eval and Backfill

```json
{
  "submit_at_offset_minutes": 10,
  "description": "deadline batch begins after online warmup",
  "jobs": [
    {
      "user_id": "tenant_eval_platform",
      "kind": "batch",
      "spec_json": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "job_features": {
          "isl_token_avg": 1800,
          "isl_token_min": 450,
          "isl_token_max": 6000,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 1200,
          "osl_token_min": 300,
          "osl_token_max": 4000,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Constant",
          "peak_to_mean_ratio": 1,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "total_token_budget": 48000000,
          "priority_class": "STANDARD",
          "deadline_hrs": 7.8333,
          "target_p99_ttft_ms": 10000,
          "target_p99_tpot_ms": 120
        }
      },
      "input_source": {
        "type": "dataset",
        "dataset_id": "real_12h_eval_llama70b",
        "partitioning": "fixed_shards",
        "shard_count": 80,
        "requests_per_shard": 200
      },
      "output_target": {
        "sink": "eval_store"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-10-1",
      "schedule": {
        "release_at_offset_minutes": 10,
        "deadline_at_offset_minutes": 480,
        "campaign_id": "campaign-benchmark_timing_real_12hr_192gpu-10-1",
        "shard_count": 80,
        "requests_per_shard": 200,
        "expansion_policy": "required",
        "checkpointable": true
      }
    },
    {
      "user_id": "tenant_eval_platform",
      "kind": "batch",
      "spec_json": {
        "model_id": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "job_features": {
          "isl_token_avg": 1800,
          "isl_token_min": 450,
          "isl_token_max": 6000,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 1200,
          "osl_token_min": 300,
          "osl_token_max": 4000,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Constant",
          "peak_to_mean_ratio": 1,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "total_token_budget": 42000000,
          "priority_class": "STANDARD",
          "deadline_hrs": 7.8333,
          "target_p99_ttft_ms": 10000,
          "target_p99_tpot_ms": 120
        }
      },
      "input_source": {
        "type": "dataset",
        "dataset_id": "real_12h_eval_qwen72b",
        "partitioning": "fixed_shards",
        "shard_count": 70,
        "requests_per_shard": 200
      },
      "output_target": {
        "sink": "eval_store"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-10-2",
      "schedule": {
        "release_at_offset_minutes": 10,
        "deadline_at_offset_minutes": 480,
        "campaign_id": "campaign-benchmark_timing_real_12hr_192gpu-10-2",
        "shard_count": 70,
        "requests_per_shard": 200,
        "expansion_policy": "required",
        "checkpointable": true
      }
    }
  ],
  "schema_version": "koibench/v1",
  "event_id": "event-benchmark_timing_real_12hr_192gpu-10"
}
```

## T+00:45 - Regional Support Comes Online

```json
{
  "submit_at_offset_minutes": 45,
  "description": "customer support traffic joins the always-on services",
  "jobs": [
    {
      "user_id": "tenant_support",
      "kind": "online",
      "spec_json": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "job_features": {
          "isl_token_avg": 1400,
          "isl_token_min": 350,
          "isl_token_max": 6000,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 350,
          "osl_token_min": 87,
          "osl_token_max": 1000,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Bursty",
          "peak_to_mean_ratio": 3.5,
          "workload_prefix_concentration": 0.5,
          "multi_turn_ratio": 0.3,
          "shared_prefix_length_avg": 350,
          "is_session_affinity": true,
          "priority_class": "STANDARD",
          "max_concurrent_streaming": 64,
          "target_p99_ttft_ms": 700,
          "target_p99_tpot_ms": 50
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_support",
        "request_arrival_pattern": "Bursty",
        "max_concurrent_streaming": 64,
        "seed": 103
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-45-3",
      "schedule": {
        "release_at_offset_minutes": 45,
        "active_end_at_offset_minutes": 720,
        "duration_minutes": 675,
        "operation": "base_service",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-45-3"
      }
    }
  ],
  "schema_version": "koibench/v1",
  "event_id": "event-benchmark_timing_real_12hr_192gpu-45"
}
```

## T+01:30 - Synthetic Teacher and Quant Probes

```json
{
  "submit_at_offset_minutes": 90,
  "description": "large batch work plus explicit quantization exploration",
  "jobs": [
    {
      "user_id": "tenant_synthetic_data",
      "kind": "batch",
      "spec_json": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "job_features": {
          "isl_token_avg": 1200,
          "isl_token_min": 300,
          "isl_token_max": 4096,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 2400,
          "osl_token_min": 600,
          "osl_token_max": 8000,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Constant",
          "peak_to_mean_ratio": 1,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "total_token_budget": 64800000,
          "priority_class": "LOW",
          "deadline_hrs": 9.5,
          "target_p99_ttft_ms": 12000,
          "target_p99_tpot_ms": 160
        }
      },
      "input_source": {
        "type": "dataset",
        "dataset_id": "real_12h_synthetic_teacher",
        "partitioning": "fixed_shards",
        "shard_count": 90,
        "requests_per_shard": 200
      },
      "output_target": {
        "sink": "synth_store"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-90-1",
      "schedule": {
        "release_at_offset_minutes": 90,
        "deadline_at_offset_minutes": 660,
        "campaign_id": "campaign-benchmark_timing_real_12hr_192gpu-90-1",
        "shard_count": 90,
        "requests_per_shard": 200,
        "expansion_policy": "required",
        "checkpointable": true
      }
    },
    {
      "user_id": "tenant_koi_profiling",
      "kind": "batch",
      "spec_json": {
        "model_id": "microsoft/phi-4",
        "job_features": {
          "isl_token_avg": 1000,
          "isl_token_min": 250,
          "isl_token_max": 4096,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 512,
          "osl_token_min": 128,
          "osl_token_max": 2048,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Constant",
          "peak_to_mean_ratio": 1,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "total_token_budget": 36288000,
          "priority_class": "LOW",
          "deadline_hrs": 3.5,
          "target_p99_ttft_ms": 12000,
          "target_p99_tpot_ms": 200
        }
      },
      "input_source": {
        "type": "dataset",
        "dataset_id": "real_12h_quant_probe",
        "partitioning": "fixed_shards",
        "shard_count": 48,
        "requests_per_shard": 500
      },
      "output_target": {
        "sink": "perfdb"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-90-2",
      "schedule": {
        "release_at_offset_minutes": 90,
        "deadline_at_offset_minutes": 300,
        "campaign_id": "campaign-benchmark_timing_real_12hr_192gpu-90-2",
        "shard_count": 48,
        "requests_per_shard": 500,
        "expansion_policy": "required",
        "checkpointable": true
      }
    }
  ],
  "schema_version": "koibench/v1",
  "event_id": "event-benchmark_timing_real_12hr_192gpu-90"
}
```

## T+02:30 - EU Business Ramp

```json
{
  "submit_at_offset_minutes": 150,
  "description": "EU enterprise demand ramps into the business-day peak",
  "jobs": [
    {
      "user_id": "tenant_enterprise_chat",
      "kind": "online",
      "spec_json": {
        "model_id": "microsoft/phi-4",
        "job_features": {
          "isl_token_avg": 1500,
          "isl_token_min": 375,
          "isl_token_max": 5500,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 420,
          "osl_token_min": 105,
          "osl_token_max": 1200,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Diurnal",
          "peak_to_mean_ratio": 2,
          "workload_prefix_concentration": 0.5,
          "multi_turn_ratio": 0.7,
          "shared_prefix_length_avg": 375,
          "is_session_affinity": true,
          "priority_class": "STANDARD",
          "max_concurrent_streaming": 24,
          "target_p99_ttft_ms": 500,
          "target_p99_tpot_ms": 42
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_eu_enterprise",
        "request_arrival_pattern": "Diurnal",
        "max_concurrent_streaming": 24,
        "seed": 102
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-150-2",
      "schedule": {
        "release_at_offset_minutes": 150,
        "active_end_at_offset_minutes": 480,
        "duration_minutes": 330,
        "operation": "base_service",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-150-2"
      }
    }
  ],
  "schema_version": "koibench/v1",
  "event_id": "event-benchmark_timing_real_12hr_192gpu-150"
}
```

## T+03:30 - Code Peak

```json
{
  "submit_at_offset_minutes": 210,
  "description": "TTFT-sensitive code job peak",
  "jobs": [
    {
      "user_id": "tenant_code_autocomplete",
      "kind": "online",
      "spec_json": {
        "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "job_features": {
          "isl_token_avg": 2200,
          "isl_token_min": 550,
          "isl_token_max": 6000,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 96,
          "osl_token_min": 24,
          "osl_token_max": 280,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Bursty",
          "peak_to_mean_ratio": 3.5,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "priority_class": "STANDARD",
          "max_concurrent_streaming": 96,
          "target_p99_ttft_ms": 160,
          "target_p99_tpot_ms": 45
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_code_peak",
        "request_arrival_pattern": "Bursty",
        "max_concurrent_streaming": 96,
        "seed": 101
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-210-1",
      "schedule": {
        "release_at_offset_minutes": 210,
        "active_end_at_offset_minutes": 480,
        "duration_minutes": 270,
        "operation": "additive_stream",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-210-1"
      }
    },
    {
      "user_id": "tenant_code_chat",
      "kind": "online",
      "spec_json": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "job_features": {
          "isl_token_avg": 2600,
          "isl_token_min": 650,
          "isl_token_max": 10000,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 900,
          "osl_token_min": 225,
          "osl_token_max": 3200,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Bursty",
          "peak_to_mean_ratio": 3.5,
          "workload_prefix_concentration": 0.5,
          "multi_turn_ratio": 0.7,
          "shared_prefix_length_avg": 650,
          "is_session_affinity": true,
          "priority_class": "STANDARD",
          "max_concurrent_streaming": 48,
          "target_p99_ttft_ms": 800,
          "target_p99_tpot_ms": 42
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_code_chat",
        "request_arrival_pattern": "Bursty",
        "max_concurrent_streaming": 48,
        "seed": 102
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-210-2",
      "schedule": {
        "release_at_offset_minutes": 210,
        "active_end_at_offset_minutes": 480,
        "duration_minutes": 270,
        "operation": "base_service",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-210-2"
      }
    }
  ],
  "schema_version": "koibench/v1",
  "event_id": "event-benchmark_timing_real_12hr_192gpu-210"
}
```

## T+04:30 - Engine Canary

```json
{
  "submit_at_offset_minutes": 270,
  "description": "canary plus long-input online burst stresses prediction calibration",
  "jobs": [
    {
      "user_id": "tenant_engine_canary",
      "kind": "batch",
      "spec_json": {
        "model_id": "microsoft/phi-4",
        "job_features": {
          "isl_token_avg": 4000,
          "isl_token_min": 1000,
          "isl_token_max": 16000,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 400,
          "osl_token_min": 100,
          "osl_token_max": 1200,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Constant",
          "peak_to_mean_ratio": 1,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "total_token_budget": 88000000,
          "priority_class": "HIGH",
          "deadline_hrs": 2.5,
          "target_p99_ttft_ms": 8000,
          "target_p99_tpot_ms": 120
        }
      },
      "input_source": {
        "type": "dataset",
        "dataset_id": "real_12h_engine_canary",
        "partitioning": "fixed_shards",
        "shard_count": 40,
        "requests_per_shard": 500
      },
      "output_target": {
        "sink": "perfdb"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-270-1",
      "schedule": {
        "release_at_offset_minutes": 270,
        "deadline_at_offset_minutes": 420,
        "campaign_id": "campaign-benchmark_timing_real_12hr_192gpu-270-1",
        "shard_count": 40,
        "requests_per_shard": 500,
        "expansion_policy": "required",
        "checkpointable": true
      }
    }
  ],
  "schema_version": "koibench/v1",
  "event_id": "event-benchmark_timing_real_12hr_192gpu-270"
}
```

## T+06:00 - Product Launch Burst

```json
{
  "submit_at_offset_minutes": 360,
  "description": "real-size launch burst with enough pressure to require on-demand",
  "jobs": [
    {
      "user_id": "tenant_chat_free",
      "kind": "online",
      "spec_json": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "job_features": {
          "isl_token_avg": 850,
          "isl_token_min": 212,
          "isl_token_max": 2200,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 280,
          "osl_token_min": 70,
          "osl_token_max": 650,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Bursty",
          "peak_to_mean_ratio": 3.5,
          "workload_prefix_concentration": 0.5,
          "multi_turn_ratio": 0.7,
          "shared_prefix_length_avg": 212,
          "is_session_affinity": true,
          "priority_class": "STANDARD",
          "max_concurrent_streaming": 64,
          "target_p99_ttft_ms": 650,
          "target_p99_tpot_ms": 50
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_launch_chat",
        "request_arrival_pattern": "Bursty",
        "max_concurrent_streaming": 64,
        "seed": 101
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-360-1",
      "schedule": {
        "release_at_offset_minutes": 360,
        "active_end_at_offset_minutes": 450,
        "duration_minutes": 90,
        "operation": "additive_stream",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-360-1"
      }
    },
    {
      "user_id": "tenant_moderation_p0",
      "kind": "online",
      "spec_json": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "job_features": {
          "isl_token_avg": 320,
          "isl_token_min": 80,
          "isl_token_max": 1024,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 8,
          "osl_token_min": 2,
          "osl_token_max": 32,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Bursty",
          "peak_to_mean_ratio": 3,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "priority_class": "HIGH",
          "max_concurrent_streaming": 32,
          "target_p99_ttft_ms": 90,
          "target_p99_tpot_ms": 30
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_launch_moderation",
        "request_arrival_pattern": "Bursty",
        "max_concurrent_streaming": 32,
        "seed": 102
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-360-2",
      "schedule": {
        "release_at_offset_minutes": 360,
        "active_end_at_offset_minutes": 450,
        "duration_minutes": 90,
        "operation": "additive_stream",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-360-2"
      }
    }
  ],
  "schema_version": "koibench/v1",
  "event_id": "event-benchmark_timing_real_12hr_192gpu-360"
}
```

## T+07:00 - Premium Swap and P0 Safety Replay

```json
{
  "submit_at_offset_minutes": 420,
  "description": "new 80B-A3B premium stream overlaps the 70B service for 20 minutes before cutover; urgent safety batch arrives simultaneously",
  "jobs": [
    {
      "user_id": "tenant_chat_premium",
      "kind": "online",
      "spec_json": {
        "model_id": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "job_features": {
          "isl_token_avg": 1200,
          "isl_token_min": 300,
          "isl_token_max": 4096,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 550,
          "osl_token_min": 137,
          "osl_token_max": 1600,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Bursty",
          "peak_to_mean_ratio": 3,
          "workload_prefix_concentration": 0.5,
          "multi_turn_ratio": 0.7,
          "shared_prefix_length_avg": 300,
          "is_session_affinity": true,
          "priority_class": "HIGH",
          "max_concurrent_streaming": 32,
          "target_p99_ttft_ms": 700,
          "target_p99_tpot_ms": 42
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_qwen72b_premium",
        "request_arrival_pattern": "Bursty",
        "max_concurrent_streaming": 32,
        "seed": 101
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-420-1",
      "schedule": {
        "release_at_offset_minutes": 420,
        "active_end_at_offset_minutes": 720,
        "duration_minutes": 300,
        "operation": "replace_service",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-420-1",
        "supersedes_service_id": "service-benchmark_timing_real_12hr_192gpu-0-3"
      }
    },
    {
      "user_id": "tenant_safety_p0",
      "kind": "batch",
      "spec_json": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "job_features": {
          "isl_token_avg": 2200,
          "isl_token_min": 550,
          "isl_token_max": 8000,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 700,
          "osl_token_min": 175,
          "osl_token_max": 2500,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Constant",
          "peak_to_mean_ratio": 1,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "total_token_budget": 27840000,
          "priority_class": "HIGH",
          "deadline_hrs": 2,
          "target_p99_ttft_ms": 8000,
          "target_p99_tpot_ms": 100
        }
      },
      "input_source": {
        "type": "dataset",
        "dataset_id": "real_12h_p0_safety",
        "partitioning": "fixed_shards",
        "shard_count": 48,
        "requests_per_shard": 200
      },
      "output_target": {
        "sink": "safety_store"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-420-2",
      "schedule": {
        "release_at_offset_minutes": 420,
        "deadline_at_offset_minutes": 540,
        "campaign_id": "campaign-benchmark_timing_real_12hr_192gpu-420-2",
        "shard_count": 48,
        "requests_per_shard": 200,
        "expansion_policy": "required",
        "checkpointable": true
      }
    }
  ],
  "schema_version": "koibench/v1",
  "event_id": "event-benchmark_timing_real_12hr_192gpu-420"
}
```

## T+08:00 - Frontier Reasoning Wave

```json
{
  "submit_at_offset_minutes": 480,
  "description": "GLM and Qwen3-Next put pressure on large-memory and multi-node planning",
  "jobs": [
    {
      "user_id": "tenant_agents_premium",
      "kind": "online",
      "spec_json": {
        "model_id": "zai-org/GLM-4.5-Air",
        "job_features": {
          "isl_token_avg": 1400,
          "isl_token_min": 350,
          "isl_token_max": 4096,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 4096,
          "osl_token_min": 1024,
          "osl_token_max": 12000,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Bursty",
          "peak_to_mean_ratio": 5,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": true,
          "priority_class": "HIGH",
          "max_concurrent_streaming": 16,
          "target_p99_ttft_ms": 2500,
          "target_p99_tpot_ms": 50
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_glm_premium_agents",
        "request_arrival_pattern": "Bursty",
        "max_concurrent_streaming": 16,
        "seed": 101
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-480-1",
      "schedule": {
        "release_at_offset_minutes": 480,
        "active_end_at_offset_minutes": 660,
        "duration_minutes": 180,
        "operation": "base_service",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-480-1"
      }
    },
    {
      "user_id": "tenant_frontier_glm",
      "kind": "online",
      "spec_json": {
        "model_id": "zai-org/GLM-4.5-Air",
        "job_features": {
          "isl_token_avg": 1800,
          "isl_token_min": 450,
          "isl_token_max": 6000,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 2500,
          "osl_token_min": 625,
          "osl_token_max": 9000,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Bursty",
          "peak_to_mean_ratio": 5,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": true,
          "priority_class": "STANDARD",
          "max_concurrent_streaming": 16,
          "target_p99_ttft_ms": 2300,
          "target_p99_tpot_ms": 55
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_glm_agents",
        "request_arrival_pattern": "Bursty",
        "max_concurrent_streaming": 16,
        "seed": 102
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-480-2",
      "schedule": {
        "release_at_offset_minutes": 480,
        "active_end_at_offset_minutes": 660,
        "duration_minutes": 180,
        "operation": "base_service",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-480-2"
      }
    },
    {
      "user_id": "tenant_new_model_eval",
      "kind": "batch",
      "spec_json": {
        "model_id": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "job_features": {
          "isl_token_avg": 1800,
          "isl_token_min": 450,
          "isl_token_max": 6000,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 1600,
          "osl_token_min": 400,
          "osl_token_max": 6000,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Constant",
          "peak_to_mean_ratio": 1,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "total_token_budget": 21760000,
          "priority_class": "STANDARD",
          "deadline_hrs": 3.5,
          "target_p99_ttft_ms": 12000,
          "target_p99_tpot_ms": 180
        }
      },
      "input_source": {
        "type": "dataset",
        "dataset_id": "real_12h_qwen3_next_probe",
        "partitioning": "fixed_shards",
        "shard_count": 32,
        "requests_per_shard": 200
      },
      "output_target": {
        "sink": "eval_store"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-480-3",
      "schedule": {
        "release_at_offset_minutes": 480,
        "deadline_at_offset_minutes": 690,
        "campaign_id": "campaign-benchmark_timing_real_12hr_192gpu-480-3",
        "shard_count": 32,
        "requests_per_shard": 200,
        "expansion_policy": "required",
        "checkpointable": true
      }
    }
  ],
  "schema_version": "koibench/v1",
  "event_id": "event-benchmark_timing_real_12hr_192gpu-480"
}
```

## T+09:30 - Long Context and MoE Batch

```json
{
  "submit_at_offset_minutes": 570,
  "description": "long context and MoE campaigns arrive while online frontier work is active",
  "jobs": [
    {
      "user_id": "tenant_long_context",
      "kind": "batch",
      "spec_json": {
        "model_id": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "job_features": {
          "isl_token_avg": 32000,
          "isl_token_min": 8000,
          "isl_token_max": 128000,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 1200,
          "osl_token_min": 300,
          "osl_token_max": 4000,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Constant",
          "peak_to_mean_ratio": 1,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "total_token_budget": 139440000,
          "priority_class": "STANDARD",
          "deadline_hrs": 2.1667,
          "target_p99_ttft_ms": 10000,
          "target_p99_tpot_ms": 160
        }
      },
      "input_source": {
        "type": "dataset",
        "dataset_id": "real_12h_long_context",
        "partitioning": "fixed_shards",
        "shard_count": 42,
        "requests_per_shard": 100
      },
      "output_target": {
        "sink": "eval_store"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-570-1",
      "schedule": {
        "release_at_offset_minutes": 570,
        "deadline_at_offset_minutes": 700,
        "campaign_id": "campaign-benchmark_timing_real_12hr_192gpu-570-1",
        "shard_count": 42,
        "requests_per_shard": 100,
        "expansion_policy": "required",
        "checkpointable": true
      }
    },
    {
      "user_id": "tenant_moe_eval",
      "kind": "batch",
      "spec_json": {
        "model_id": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "job_features": {
          "isl_token_avg": 1000,
          "isl_token_min": 250,
          "isl_token_max": 4096,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 1200,
          "osl_token_min": 300,
          "osl_token_max": 5000,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Constant",
          "peak_to_mean_ratio": 1,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "total_token_budget": 66000000,
          "priority_class": "STANDARD",
          "deadline_hrs": 2.1667,
          "target_p99_ttft_ms": 10000,
          "target_p99_tpot_ms": 150
        }
      },
      "input_source": {
        "type": "dataset",
        "dataset_id": "real_12h_moe_eval",
        "partitioning": "fixed_shards",
        "shard_count": 60,
        "requests_per_shard": 500
      },
      "output_target": {
        "sink": "eval_store"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-570-2",
      "schedule": {
        "release_at_offset_minutes": 570,
        "deadline_at_offset_minutes": 700,
        "campaign_id": "campaign-benchmark_timing_real_12hr_192gpu-570-2",
        "shard_count": 60,
        "requests_per_shard": 500,
        "expansion_policy": "required",
        "checkpointable": true
      }
    }
  ],
  "schema_version": "koibench/v1",
  "event_id": "event-benchmark_timing_real_12hr_192gpu-570"
}
```

## T+10:30 - Reports and Transcripts

```json
{
  "submit_at_offset_minutes": 630,
  "description": "late-run deadline work tests whether batch can finish without hurting online SLOs",
  "jobs": [
    {
      "user_id": "tenant_reports",
      "kind": "batch",
      "spec_json": {
        "model_id": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "job_features": {
          "isl_token_avg": 18000,
          "isl_token_min": 4500,
          "isl_token_max": 64000,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 1400,
          "osl_token_min": 350,
          "osl_token_max": 5000,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Constant",
          "peak_to_mean_ratio": 1,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "total_token_budget": 155200000,
          "priority_class": "LOW",
          "deadline_hrs": 1.5,
          "target_p99_ttft_ms": 12000,
          "target_p99_tpot_ms": 160
        }
      },
      "input_source": {
        "type": "dataset",
        "dataset_id": "real_12h_reports",
        "partitioning": "fixed_shards",
        "shard_count": 80,
        "requests_per_shard": 100
      },
      "output_target": {
        "sink": "report_store"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-630-1",
      "schedule": {
        "release_at_offset_minutes": 630,
        "deadline_at_offset_minutes": 720,
        "campaign_id": "campaign-benchmark_timing_real_12hr_192gpu-630-1",
        "shard_count": 80,
        "requests_per_shard": 100,
        "expansion_policy": "required",
        "checkpointable": true
      }
    },
    {
      "user_id": "tenant_transcripts",
      "kind": "batch",
      "spec_json": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "job_features": {
          "isl_token_avg": 16000,
          "isl_token_min": 4000,
          "isl_token_max": 64000,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 900,
          "osl_token_min": 225,
          "osl_token_max": 3000,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Constant",
          "peak_to_mean_ratio": 1,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "total_token_budget": 108160000,
          "priority_class": "LOW",
          "deadline_hrs": 1.5,
          "target_p99_ttft_ms": 8000,
          "target_p99_tpot_ms": 120
        }
      },
      "input_source": {
        "type": "dataset",
        "dataset_id": "real_12h_transcripts",
        "partitioning": "fixed_shards",
        "shard_count": 64,
        "requests_per_shard": 100
      },
      "output_target": {
        "sink": "summary_store"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-630-2",
      "schedule": {
        "release_at_offset_minutes": 630,
        "deadline_at_offset_minutes": 720,
        "campaign_id": "campaign-benchmark_timing_real_12hr_192gpu-630-2",
        "shard_count": 64,
        "requests_per_shard": 100,
        "expansion_policy": "required",
        "checkpointable": true
      }
    }
  ],
  "schema_version": "koibench/v1",
  "event_id": "event-benchmark_timing_real_12hr_192gpu-630"
}
```

## T+11:30 - APAC Ramp and Cooldown

```json
{
  "submit_at_offset_minutes": 690,
  "description": "APAC online load appears while low-priority probes test remaining capacity",
  "jobs": [
    {
      "user_id": "tenant_chat_free",
      "kind": "online",
      "spec_json": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "job_features": {
          "isl_token_avg": 900,
          "isl_token_min": 225,
          "isl_token_max": 2200,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 300,
          "osl_token_min": 75,
          "osl_token_max": 700,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Diurnal",
          "peak_to_mean_ratio": 2,
          "workload_prefix_concentration": 0.5,
          "multi_turn_ratio": 0.7,
          "shared_prefix_length_avg": 225,
          "is_session_affinity": true,
          "priority_class": "STANDARD",
          "max_concurrent_streaming": 64,
          "target_p99_ttft_ms": 800,
          "target_p99_tpot_ms": 60
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "prompt_dataset_id": "real_12h_apac_chat",
        "request_arrival_pattern": "Diurnal",
        "max_concurrent_streaming": 64,
        "seed": 101
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-690-1",
      "schedule": {
        "release_at_offset_minutes": 690,
        "active_end_at_offset_minutes": 720,
        "duration_minutes": 30,
        "operation": "base_service",
        "service_id": "service-benchmark_timing_real_12hr_192gpu-690-1"
      }
    },
    {
      "user_id": "tenant_koi_profiling",
      "kind": "batch",
      "spec_json": {
        "model_id": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "job_features": {
          "isl_token_avg": 1000,
          "isl_token_min": 250,
          "isl_token_max": 4096,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 800,
          "osl_token_min": 200,
          "osl_token_max": 3000,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Constant",
          "peak_to_mean_ratio": 1,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "total_token_budget": 21600000,
          "priority_class": "LOW",
          "deadline_hrs": 0.5,
          "target_p99_ttft_ms": 12000,
          "target_p99_tpot_ms": 200
        }
      },
      "input_source": {
        "type": "dataset",
        "dataset_id": "real_12h_final_probe",
        "partitioning": "fixed_shards",
        "shard_count": 24,
        "requests_per_shard": 500
      },
      "output_target": {
        "sink": "perfdb"
      },
      "job_id": "job-benchmark_timing_real_12hr_192gpu-690-2",
      "schedule": {
        "release_at_offset_minutes": 690,
        "deadline_at_offset_minutes": 720,
        "campaign_id": "campaign-benchmark_timing_real_12hr_192gpu-690-2",
        "shard_count": 24,
        "requests_per_shard": 500,
        "expansion_policy": "required",
        "checkpointable": true
      }
    }
  ],
  "schema_version": "koibench/v1",
  "event_id": "event-benchmark_timing_real_12hr_192gpu-690"
}
```
