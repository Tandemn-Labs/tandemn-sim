# Real-life 3-hour benchmark

Benchmark ID: `real_life_3h`

Benchmark name: `KoiBench-Real-3h-192GPU-Smoke`

Purpose: synthetic job replay on a real-cluster smoke run for the 192 GPU representative fleet. This checks that the spawner, vLLM/Dynamo stack, LMCache, TP/PP/DP/EP choices, quantization filters, and online load generator behave before longer runs.

Horizon: 180 minutes

Tick: 5 minutes

Engine: `vllm` behind Dynamo-style serving

Markets: `reserved`, `on_demand`

Serving mode: aggregate serving with LMCache, no prefill/decode disaggregation

Real fleet target: 192 GPUs



## Canonical Lifecycle Contract

Schema: `koibench/v1`

Dispatch: releases occur at their exact offset. A job arrival triggers the admission path immediately; the 5-minute tick remains the slow reoptimization loop.

Online lifecycle: `active_start_at_offset_minutes` and `active_end_at_offset_minutes` define a half-open traffic window. `duration_minutes` is retained only as a consistency check. At the end offset the load generator stops new requests, drains in-flight requests, and the executor records `terminal_at`.

Batch lifecycle: `deadline_at_offset_minutes` is an absolute offset from benchmark start. `shard_count` is expanded deterministically because `expansion_policy` is `required`. `requests_per_shard` fixes campaign size. Runtime `started_at`, `completed_at`, and `terminal_at` are produced by the executor, not authored in the job definition.

Overlap: `base_service` remains active, `additive_stream` adds demand to an existing service, and `replace_service` identifies a model/service replacement. Overlap is intentional and additive unless explicitly marked as replacement.

Planner fairness: descriptions, trace names, human-readable tenant labels, and `input_source` are not exposed to planners. Every planner receives the same sanitized job features. Hardware placement and market preferences are deliberately absent from job inputs.

Traffic: `max_concurrent_streaming` is the planner-facing service cap. Client submission rates may be fixed in `input_source`; they remain excluded from the Orca payload so Orca stays in concurrency mode. Patterns are limited to `Poisson`, `Constant`, `Bursty`, and `Diurnal`.

Input modes: every online job uses `api_load_generator` to ping an OpenAI-compatible endpoint under its concurrency profile. Every batch campaign uses a finite `dataset` partitioned into required shards.

Drain: after minute 180, no new work arrives. A 30-minute drain period records batch tardiness and in-flight completion; work unfinished at its deadline is a miss.

Exact manifest counts: **14 online jobs**, **5 batch campaigns**, **132 expanded shards**.

## Real-Run Capacity Gate

The listed concurrency caps are frozen only after an isolated preflight on the exact engine/model build. Apply one global concurrency multiplier per run so normal offered load consumes 55-65% of measured qualified capacity, business peaks consume 80-90%, and the designated overload burst reaches 105-120%. Freeze that multiplier before comparing systems.
## Frozen Real Cluster Resources

Maximum provisioned capacity: **192 GPUs**. Reserved capacity is pre-provisioned; on-demand capacity is a hard launch quota.

| Accelerator | Node/Slice Shape | Nodes/Slices | Reserved | On-Demand Cap | Total |
|---|---:|---:|---:|---:|---:|
| `H100-SXM-80GB` | 8 GPUs, NVSwitch | 5 | 32 | 8 | 40 |
| `A100-SXM-80GB` | 8 GPUs, NVLink | 5 | 32 | 8 | 40 |
| `A100-PCIe-40GB` | 8 GPUs, PCIe | 2 | 8 | 8 | 16 |
| `A10G-24GB` | 8 GPUs, PCIe | 4 | 24 | 8 | 32 |
| `L40S-48GB` | 8 GPUs, PCIe | 5 | 32 | 8 | 40 |
| `MI300X-192GB` | 8 GPUs, Infinity Fabric | 3 | 8 | 16 | 24 |
| **Total** |  | **24** | **136** | **56** | **192** |

| Accelerator | `us-east` | `eu-west` | `apac` | Total |
|---|---:|---:|---:|---:|
| H100 | 24 | 8 | 8 | 40 |
| A100-SXM | 16 | 16 | 8 | 40 |
| A100-PCIe | 8 | 8 | 0 | 16 |
| A10G | 16 | 8 | 8 | 32 |
| L40S | 16 | 8 | 16 | 40 |
| MI300X | 16 | 8 | 0 | 24 |
| **Region total** | **96** | **56** | **40** | **192** |

| Region | Reserved Devices | On-Demand Cap | Total Devices |
|---|---:|---:|---:|
| `us-east` | 80 | 16 | 96 |
| `eu-west` | 32 | 24 | 56 |
| `apac` | 24 | 16 | 40 |

Network caps: 400 Gbps RDMA for H100/A100-SXM/MI300X nodes; 100 Gbps RoCE for A100-PCIe/A10G/L40S nodes. Inter-region caps are 25 Gbps/75 ms RTT (`us-east<->eu-west`), 20 Gbps/145 ms (`us-east<->apac`), and 15 Gbps/220 ms (`eu-west<->apac`). Inferentia and Trainium are not allocated in this real manifest.

## T+00:00 - Warm Core APIs

```json
{
    "submit_at_offset_minutes":  0,
    "description":  "persistent core APIs start warm on the real 160-GPU fleet",
    "jobs":  [
                 {
                     "user_id":  "tenant_chat_free",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.2-1B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  900,
                                                            "isl_token_min":  225,
                                                            "isl_token_max":  2200,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  300,
                                                            "osl_token_min":  75,
                                                            "osl_token_max":  700,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.7,
                                                            "shared_prefix_length_avg":  225,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  2800,
                                                            "target_p99_ttft_ms":  700,
                                                            "target_p99_tpot_ms":  55
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_3h_free_chat_us",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  2800,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-0-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  180,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_192gpu-0-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_chat_premium",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  1200,
                                                            "isl_token_min":  300,
                                                            "isl_token_max":  4096,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  500,
                                                            "osl_token_min":  125,
                                                            "osl_token_max":  1400,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Diurnal",
                                                            "peak_to_mean_ratio":  2,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.7,
                                                            "shared_prefix_length_avg":  300,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "HIGH",
                                                            "max_concurrent_streaming":  180,
                                                            "target_p99_ttft_ms":  650,
                                                            "target_p99_tpot_ms":  40
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_3h_premium_chat",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  180,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-0-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  180,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_192gpu-0-2"
                                  }
                 },
                 {
                     "user_id":  "tenant_code_autocomplete",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-Coder-32B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  1800,
                                                            "isl_token_min":  450,
                                                            "isl_token_max":  5000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  120,
                                                            "osl_token_min":  30,
                                                            "osl_token_max":  350,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  900,
                                                            "target_p99_ttft_ms":  180,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_3h_code_autocomplete",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  900,
                                          "seed":  103
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-0-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  180,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_192gpu-0-3"
                                  }
                 },
                 {
                     "user_id":  "tenant_moderation_p0",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.2-1B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  256,
                                                            "isl_token_min":  64,
                                                            "isl_token_max":  1024,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  8,
                                                            "osl_token_min":  2,
                                                            "osl_token_max":  32,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "HIGH",
                                                            "max_concurrent_streaming":  9000,
                                                            "target_p99_ttft_ms":  100,
                                                            "target_p99_tpot_ms":  30
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_3h_moderation",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  9000,
                                          "seed":  104
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-0-4",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  180,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_192gpu-0-4"
                                  }
                 },
                 {
                     "user_id":  "tenant_small_model",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.2-1B-Instruct",
                                       "precision":  "fp16",
                                       "job_features":  {
                                                            "isl_token_avg":  256,
                                                            "isl_token_min":  64,
                                                            "isl_token_max":  512,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  64,
                                                            "osl_token_min":  16,
                                                            "osl_token_max":  128,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  8,
                                                            "target_p99_ttft_ms":  300,
                                                            "target_p99_tpot_ms":  40
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "small_model_chat",
                                          "request_arrival_pattern":  "Constant",
                                          "request_rate_per_second":  2,
                                          "max_concurrent_streaming":  8,
                                          "seed":  105
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-0-5",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  180,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_192gpu-0-5"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_192gpu-0"
}
```

## T+00:12 - Search and Support APIs

```json
{
    "submit_at_offset_minutes":  12,
    "description":  "RAG, reranking, and support enter after core warmup",
    "jobs":  [
                 {
                     "user_id":  "tenant_search_rag",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "microsoft/phi-4",
                                       "job_features":  {
                                                            "isl_token_avg":  9000,
                                                            "isl_token_min":  2250,
                                                            "isl_token_max":  32000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  220,
                                                            "osl_token_min":  55,
                                                            "osl_token_max":  700,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Poisson",
                                                            "peak_to_mean_ratio":  2,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.3,
                                                            "shared_prefix_length_avg":  2250,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  700,
                                                            "target_p99_ttft_ms":  1500,
                                                            "target_p99_tpot_ms":  55
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_3h_rag",
                                          "request_arrival_pattern":  "Poisson",
                                          "max_concurrent_streaming":  700,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-12-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  12,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  168,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_192gpu-12-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_search_rerank",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "BAAI/bge-reranker-v2-m3",
                                       "job_features":  {
                                                            "isl_token_avg":  512,
                                                            "isl_token_min":  128,
                                                            "isl_token_max":  2048,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  1,
                                                            "osl_token_min":  1,
                                                            "osl_token_max":  1,
                                                            "osl_distribution_type":  "Constant",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  3200,
                                                            "target_p99_ttft_ms":  160,
                                                            "target_p99_tpot_ms":  1
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/rerank",
                                          "prompt_dataset_id":  "real_3h_reranker",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  3200,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-12-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  12,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  168,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_192gpu-12-2"
                                  }
                 },
                 {
                     "user_id":  "tenant_support",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "mistralai/Mixtral-8x7B-Instruct-v0.1",
                                       "job_features":  {
                                                            "isl_token_avg":  1400,
                                                            "isl_token_min":  350,
                                                            "isl_token_max":  6000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  350,
                                                            "osl_token_min":  87,
                                                            "osl_token_max":  1000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.3,
                                                            "shared_prefix_length_avg":  350,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  900,
                                                            "target_p99_ttft_ms":  700,
                                                            "target_p99_tpot_ms":  50
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_3h_support",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  900,
                                          "seed":  103
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-12-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  12,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  168,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_192gpu-12-3"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_192gpu-12"
}
```

## T+00:25 - Initial Batch Campaigns

```json
{
    "submit_at_offset_minutes":  25,
    "description":  "small batch campaigns start filling low-priority capacity",
    "jobs":  [
                 {
                     "user_id":  "tenant_eval_platform",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  1800,
                                                            "isl_token_min":  450,
                                                            "isl_token_max":  6000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  1200,
                                                            "osl_token_min":  300,
                                                            "osl_token_max":  4000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  19200000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  2.25,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_3h_eval_llama70b",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  32,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-25-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  25,
                                      "deadline_at_offset_minutes":  160,
                                      "campaign_id":  "campaign-benchmark_timing_real_3hr_192gpu-25-1",
                                      "shard_count":  32,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_koi_profiling",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "microsoft/phi-4",
                                       "job_features":  {
                                                            "isl_token_avg":  1000,
                                                            "isl_token_min":  250,
                                                            "isl_token_max":  4096,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  512,
                                                            "osl_token_min":  128,
                                                            "osl_token_max":  2048,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  15120000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  1.5833,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  200
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_3h_quant_probe_phi4",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  20,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-25-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  25,
                                      "deadline_at_offset_minutes":  120,
                                      "campaign_id":  "campaign-benchmark_timing_real_3hr_192gpu-25-3",
                                      "shard_count":  20,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_192gpu-25"
}
```

## T+00:45 - Launch Burst

```json
{
    "submit_at_offset_minutes":  45,
    "description":  "short launch burst with high small-model load and strict moderation SLO",
    "jobs":  [
                 {
                     "user_id":  "tenant_chat_free",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "mistralai/Mixtral-8x7B-Instruct-v0.1",
                                       "job_features":  {
                                                            "isl_token_avg":  850,
                                                            "isl_token_min":  212,
                                                            "isl_token_max":  2200,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  280,
                                                            "osl_token_min":  70,
                                                            "osl_token_max":  650,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.7,
                                                            "shared_prefix_length_avg":  212,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  8000,
                                                            "target_p99_ttft_ms":  650,
                                                            "target_p99_tpot_ms":  50
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_3h_launch_chat",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  8000,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-45-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  45,
                                      "active_end_at_offset_minutes":  90,
                                      "duration_minutes":  45,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_real_3hr_192gpu-45-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_moderation_p0",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.2-1B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  320,
                                                            "isl_token_min":  80,
                                                            "isl_token_max":  1024,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  8,
                                                            "osl_token_min":  2,
                                                            "osl_token_max":  32,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "HIGH",
                                                            "max_concurrent_streaming":  20000,
                                                            "target_p99_ttft_ms":  90,
                                                            "target_p99_tpot_ms":  30
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_3h_launch_moderation",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  20000,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-45-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  45,
                                      "active_end_at_offset_minutes":  90,
                                      "duration_minutes":  45,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_real_3hr_192gpu-45-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_192gpu-45"
}
```

## T+01:10 - Frontier and MoE Reasoning

```json
{
    "submit_at_offset_minutes":  70,
    "description":  "low-concurrency frontier jobs still consume large chunks of H100/A100/MI300X capacity",
    "jobs":  [
                 {
                     "user_id":  "tenant_agents_premium",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "moonshotai/Kimi-K2-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  1400,
                                                            "isl_token_min":  350,
                                                            "isl_token_max":  4096,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  4096,
                                                            "osl_token_min":  1024,
                                                            "osl_token_max":  12000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  5,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "HIGH",
                                                            "max_concurrent_streaming":  80,
                                                            "target_p99_ttft_ms":  2500,
                                                            "target_p99_tpot_ms":  50
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_3h_kimi_reasoning",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  80,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-70-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  70,
                                      "active_end_at_offset_minutes":  140,
                                      "duration_minutes":  70,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_192gpu-70-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_frontier_glm",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "zai-org/GLM-4.5-Air",
                                       "job_features":  {
                                                            "isl_token_avg":  1800,
                                                            "isl_token_min":  450,
                                                            "isl_token_max":  6000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  2500,
                                                            "osl_token_min":  625,
                                                            "osl_token_max":  9000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  5,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  90,
                                                            "target_p99_ttft_ms":  2300,
                                                            "target_p99_tpot_ms":  55
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_3h_glm_reasoning",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  90,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-70-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  70,
                                      "active_end_at_offset_minutes":  140,
                                      "duration_minutes":  70,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_192gpu-70-2"
                                  }
                 },
                 {
                     "user_id":  "tenant_moe_routing",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-Next-80B-A3B-Thinking",
                                       "job_features":  {
                                                            "isl_token_avg":  1200,
                                                            "isl_token_min":  300,
                                                            "isl_token_max":  4096,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  900,
                                                            "osl_token_min":  225,
                                                            "osl_token_max":  3000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  5,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  260,
                                                            "target_p99_ttft_ms":  1000,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_3h_qwen3_moe",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  260,
                                          "seed":  103
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-70-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  70,
                                      "active_end_at_offset_minutes":  140,
                                      "duration_minutes":  70,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_192gpu-70-3"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_192gpu-70"
}
```

## T+01:35 - Long Context and Reports

```json
{
    "submit_at_offset_minutes":  95,
    "description":  "long-context online summary plus deadline report batch",
    "jobs":  [
                 {
                     "user_id":  "tenant_meeting_summary",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-Next-80B-A3B-Thinking",
                                       "job_features":  {
                                                            "isl_token_avg":  24000,
                                                            "isl_token_min":  6000,
                                                            "isl_token_max":  128000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  1200,
                                                            "osl_token_min":  300,
                                                            "osl_token_max":  4000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  60,
                                                            "target_p99_ttft_ms":  3000,
                                                            "target_p99_tpot_ms":  70
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_3h_long_context_online",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  60,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-95-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  95,
                                      "active_end_at_offset_minutes":  150,
                                      "duration_minutes":  55,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_192gpu-95-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_reports",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-Next-80B-A3B-Thinking",
                                       "job_features":  {
                                                            "isl_token_avg":  18000,
                                                            "isl_token_min":  4500,
                                                            "isl_token_max":  64000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  1400,
                                                            "osl_token_min":  350,
                                                            "osl_token_max":  5000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  69840000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  1.25,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  140
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_3h_reports",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  36,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "report_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-95-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  95,
                                      "deadline_at_offset_minutes":  170,
                                      "campaign_id":  "campaign-benchmark_timing_real_3hr_192gpu-95-2",
                                      "shard_count":  36,
                                      "requests_per_shard":  100,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_192gpu-95"
}
```

## T+02:00 - P0 Safety Replay

```json
{
    "submit_at_offset_minutes":  120,
    "description":  "urgent safety batch should win over P3/P4 work without collapsing online APIs",
    "jobs":  [
                 {
                     "user_id":  "tenant_safety_p0",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  2200,
                                                            "isl_token_min":  550,
                                                            "isl_token_max":  8000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  700,
                                                            "osl_token_min":  175,
                                                            "osl_token_max":  2500,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  16240000,
                                                            "priority_class":  "HIGH",
                                                            "deadline_hrs":  0.75,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  100
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_3h_p0_safety_replay",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  28,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "safety_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-120-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  120,
                                      "deadline_at_offset_minutes":  165,
                                      "campaign_id":  "campaign-benchmark_timing_real_3hr_192gpu-120-1",
                                      "shard_count":  28,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_192gpu-120"
}
```

## T+02:25 - DeepSeek Held-Out Probe and Cooldown

```json
{
    "submit_at_offset_minutes":  145,
    "description":  "held-out frontier model probes transfer/generalization late in the run",
    "jobs":  [
                 {
                     "user_id":  "tenant_new_model_eval",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "deepseek-ai/DeepSeek-V3",
                                       "job_features":  {
                                                            "isl_token_avg":  1800,
                                                            "isl_token_min":  450,
                                                            "isl_token_max":  6000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  1600,
                                                            "osl_token_min":  400,
                                                            "osl_token_max":  6000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  10880000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  0.5833,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  180
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_3h_deepseek_probe",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  16,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_192gpu-145-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  145,
                                      "deadline_at_offset_minutes":  180,
                                      "campaign_id":  "campaign-benchmark_timing_real_3hr_192gpu-145-1",
                                      "shard_count":  16,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_192gpu-145"
}
```
