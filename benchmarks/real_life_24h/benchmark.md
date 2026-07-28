# Real-life 24-hour benchmark

Benchmark ID: `real_life_24h`

Benchmark name: `KoiBench-Real-24h-160GPU-TPU-MegaCo`

Purpose: synthetic job replay on a real-cluster 24 hour representative company-day run. This is the real-hardware companion to the larger simulated paper trace. It keeps the same job structure but scales load to 160 GPUs plus 32 TPU chips.

Horizon: 1440 minutes

Tick: 5 minutes

Engine: `vllm` behind Dynamo-style serving

Markets: `reserved`, `on_demand`

Serving mode: aggregate serving with LMCache, no prefill/decode disaggregation

Real fleet target: 160 GPUs + 32 TPU chips = 192 accelerator units



## Canonical Lifecycle Contract

Schema: `koibench/v1`

Dispatch: releases occur at their exact offset. A job arrival triggers the admission path immediately; the 5-minute tick remains the slow reoptimization loop.

Online lifecycle: `active_start_at_offset_minutes` and `active_end_at_offset_minutes` define a half-open traffic window. `duration_minutes` is retained only as a consistency check. At the end offset the load generator stops new requests, drains in-flight requests, and the executor records `terminal_at`.

Batch lifecycle: `deadline_at_offset_minutes` is an absolute offset from benchmark start. `shard_count` is expanded deterministically because `expansion_policy` is `required`. `requests_per_shard` fixes campaign size. Runtime `started_at`, `completed_at`, and `terminal_at` are produced by the executor, not authored in the job definition.

Overlap: `base_service` remains active, `additive_stream` adds demand to an existing service, and `replace_service` identifies a model/service replacement. Overlap is intentional and additive unless explicitly marked as replacement.

Planner fairness: descriptions, trace names, human-readable tenant labels, and `input_source` are not exposed to planners. Every planner receives the same sanitized job features. Hardware placement and market preferences are deliberately absent from job inputs.

Traffic: `max_concurrent_streaming` is the sole authoritative submitted traffic field. `request_arrival_rate` is omitted so Orca selects concurrency mode and derives request rate. Patterns are limited to `Poisson`, `Constant`, `Bursty`, and `Diurnal`.

Input modes: every online job uses `api_load_generator` to ping an OpenAI-compatible endpoint under its concurrency profile. Every batch campaign uses a finite `dataset` partitioned into required shards.

Drain: after minute 1440, no new work arrives. A 120-minute drain period records batch tardiness and in-flight completion; work unfinished at its deadline is a miss.

Exact manifest counts: **23 online jobs**, **19 batch campaigns**, **1,614 expanded shards**.

## Real-Run Capacity Gate

The listed concurrency caps are frozen only after an isolated preflight on the exact engine/model build. Apply one global concurrency multiplier per run so normal offered load consumes 55-65% of measured qualified capacity, business peaks consume 80-90%, and the designated overload burst reaches 105-120%. Freeze that multiplier before comparing systems. TPU results require a validated vLLM-TPU path; otherwise report TPU separately.
## Frozen Real Cluster Resources

Maximum provisioned capacity: **160 GPUs + 32 TPU chips = 192 typed accelerator devices**. Reserved capacity is pre-provisioned; on-demand capacity is a hard launch quota.

| Accelerator | Node/Slice Shape | Nodes/Slices | Reserved | On-Demand Cap | Total |
|---|---:|---:|---:|---:|---:|
| `H100-SXM-80GB` | 8 GPUs, NVSwitch | 4 | 24 | 8 | 32 |
| `A100-SXM-80GB` | 8 GPUs, NVLink | 4 | 24 | 8 | 32 |
| `A100-PCIe-40GB` | 8 GPUs, PCIe | 2 | 8 | 8 | 16 |
| `A10G-24GB` | 8 GPUs, PCIe | 4 | 24 | 8 | 32 |
| `L40S-48GB` | 8 GPUs, PCIe | 4 | 24 | 8 | 32 |
| `MI300X-192GB` | 8 GPUs, Infinity Fabric | 2 | 8 | 8 | 16 |
| `TPU-v5e` | 8-chip slice, ICI | 4 | 24 | 8 | 32 |
| **Total** |  | **24** | **136** | **56** | **192** |

| Accelerator | `us-east` | `eu-west` | `apac` | Total |
|---|---:|---:|---:|---:|
| H100 | 16 | 8 | 8 | 32 |
| A100-SXM | 16 | 8 | 8 | 32 |
| A100-PCIe | 8 | 8 | 0 | 16 |
| A10G | 16 | 8 | 8 | 32 |
| L40S | 16 | 8 | 8 | 32 |
| MI300X | 8 | 8 | 0 | 16 |
| TPU-v5e | 16 | 8 | 8 | 32 |
| **Region total** | **96** | **56** | **40** | **192** |

| Region | Reserved Devices | On-Demand Cap | Total Devices |
|---|---:|---:|---:|
| `us-east` | 80 | 16 | 96 |
| `eu-west` | 32 | 24 | 56 |
| `apac` | 24 | 16 | 40 |

Network caps: 400 Gbps RDMA for H100/A100-SXM/MI300X nodes; 100 Gbps RoCE for A100-PCIe/A10G/L40S nodes; TPU ICI within each slice. Inter-region caps are 25 Gbps/75 ms RTT (`us-east<->eu-west`), 20 Gbps/145 ms (`us-east<->apac`), and 15 Gbps/220 ms (`eu-west<->apac`). Inferentia and Trainium are not allocated in this real manifest.

## T+00:00 - Persistent Online APIs Warm

```json
{
    "submit_at_offset_minutes":  0,
    "description":  "always-on online services across all regions",
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
                                                            "max_concurrent_streaming":  4000,
                                                            "target_p99_ttft_ms":  700,
                                                            "target_p99_tpot_ms":  55
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_free_chat_us",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  4000,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-0-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-0-1"
                                  }
                 },
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
                                                            "request_arrival_pattern":  "Diurnal",
                                                            "peak_to_mean_ratio":  2,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.7,
                                                            "shared_prefix_length_avg":  225,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  2200,
                                                            "target_p99_ttft_ms":  750,
                                                            "target_p99_tpot_ms":  60
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_free_chat_eu",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  2200,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-0-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-0-2"
                                  }
                 },
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
                                                            "request_arrival_pattern":  "Diurnal",
                                                            "peak_to_mean_ratio":  2,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.7,
                                                            "shared_prefix_length_avg":  225,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  1900,
                                                            "target_p99_ttft_ms":  800,
                                                            "target_p99_tpot_ms":  60
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_free_chat_apac",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  1900,
                                          "seed":  103
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-0-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-0-3"
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
                                                            "max_concurrent_streaming":  240,
                                                            "target_p99_ttft_ms":  650,
                                                            "target_p99_tpot_ms":  40
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_premium_chat",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  240,
                                          "seed":  104
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-0-4",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-0-4"
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
                                                            "max_concurrent_streaming":  1200,
                                                            "target_p99_ttft_ms":  180,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_code_base",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  1200,
                                          "seed":  105
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-0-5",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-0-5"
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
                                                            "max_concurrent_streaming":  12000,
                                                            "target_p99_ttft_ms":  100,
                                                            "target_p99_tpot_ms":  30
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_moderation",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  12000,
                                          "seed":  106
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-0-6",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-0-6"
                                  }
                 },
                 {
                     "user_id":  "tenant_embeddings",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "BAAI/bge-m3",
                                       "job_features":  {
                                                            "isl_token_avg":  384,
                                                            "isl_token_min":  96,
                                                            "isl_token_max":  1024,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  1,
                                                            "osl_token_min":  1,
                                                            "osl_token_max":  1,
                                                            "osl_distribution_type":  "Constant",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  16000,
                                                            "target_p99_ttft_ms":  120,
                                                            "target_p99_tpot_ms":  1
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/embeddings",
                                          "prompt_dataset_id":  "real_24h_embeddings",
                                          "request_arrival_pattern":  "Constant",
                                          "max_concurrent_streaming":  16000,
                                          "seed":  107
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-0-7",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-0-7"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-0"
}
```

## T+00:10 - Nightly Regression Eval

```json
{
    "submit_at_offset_minutes":  10,
    "description":  "nightly eval across core model families",
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
                                                            "total_token_budget":  60000000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  7.8333,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_eval_llama70b",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  100,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-10-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  10,
                                      "deadline_at_offset_minutes":  480,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-10-1",
                                      "shard_count":  100,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_eval_platform",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-72B-Instruct",
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
                                                            "total_token_budget":  54000000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  7.8333,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_eval_qwen72b",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  90,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-10-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  10,
                                      "deadline_at_offset_minutes":  480,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-10-2",
                                      "shard_count":  90,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_eval_platform",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-Coder-32B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  2500,
                                                            "isl_token_min":  625,
                                                            "isl_token_max":  10000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  900,
                                                            "osl_token_min":  225,
                                                            "osl_token_max":  3000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  119000000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  6.8333,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_eval_code",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  70,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-10-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  10,
                                      "deadline_at_offset_minutes":  420,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-10-3",
                                      "shard_count":  70,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-10"
}
```

## T+00:30 - Engine Canary and Quant Calibration

```json
{
    "submit_at_offset_minutes":  30,
    "description":  "calibration job for engine version and quantization behavior",
    "jobs":  [
                 {
                     "user_id":  "tenant_engine_canary",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-14B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  4000,
                                                            "isl_token_min":  1000,
                                                            "isl_token_max":  16000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  400,
                                                            "osl_token_min":  100,
                                                            "osl_token_max":  1200,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  88000000,
                                                            "priority_class":  "HIGH",
                                                            "deadline_hrs":  2.5,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_initial_canary",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  40,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-30-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  30,
                                      "deadline_at_offset_minutes":  180,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-30-1",
                                      "shard_count":  40,
                                      "requests_per_shard":  500,
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
                                                            "total_token_budget":  42336000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  5.5,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  200
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_quant_calibration",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  56,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-30-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  30,
                                      "deadline_at_offset_minutes":  360,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-30-2",
                                      "shard_count":  56,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-30"
}
```

## T+01:00 - Synthetic and Embedding Backfill

```json
{
    "submit_at_offset_minutes":  60,
    "description":  "large low-priority backfills consume off-peak capacity",
    "jobs":  [
                 {
                     "user_id":  "tenant_synthetic_data",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  1200,
                                                            "isl_token_min":  300,
                                                            "isl_token_max":  4096,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  2400,
                                                            "osl_token_min":  600,
                                                            "osl_token_max":  8000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  100800000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  17,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_synthetic_teacher",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  140,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "synth_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-60-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  60,
                                      "deadline_at_offset_minutes":  1080,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-60-1",
                                      "shard_count":  140,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_embeddings",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "BAAI/bge-m3",
                                       "job_features":  {
                                                            "isl_token_avg":  384,
                                                            "isl_token_min":  96,
                                                            "isl_token_max":  1024,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  1,
                                                            "osl_token_min":  1,
                                                            "osl_token_max":  1,
                                                            "osl_distribution_type":  "Constant",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  1001000000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  23,
                                                            "target_p99_ttft_ms":  1000
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_embedding_backfill",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  260,
                                          "requests_per_shard":  10000
                                      },
                     "output_target":  {
                                           "sink":  "vector_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-60-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  60,
                                      "deadline_at_offset_minutes":  1440,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-60-2",
                                      "shard_count":  260,
                                      "requests_per_shard":  10000,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-60"
}
```

## T+02:00 - RAG Index Refresh

```json
{
    "submit_at_offset_minutes":  120,
    "description":  "long-input RAG batch plus online RAG service",
    "jobs":  [
                 {
                     "user_id":  "tenant_search_rag",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-14B-Instruct",
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
                                                            "max_concurrent_streaming":  900,
                                                            "target_p99_ttft_ms":  1500,
                                                            "target_p99_tpot_ms":  55
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_rag_us",
                                          "request_arrival_pattern":  "Poisson",
                                          "max_concurrent_streaming":  900,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-120-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  120,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1320,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-120-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_search_rag",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-14B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  16000,
                                                            "isl_token_min":  4000,
                                                            "isl_token_max":  64000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  350,
                                                            "osl_token_min":  87,
                                                            "osl_token_max":  1200,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  196200000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  6,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_rag_index_refresh",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  120,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "rag_index"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-120-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  120,
                                      "deadline_at_offset_minutes":  480,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-120-2",
                                      "shard_count":  120,
                                      "requests_per_shard":  100,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-120"
}
```

## T+03:00 - Safety and Code Batch

```json
{
    "submit_at_offset_minutes":  180,
    "description":  "safety red-team and code benchmark campaigns",
    "jobs":  [
                 {
                     "user_id":  "tenant_safety_redteam",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  2200,
                                                            "isl_token_min":  550,
                                                            "isl_token_max":  8000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  900,
                                                            "osl_token_min":  225,
                                                            "osl_token_max":  3000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  43400000,
                                                            "priority_class":  "HIGH",
                                                            "deadline_hrs":  12,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_redteam",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  70,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "safety_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-180-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  180,
                                      "deadline_at_offset_minutes":  900,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-180-1",
                                      "shard_count":  70,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_code_eval",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-Coder-32B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  2500,
                                                            "isl_token_min":  625,
                                                            "isl_token_max":  10000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  900,
                                                            "osl_token_min":  225,
                                                            "osl_token_max":  3000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  136000000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  6,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_code_benchmark",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  80,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-180-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  180,
                                      "deadline_at_offset_minutes":  540,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-180-2",
                                      "shard_count":  80,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-180"
}
```

## T+05:00 - EU Business Ramp

```json
{
    "submit_at_offset_minutes":  300,
    "description":  "EU traffic becomes dominant for several hours",
    "jobs":  [
                 {
                     "user_id":  "tenant_search_rag",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-14B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  10000,
                                                            "isl_token_min":  2500,
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
                                                            "shared_prefix_length_avg":  2500,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  1400,
                                                            "target_p99_ttft_ms":  1600,
                                                            "target_p99_tpot_ms":  60
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_eu_rag",
                                          "request_arrival_pattern":  "Poisson",
                                          "max_concurrent_streaming":  1400,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-300-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  300,
                                      "active_end_at_offset_minutes":  660,
                                      "duration_minutes":  360,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-300-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_enterprise_chat",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "microsoft/phi-4",
                                       "job_features":  {
                                                            "isl_token_avg":  1500,
                                                            "isl_token_min":  375,
                                                            "isl_token_max":  5500,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  420,
                                                            "osl_token_min":  105,
                                                            "osl_token_max":  1200,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.7,
                                                            "shared_prefix_length_avg":  375,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  650,
                                                            "target_p99_ttft_ms":  500,
                                                            "target_p99_tpot_ms":  42
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_eu_enterprise",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  650,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-300-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  300,
                                      "active_end_at_offset_minutes":  720,
                                      "duration_minutes":  420,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-300-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-300"
}
```

## T+07:00 - Code Workday Peak

```json
{
    "submit_at_offset_minutes":  420,
    "description":  "code traffic reaches hard-TTFT peak",
    "jobs":  [
                 {
                     "user_id":  "tenant_code_autocomplete",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-Coder-32B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  2200,
                                                            "isl_token_min":  550,
                                                            "isl_token_max":  6000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  96,
                                                            "osl_token_min":  24,
                                                            "osl_token_max":  280,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  3300,
                                                            "target_p99_ttft_ms":  160,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_code_peak",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  3300,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-420-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  420,
                                      "active_end_at_offset_minutes":  900,
                                      "duration_minutes":  480,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-420-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_code_chat",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  2600,
                                                            "isl_token_min":  650,
                                                            "isl_token_max":  10000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  900,
                                                            "osl_token_min":  225,
                                                            "osl_token_max":  3200,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.7,
                                                            "shared_prefix_length_avg":  650,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  180,
                                                            "target_p99_ttft_ms":  800,
                                                            "target_p99_tpot_ms":  42
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_code_chat",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  180,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-420-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  420,
                                      "active_end_at_offset_minutes":  900,
                                      "duration_minutes":  480,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-420-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-420"
}
```

## T+09:00 - Enterprise Peak

```json
{
    "submit_at_offset_minutes":  540,
    "description":  "enterprise chat, docs QA, and legal jobs overlap",
    "jobs":  [
                 {
                     "user_id":  "tenant_enterprise_chat",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-72B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  1600,
                                                            "isl_token_min":  400,
                                                            "isl_token_max":  6000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  500,
                                                            "osl_token_min":  125,
                                                            "osl_token_max":  1600,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.7,
                                                            "shared_prefix_length_avg":  400,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  220,
                                                            "target_p99_ttft_ms":  700,
                                                            "target_p99_tpot_ms":  42
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_enterprise_72b",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  220,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-540-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  540,
                                      "active_end_at_offset_minutes":  960,
                                      "duration_minutes":  420,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-540-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_legal_finance",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  3000,
                                                            "isl_token_min":  750,
                                                            "isl_token_max":  12000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  700,
                                                            "osl_token_min":  175,
                                                            "osl_token_max":  2200,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Diurnal",
                                                            "peak_to_mean_ratio":  2,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "HIGH",
                                                            "max_concurrent_streaming":  90,
                                                            "target_p99_ttft_ms":  1000,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_legal_finance",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  90,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-540-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  540,
                                      "active_end_at_offset_minutes":  900,
                                      "duration_minutes":  360,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-540-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-540"
}
```

## T+10:00 - Engine Regression Probe

```json
{
    "submit_at_offset_minutes":  600,
    "description":  "engine-version probe repeats under peak load",
    "jobs":  [
                 {
                     "user_id":  "tenant_engine_canary",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-14B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  4000,
                                                            "isl_token_min":  1000,
                                                            "isl_token_max":  16000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  400,
                                                            "osl_token_min":  100,
                                                            "osl_token_max":  1200,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  105600000,
                                                            "priority_class":  "HIGH",
                                                            "deadline_hrs":  3,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_engine_regression_probe",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  48,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-600-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  600,
                                      "deadline_at_offset_minutes":  780,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-600-1",
                                      "shard_count":  48,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-600"
}
```

## T+11:30 - Long-Input RAG Burst

```json
{
    "submit_at_offset_minutes":  690,
    "description":  "largest prefill-heavy online burst in the real run",
    "jobs":  [
                 {
                     "user_id":  "tenant_search_rag",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-72B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  24000,
                                                            "isl_token_min":  6000,
                                                            "isl_token_max":  96000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  260,
                                                            "osl_token_min":  65,
                                                            "osl_token_max":  900,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.3,
                                                            "shared_prefix_length_avg":  4000,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  260,
                                                            "target_p99_ttft_ms":  1800,
                                                            "target_p99_tpot_ms":  65
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_rag_lunch_burst",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  260,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-690-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  690,
                                      "active_end_at_offset_minutes":  780,
                                      "duration_minutes":  90,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-690-1"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-690"
}
```

## T+12:00 - Midday Batch Wave

```json
{
    "submit_at_offset_minutes":  720,
    "description":  "second index refresh and MoE eval arrive mid-day",
    "jobs":  [
                 {
                     "user_id":  "tenant_search_rag",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-14B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  16000,
                                                            "isl_token_min":  4000,
                                                            "isl_token_max":  64000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  350,
                                                            "osl_token_min":  87,
                                                            "osl_token_max":  1200,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  147150000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  6,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_midday_index_refresh",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  90,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "rag_index"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-720-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  720,
                                      "deadline_at_offset_minutes":  1080,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-720-1",
                                      "shard_count":  90,
                                      "requests_per_shard":  100,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_moe_eval",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-30B-A3B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  1000,
                                                            "isl_token_min":  250,
                                                            "isl_token_max":  4096,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  1200,
                                                            "osl_token_min":  300,
                                                            "osl_token_max":  5000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  77000000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  10,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  150
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_midday_moe_eval",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  70,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-720-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  720,
                                      "deadline_at_offset_minutes":  1320,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-720-2",
                                      "shard_count":  70,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-720"
}
```

## T+14:00 - Product Launch Burst

```json
{
    "submit_at_offset_minutes":  840,
    "description":  "highest real online burst, should use on-demand without SLO collapse",
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
                                                            "max_concurrent_streaming":  10000,
                                                            "target_p99_ttft_ms":  650,
                                                            "target_p99_tpot_ms":  50
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_launch_chat",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  10000,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-840-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  840,
                                      "active_end_at_offset_minutes":  920,
                                      "duration_minutes":  80,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-840-1"
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
                                                            "max_concurrent_streaming":  30000,
                                                            "target_p99_ttft_ms":  90,
                                                            "target_p99_tpot_ms":  30
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_launch_moderation",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  30000,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-840-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  840,
                                      "active_end_at_offset_minutes":  920,
                                      "duration_minutes":  80,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-840-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-840"
}
```

## T+15:30 - New Frontier Model Arrival

```json
{
    "submit_at_offset_minutes":  930,
    "description":  "DeepSeek appears late as held-out frontier work",
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
                                                            "total_token_budget":  29920000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  6.5,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  180
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_deepseek_eval",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  44,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-930-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  930,
                                      "deadline_at_offset_minutes":  1320,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-930-1",
                                      "shard_count":  44,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_agents_premium",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "deepseek-ai/DeepSeek-V3",
                                       "job_features":  {
                                                            "isl_token_avg":  1400,
                                                            "isl_token_min":  350,
                                                            "isl_token_max":  4096,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  3800,
                                                            "osl_token_min":  950,
                                                            "osl_token_max":  12000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "HIGH",
                                                            "max_concurrent_streaming":  60,
                                                            "target_p99_ttft_ms":  2800,
                                                            "target_p99_tpot_ms":  55
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_deepseek_online",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  60,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-930-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  930,
                                      "active_end_at_offset_minutes":  1110,
                                      "duration_minutes":  180,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-930-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-930"
}
```

## T+16:00 - P0 Safety Incident

```json
{
    "submit_at_offset_minutes":  960,
    "description":  "urgent safety work should displace P3/P4 batch but preserve online SLOs",
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
                                                            "total_token_budget":  34800000,
                                                            "priority_class":  "HIGH",
                                                            "deadline_hrs":  2,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  100
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_p0_safety",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  60,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "safety_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-960-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  960,
                                      "deadline_at_offset_minutes":  1080,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-960-1",
                                      "shard_count":  60,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_safety_reasoner",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "microsoft/phi-4",
                                       "job_features":  {
                                                            "isl_token_avg":  900,
                                                            "isl_token_min":  225,
                                                            "isl_token_max":  3000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  160,
                                                            "osl_token_min":  40,
                                                            "osl_token_max":  600,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "HIGH",
                                                            "max_concurrent_streaming":  1200,
                                                            "target_p99_ttft_ms":  350,
                                                            "target_p99_tpot_ms":  40
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_p0_safety_live",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  1200,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-960-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  960,
                                      "active_end_at_offset_minutes":  1050,
                                      "duration_minutes":  90,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-960-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-960"
}
```

## T+18:00 - Decode-Heavy Agent Wave

```json
{
    "submit_at_offset_minutes":  1080,
    "description":  "long-output frontier traffic becomes the binding job",
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
                                                            "max_concurrent_streaming":  140,
                                                            "target_p99_ttft_ms":  2500,
                                                            "target_p99_tpot_ms":  50
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_kimi_evening",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  140,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-1080-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  1080,
                                      "active_end_at_offset_minutes":  1260,
                                      "duration_minutes":  180,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-1080-1"
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
                                                            "max_concurrent_streaming":  150,
                                                            "target_p99_ttft_ms":  2300,
                                                            "target_p99_tpot_ms":  55
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_glm_evening",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  150,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-1080-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  1080,
                                      "active_end_at_offset_minutes":  1260,
                                      "duration_minutes":  180,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-1080-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-1080"
}
```

## T+20:00 - APAC Online Ramp

```json
{
    "submit_at_offset_minutes":  1200,
    "description":  "APAC becomes the dominant region while US cools",
    "jobs":  [
                 {
                     "user_id":  "tenant_chat_free",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "mistralai/Mixtral-8x7B-Instruct-v0.1",
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
                                                            "max_concurrent_streaming":  6000,
                                                            "target_p99_ttft_ms":  800,
                                                            "target_p99_tpot_ms":  60
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_apac_chat_peak",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  6000,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-1200-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  1200,
                                      "active_end_at_offset_minutes":  1410,
                                      "duration_minutes":  210,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-1200-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_search_rag",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-14B-Instruct",
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
                                                            "max_concurrent_streaming":  1200,
                                                            "target_p99_ttft_ms":  1700,
                                                            "target_p99_tpot_ms":  60
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "real_24h_apac_rag",
                                          "request_arrival_pattern":  "Poisson",
                                          "max_concurrent_streaming":  1200,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-1200-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  1200,
                                      "active_end_at_offset_minutes":  1410,
                                      "duration_minutes":  210,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_24hr_160gpu_tpu-1200-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-1200"
}
```

## T+22:00 - Report Generation Wave

```json
{
    "submit_at_offset_minutes":  1320,
    "description":  "end-of-day deadline batch must complete during cooldown",
    "jobs":  [
                 {
                     "user_id":  "tenant_reports",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-72B-Instruct",
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
                                                            "total_token_budget":  232800000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  2,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_reports_eod",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  120,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "report_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-1320-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  1320,
                                      "deadline_at_offset_minutes":  1440,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-1320-1",
                                      "shard_count":  120,
                                      "requests_per_shard":  100,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_transcripts",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "mistralai/Mixtral-8x7B-Instruct-v0.1",
                                       "job_features":  {
                                                            "isl_token_avg":  16000,
                                                            "isl_token_min":  4000,
                                                            "isl_token_max":  64000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  900,
                                                            "osl_token_min":  225,
                                                            "osl_token_max":  3000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  169000000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  2,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_transcripts_eod",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  100,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "summary_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-1320-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  1320,
                                      "deadline_at_offset_minutes":  1440,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-1320-2",
                                      "shard_count":  100,
                                      "requests_per_shard":  100,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-1320"
}
```

## T+23:30 - Cooldown and Final Probes

```json
{
    "submit_at_offset_minutes":  1410,
    "description":  "low-priority probes fill remaining capacity while measuring final churn/convergence",
    "jobs":  [
                 {
                     "user_id":  "tenant_koi_profiling",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-30B-A3B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  1000,
                                                            "isl_token_min":  250,
                                                            "isl_token_max":  4096,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  800,
                                                            "osl_token_min":  200,
                                                            "osl_token_max":  3000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  28800000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  0.5,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  200
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_final_moe_probe",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  32,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-1410-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  1410,
                                      "deadline_at_offset_minutes":  1440,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-1410-1",
                                      "shard_count":  32,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_engine_canary",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-14B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  4000,
                                                            "isl_token_min":  1000,
                                                            "isl_token_max":  16000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  400,
                                                            "osl_token_min":  100,
                                                            "osl_token_max":  1200,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  52800000,
                                                            "priority_class":  "HIGH",
                                                            "deadline_hrs":  0.5,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  200
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "real_24h_final_canary",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  24,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_real_24hr_160gpu_tpu-1410-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  1410,
                                      "deadline_at_offset_minutes":  1440,
                                      "campaign_id":  "campaign-benchmark_timing_real_24hr_160gpu_tpu-1410-2",
                                      "shard_count":  24,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_24hr_160gpu_tpu-1410"
}
```
