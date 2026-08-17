# Real 3-hour homogeneous L40S benchmark

Benchmark ID: `real_3h_homogenous`

Purpose: real-hardware enterprise workload replay on a homogeneous, on-demand L40S cluster. It preserves the simulation 3-hour benchmark's online and batch mix, timing, traffic patterns, seeds, SLO classes, quantization probes, and cache behavior so hardware and scheduling results remain directly comparable.

Benchmark name: `KoiBench-Real-3h-OnDemand-L40S`

Horizon: 180 minutes

Tick: 5 minutes

Market: `on_demand`

Engine: `vllm` behind Dynamo-style serving

Serving mode: aggregate serving with LMCache, no prefill/decode disaggregation

Region: `us-east`

Active accelerator catalog: `L40S-48GB`

Model precision: workload-defined `fp16`, `bf16`, or `bf16-fp8` according to the placement policy

## Frozen Cluster Resources

Maximum provisioned capacity: **128 on-demand L40S GPUs**. The complete fleet is provisioned before T+00:00; provisioning time is outside the benchmark horizon.

| Accelerator | Node Shape | Nodes | On-Demand | Total |
|---|---:|---:|---:|---:|
| `L40S-48GB` | 8 GPUs, PCIe | 16 | 128 | 128 |

All nodes are in `us-east` and use 100 Gbps RoCE. There is no reserved capacity, heterogeneous fallback, or inter-region placement.

The fixed-placement envelope peaks at **119 GPUs**, leaving nine devices of cluster headroom. This intentionally keeps the run capacity-bound while allowing scheduler fragmentation and short-lived overlap to surface.

## Workload Compatibility

The workload is the same 18 online services and 7 batch campaigns used by `simulation_3h`. Benchmark-specific job, service, campaign, and event IDs prevent cross-run collisions while unchanged seeds and phase definitions preserve comparison fidelity.

The L40S policy covers every model in the inherited workload. Large frontier models may span multiple eight-GPU nodes; no job may fall back to another accelerator type or market.

## Canonical Lifecycle Contract

Schema: `koibench/v1`

Dispatch: releases occur at their exact offset. A job arrival triggers the admission path immediately; the 5-minute tick remains the slow reoptimization loop.

Online lifecycle: `active_start_at_offset_minutes` and `active_end_at_offset_minutes` define a half-open traffic window. `duration_minutes` is retained only as a consistency check. At the end offset the load generator stops new requests, drains in-flight requests, and the executor records `terminal_at`.

Batch lifecycle: `deadline_at_offset_minutes` is an absolute offset from benchmark start. `shard_count` is expanded deterministically because `expansion_policy` is `required`. `requests_per_shard` fixes campaign size and `max_in_flight_requests` bounds each simulator dispatch batch. Runtime `started_at`, `completed_at`, and `terminal_at` are produced by the executor, not authored in the job definition.

Overlap: `base_service` remains active, `additive_stream` adds demand to an existing service, and `replace_service` identifies a model/service replacement. Overlap is intentional and additive unless explicitly marked as replacement.

Planner fairness: descriptions, trace names, human-readable tenant labels, and `input_source` are not exposed to planners. Every planner receives the same sanitized job features. Hardware placement and market preferences are deliberately absent from job inputs.

Traffic: `max_concurrent_streaming` is the planner-facing service cap; the matching AIPerf phase `concurrency` is the client cap. AIPerf phase `rate` is the submitted request rate and is deliberately bounded by the capacity envelope. `request_arrival_rate` is omitted from the Orca payload so Orca remains in concurrency mode. Patterns are limited to `Poisson`, `Constant`, `Bursty`, and `Diurnal`.

Input modes: every online job uses `api_load_generator` to ping an OpenAI-compatible endpoint under its concurrency profile. Every batch campaign uses a finite `dataset` partitioned into required shards.

Drain: after minute 180, no new work arrives. A 30-minute drain period records batch tardiness and in-flight completion; work unfinished at its deadline is a miss.

Exact manifest counts: **18 online jobs**, **7 batch campaigns**, **332 expanded shards**.

## T+00:00 - Warm persistent core APIs

```json
{
    "submit_at_offset_minutes":  0,
    "description":  "warm always-on online APIs in us-east",
    "jobs":  [
                 {
                     "user_id":  "tenant_chat_free",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.2-1B-Instruct",
                                       "precision":  "fp16",
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
                                                            "priority_class":  "LOW",
                                                             "max_concurrent_streaming":  64,
                                                            "target_p99_ttft_ms":  700,
                                                            "target_p99_tpot_ms":  55
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "synthetic_chat_multiturn",
                                          "request_arrival_pattern":  "Diurnal",
                                           "max_concurrent_streaming":  64,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-0-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  180,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-0-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_chat_premium",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "precision":  "fp16",
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
                                                             "max_concurrent_streaming":  24,
                                                            "target_p99_ttft_ms":  600,
                                                            "target_p99_tpot_ms":  38
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "premium_chat",
                                          "request_arrival_pattern":  "Diurnal",
                                           "max_concurrent_streaming":  24,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-0-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  180,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-0-2"
                                  }
                 },
                 {
                     "user_id":  "tenant_code_autocomplete",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-Coder-32B-Instruct",
                                       "precision":  "fp16",
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
                                                             "max_concurrent_streaming":  32,
                                                            "target_p99_ttft_ms":  180,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "code_completion",
                                          "request_arrival_pattern":  "Bursty",
                                           "max_concurrent_streaming":  32,
                                          "seed":  103
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-0-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  180,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-0-3"
                                  }
                 },
                 {
                     "user_id":  "tenant_safety_moderation",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.2-1B-Instruct",
                                       "precision":  "fp16",
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
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "HIGH",
                                                             "max_concurrent_streaming":  128,
                                                            "target_p99_ttft_ms":  100,
                                                            "target_p99_tpot_ms":  30
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "moderation_stream",
                                          "request_arrival_pattern":  "Bursty",
                                           "max_concurrent_streaming":  128,
                                          "seed":  104
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-0-4",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  180,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-0-4"
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
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-0-5",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  180,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-0-5"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_homogenous-0"
}
```

## T+00:07 - Search and support APIs

```json
{
    "submit_at_offset_minutes":  7,
    "description":  "search, reranking, support, and translation come online",
    "jobs":  [
                 {
                     "user_id":  "tenant_search_rag",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "microsoft/phi-4",
                                       "precision":  "fp16",
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
                                                             "max_concurrent_streaming":  24,
                                                            "target_p99_ttft_ms":  1500,
                                                            "target_p99_tpot_ms":  55
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "rag_long_input",
                                          "request_arrival_pattern":  "Poisson",
                                           "max_concurrent_streaming":  24,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-7-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  7,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  173,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-7-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_search_rerank",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "BAAI/bge-reranker-v2-m3",
                                       "precision":  "fp16",
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
                                                             "max_concurrent_streaming":  32,
                                                            "target_p99_ttft_ms":  160,
                                                            "target_p99_tpot_ms":  1
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/rerank",
                                          "prompt_dataset_id":  "reranker_api",
                                          "request_arrival_pattern":  "Bursty",
                                           "max_concurrent_streaming":  32,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-7-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  7,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  173,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-7-2"
                                  }
                 },
                 {
                     "user_id":  "tenant_support",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "mistralai/Mixtral-8x7B-Instruct-v0.1",
                                       "precision":  "fp16",
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
                                                             "max_concurrent_streaming":  64,
                                                            "target_p99_ttft_ms":  700,
                                                            "target_p99_tpot_ms":  50
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "support_assistant",
                                          "request_arrival_pattern":  "Bursty",
                                           "max_concurrent_streaming":  64,
                                          "seed":  103
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-7-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  7,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  173,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-7-3"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_homogenous-7"
}
```

## T+00:15 - Batch campaigns begin

```json
{
    "submit_at_offset_minutes":  15,
    "description":  "small eval and quant profiling campaigns",
    "jobs":  [
                 {
                     "user_id":  "tenant_eval_platform",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "precision":  "fp16",
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
                                                            "total_token_budget":  43200000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  2.25,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "mini_nightly_eval",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  72,
                                           "requests_per_shard":  200,
                                           "max_in_flight_requests":  4
                                       },
                                       "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-15-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  15,
                                      "deadline_at_offset_minutes":  150,
                                      "campaign_id":  "campaign-benchmark_timing_real_3hr_homogenous-15-1",
                                      "shard_count":  72,
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
                                       "precision":  "fp16",
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
                                                            "total_token_budget":  27216000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  1.75,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  200
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "quant_probe_phi4",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  36,
                                           "requests_per_shard":  500,
                                           "max_in_flight_requests":  16
                                       },
                                       "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-15-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  15,
                                      "deadline_at_offset_minutes":  120,
                                      "campaign_id":  "campaign-benchmark_timing_real_3hr_homogenous-15-3",
                                      "shard_count":  36,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_homogenous-15"
}
```

## T+00:32 - Code and enterprise burst

```json
{
    "submit_at_offset_minutes":  32,
    "description":  "regional code burst and enterprise chat start",
    "jobs":  [
                 {
                     "user_id":  "tenant_code_autocomplete",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-Coder-32B-Instruct",
                                       "precision":  "fp16",
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
                                                             "max_concurrent_streaming":  96,
                                                            "target_p99_ttft_ms":  160,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "code_morning_burst",
                                          "request_arrival_pattern":  "Bursty",
                                           "max_concurrent_streaming":  96,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-32-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  32,
                                      "active_end_at_offset_minutes":  87,
                                      "duration_minutes":  55,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-32-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_enterprise_chat",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "microsoft/phi-4",
                                       "precision":  "fp16",
                                       "job_features":  {
                                                            "isl_token_avg":  1500,
                                                            "isl_token_min":  375,
                                                            "isl_token_max":  5500,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  420,
                                                            "osl_token_min":  105,
                                                            "osl_token_max":  1200,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Diurnal",
                                                            "peak_to_mean_ratio":  2,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.7,
                                                            "shared_prefix_length_avg":  375,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                             "max_concurrent_streaming":  24,
                                                            "target_p99_ttft_ms":  500,
                                                            "target_p99_tpot_ms":  42
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "enterprise_chat",
                                          "request_arrival_pattern":  "Diurnal",
                                           "max_concurrent_streaming":  24,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-32-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  32,
                                      "active_end_at_offset_minutes":  152,
                                      "duration_minutes":  120,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-32-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_homogenous-32"
}
```

## T+00:49 - Product launch burst

```json
{
    "submit_at_offset_minutes":  49,
    "description":  "short product-launch burst, tight tail-latency target",
    "jobs":  [
                 {
                     "user_id":  "tenant_chat_free",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "mistralai/Mixtral-8x7B-Instruct-v0.1",
                                       "precision":  "fp16",
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
                                                             "max_concurrent_streaming":  64,
                                                            "target_p99_ttft_ms":  650,
                                                            "target_p99_tpot_ms":  50
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "product_launch_chat_burst",
                                          "request_arrival_pattern":  "Bursty",
                                           "max_concurrent_streaming":  64,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-49-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  49,
                                      "active_end_at_offset_minutes":  84,
                                      "duration_minutes":  35,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-49-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_safety_moderation",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.2-1B-Instruct",
                                       "precision":  "fp16",
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
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "HIGH",
                                                             "max_concurrent_streaming":  128,
                                                            "target_p99_ttft_ms":  90,
                                                            "target_p99_tpot_ms":  30
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "moderation_launch_burst",
                                          "request_arrival_pattern":  "Bursty",
                                           "max_concurrent_streaming":  128,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-49-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  49,
                                      "active_end_at_offset_minutes":  84,
                                      "duration_minutes":  35,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-49-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_homogenous-49"
}
```

## T+01:07 - Frontier reasoning wave

```json
{
    "submit_at_offset_minutes":  67,
    "description":  "frontier reasoning jobs exercise MoE and large-model planning",
    "jobs":  [
                 {
                     "user_id":  "tenant_agents_premium",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "moonshotai/Kimi-K2-Instruct",
                                       "precision":  "fp16",
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
                                                             "max_concurrent_streaming":  16,
                                                            "target_p99_ttft_ms":  2500,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "agentic_decode_heavy",
                                          "request_arrival_pattern":  "Bursty",
                                           "max_concurrent_streaming":  16,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-67-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  67,
                                      "active_end_at_offset_minutes":  157,
                                      "duration_minutes":  90,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-67-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_frontier_glm",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "zai-org/GLM-4.5-Air",
                                       "precision":  "fp16",
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
                                                             "max_concurrent_streaming":  16,
                                                            "target_p99_ttft_ms":  2200,
                                                            "target_p99_tpot_ms":  50
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "frontier_glm_reasoning",
                                          "request_arrival_pattern":  "Bursty",
                                           "max_concurrent_streaming":  16,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-67-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  67,
                                      "active_end_at_offset_minutes":  142,
                                      "duration_minutes":  75,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-67-2"
                                  }
                 },
                 {
                     "user_id":  "tenant_moe_routing",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-Next-80B-A3B-Thinking",
                                       "precision":  "fp16",
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
                                                             "max_concurrent_streaming":  16,
                                                            "target_p99_ttft_ms":  1000,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "small_active_moe",
                                          "request_arrival_pattern":  "Bursty",
                                           "max_concurrent_streaming":  16,
                                          "seed":  103
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-67-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  67,
                                      "active_end_at_offset_minutes":  142,
                                      "duration_minutes":  75,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-67-3"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_homogenous-67"
}
```

## T+01:25 - Calibration and quantization probes

```json
{
    "submit_at_offset_minutes":  85,
    "description":  "controlled probes for fp32, fp4, int4, and fp8 behavior",
    "jobs":  [
                 {
                     "user_id":  "tenant_koi_profiling",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "microsoft/phi-4",
                                       "precision":  "fp16",
                                       "job_features":  {
                                                            "isl_token_avg":  1400,
                                                            "isl_token_min":  350,
                                                            "isl_token_max":  4096,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  512,
                                                            "osl_token_min":  128,
                                                            "osl_token_max":  1600,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  22944000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  1.4167,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  200
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "calibration_probe",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  24,
                                           "requests_per_shard":  500,
                                           "max_in_flight_requests":  16
                                       },
                                       "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-85-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  85,
                                      "deadline_at_offset_minutes":  170,
                                      "campaign_id":  "campaign-benchmark_timing_real_3hr_homogenous-85-1",
                                      "shard_count":  24,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_homogenous-85"
}
```

## T+01:40 - P0 safety incident

```json
{
    "submit_at_offset_minutes":  100,
    "description":  "urgent safety replay should displace low-priority work if needed",
    "jobs":  [
                 {
                     "user_id":  "tenant_safety_p0",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "precision":  "fp16",
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
                                                            "total_token_budget":  23200000,
                                                            "priority_class":  "HIGH",
                                                            "deadline_hrs":  1,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  100
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "abuse_incident_replay",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  40,
                                           "requests_per_shard":  200,
                                           "max_in_flight_requests":  12
                                       },
                                       "output_target":  {
                                           "sink":  "safety_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-100-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  100,
                                      "deadline_at_offset_minutes":  160,
                                      "campaign_id":  "campaign-benchmark_timing_real_3hr_homogenous-100-1",
                                      "shard_count":  40,
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
                                       "precision":  "fp16",
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
                                                             "max_concurrent_streaming":  32,
                                                            "target_p99_ttft_ms":  350,
                                                            "target_p99_tpot_ms":  40
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "safety_p0_live",
                                          "request_arrival_pattern":  "Bursty",
                                           "max_concurrent_streaming":  32,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-100-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  100,
                                      "active_end_at_offset_minutes":  150,
                                      "duration_minutes":  50,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-100-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_homogenous-100"
}
```

## T+02:05 - Long-context and reports

```json
{
    "submit_at_offset_minutes":  125,
    "description":  "long-context summarization and report batch stress memory/KV behavior",
    "jobs":  [
                 {
                     "user_id":  "tenant_meeting_summary",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-Next-80B-A3B-Thinking",
                                       "precision":  "fp16",
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
                                                             "max_concurrent_streaming":  8,
                                                            "target_p99_ttft_ms":  3000,
                                                            "target_p99_tpot_ms":  65
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "long_context_meetings",
                                          "request_arrival_pattern":  "Bursty",
                                           "max_concurrent_streaming":  8,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-125-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  125,
                                      "active_end_at_offset_minutes":  170,
                                      "duration_minutes":  45,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-125-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_reports",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-Next-80B-A3B-Thinking",
                                       "precision":  "fp16",
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
                                                            "total_token_budget":  155200000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  0.9167,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "sales_reports_short",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  80,
                                           "requests_per_shard":  100,
                                           "max_in_flight_requests":  64
                                       },
                                       "output_target":  {
                                           "sink":  "report_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-125-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  125,
                                      "deadline_at_offset_minutes":  180,
                                      "campaign_id":  "campaign-benchmark_timing_real_3hr_homogenous-125-2",
                                      "shard_count":  80,
                                      "requests_per_shard":  100,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_homogenous-125"
}
```

## T+02:25 - Legal audit and DeepSeek held-out arrival

```json
{
    "submit_at_offset_minutes":  145,
    "description":  "strict-quality legal work and a held-out frontier model arrival",
    "jobs":  [
                 {
                     "user_id":  "tenant_legal_finance",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "precision":  "fp16",
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
                                                             "max_concurrent_streaming":  16,
                                                            "target_p99_ttft_ms":  1000,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "legal_review",
                                          "request_arrival_pattern":  "Diurnal",
                                           "max_concurrent_streaming":  16,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-145-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  145,
                                      "active_end_at_offset_minutes":  180,
                                      "duration_minutes":  35,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_real_3hr_homogenous-145-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_new_model_eval",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "deepseek-ai/DeepSeek-V3",
                                       "precision":  "fp16",
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
                                                            "total_token_budget":  21760000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  0.5833,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "heldout_frontier_eval",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  32,
                                           "requests_per_shard":  200,
                                           "max_in_flight_requests":  8
                                       },
                                       "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-145-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  145,
                                      "deadline_at_offset_minutes":  180,
                                      "campaign_id":  "campaign-benchmark_timing_real_3hr_homogenous-145-2",
                                      "shard_count":  32,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_homogenous-145"
}
```

## T+02:40 - Cooldown and exploration probes

```json
{
    "submit_at_offset_minutes":  160,
    "description":  "low-priority probes fill remaining capacity while online APIs continue",
    "jobs":  [
                 {
                     "user_id":  "tenant_koi_profiling",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-Next-80B-A3B-Thinking",
                                       "precision":  "fp16",
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
                                                            "total_token_budget":  43200000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  0.3333,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  200
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "moe_quant_probe",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  48,
                                           "requests_per_shard":  500,
                                           "max_in_flight_requests":  32
                                       },
                                       "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_real_3hr_homogenous-160-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  160,
                                      "deadline_at_offset_minutes":  180,
                                      "campaign_id":  "campaign-benchmark_timing_real_3hr_homogenous-160-1",
                                      "shard_count":  48,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_real_3hr_homogenous-160"
}
```
