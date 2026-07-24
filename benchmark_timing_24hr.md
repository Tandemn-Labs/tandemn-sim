# benchmark_timing_24hr.md

Purpose: paper-scale daily trace for a large enterprise LLM inference cluster. This is the comprehensive 24 hour benchmark timing file. It models persistent product APIs, regional business-day ramps, batch campaigns decomposed into checkpointable shards, strict quality floors, quantization choices, reserved plus on-demand markets, LMCache, and vLLM/Dynamo-only serving.

Benchmark name: `KoiBench-24h-ReservedFleet-MegaCo`

Horizon: 1440 minutes

Tick: 5 minutes

Markets: `reserved`, `on_demand`

Engine: `vllm` behind Dynamo-style serving

Serving mode: aggregate serving with LMCache, no prefill/decode disaggregation

Regions: `us-east`, `eu-west`, `apac`

Active accelerator catalog: `H100-SXM-80GB`, `A100-SXM-80GB`, `A100-PCIe-40GB`, `A10G-24GB`, `L40S-48GB`, `MI300X-192GB`, `TPU-v5e`

Model catalog: `Qwen/Qwen2.5-1.5B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `microsoft/phi-4`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-Coder-32B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`, `Qwen/Qwen2.5-72B-Instruct`, `Qwen/Qwen3-30B-A3B-Instruct`, `zai-org/GLM-4.5-Air`, `moonshotai/Kimi-K2-Instruct`, `deepseek-ai/DeepSeek-V3`, `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`

## Frozen Cluster Resources

Maximum provisioned capacity: **736 GPUs + 64 TPU chips = 800 typed accelerator devices**. This is the paper-scale simulation fleet and exactly 4x the 3-hour fleet.

| Accelerator | Node/Slice Shape | Nodes/Slices | Reserved | On-Demand Cap | Total |
|---|---:|---:|---:|---:|---:|
| `H100-SXM-80GB` | 8 GPUs, NVSwitch | 16 | 96 | 32 | 128 |
| `A100-SXM-80GB` | 8 GPUs, NVLink | 16 | 96 | 32 | 128 |
| `A100-PCIe-40GB` | 8 GPUs, PCIe | 8 | 32 | 32 | 64 |
| `A10G-24GB` | 8 GPUs, PCIe | 24 | 128 | 64 | 192 |
| `L40S-48GB` | 8 GPUs, PCIe | 20 | 128 | 32 | 160 |
| `MI300X-192GB` | 8 GPUs, Infinity Fabric | 8 | 32 | 32 | 64 |
| `TPU-v5e` | 8-chip slice, ICI | 8 | 32 | 32 | 64 |
| **Total** |  | **100** | **544** | **256** | **800** |

| Accelerator | `us-east` | `eu-west` | `apac` | Total |
|---|---:|---:|---:|---:|
| H100 | 64 | 32 | 32 | 128 |
| A100-SXM | 64 | 32 | 32 | 128 |
| A100-PCIe | 32 | 32 | 0 | 64 |
| A10G | 96 | 64 | 32 | 192 |
| L40S | 64 | 64 | 32 | 160 |
| MI300X | 32 | 32 | 0 | 64 |
| TPU-v5e | 32 | 0 | 32 | 64 |
| **Region total** | **384** | **256** | **160** | **800** |

Network caps: 400 Gbps RDMA between H100/A100-SXM/MI300X nodes within a region; 100 Gbps RoCE for PCIe/A10G/L40S nodes; TPU ICI inside each slice. Inter-region caps are `us-east<->eu-west` 25 Gbps/75 ms RTT, `us-east<->apac` 20 Gbps/145 ms, and `eu-west<->apac` 15 Gbps/220 ms.

Inactive in this manifest: TPU-v5p, Inferentia2, and Trainium2.



## Simulation Independence Gate

The environment emulator used to realize latency and throughput must be independently calibrated from the surrogate exposed to Koi. Test residuals, perturbations, and held-out model/engine shifts must remain hidden from every planner.
## Canonical Lifecycle Contract

Schema: `koibench/v1`

Dispatch: releases occur at their exact offset. A job arrival triggers the admission path immediately; the 5-minute tick remains the slow reoptimization loop.

Online lifecycle: `active_start_at_offset_minutes` and `active_end_at_offset_minutes` define a half-open traffic window. `duration_minutes` is retained only as a consistency check. At the end offset the load generator stops new requests, drains in-flight requests, and the executor records `terminal_at`.

Batch lifecycle: `deadline_at_offset_minutes` is an absolute offset from benchmark start. `shard_count` is expanded deterministically because `expansion_policy` is `required`. `requests_per_shard` fixes campaign size. Runtime `started_at`, `completed_at`, and `terminal_at` are produced by the executor, not authored in the workload.

Overlap: `base_service` remains active, `additive_stream` adds demand to an existing service, and `replace_service` identifies a model/service replacement. Overlap is intentional and additive unless explicitly marked as replacement.

Planner fairness: descriptions, trace names, human-readable tenant labels, and `input_source` are not exposed to planners. Every planner receives the same sanitized workload features. Hardware placement and market preferences are deliberately absent from job inputs.

Traffic: `max_concurrent_streaming` is the sole authoritative submitted traffic field. `request_arrival_rate` is omitted so Orca selects concurrency mode and derives request rate. Patterns are limited to `Poisson`, `Constant`, `Bursty`, and `Diurnal`.

Input modes: every online job uses `api_load_generator` to ping an OpenAI-compatible endpoint under its concurrency profile. Every batch campaign uses a finite `dataset` partitioned into required shards.

Drain: after minute 1440, no new work arrives. A 120-minute drain period records batch tardiness and in-flight completion; work unfinished at its deadline is a miss.

Exact manifest counts: **24 online jobs**, **25 batch campaigns**, **5,344 expanded shards**.

## T+00:00 - All persistent online APIs warm

```json
{
    "submit_at_offset_minutes":  0,
    "description":  "core 24h services across us-east, eu-west, and apac",
    "jobs":  [
                 {
                     "user_id":  "tenant_chat_free",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-1.5B-Instruct",
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
                                                            "max_concurrent_streaming":  17000,
                                                            "target_p99_ttft_ms":  700,
                                                            "target_p99_tpot_ms":  55
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "free_chat_us",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  17000,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-0-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-0-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_chat_free",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-1.5B-Instruct",
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
                                                            "max_concurrent_streaming":  10000,
                                                            "target_p99_ttft_ms":  750,
                                                            "target_p99_tpot_ms":  60
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "free_chat_eu",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  10000,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-0-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-0-2"
                                  }
                 },
                 {
                     "user_id":  "tenant_chat_free",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-1.5B-Instruct",
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
                                                            "max_concurrent_streaming":  9000,
                                                            "target_p99_ttft_ms":  800,
                                                            "target_p99_tpot_ms":  60
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "free_chat_apac",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  9000,
                                          "seed":  103
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-0-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-0-3"
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
                                                            "max_concurrent_streaming":  1300,
                                                            "target_p99_ttft_ms":  600,
                                                            "target_p99_tpot_ms":  38
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "premium_chat_us",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  1300,
                                          "seed":  104
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-0-4",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-0-4"
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
                                                            "max_concurrent_streaming":  6000,
                                                            "target_p99_ttft_ms":  180,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "code_autocomplete_us",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  6000,
                                          "seed":  105
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-0-5",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-0-5"
                                  }
                 },
                 {
                     "user_id":  "tenant_safety_moderation",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-1.5B-Instruct",
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
                                                            "max_concurrent_streaming":  36000,
                                                            "target_p99_ttft_ms":  100,
                                                            "target_p99_tpot_ms":  30
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "moderation_us",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  36000,
                                          "seed":  106
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-0-6",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-0-6"
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
                                                            "max_concurrent_streaming":  65000,
                                                            "target_p99_ttft_ms":  120,
                                                            "target_p99_tpot_ms":  1
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/embeddings",
                                          "prompt_dataset_id":  "embeddings_us",
                                          "request_arrival_pattern":  "Constant",
                                          "max_concurrent_streaming":  65000,
                                          "seed":  107
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-0-7",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-0-7"
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
                                                            "max_concurrent_streaming":  15000,
                                                            "target_p99_ttft_ms":  160,
                                                            "target_p99_tpot_ms":  1
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/rerank",
                                          "prompt_dataset_id":  "rerank_us",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  15000,
                                          "seed":  108
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-0-8",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  1440,
                                      "duration_minutes":  1440,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-0-8"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-0"
}
```

## T+00:10 - Nightly regression eval

```json
{
    "submit_at_offset_minutes":  10,
    "description":  "nightly eval across dense, code, MoE, embedding, and reranker models",
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
                                                            "total_token_budget":  144000000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  7.8333,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "nightly_llama70b",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  240,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-10-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  10,
                                      "deadline_at_offset_minutes":  480,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-10-1",
                                      "shard_count":  240,
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
                                                            "total_token_budget":  132000000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  7.8333,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "nightly_qwen72b",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  220,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-10-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  10,
                                      "deadline_at_offset_minutes":  480,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-10-2",
                                      "shard_count":  220,
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
                                                            "total_token_budget":  272000000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  6.8333,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "nightly_code",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  160,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-10-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  10,
                                      "deadline_at_offset_minutes":  420,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-10-3",
                                      "shard_count":  160,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_eval_platform",
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
                                                            "total_token_budget":  132000000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  7.8333,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  150
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "nightly_qwen3_moe",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  120,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-10-4",
                     "schedule":  {
                                      "release_at_offset_minutes":  10,
                                      "deadline_at_offset_minutes":  480,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-10-4",
                                      "shard_count":  120,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-10"
}
```

## T+00:30 - Engine canary and calibration

```json
{
    "submit_at_offset_minutes":  30,
    "description":  "small canaries populate residual and quantization calibration data",
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
                                                            "total_token_budget":  176000000,
                                                            "priority_class":  "HIGH",
                                                            "deadline_hrs":  2.5,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "engine_canary_initial",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  80,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-30-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  30,
                                      "deadline_at_offset_minutes":  180,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-30-1",
                                      "shard_count":  80,
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
                                                            "total_token_budget":  90720000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  5.5,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  200
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "quant_calibration_phi4",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  120,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-30-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  30,
                                      "deadline_at_offset_minutes":  360,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-30-2",
                                      "shard_count":  120,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-30"
}
```

## T+01:00 - Synthetic data and embedding backfill

```json
{
    "submit_at_offset_minutes":  60,
    "description":  "large low-priority batch campaigns begin filling off-peak capacity",
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
                                                            "total_token_budget":  259200000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  17,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "synthetic_teacher_day",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  360,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "synth_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-60-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  60,
                                      "deadline_at_offset_minutes":  1080,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-60-1",
                                      "shard_count":  360,
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
                                                            "total_token_budget":  2464000000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  23,
                                                            "target_p99_ttft_ms":  1000
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "embedding_backfill_day",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  640,
                                          "requests_per_shard":  10000
                                      },
                     "output_target":  {
                                           "sink":  "vector_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-60-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  60,
                                      "deadline_at_offset_minutes":  1440,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-60-2",
                                      "shard_count":  640,
                                      "requests_per_shard":  10000,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-60"
}
```

## T+02:00 - RAG index refresh

```json
{
    "submit_at_offset_minutes":  120,
    "description":  "long-input RAG index refresh competes with always-on search APIs",
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
                                                            "total_token_budget":  490500000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  6,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "rag_index_refresh_morning",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  300,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "rag_index"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-120-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  120,
                                      "deadline_at_offset_minutes":  480,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-120-1",
                                      "shard_count":  300,
                                      "requests_per_shard":  100,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_product_enrichment",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.1-8B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  1200,
                                                            "isl_token_min":  300,
                                                            "isl_token_max":  4096,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  500,
                                                            "osl_token_min":  125,
                                                            "osl_token_max":  1600,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  306000000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  7,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "product_enrichment",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  180,
                                          "requests_per_shard":  1000
                                      },
                     "output_target":  {
                                           "sink":  "product_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-120-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  120,
                                      "deadline_at_offset_minutes":  540,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-120-2",
                                      "shard_count":  180,
                                      "requests_per_shard":  1000,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-120"
}
```

## T+03:00 - Safety red-team and code benchmark

```json
{
    "submit_at_offset_minutes":  180,
    "description":  "safety and code batch work before business peak",
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
                                                            "total_token_budget":  111600000,
                                                            "priority_class":  "HIGH",
                                                            "deadline_hrs":  12,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "redteam_eval_morning",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  180,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "safety_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-180-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  180,
                                      "deadline_at_offset_minutes":  900,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-180-1",
                                      "shard_count":  180,
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
                                                            "total_token_budget":  340000000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  6,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "code_benchmark_morning",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  200,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-180-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  180,
                                      "deadline_at_offset_minutes":  540,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-180-2",
                                      "shard_count":  200,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-180"
}
```

## T+04:00 - Transcript summary wave

```json
{
    "submit_at_offset_minutes":  240,
    "description":  "long-input summarization batch stresses LMCache and memory pressure",
    "jobs":  [
                 {
                     "user_id":  "tenant_transcripts",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.1-8B-Instruct",
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
                                                            "total_token_budget":  473200000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  4,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "transcript_summary_morning",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  280,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "summary_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-240-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  240,
                                      "deadline_at_offset_minutes":  480,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-240-1",
                                      "shard_count":  280,
                                      "requests_per_shard":  100,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-240"
}
```

## T+05:00 - EU business ramp

```json
{
    "submit_at_offset_minutes":  300,
    "description":  "EU user-facing traffic ramps sharply",
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
                                                            "max_concurrent_streaming":  8000,
                                                            "target_p99_ttft_ms":  1500,
                                                            "target_p99_tpot_ms":  55
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "eu_rag_ramp",
                                          "request_arrival_pattern":  "Poisson",
                                          "max_concurrent_streaming":  8000,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-300-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  300,
                                      "active_end_at_offset_minutes":  660,
                                      "duration_minutes":  360,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-300-1"
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
                                                            "request_arrival_pattern":  "Diurnal",
                                                            "peak_to_mean_ratio":  2,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.7,
                                                            "shared_prefix_length_avg":  375,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  2400,
                                                            "target_p99_ttft_ms":  500,
                                                            "target_p99_tpot_ms":  42
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "eu_enterprise_chat",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  2400,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-300-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  300,
                                      "active_end_at_offset_minutes":  720,
                                      "duration_minutes":  420,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-300-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-300"
}
```

## T+07:00 - Code traffic ramp

```json
{
    "submit_at_offset_minutes":  420,
    "description":  "code autocomplete and code chat hit business-hour load",
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
                                                            "max_concurrent_streaming":  26000,
                                                            "target_p99_ttft_ms":  160,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "code_peak_us",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  26000,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-420-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  420,
                                      "active_end_at_offset_minutes":  900,
                                      "duration_minutes":  480,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_24hr-420-1"
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
                                                            "max_concurrent_streaming":  900,
                                                            "target_p99_ttft_ms":  800,
                                                            "target_p99_tpot_ms":  42
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "code_chat_peak",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  900,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-420-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  420,
                                      "active_end_at_offset_minutes":  900,
                                      "duration_minutes":  480,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-420-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-420"
}
```

## T+09:00 - Enterprise peak begins

```json
{
    "submit_at_offset_minutes":  540,
    "description":  "US enterprise chat, docs QA, legal, and support peak",
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
                                                            "priority_class":  "HIGH",
                                                            "max_concurrent_streaming":  1200,
                                                            "target_p99_ttft_ms":  650,
                                                            "target_p99_tpot_ms":  42
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "enterprise_premium_peak",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  1200,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-540-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  540,
                                      "active_end_at_offset_minutes":  960,
                                      "duration_minutes":  420,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-540-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_docs_qa",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-14B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  7000,
                                                            "isl_token_min":  1750,
                                                            "isl_token_max":  24000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  300,
                                                            "osl_token_min":  75,
                                                            "osl_token_max":  1000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1750,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  3500,
                                                            "target_p99_ttft_ms":  1300,
                                                            "target_p99_tpot_ms":  55
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "docs_qa_peak",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  3500,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-540-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  540,
                                      "active_end_at_offset_minutes":  960,
                                      "duration_minutes":  420,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-540-2"
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
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "priority_class":  "HIGH",
                                                            "max_concurrent_streaming":  360,
                                                            "target_p99_ttft_ms":  1000,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "legal_finance_peak",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  360,
                                          "seed":  103
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-540-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  540,
                                      "active_end_at_offset_minutes":  900,
                                      "duration_minutes":  360,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-540-3"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-540"
}
```

## T+10:00 - vLLM/Dynamo regression probe

```json
{
    "submit_at_offset_minutes":  600,
    "description":  "second canary simulates an engine-version performance change on one model family",
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
                                                            "total_token_budget":  211200000,
                                                            "priority_class":  "HIGH",
                                                            "deadline_hrs":  3,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "engine_regression_probe",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  96,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-600-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  600,
                                      "deadline_at_offset_minutes":  780,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-600-1",
                                      "shard_count":  96,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-600"
}
```

## T+11:30 - RAG long-input burst

```json
{
    "submit_at_offset_minutes":  690,
    "description":  "prefill-heavy long input burst without PD; LMCache should matter",
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
                                                            "max_concurrent_streaming":  1600,
                                                            "target_p99_ttft_ms":  1800,
                                                            "target_p99_tpot_ms":  60
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "rag_lunch_long_input_burst",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  1600,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-690-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  690,
                                      "active_end_at_offset_minutes":  780,
                                      "duration_minutes":  90,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-690-1"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-690"
}
```

## T+12:00 - Second index refresh and MoE eval

```json
{
    "submit_at_offset_minutes":  720,
    "description":  "midday batch wave arrives while online traffic is still high",
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
                                                            "total_token_budget":  425100000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  6,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "rag_index_refresh_midday",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  260,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "rag_index"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-720-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  720,
                                      "deadline_at_offset_minutes":  1080,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-720-1",
                                      "shard_count":  260,
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
                                                            "total_token_budget":  198000000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  10,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  150
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "moe_eval_midday",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  180,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-720-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  720,
                                      "deadline_at_offset_minutes":  1320,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-720-2",
                                      "shard_count":  180,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-720"
}
```

## T+14:00 - Product launch chat burst

```json
{
    "submit_at_offset_minutes":  840,
    "description":  "main product launch burst, high concurrency and strict moderation tail latency",
    "jobs":  [
                 {
                     "user_id":  "tenant_chat_free",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.1-8B-Instruct",
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
                                                            "max_concurrent_streaming":  56000,
                                                            "target_p99_ttft_ms":  650,
                                                            "target_p99_tpot_ms":  50
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "product_launch_chat",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  56000,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-840-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  840,
                                      "active_end_at_offset_minutes":  920,
                                      "duration_minutes":  80,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_24hr-840-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_safety_moderation",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-1.5B-Instruct",
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
                                                            "max_concurrent_streaming":  120000,
                                                            "target_p99_ttft_ms":  90,
                                                            "target_p99_tpot_ms":  30
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "launch_moderation",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  120000,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-840-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  840,
                                      "active_end_at_offset_minutes":  920,
                                      "duration_minutes":  80,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_24hr-840-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-840"
}
```

## T+15:30 - New frontier model arrival

```json
{
    "submit_at_offset_minutes":  930,
    "description":  "DeepSeek arrives as a held-out frontier model after the system has already learned",
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
                                                            "total_token_budget":  81600000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  6.5,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "heldout_deepseek_day",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  120,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-930-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  930,
                                      "deadline_at_offset_minutes":  1320,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-930-1",
                                      "shard_count":  120,
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
                                                            "max_concurrent_streaming":  180,
                                                            "target_p99_ttft_ms":  2600,
                                                            "target_p99_tpot_ms":  50
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "deepseek_new_model_online",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  180,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-930-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  930,
                                      "active_end_at_offset_minutes":  1140,
                                      "duration_minutes":  210,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-930-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-930"
}
```

## T+16:00 - P0 safety incident

```json
{
    "submit_at_offset_minutes":  960,
    "description":  "P0 abuse replay and live safety traffic should get priority over P3/P4 batch",
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
                                                            "total_token_budget":  81200000,
                                                            "priority_class":  "HIGH",
                                                            "deadline_hrs":  2,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  100
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "p0_abuse_replay_day",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  140,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "safety_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-960-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  960,
                                      "deadline_at_offset_minutes":  1080,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-960-1",
                                      "shard_count":  140,
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
                                                            "max_concurrent_streaming":  3800,
                                                            "target_p99_ttft_ms":  350,
                                                            "target_p99_tpot_ms":  40
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "p0_safety_live_day",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  3800,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-960-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  960,
                                      "active_end_at_offset_minutes":  1050,
                                      "duration_minutes":  90,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-960-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-960"
}
```

## T+17:00 - Sales reports and RLHF labels

```json
{
    "submit_at_offset_minutes":  1020,
    "description":  "late-day deadline batch wave arrives as business traffic starts to cool",
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
                                                            "total_token_budget":  620800000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  7,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "sales_reports_day",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  320,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "report_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-1020-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  1020,
                                      "deadline_at_offset_minutes":  1440,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-1020-1",
                                      "shard_count":  320,
                                      "requests_per_shard":  100,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_rlhf",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "zai-org/GLM-4.5-Air",
                                       "job_features":  {
                                                            "isl_token_avg":  1800,
                                                            "isl_token_min":  450,
                                                            "isl_token_max":  6000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  1000,
                                                            "osl_token_min":  250,
                                                            "osl_token_max":  4000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  100800000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  5,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "rlhf_label_generation",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  180,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "label_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-1020-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  1020,
                                      "deadline_at_offset_minutes":  1320,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-1020-2",
                                      "shard_count":  180,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-1020"
}
```

## T+18:00 - Decode-heavy agent wave

```json
{
    "submit_at_offset_minutes":  1080,
    "description":  "agentic workloads shift pressure toward long output and decode bandwidth",
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
                                                            "max_concurrent_streaming":  540,
                                                            "target_p99_ttft_ms":  2500,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "agent_wave_kimi_evening",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  540,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-1080-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  1080,
                                      "active_end_at_offset_minutes":  1260,
                                      "duration_minutes":  180,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_24hr-1080-1"
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
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  560,
                                                            "target_p99_ttft_ms":  2200,
                                                            "target_p99_tpot_ms":  50
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "agent_wave_glm_evening",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  560,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-1080-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  1080,
                                      "active_end_at_offset_minutes":  1260,
                                      "duration_minutes":  180,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-1080-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-1080"
}
```

## T+19:00 - APAC localization batch

```json
{
    "submit_at_offset_minutes":  1140,
    "description":  "APAC batch and translation work starts before APAC online ramp",
    "jobs":  [
                 {
                     "user_id":  "tenant_localization",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.1-8B-Instruct",
                                       "job_features":  {
                                                            "isl_token_avg":  1200,
                                                            "isl_token_min":  300,
                                                            "isl_token_max":  4096,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  800,
                                                            "osl_token_min":  200,
                                                            "osl_token_max":  2500,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  480000000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  4,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "apac_localization",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  240,
                                          "requests_per_shard":  1000
                                      },
                     "output_target":  {
                                           "sink":  "localization_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-1140-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  1140,
                                      "deadline_at_offset_minutes":  1380,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-1140-1",
                                      "shard_count":  240,
                                      "requests_per_shard":  1000,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-1140"
}
```

## T+20:00 - APAC online ramp

```json
{
    "submit_at_offset_minutes":  1200,
    "description":  "APAC online traffic becomes dominant while US cools",
    "jobs":  [
                 {
                     "user_id":  "tenant_chat_free",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.1-8B-Instruct",
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
                                                            "max_concurrent_streaming":  26000,
                                                            "target_p99_ttft_ms":  800,
                                                            "target_p99_tpot_ms":  60
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "apac_chat_peak",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  26000,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-1200-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  1200,
                                      "active_end_at_offset_minutes":  1410,
                                      "duration_minutes":  210,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_24hr-1200-1"
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
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3.5,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.3,
                                                            "shared_prefix_length_avg":  2250,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "STANDARD",
                                                            "max_concurrent_streaming":  5200,
                                                            "target_p99_ttft_ms":  1700,
                                                            "target_p99_tpot_ms":  60
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "apac_rag_peak",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  5200,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-1200-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  1200,
                                      "active_end_at_offset_minutes":  1410,
                                      "duration_minutes":  210,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_24hr-1200-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-1200"
}
```

## T+22:00 - Report generation wave

```json
{
    "submit_at_offset_minutes":  1320,
    "description":  "end-of-day reports, transcripts, and final eval work",
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
                                                            "total_token_budget":  698400000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  2,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "end_of_day_reports",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  360,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "report_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-1320-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  1320,
                                      "deadline_at_offset_minutes":  1440,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-1320-1",
                                      "shard_count":  360,
                                      "requests_per_shard":  100,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_transcripts",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.1-8B-Instruct",
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
                                                            "total_token_budget":  405600000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  2,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "transcript_summary_eod",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  240,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "summary_store"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-1320-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  1320,
                                      "deadline_at_offset_minutes":  1440,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-1320-2",
                                      "shard_count":  240,
                                      "requests_per_shard":  100,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-1320"
}
```

## T+23:30 - Cooldown and convergence probes

```json
{
    "submit_at_offset_minutes":  1410,
    "description":  "low-priority profiling while online load cools; useful for measuring final churn and convergence",
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
                                                            "total_token_budget":  72000000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  0.5,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  200
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "final_moe_probe",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  80,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-1410-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  1410,
                                      "deadline_at_offset_minutes":  1440,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-1410-1",
                                      "shard_count":  80,
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
                                                            "total_token_budget":  105600000,
                                                            "priority_class":  "HIGH",
                                                            "deadline_hrs":  0.5,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  200
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "final_engine_canary",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  48,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_24hr-1410-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  1410,
                                      "deadline_at_offset_minutes":  1440,
                                      "campaign_id":  "campaign-benchmark_timing_24hr-1410-2",
                                      "shard_count":  48,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_24hr-1410"
}
```
