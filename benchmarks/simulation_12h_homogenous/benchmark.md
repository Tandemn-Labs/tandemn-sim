# Simulation 12-hour homogeneous H100 benchmark

Benchmark ID: `simulation_12h_homogenous`

Purpose: calibrated Koi/Tandemn workload replay on a homogeneous 400-GPU H100 simulation fleet. This second calibration preserves persistent online APIs, regional traffic identities, deadline batch campaigns, canary and quantization probes, a product-launch burst, a P0 incident, and frontier/MoE arrivals while avoiding the prior universal SLO and deadline floor.

Benchmark name: `KoiBench-Sim-12h-OnDemand-H100`

Horizon: 720 minutes

Tick: 5 minutes

Market: `on_demand`

Engine: `vllm` behind Dynamo-style serving

Serving mode: aggregate serving with LMCache, no prefill/decode disaggregation

Region: `us-east`

Active accelerator catalog: `H100`

Model catalog: `meta-llama/Llama-3.2-1B-Instruct`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `microsoft/phi-4`, `Qwen/Qwen2.5-Coder-32B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`, `Qwen/Qwen3-Next-80B-A3B-Thinking`, `zai-org/GLM-4.5-Air`, `moonshotai/Kimi-K2-Instruct`, `deepseek-ai/DeepSeek-V3`

Model precision: the launch policy selects `bf16` or `bf16-fp8` per model.

## Frozen Cluster Resources

Maximum provisioned capacity: **400 on-demand H100 GPUs**. The complete fleet is available before T+00:00, outside the benchmark horizon.

| Accelerator | Node Shape | Nodes | On-Demand | Total |
|---|---:|---:|---:|---:|
| `H100` | 8 GPUs, NVSwitch | 50 | 400 | 400 |

All nodes are in `us-east` with 400 Gbps RDMA. There is no reserved capacity, heterogeneous fallback, or inter-region placement.
Each p5.48xlarge is an eight-GPU allocation unit; TP/PP groups must fit within one node.

## Capacity Envelope

Online concurrency is an in-flight request ceiling, not a user-population count. Persistent chat and support remain at tens of streams, burst peaks are at most 64 streams, and each moderation workload runs at 12 requests/s with a 16-stream cap. At the overlap peak, the benchmark declares at most **504 online streams** and approximately **43.9 mean requests/s**. The executable online population falls from roughly 2.26 million to **1.43 million requests** while preserving every workload identity, release wave, and SLO target.

Logical TP/PP/DP counts in the preferred-placement map still peak at **286 GPUs** while campaigns meet deadlines and **472 GPUs** if every campaign remains until hard stop. These are logical counts, not node-rounded p5 allocation footprints; physical instance accounting and unsupported multi-node shapes remain separate harness constraints.

## Simulation Independence Gate

The environment emulator used to realize latency and throughput must be independently calibrated from the surrogate exposed to Koi. Test residuals, perturbations, and held-out model/engine shifts must remain hidden from every planner.
## Canonical Lifecycle Contract

Schema: `koibench/v1`

Dispatch: releases occur at their exact offset. A job arrival triggers the admission path immediately; the 5-minute tick remains the slow reoptimization loop.

Online lifecycle: `active_start_at_offset_minutes` and `active_end_at_offset_minutes` define a half-open traffic window. `duration_minutes` is retained only as a consistency check. At the end offset the load generator stops new requests, drains in-flight requests, and the executor records `terminal_at`.

Batch lifecycle: `deadline_at_offset_minutes` is an absolute offset from benchmark start. `record_count` fixes campaign size, while `requests_per_shard` fixes the representative probe and `shard_count` preserves their exact product. Runtime `started_at`, `completed_at`, and `terminal_at` are produced by the executor, not authored in the job definition.

Overlap: `base_service` remains active, `additive_stream` adds demand to an existing service, and `replace_service` identifies a model/service replacement. Overlap is intentional and additive unless explicitly marked as replacement.

Planner fairness: descriptions, trace names, human-readable tenant labels, and `input_source` are not exposed to planners. Every planner receives the same sanitized job features. Hardware placement and market preferences are deliberately absent from job inputs.

Traffic: `max_concurrent_streaming` is the planner-facing service cap; the matching AIPerf phase `concurrency` is the client cap. AIPerf phase `rate` is the submitted request rate and remains excluded from the Orca payload so Orca stays in concurrency mode. Patterns are limited to `Poisson`, `Constant`, `Bursty`, and `Diurnal`.

Traffic calibration: each online phase uses a production-trace window selected for the intended workload class. Constant traffic maps to AIPerf `constant`, Poisson traffic maps to `poisson`, and Bursty or Diurnal traffic maps to `gamma`; gamma smoothness is `1 / interarrival CV²` with a `0.001` floor. Diurnal phases use a full-window `rateRamp`, while correlated launch workloads reuse the same source window.

Input modes: every online job uses `api_load_generator` to ping an OpenAI-compatible endpoint under its concurrency profile. Every batch campaign uses a finite `dataset` partitioned into required shards.

Drain: after minute 720, no new work arrives. A 60-minute drain period records batch tardiness and in-flight completion; work unfinished at its deadline is a miss.

Exact manifest counts: **23 online jobs**, **14 batch campaigns**, **796 expanded shards**, and **187,700 batch records**.

## T+00:00 - Always-on APIs across three regions

```json
{
    "submit_at_offset_minutes":  0,
    "description":  "baseline always-on product APIs in all regions",
    "jobs":  [
                 {
                     "user_id":  "tenant_chat_free",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.2-1B-Instruct",
                                       "precision":  "bf16",
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
                                                            "max_concurrent_streaming":  64,
                                                            "target_p99_ttft_ms":  700,
                                                            "target_p99_tpot_ms":  55
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "chat_us",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  64,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-0-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  720,
                                      "duration_minutes":  720,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-0-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_chat_free",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.2-1B-Instruct",
                                       "precision":  "bf16",
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
                                                            "max_concurrent_streaming":  64,
                                                            "target_p99_ttft_ms":  750,
                                                            "target_p99_tpot_ms":  60
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "chat_eu",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  64,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-0-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  720,
                                      "duration_minutes":  720,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-0-2"
                                  }
                 },
                 {
                     "user_id":  "tenant_chat_premium",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "precision":  "bf16",
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
                                          "prompt_dataset_id":  "premium_us",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  24,
                                          "seed":  103
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-0-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  465,
                                      "duration_minutes":  485,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-0-3"
                                  }
                 },
                 {
                     "user_id":  "tenant_code_autocomplete",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-Coder-32B-Instruct",
                                       "precision":  "bf16",
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
                                          "prompt_dataset_id":  "code_us",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  32,
                                          "seed":  104
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-0-4",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  720,
                                      "duration_minutes":  720,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-0-4"
                                  }
                 },
                 {
                     "user_id":  "tenant_safety_moderation",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.2-1B-Instruct",
                                       "precision":  "bf16",
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
                                                            "max_concurrent_streaming":  16,
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
                                          "max_concurrent_streaming":  16,
                                          "seed":  105
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-0-5",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  720,
                                      "duration_minutes":  720,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-0-5"
                                  }
                 },
                 {
                     "user_id":  "tenant_small_model",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.2-1B-Instruct",
                                       "precision":  "bf16",
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
                                          "request_rate_per_second":  2.0,
                                          "max_concurrent_streaming":  8,
                                          "seed":  106
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-0-6",
                     "schedule":  {
                                      "release_at_offset_minutes":  0,
                                      "active_end_at_offset_minutes":  720,
                                      "duration_minutes":  720,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-0-6"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-0"
}
```

## T+00:10 - Nightly evaluation campaigns

```json
{
    "submit_at_offset_minutes":  10,
    "description":  "nightly eval shards arrive after warm start",
    "jobs":  [
                 {
                     "user_id":  "tenant_eval_platform",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "precision":  "bf16",
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
                                          "dataset_id":  "nightly_eval_llama70b",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  90,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-10-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  10,
                                      "deadline_at_offset_minutes":  480,
                                      "campaign_id":  "campaign-benchmark_timing_12hr_homogenous-10-1",
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
                                       "model_id":  "Qwen/Qwen3-Next-80B-A3B-Thinking",
                                       "precision":  "bf16",
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
                                                            "total_token_budget":  48000000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  7.8333,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "nightly_eval_qwen72b",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  80,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-10-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  10,
                                      "deadline_at_offset_minutes":  480,
                                      "campaign_id":  "campaign-benchmark_timing_12hr_homogenous-10-2",
                                      "shard_count":  80,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_eval_platform",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "microsoft/phi-4",
                                       "precision":  "bf16",
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
                                                            "total_token_budget":  60000000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  5.8333,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "nightly_eval_phi4",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  60,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-10-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  10,
                                      "deadline_at_offset_minutes":  360,
                                      "campaign_id":  "campaign-benchmark_timing_12hr_homogenous-10-3",
                                      "shard_count":  60,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-10"
}
```

## T+00:25 - Chat and support APIs

```json
{
    "submit_at_offset_minutes":  25,
    "description":  "Phi-4 long-context chat, small-model chat, docs QA, and support APIs start",
    "jobs":  [
                 {
                     "user_id":  "tenant_phi4_long_context_chat",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "microsoft/phi-4",
                                       "precision":  "bf16",
                                       "job_features":  {
                                                            "isl_token_avg":  9000,
                                                            "isl_token_min":  2250,
                                                            "isl_token_max":  15684,
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
                                          "prompt_dataset_id":  "phi4_long_context_chat_us",
                                          "request_arrival_pattern":  "Poisson",
                                          "max_concurrent_streaming":  24,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-25-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  25,
                                      "active_end_at_offset_minutes":  720,
                                      "duration_minutes":  695,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-25-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_small_model",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.2-1B-Instruct",
                                       "precision":  "bf16",
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
                                          "request_rate_per_second":  2.0,
                                          "max_concurrent_streaming":  8,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-25-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  25,
                                      "active_end_at_offset_minutes":  720,
                                      "duration_minutes":  695,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-25-2"
                                  }
                 },
                 {
                     "user_id":  "tenant_docs_qa",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "microsoft/phi-4",
                                       "precision":  "bf16",
                                       "job_features":  {
                                                            "isl_token_avg":  7000,
                                                            "isl_token_min":  1750,
                                                            "isl_token_max":  15384,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  300,
                                                            "osl_token_min":  75,
                                                            "osl_token_max":  1000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Poisson",
                                                            "peak_to_mean_ratio":  2,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1750,
                                                            "is_session_affinity":  false,
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
                                          "prompt_dataset_id":  "docs_qa",
                                          "request_arrival_pattern":  "Poisson",
                                          "max_concurrent_streaming":  24,
                                          "seed":  103
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-25-3",
                     "schedule":  {
                                      "release_at_offset_minutes":  25,
                                      "active_end_at_offset_minutes":  645,
                                      "duration_minutes":  620,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-25-3"
                                  }
                 },
                 {
                     "user_id":  "tenant_support",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "mistralai/Mixtral-8x7B-Instruct-v0.1",
                                       "precision":  "bf16",
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
                                          "prompt_dataset_id":  "support",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  64,
                                          "seed":  104
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-25-4",
                     "schedule":  {
                                      "release_at_offset_minutes":  25,
                                      "active_end_at_offset_minutes":  720,
                                      "duration_minutes":  695,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-25-4"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-25"
}
```

## T+00:45 - Long-input document processing

```json
{
    "submit_at_offset_minutes":  45,
    "description":  "long-input document refresh",
    "jobs":  [
                 {
                     "user_id":  "tenant_document_processing",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "microsoft/phi-4",
                                       "precision":  "bf16",
                                       "job_features":  {
                                                            "isl_token_avg":  14000,
                                                            "isl_token_min":  4000,
                                                            "isl_token_max":  15184,
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
                                                            "total_token_budget":  114800000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  6.25,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "document_refresh",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  80,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "document_store"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-45-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  45,
                                      "deadline_at_offset_minutes":  420,
                                      "campaign_id":  "campaign-benchmark_timing_12hr_homogenous-45-2",
                                      "shard_count":  80,
                                      "requests_per_shard":  100,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-45"
}
```

## T+01:30 - Synthetic teacher generation

```json
{
    "submit_at_offset_minutes":  90,
    "description":  "large-model batch generation fills low-priority capacity",
    "jobs":  [
                 {
                     "user_id":  "tenant_synthetic_data",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "precision":  "bf16",
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
                                                            "total_token_budget":  64800000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  9.5,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "synthetic_teacher_llama70b",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  90,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "synth_store"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-90-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  90,
                                      "deadline_at_offset_minutes":  660,
                                      "campaign_id":  "campaign-benchmark_timing_12hr_homogenous-90-1",
                                      "shard_count":  90,
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
                                       "precision":  "bf16",
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
                                                            "total_token_budget":  30240000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  3.5,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  200
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "quant_probe_12h",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  40,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-90-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  90,
                                      "deadline_at_offset_minutes":  300,
                                      "campaign_id":  "campaign-benchmark_timing_12hr_homogenous-90-2",
                                      "shard_count":  40,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-90"
}
```

## T+02:15 - EU ramp

```json
{
    "submit_at_offset_minutes":  135,
    "description":  "EU business traffic ramps before US peak",
    "jobs":  [
                 {
                     "user_id":  "tenant_phi4_long_context_chat",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "microsoft/phi-4",
                                       "precision":  "bf16",
                                       "job_features":  {
                                                            "isl_token_avg":  10000,
                                                            "isl_token_min":  2500,
                                                            "isl_token_max":  15684,
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
                                                            "max_concurrent_streaming":  24,
                                                            "target_p99_ttft_ms":  1500,
                                                            "target_p99_tpot_ms":  55
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "phi4_long_context_chat_eu",
                                          "request_arrival_pattern":  "Poisson",
                                          "max_concurrent_streaming":  24,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-135-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  135,
                                      "active_end_at_offset_minutes":  395,
                                      "duration_minutes":  260,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-135-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_enterprise_chat",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "microsoft/phi-4",
                                       "precision":  "bf16",
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
                                          "prompt_dataset_id":  "eu_enterprise_chat",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  24,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-135-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  135,
                                      "active_end_at_offset_minutes":  435,
                                      "duration_minutes":  300,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-135-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-135"
}
```

## T+03:00 - Code traffic ramp

```json
{
    "submit_at_offset_minutes":  180,
    "description":  "code autocomplete becomes the dominant TTFT-sensitive job",
    "jobs":  [
                 {
                     "user_id":  "tenant_code_autocomplete",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen2.5-Coder-32B-Instruct",
                                       "precision":  "bf16",
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
                                                            "max_concurrent_streaming":  64,
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
                                          "max_concurrent_streaming":  64,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-180-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  180,
                                      "active_end_at_offset_minutes":  400,
                                      "duration_minutes":  220,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-180-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_code_chat",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "precision":  "bf16",
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
                                                            "max_concurrent_streaming":  24,
                                                            "target_p99_ttft_ms":  1000,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "code_chat",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  24,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-180-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  180,
                                      "active_end_at_offset_minutes":  420,
                                      "duration_minutes":  240,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-180-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-180"
}
```

## T+04:00 - Enterprise peak

```json
{
    "submit_at_offset_minutes":  240,
    "description":  "enterprise chat, legal, finance, and support all contend for workhorse GPUs",
    "jobs":  [
                 {
                     "user_id":  "tenant_enterprise_chat",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-Next-80B-A3B-Thinking",
                                       "precision":  "bf16",
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
                                                            "max_concurrent_streaming":  16,
                                                            "target_p99_ttft_ms":  1000,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "enterprise_premium_peak",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  16,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-240-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  240,
                                      "active_end_at_offset_minutes":  460,
                                      "duration_minutes":  220,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-240-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_legal_finance",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "precision":  "bf16",
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
                                                            "peak_to_mean_ratio":  3.5,
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
                                          "prompt_dataset_id":  "legal_finance_peak",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  16,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-240-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  240,
                                      "active_end_at_offset_minutes":  420,
                                      "duration_minutes":  180,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-240-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-240"
}
```

## T+05:00 - vLLM/Dynamo canary and regression probes

```json
{
    "submit_at_offset_minutes":  300,
    "description":  "canary jobs exercise engine-version sensitivity and residual calibration",
    "jobs":  [
                 {
                     "user_id":  "tenant_engine_canary",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "microsoft/phi-4",
                                       "precision":  "bf16",
                                       "job_features":  {
                                                            "isl_token_avg":  4000,
                                                            "isl_token_min":  1000,
                                                            "isl_token_max":  15184,
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
                                                            "total_token_budget":  70400000,
                                                            "priority_class":  "HIGH",
                                                            "deadline_hrs":  2,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "engine_regression_probe",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  32,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-300-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  300,
                                      "deadline_at_offset_minutes":  420,
                                      "campaign_id":  "campaign-benchmark_timing_12hr_homogenous-300-1",
                                      "shard_count":  32,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-300"
}
```

## T+05:30 - Prefill-heavy long-context burst without PD

```json
{
    "submit_at_offset_minutes":  330,
    "description":  "long-input chat burst stresses batching, LMCache, and TTFT without PD",
    "jobs":  [
                 {
                     "user_id":  "tenant_qwen_long_context_chat",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-Next-80B-A3B-Thinking",
                                       "precision":  "bf16",
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
                                                            "max_concurrent_streaming":  8,
                                                            "target_p99_ttft_ms":  3000,
                                                            "target_p99_tpot_ms":  65
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "qwen_long_context_chat_burst",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  8,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-330-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  330,
                                      "active_end_at_offset_minutes":  410,
                                      "duration_minutes":  80,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-330-1"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-330"
}
```

## T+06:15 - Legal audit batch

```json
{
    "submit_at_offset_minutes":  375,
    "description":  "strict quality batch cannot use aggressive low precision",
    "jobs":  [
                 {
                     "user_id":  "tenant_legal_finance",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "precision":  "bf16",
                                       "job_features":  {
                                                            "isl_token_avg":  8000,
                                                            "isl_token_min":  2000,
                                                            "isl_token_max":  32000,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  600,
                                                            "osl_token_min":  150,
                                                            "osl_token_max":  2000,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Constant",
                                                            "peak_to_mean_ratio":  1,
                                                            "workload_prefix_concentration":  0,
                                                            "multi_turn_ratio":  0,
                                                            "shared_prefix_length_avg":  1,
                                                            "is_session_affinity":  false,
                                                            "total_token_budget":  77400000,
                                                            "priority_class":  "HIGH",
                                                            "deadline_hrs":  3.75,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "legal_audit",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  45,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "audit_store"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-375-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  375,
                                      "deadline_at_offset_minutes":  600,
                                      "campaign_id":  "campaign-benchmark_timing_12hr_homogenous-375-1",
                                      "shard_count":  45,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-375"
}
```

## T+07:00 - Product launch chat burst

```json
{
    "submit_at_offset_minutes":  420,
    "description":  "large free-chat and moderation burst uses on-demand capacity if needed",
    "jobs":  [
                 {
                     "user_id":  "tenant_chat_free",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "mistralai/Mixtral-8x7B-Instruct-v0.1",
                                       "precision":  "bf16",
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
                                                            "max_concurrent_streaming":  48,
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
                                          "max_concurrent_streaming":  48,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-420-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  420,
                                      "active_end_at_offset_minutes":  490,
                                      "duration_minutes":  70,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-420-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_safety_moderation",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.2-1B-Instruct",
                                       "precision":  "bf16",
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
                                                            "max_concurrent_streaming":  16,
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
                                          "max_concurrent_streaming":  16,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-420-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  420,
                                      "active_end_at_offset_minutes":  490,
                                      "duration_minutes":  70,
                                      "operation":  "additive_stream",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-420-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-420"
}
```

## T+07:45 - Premium Model Swap

```json
{
    "submit_at_offset_minutes":  465,
    "description":  "new 80B-A3B premium variant overlaps the 70B service for 20 minutes, then replaces it",
    "jobs":  [
                 {
                     "user_id":  "tenant_chat_premium",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-Next-80B-A3B-Thinking",
                                       "precision":  "bf16",
                                       "job_features":  {
                                                            "isl_token_avg":  1200,
                                                            "isl_token_min":  300,
                                                            "isl_token_max":  4096,
                                                            "isl_distribution_type":  "LogNormal",
                                                            "osl_token_avg":  550,
                                                            "osl_token_min":  137,
                                                            "osl_token_max":  1600,
                                                            "osl_distribution_type":  "LogNormal",
                                                            "request_arrival_pattern":  "Bursty",
                                                            "peak_to_mean_ratio":  3,
                                                            "workload_prefix_concentration":  0.5,
                                                            "multi_turn_ratio":  0.7,
                                                            "shared_prefix_length_avg":  300,
                                                            "is_session_affinity":  true,
                                                            "priority_class":  "HIGH",
                                                            "max_concurrent_streaming":  24,
                                                            "target_p99_ttft_ms":  1000,
                                                            "target_p99_tpot_ms":  45
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "premium_qwen72b_swap",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  24,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-465-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  465,
                                      "active_end_at_offset_minutes":  720,
                                      "duration_minutes":  255,
                                      "operation":  "replace_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-465-1",
                                      "supersedes_service_id":  "service-benchmark_timing_12hr_homogenous-0-3"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-465"
}
```

## T+08:00 - P0 abuse replay

```json
{
    "submit_at_offset_minutes":  480,
    "description":  "urgent P0 safety replay tests priority and deadline handling",
    "jobs":  [
                 {
                     "user_id":  "tenant_safety_p0",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "meta-llama/Llama-3.3-70B-Instruct",
                                       "precision":  "bf16",
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
                                                            "deadline_hrs":  2,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  100
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "p0_abuse_replay",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  40,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "safety_store"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-480-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  480,
                                      "deadline_at_offset_minutes":  600,
                                      "campaign_id":  "campaign-benchmark_timing_12hr_homogenous-480-1",
                                      "shard_count":  40,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-480"
}
```

## T+08:30 - Decode-heavy agent wave

```json
{
    "submit_at_offset_minutes":  510,
    "description":  "frontier and MoE reasoning with long outputs",
    "jobs":  [
                 {
                     "user_id":  "tenant_agents_premium",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "moonshotai/Kimi-K2-Instruct",
                                       "precision":  "bf16-fp8",
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
                                          "prompt_dataset_id":  "agent_wave_kimi",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  16,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-510-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  510,
                                      "active_end_at_offset_minutes":  650,
                                      "duration_minutes":  140,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-510-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_frontier_glm",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "zai-org/GLM-4.5-Air",
                                       "precision":  "bf16",
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
                                                            "max_concurrent_streaming":  16,
                                                            "target_p99_ttft_ms":  2200,
                                                            "target_p99_tpot_ms":  50
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "agent_wave_glm",
                                          "request_arrival_pattern":  "Bursty",
                                          "max_concurrent_streaming":  16,
                                          "seed":  102
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-510-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  510,
                                      "active_end_at_offset_minutes":  630,
                                      "duration_minutes":  120,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-510-2"
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-510"
}
```

## T+09:30 - MoE and long-context campaigns

```json
{
    "submit_at_offset_minutes":  570,
    "description":  "MoE eval plus long-context benchmark",
    "jobs":  [
                 {
                     "user_id":  "tenant_moe_eval",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-Next-80B-A3B-Thinking",
                                       "precision":  "bf16",
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
                                                            "total_token_budget":  52800000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  2.1667,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  150
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "moe_eval_qwen3",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  48,
                                          "requests_per_shard":  500
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-570-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  570,
                                      "deadline_at_offset_minutes":  700,
                                      "campaign_id":  "campaign-benchmark_timing_12hr_homogenous-570-1",
                                      "shard_count":  48,
                                      "requests_per_shard":  500,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 },
                 {
                     "user_id":  "tenant_long_context",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-Next-80B-A3B-Thinking",
                                       "precision":  "bf16",
                                       "job_features":  {
                                                            "isl_token_avg":  32000,
                                                            "isl_token_min":  8000,
                                                            "isl_token_max":  128000,
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
                                                            "total_token_budget":  116200000,
                                                            "priority_class":  "STANDARD",
                                                            "deadline_hrs":  2.1667,
                                                            "target_p99_ttft_ms":  10000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "long_context_benchmark",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  35,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "eval_store"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-570-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  570,
                                      "deadline_at_offset_minutes":  700,
                                      "campaign_id":  "campaign-benchmark_timing_12hr_homogenous-570-2",
                                      "shard_count":  35,
                                      "requests_per_shard":  100,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-570"
}
```

## T+10:30 - Reports and transcript summaries

```json
{
    "submit_at_offset_minutes":  630,
    "description":  "late-day report and transcript batch work",
    "jobs":  [
                 {
                     "user_id":  "tenant_reports",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "Qwen/Qwen3-Next-80B-A3B-Thinking",
                                       "precision":  "bf16",
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
                                                            "deadline_hrs":  1.5,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  160
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "reports_12h",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  80,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "report_store"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-630-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  630,
                                      "deadline_at_offset_minutes":  720,
                                      "campaign_id":  "campaign-benchmark_timing_12hr_homogenous-630-1",
                                      "shard_count":  80,
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
                                       "precision":  "bf16",
                                       "job_features":  {
                                                            "isl_token_avg":  16000,
                                                            "isl_token_min":  4000,
                                                            "isl_token_max":  29768,
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
                                                            "total_token_budget":  101400000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  1.5,
                                                            "target_p99_ttft_ms":  8000,
                                                            "target_p99_tpot_ms":  120
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "transcript_summary_12h",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  60,
                                          "requests_per_shard":  100
                                      },
                     "output_target":  {
                                           "sink":  "summary_store"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-630-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  630,
                                      "deadline_at_offset_minutes":  720,
                                      "campaign_id":  "campaign-benchmark_timing_12hr_homogenous-630-2",
                                      "shard_count":  60,
                                      "requests_per_shard":  100,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-630"
}
```

## T+11:30 - APAC ramp and cooldown probes

```json
{
    "submit_at_offset_minutes":  690,
    "description":  "APAC ramp begins while low-priority profiling consumes leftovers",
    "jobs":  [
                 {
                     "user_id":  "tenant_chat_free",
                     "kind":  "online",
                     "spec_json":  {
                                       "model_id":  "mistralai/Mixtral-8x7B-Instruct-v0.1",
                                       "precision":  "bf16",
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
                                                            "max_concurrent_streaming":  64,
                                                            "target_p99_ttft_ms":  800,
                                                            "target_p99_tpot_ms":  60
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "api_load_generator",
                                          "protocol":  "openai_compatible_http",
                                          "endpoint":  "/v1/chat/completions",
                                          "prompt_dataset_id":  "apac_chat_ramp",
                                          "request_arrival_pattern":  "Diurnal",
                                          "max_concurrent_streaming":  64,
                                          "seed":  101
                                      },
                     "output_target":  {
                                           "sink":  "metrics"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-690-1",
                     "schedule":  {
                                      "release_at_offset_minutes":  690,
                                      "active_end_at_offset_minutes":  720,
                                      "duration_minutes":  30,
                                      "operation":  "base_service",
                                      "service_id":  "service-benchmark_timing_12hr_homogenous-690-1"
                                  }
                 },
                 {
                     "user_id":  "tenant_koi_profiling",
                     "kind":  "batch",
                     "spec_json":  {
                                       "model_id":  "deepseek-ai/DeepSeek-V3",
                                       "precision":  "bf16-fp8",
                                       "job_features":  {
                                                            "isl_token_avg":  1000,
                                                            "isl_token_min":  250,
                                                            "isl_token_max":  4096,
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
                                                            "total_token_budget":  6400000,
                                                            "priority_class":  "LOW",
                                                            "deadline_hrs":  0.5,
                                                            "target_p99_ttft_ms":  12000,
                                                            "target_p99_tpot_ms":  200
                                                        }
                                   },
                     "input_source":  {
                                          "type":  "dataset",
                                          "dataset_id":  "heldout_deepseek_probe",
                                          "partitioning":  "fixed_shards",
                                          "shard_count":  16,
                                          "requests_per_shard":  200
                                      },
                     "output_target":  {
                                           "sink":  "perfdb"
                                       },
                     "job_id":  "job-benchmark_timing_12hr_homogenous-690-2",
                     "schedule":  {
                                      "release_at_offset_minutes":  690,
                                      "deadline_at_offset_minutes":  720,
                                      "campaign_id":  "campaign-benchmark_timing_12hr_homogenous-690-2",
                                      "shard_count":  16,
                                      "requests_per_shard":  200,
                                      "expansion_policy":  "required",
                                      "checkpointable":  true
                                  }
                 }
             ],
    "schema_version":  "koibench/v1",
    "event_id":  "event-benchmark_timing_12hr_homogenous-690"
}
```
