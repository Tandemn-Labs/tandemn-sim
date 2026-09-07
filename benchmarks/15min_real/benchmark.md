# Real 15-minute, 16-GPU Orca benchmark

Benchmark ID: `15min_real`

Benchmark name: `KoiBench-Real-15min-16GPU-Orca`

Purpose: validate the real Store, planner, Orca, Kubernetes, vLLM, and AIPerf
path against a representative heterogeneous test fleet.

Horizon: 15 minutes

Tick: 1 minute

Engine: `vllm` with AIPerf traffic generation

Markets: `on_demand`

Serving mode: one streaming service and one finite batch evaluation

Real fleet target: 16 GPUs

## Canonical Lifecycle Contract

Schema: `benchmark-manifest/v1`

Dispatch: the executor polls every 10 seconds and starts both jobs at T+00:00.
The one-minute tick is the reoptimization loop.

Online lifecycle: `start_at_offset_minutes` and `stop_at_offset_minutes` define
the streaming traffic window. At minute 15, AIPerf stops new requests and drains
in-flight work.

Batch lifecycle: the evaluation starts at T+00:00, has an absolute deadline at
minute 15, and is split into four required 64-request shards.

Drain: the measured window ends at minute 15. The executor allows one minute
for the online drain and batch completion, then hard-stops at minute 16.

Exact manifest counts: **1 online job**, **1 batch campaign**, **4 batch shards**.

## Frozen Real Cluster Resources

Maximum provisioned capacity: **16 on-demand GPUs** in `central-1`.

| Accelerator | Node Shape | Nodes | On-Demand | Total |
|---|---:|---:|---:|---:|
| `RTXPRO6000` | 8 GPUs, 96 GB | 1 | 8 | 8 |
| `L4` | 8 GPUs, 24 GB | 1 | 8 | 8 |
| **Total** |  | **2** | **16** | **16** |

## Job Characteristics

| Job | Type | Model | Precision | Start | End / Deadline |
|---|---|---|---|---:|---:|
| `job-benchmark_timing_15min_real-0-1` | Streaming | `Qwen/Qwen2.5-Coder-32B-Instruct` | `fp16` | T+00:00 | T+15:00 |
| `job-benchmark_timing_15min_real-0-2` | Batch | `meta-llama/Llama-3.2-1B-Instruct` | `bf16` | T+00:00 | T+15:00 |

The streaming service uses Poisson traffic at 0.5 requests/second, concurrency
8, with 1,800 average input and 120 average output tokens. The batch evaluation
contains 256 records, four 64-request shards, and up to eight in-flight requests.

## T+00:00 - Code Chat and Small Batch Evaluation

```json
{
  "submit_at_offset_minutes": 0,
  "description": "streaming code chat and a small batch evaluation begin together",
  "jobs": [
    {
      "user_id": "tenant_smoke",
      "kind": "online",
      "spec_json": {
        "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "precision": "fp16",
        "job_features": {
          "isl_token_avg": 1800,
          "isl_token_min": 450,
          "isl_token_max": 5000,
          "isl_distribution_type": "LogNormal",
          "osl_token_avg": 120,
          "osl_token_min": 30,
          "osl_token_max": 350,
          "osl_distribution_type": "LogNormal",
          "request_arrival_pattern": "Poisson",
          "peak_to_mean_ratio": 1,
          "workload_prefix_concentration": 0,
          "multi_turn_ratio": 0,
          "shared_prefix_length_avg": 1,
          "is_session_affinity": false,
          "priority_class": "STANDARD",
          "max_concurrent_streaming": 8,
          "target_p99_ttft_ms": 500,
          "target_p99_tpot_ms": 50
        }
      },
      "input_source": {
        "type": "api_load_generator",
        "protocol": "openai_compatible_http",
        "endpoint": "/v1/chat/completions",
        "request_arrival_pattern": "Poisson",
        "rate_requests_per_second": 0.5,
        "max_concurrent_streaming": 8,
        "seed": 24015
      },
      "output_target": {
        "sink": "metrics"
      },
      "job_id": "job-benchmark_timing_15min_real-0-1",
      "schedule": {
        "release_at_offset_minutes": 0,
        "active_end_at_offset_minutes": 30,
        "duration_minutes": 30,
        "operation": "base_service",
        "service_id": "service-benchmark_timing_15min_real-0-1"
      }
    },
    {
      "user_id": "tenant_batch_eval",
      "kind": "batch",
      "spec_json": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "precision": "bf16",
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
          "total_token_budget": 81920,
          "priority_class": "LOW",
          "deadline_hrs": 0.25,
          "target_p99_ttft_ms": 12000,
          "target_p99_tpot_ms": 200
        }
      },
      "input_source": {
        "type": "dataset",
        "dataset_id": "small_batch_eval",
        "record_count": 256,
        "partitioning": "fixed_shards",
        "shard_count": 4,
        "requests_per_shard": 64,
        "max_in_flight_requests": 8,
        "seed": 24016
      },
      "output_target": {
        "sink": "eval_store"
      },
      "job_id": "job-benchmark_timing_15min_real-0-2",
      "schedule": {
        "release_at_offset_minutes": 0,
        "deadline_at_offset_minutes": 30,
        "hard_stop_at_offset_minutes": 30,
        "completion_policy": "stop_when_complete",
        "campaign_id": "campaign-benchmark_timing_15min_real-0-2",
        "shard_count": 4,
        "requests_per_shard": 64,
        "expansion_policy": "required"
      }
    }
  ],
  "schema_version": "koibench/v1",
  "event_id": "event-benchmark_timing_15min_real-0"
}
```

Provision the fleet and cache the container/model before starting the measured
window. Provisioning and shutdown time are additional.

Seed and clone the matching 16-GPU Store map with:

```bash
uv run python benchmarks/benchmark_static_seed.py \
  --full-reset \
  --benchmark-id 15min_real

uv run python benchmarks/baseline_user_setup.py \
  --baseline bestfit \
  --run-id smoke-1 \
  --benchmark-id 15min_real
```
