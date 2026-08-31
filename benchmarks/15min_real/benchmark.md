# Real 15-minute, 16-GPU Orca benchmark

Benchmark ID: `15min_real`

Purpose: validate the real Store, planner, Orca, Kubernetes, vLLM, and AIPerf
path against a representative heterogeneous test fleet.

- Fleet: one on-demand GCP `g4-standard-384` node with 8 RTX PRO 6000
  Blackwell 96GB GPUs and one `g2-standard-96` node with 8 L4 24GB GPUs.
- Workload: one `Qwen/Qwen2.5-Coder-32B-Instruct` streaming service and one
  256-record `meta-llama/Llama-3.2-1B-Instruct` batch evaluation, both at T=0.
- Traffic: Poisson arrivals averaging 0.5 requests/second at concurrency 8 for
  15 minutes.
- Shutdown: stop new requests at minute 15 and hard-stop after one drain minute.

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
