# Real 15-minute, 24-GPU Orca smoke benchmark

Benchmark ID: `15min_real`

Purpose: cheaply validate the real Store, planner, Orca, Kubernetes, vLLM, and
AIPerf path before running the 128+ GPU benchmarks.

- Fleet: three on-demand `g6e.48xlarge` nodes, 24 L40S GPUs total.
- Workload: one `Qwen/Qwen2.5-Coder-32B-Instruct` streaming service.
- Placement: TP8, PP1, DP3; three replicas consume three 8-GPU nodes.
- Traffic: 5 requests/second at concurrency 48 for 15 minutes.
- Shutdown: stop new requests at minute 15 and hard-stop after one drain minute.

Provision the fleet and cache the container/model before starting the measured
window. Using the repository's frozen AWS price, the 15-minute fleet cost is
about $22.60; provisioning and shutdown time are additional.

Seed and clone the matching 24-GPU Store map with:

```bash
uv run python benchmarks/benchmark_static_seed.py \
  --full-reset \
  --benchmark-id 15min_real

uv run python benchmarks/baseline_user_setup.py \
  --baseline bestfit \
  --run-id smoke-1 \
  --benchmark-id 15min_real
```
