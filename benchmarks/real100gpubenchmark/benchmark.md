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

The manifest contains 13 online jobs and 4 batch campaigns. The selected batch campaigns are Mini Nightly Eval, Phi-4 Quant Probe, Calibration Probe, and Sales Reports, totaling exactly 26,200 records across 106 shards. Their definitions are unchanged from `simulation_3h`; the external pipeline owns batch dataset staging.

All online traffic uses the original `simulation_3h` phase rates and concurrency caps without rate ramps. The maximum simultaneous online declaration is 392 streams at T+100. Product Launch Chat and Product Launch Moderation run through T+109, while Legal/Finance Review runs from T+135 through T+180, so cold starts do not consume their measurement windows. Online traffic stops at minute 180 and in-flight work drains through the hard stop at minute 210.

## Model Allowlist

- `meta-llama/Llama-3.2-1B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `microsoft/phi-4`
- `Qwen/Qwen2.5-Coder-32B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`
- `Qwen/Qwen3-Next-80B-A3B-Thinking`
- `zai-org/GLM-4.5-Air`
- `QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ`

`QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ` is allowed but unused by this workload.

## Difficulty Contract

All 120 GPUs remain schedulable. The T+32, T+49, T+67, T+100, T+125, and T+135 arrival waves create persistent code, product-launch, high-priority safety, and Qwen3 placement pressure. Completion of every batch campaign before its deadline is not guaranteed. Sales Reports intentionally creates late deadline pressure. A plan-only preflight is required before a live run to confirm admission contention on the exact fleet.
