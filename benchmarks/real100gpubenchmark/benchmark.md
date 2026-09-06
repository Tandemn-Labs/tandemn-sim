# Real 100-GPU benchmark

Benchmark ID: `real100gpubenchmark`

This executable real-hardware benchmark runs a 180-minute workload followed by a 30-minute drain on an exact 96-GPU, 12-node reserved fleet. No on-demand capacity is available.

## Fleet

| Accelerator | Node shape | Nodes | `us-central1` | `europe-west2` | Total GPUs |
|---|---:|---:|---:|---:|---:|
| `RTXPRO6000` | 8 x 96 GB | 5 | 24 | 16 | 40 |
| `H100` | 8 x 80 GB | 2 | 8 | 8 | 16 |
| `A100-80GB` | 8 x 80 GB | 2 | 8 | 8 | 16 |
| `L4` | 8 x 24 GB | 3 | 16 | 8 | 24 |
| **Total** |  | **12** | **56** | **40** | **96** |

## Workload

The manifest contains 7 online jobs and 4 batch campaigns. The selected batch campaigns are Mini Nightly Eval, Phi-4 Quant Probe, Calibration Probe, and Sales Reports, totaling exactly 26,200 records across 106 shards. Their definitions are unchanged from `simulation_3h`; the external pipeline owns batch dataset staging.

The maximum simultaneous online declaration is 60 streams, with roughly 2.4 target requests/s during the main overlap. Every online window lasts at least 45 minutes so deployment cold starts do not consume the whole measurement window. Online traffic stops at minute 180 and in-flight work drains through the hard stop at minute 210.

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

Peak demand is sized for 10 of the 12 whole nodes, retaining two nodes of headroom. Completion of every batch campaign before its deadline is not guaranteed. Sales Reports intentionally creates late deadline pressure. The workload has no deliberate overload floor.
