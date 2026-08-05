# Three-hour simulation trace index

This is the human-readable quick reference for connecting an online benchmark
job to its production trace source. The machine-readable mapping is
[`trace_mapping.yaml`](trace_mapping.yaml).

The relationship is:

```text
benchmark job → source ID → raw dataset → analysis JSON → AIPerf YAML
```

A raw dataset may calibrate multiple jobs. It belongs to a job only through the
mapping below.

All benchmarks use the same runner:

```bash
uv run python -m data_generation.benchmark_trace_analysis \
  datasets/simulation_3h/trace_mapping.yaml
```

Other benchmarks need only their own `trace_mapping.yaml`; they do not need
benchmark-specific Python.

## AIPerf calibration rules

- Stationary jobs use the measured mean RPS directly.
- Diurnal jobs use a full-window ramp ending at twice the measured mean RPS.
- Gamma smoothness is `1 / interarrival CV²`, rounded to three decimals.
- Synthetic token distributions use the measured token statistics and bounds.
- `concurrency` remains an execution safety ceiling because these trace
  artifacts do not contain a measured concurrency value.

## Source registry

| Source ID | Raw file | Coverage |
| --- | --- | --- |
| `AZ-CONV` | [`raw_datasets/azure_llm/AzureLLMInferenceTrace_conv_1week.csv`](raw_datasets/azure_llm/AzureLLMInferenceTrace_conv_1week.csv) | One week of production conversation requests |
| `AZ-CODE` | [`raw_datasets/azure_llm/AzureLLMInferenceTrace_code_1week.csv`](raw_datasets/azure_llm/AzureLLMInferenceTrace_code_1week.csv) | One week of production code requests |
| `BURST` | [`raw_datasets/burstgpt/BurstGPT_3.csv`](raw_datasets/burstgpt/BurstGPT_3.csv) | Azure OpenAI conversation and API traffic |
| `BAIL-CODER` | [`raw_datasets/bailian/qwen_coder_blksz_16.jsonl`](raw_datasets/bailian/qwen_coder_blksz_16.jsonl) | Production Qwen coding traffic |
| `BAIL-TOB` | [`raw_datasets/bailian/qwen_traceB_blksz_16.jsonl`](raw_datasets/bailian/qwen_traceB_blksz_16.jsonl) | Production business/API task automation |
| `BAIL-THINK` | [`raw_datasets/bailian/qwen_thinking_blksz_16.jsonl`](raw_datasets/bailian/qwen_thinking_blksz_16.jsonl) | Production reasoning-intensive traffic |
| `RAG` | [`raw_datasets/ragpulse/0_trace.jsonl`](raw_datasets/ragpulse/0_trace.jsonl) | Deployed retrieval-augmented question-answering service |
| `WEKA` | [`raw_datasets/weka_agents/claude_code_with_subagents.jsonl`](raw_datasets/weka_agents/claude_code_with_subagents.jsonl) | Production agent sessions and subagent fan-out |
| `MOON-CONV` | [`raw_datasets/mooncake/conversation_trace.jsonl`](raw_datasets/mooncake/conversation_trace.jsonl) | Long-context production conversation traffic |
| `SWISS` | [`raw_datasets/swissai/swissai_serving_trace.jsonl`](raw_datasets/swissai/swissai_serving_trace.jsonl) | Long-running serving trace with request duration and tokens |

## Job-to-source mapping

| Job | Benchmark ID and config | Active | Pattern | Source | Match | Trace status |
| --- | --- | ---: | --- | --- | --- | --- |
| Chat Free | [`job-benchmark_timing_3hr-0-1`](../../benchmarks/simulation_3h/jobs/chat-free-core.yaml) | 0–180 | Diurnal | `AZ-CONV` | Exact class | [`Analysis ready`](traces/chat_free_analysis.json) |
| Chat Premium | [`job-benchmark_timing_3hr-0-2`](../../benchmarks/simulation_3h/jobs/chat-premium-core.yaml) | 0–180 | Diurnal | `BURST`, GPT-4 conversation filter | Exact class | [`Analysis ready`](traces/chat_premium_analysis.json) |
| Code Autocomplete | [`job-benchmark_timing_3hr-0-3`](../../benchmarks/simulation_3h/jobs/code-autocomplete-core.yaml) | 0–180 | Bursty | `BAIL-CODER` | Exact class | [`Analysis ready`](traces/code_autocomplete_analysis.json) |
| Safety Moderation | [`job-benchmark_timing_3hr-0-4`](../../benchmarks/simulation_3h/jobs/safety-moderation-core.yaml) | 0–180 | Bursty | `BURST`, short API filter | Traffic proxy | [`Analysis ready`](traces/safety_moderation_analysis.json) |
| Embeddings | [`job-benchmark_timing_3hr-0-5`](../../benchmarks/simulation_3h/jobs/embeddings-core.yaml) | 0–180 | Constant | `RAG`, query-embedding stage | Stage proxy | [`Analysis ready`](traces/embeddings_analysis.json) |
| Search RAG | [`job-benchmark_timing_3hr-7-1`](../../benchmarks/simulation_3h/jobs/search-rag.yaml) | 7–180 | Poisson | `RAG` | Exact class | [`Analysis ready`](traces/search_rag_analysis.json) |
| Search Rerank | [`job-benchmark_timing_3hr-7-2`](../../benchmarks/simulation_3h/jobs/search-rerank.yaml) | 7–180 | Bursty | `RAG`, reranking stage | Stage proxy | [`Analysis ready`](traces/search_rerank_analysis.json) |
| Support Assistant | [`job-benchmark_timing_3hr-7-3`](../../benchmarks/simulation_3h/jobs/support-assistant.yaml) | 7–180 | Bursty | `RAG` | Exact class | [`Analysis ready`](traces/support_assistant_analysis.json) |
| Code Morning Burst | [`job-benchmark_timing_3hr-32-1`](../../benchmarks/simulation_3h/jobs/code-morning-burst.yaml) | 32–87 | Bursty | `AZ-CODE`, separate burst window | Exact class | [`Analysis ready`](traces/code_morning_burst_analysis.json) |
| Enterprise Chat | [`job-benchmark_timing_3hr-32-2`](../../benchmarks/simulation_3h/jobs/enterprise-chat.yaml) | 32–152 | Diurnal | `AZ-CONV`, separate window | Exact class | [`Analysis ready`](traces/enterprise_chat_analysis.json) |
| Product Launch Chat | [`job-benchmark_timing_3hr-49-1`](../../benchmarks/simulation_3h/jobs/product-launch-chat.yaml) | 49–84 | Bursty | `BURST`, sustained extreme window | Traffic proxy | [`Analysis ready`](traces/product_launch_chat_analysis.json) |
| Product Launch Moderation | [`job-benchmark_timing_3hr-49-2`](../../benchmarks/simulation_3h/jobs/product-launch-moderation.yaml) | 49–84 | Bursty | Same `BURST` parent events as launch chat | Correlated proxy | [`Analysis ready`](traces/product_launch_moderation_analysis.json) |
| Premium Agents Kimi | [`job-benchmark_timing_3hr-67-1`](../../benchmarks/simulation_3h/jobs/premium-agents-kimi.yaml) | 67–157 | Bursty | `WEKA` | Exact class | [`Analysis ready`](traces/premium_agents_kimi_analysis.json) |
| Frontier GLM Reasoning | [`job-benchmark_timing_3hr-67-2`](../../benchmarks/simulation_3h/jobs/frontier-reasoning-glm.yaml) | 67–142 | Bursty | `BAIL-THINK` | Exact class | [`Analysis ready`](traces/frontier_glm_reasoning_analysis.json) |
| MoE Routing Qwen3 | [`job-benchmark_timing_3hr-67-3`](../../benchmarks/simulation_3h/jobs/moe-routing-qwen3.yaml) | 67–142 | Bursty | `BAIL-TOB` | Qwen traffic proxy | [`Analysis ready`](traces/moe_routing_qwen3_analysis.json) |
| Safety P0 Live | [`job-benchmark_timing_3hr-100-2`](../../benchmarks/simulation_3h/jobs/safety-p0-live.yaml) | 100–150 | Bursty | `BURST`, short extreme spike | Traffic proxy | [`Analysis ready`](traces/safety_p0_live_analysis.json) |
| Meeting Summary | [`job-benchmark_timing_3hr-125-1`](../../benchmarks/simulation_3h/jobs/meeting-summary.yaml) | 125–170 | Bursty | `MOON-CONV`, input ≥24,000 tokens | Long-context proxy | [`Analysis ready`](traces/meeting_summary_analysis.json) |
| Legal/Finance Review | [`job-benchmark_timing_3hr-145-1`](../../benchmarks/simulation_3h/jobs/legal-finance-review.yaml) | 145–180 | Diurnal | `SWISS`, input ≥3,000 tokens | Long-context proxy | [`Analysis ready`](traces/legal_finance_review_analysis.json) |

## Status meaning

- **Source ready:** the production dataset is downloaded and parser-compatible,
  but no job-specific analysis has been attached.
- **Analysis ready:** a JSON analysis artifact exists under `traces/` and records
  its source, filters, and selected window in the analysis registry.

## Analysis registry

| Analysis | Source | Scope | Attached jobs |
| --- | --- | --- | --- |
| [`traces/BurstGPT_3_analysis.json`](traces/BurstGPT_3_analysis.json) | `BURST` | Complete source; no filters or selected window | None |
| [`traces/chat_free_analysis.json`](traces/chat_free_analysis.json) | `AZ-CONV` | No filter; `2024-05-12T22:15:00Z`–`2024-05-13T01:15:00Z`; 400,476 requests | Chat Free |
| [`traces/chat_premium_analysis.json`](traces/chat_premium_analysis.json) | `BURST` | GPT-4 conversation; `26637060`–`26647860` seconds; 122 requests | Chat Premium |
| [`traces/code_autocomplete_analysis.json`](traces/code_autocomplete_analysis.json) | `BAIL-CODER` | No filter; `0`–`7200` seconds; 43,011 requests | Code Autocomplete |
| [`traces/safety_moderation_analysis.json`](traces/safety_moderation_analysis.json) | `BURST` | API, input ≤512, output ≤32; `21456180`–`21466980` seconds; 35,485 requests | Safety Moderation |
| [`traces/embeddings_analysis.json`](traces/embeddings_analysis.json) | `RAG` | Query-embedding stage; `12000`–`22800` seconds; 272 requests | Embeddings |
| [`traces/search_rag_analysis.json`](traces/search_rag_analysis.json) | `RAG` | Generation stage; `440700`–`451080` seconds; 206 requests | Search RAG |
| [`traces/search_rerank_analysis.json`](traces/search_rerank_analysis.json) | `RAG` | Query plus passages; `215700`–`226080` seconds; 96 requests | Search Rerank |
| [`traces/support_assistant_analysis.json`](traces/support_assistant_analysis.json) | `RAG` | Generation stage; `414900`–`425280` seconds; 96 requests | Support Assistant |
| [`traces/code_morning_burst_analysis.json`](traces/code_morning_burst_analysis.json) | `AZ-CODE` | No filter; `2024-05-15T11:55:00Z`–`2024-05-15T12:50:00Z`; 76,074 requests | Code Morning Burst |
| [`traces/enterprise_chat_analysis.json`](traces/enterprise_chat_analysis.json) | `AZ-CONV` | No filter; `2024-05-15T20:15:00Z`–`2024-05-15T22:15:00Z`; 285,837 requests | Enterprise Chat |
| [`traces/product_launch_chat_analysis.json`](traces/product_launch_chat_analysis.json) | `BURST` | API; `27303060`–`27305160` seconds; 8,592 requests | Product Launch Chat |
| [`traces/product_launch_moderation_analysis.json`](traces/product_launch_moderation_analysis.json) | `BURST` | Same parent events and window as launch chat; 8,592 requests | Product Launch Moderation |
| [`traces/premium_agents_kimi_analysis.json`](traces/premium_agents_kimi_analysis.json) | `WEKA` | Session `2f7abb…85bde`; `0`–`5400` seconds; 417 requests | Premium Agents Kimi |
| [`traces/frontier_glm_reasoning_analysis.json`](traces/frontier_glm_reasoning_analysis.json) | `BAIL-THINK` | No filter; `0`–`4500` seconds; 7,384 requests | Frontier GLM Reasoning |
| [`traces/moe_routing_qwen3_analysis.json`](traces/moe_routing_qwen3_analysis.json) | `BAIL-TOB` | No filter; `600`–`5100` seconds; 108,087 requests | MoE Routing Qwen3 |
| [`traces/safety_p0_live_analysis.json`](traces/safety_p0_live_analysis.json) | `BURST` | API, input 256–2,048, output 64–512; `22271160`–`22274160` seconds; 2,649 requests | Safety P0 Live |
| [`traces/meeting_summary_analysis.json`](traces/meeting_summary_analysis.json) | `MOON-CONV` | Input ≥24,000; `3444600`–`3447300` seconds; 9 requests | Meeting Summary |
| [`traces/legal_finance_review_analysis.json`](traces/legal_finance_review_analysis.json) | `SWISS` | Input ≥3,000, output 300–1,500; `2024-12-04T09:40:00Z`–`2024-12-04T10:15:00Z`; 31 requests | Legal/Finance Review |

For source URLs, licenses, column mappings, filters, and limitations, see
[`raw_datasets/README.md`](raw_datasets/README.md).
