# Three-hour simulation production traces

These files calibrate traffic for the 18 online jobs in
`KoiBench-3h-ReservedFleet-Smoke`. They contain production-derived request
timing and token metadata. They are not prompt-content datasets.

Raw data files are intentionally ignored by Git. This README records where each
file came from and how it should be used.

## Job mapping

| Online job | Declared pattern | Primary trace | Match quality | Selection guidance |
| --- | --- | --- | --- | --- |
| Chat Free | Diurnal | Azure conversation | Exact workload class | Use the normalized daily curve and select a representative three-hour daytime window. |
| Chat Premium | Diurnal | BurstGPT | Exact workload class | Filter `Model=GPT-4` and `Log Type=Conversation log`; retain the measured daily shape. |
| Code Autocomplete | Bursty | Bailian Coder | Exact workload class | Use the production request timestamps and session turns; resample the two-hour trace to three hours later. |
| Safety Moderation | Bursty | BurstGPT | Production-traffic proxy | Use a bursty API window with short inputs and outputs. No public production moderation inference trace was found. |
| Embeddings | Constant | RAGPulse | Exact pipeline-stage proxy | Treat each RAG query as a query-embedding event, use its real timestamp and user-input length, and select the lowest-variance window. Embedding output length is intrinsically zero. |
| Search RAG | Poisson | RAGPulse | Exact application class | Select the most stationary three-hour window from the deployed question-answering service. |
| Search Rerank | Bursty | RAGPulse | Exact pipeline-stage proxy | Reuse RAG query timestamps and candidate-passage identifiers; select a bursty window. |
| Support Assistant | Bursty | RAGPulse | Exact workload class | Use the deployed university policy question-answering sessions and a bursty window. |
| Code Morning Burst | Bursty | Azure code | Exact workload class | Select a high-scoring three-hour burst window that does not overlap the Code Autocomplete calibration window. |
| Enterprise Chat | Diurnal | Azure conversation | Exact workload class | Use a separate non-overlapping window from the one-week trace so the daily traffic shape is measurable. |
| Product Launch Chat | Bursty | BurstGPT | Production-traffic proxy | Select a sustained, extreme API burst rather than a single one-second outlier. |
| Product Launch Moderation | Bursty | BurstGPT | Correlated production-traffic proxy | Derive it from the same parent requests as Product Launch Chat so the two streams remain correlated. |
| Premium Agents Kimi | Bursty | Weka Claude Code with subagents | Exact agentic workload class | Preserve intra-session request timing, API duration, and subagent fan-out. A representative 90-minute session is stored as `weka_agents/premium_agents_kimi_normalized.csv`. |
| Frontier GLM Reasoning | Bursty | Bailian Thinking | Exact workload class | Use the reasoning-intensive production trace with its measured long outputs. |
| MoE Routing Qwen3 | Bursty | Bailian To-B | Closest production Qwen proxy | Preserve the text/search/image/file request mix. A traffic trace cannot observe or prescribe internal expert routing. |
| Safety P0 Live | Bursty | BurstGPT | Production-traffic proxy | Select a short, extreme API spike with short-to-medium token lengths. This is not an authentic safety-incident trace. |
| Meeting Summary | Bursty | Mooncake conversation | Long-context production proxy | Filter for input length at least 24,000 tokens and choose a clustered arrival window. No public production meeting-summary serving trace was found. |
| Legal/Finance Review | Diurnal | Swiss AI serving | Long-context production proxy | Filter successful large-model requests with input length at least 3,000 tokens and select a low-volume window with a daily cycle. The trace has no application-domain label. |

“Exact workload class” means the production system serves the same broad kind
of workload. It does not mean the requests came from the KoiBench application.
Proxy rows are deliberately labeled and must not be described as authentic
moderation, product-launch, meeting, legal, or finance traffic.

## Downloaded sources

| Folder | Source | Local files | Published fields used |
| --- | --- | --- | --- |
| `azure_llm/` | [Microsoft Azure LLM Inference Trace 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) | One-week code and conversation CSV files | Invocation timestamp, context tokens, generated tokens |
| `bailian/` | [Alibaba Bailian Qwen usage traces](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon) | To-C, To-B, Coder, and Thinking JSON Lines files | Relative timestamp, input/output lengths, session hierarchy, request type |
| `burstgpt/` | [BurstGPT](https://github.com/HPMLL/BurstGPT) | `BurstGPT_3.csv` | Timestamp, elapsed time, model, service category, request/response tokens |
| `ragpulse/` | [RAGPulse](https://github.com/flashserve/RAGPulse) | Main trace plus system-prompt, passage, history, user-input, and web-search tables | Relative timestamp, input/output lengths, session ID, RAG component IDs |
| `weka_agents/` | [Weka Claude Code traces with subagents](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-051826) | Full source JSON Lines file and normalized Premium Agents Kimi CSV slice | Per-session timestamp, API duration, input/output tokens, subagent structure |
| `mooncake/` | [Mooncake FAST 2025 traces](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces) | `conversation_trace.jsonl` | Relative timestamp, input/output lengths, prefix block hashes |
| `swissai/` | [Swiss AI serving trace](https://huggingface.co/datasets/eth-easl/swissai-serving-trace) | `swissai_serving_trace.jsonl` | Request start/end, model, input/output tokens |

Review each upstream license before redistributing a raw file. The Azure data is
CC-BY, Bailian and Weka are Apache-2.0, RAGPulse is MIT, and Mooncake is
Apache-2.0. The Swiss AI dataset card labels its license as `other`.

## Normalized columns

`src/data_generation/traffic_trace_analysis.py` normalizes flat source records
to the following internal columns and discards unrelated content:

| Canonical column | Source examples |
| --- | --- |
| `timestamp_seconds` | `TIMESTAMP`, `timestamp`, `start_time` |
| `elapsed_seconds` | `Elapsed time`, `api_time`, or genuine `end_time - start_time` |
| `input_tokens` | `ContextTokens`, `input_length`, `Request tokens`, `in` |
| `output_tokens` | `GeneratedTokens`, `output_length`, `Response tokens`, `out` |

`elapsed_seconds` is unavailable in Azure, Bailian, RAGPulse, and Mooncake. It
remains null for those sources. Arrival-rate and token analysis still works;
maximum observed concurrency cannot be calculated from a trace without genuine
request completion times. The benchmark-authored concurrency cap remains a
separate load-generation control.

Do not fill missing elapsed times with estimated service time. That would mix a
model assumption into the measured production trace.
