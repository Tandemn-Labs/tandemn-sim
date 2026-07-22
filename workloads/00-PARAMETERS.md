# Workload parameters

This is the practical parameter reference for the workload YAML files in this folder. It targets the pinned AIPerf `0.11.0` dependency.

## Run identity

| Parameter | What it controls | Options |
|---|---|---|
| `schemaVersion` | AIPerf configuration format | Keep at `"2.0"` |
| `random_seed` | Makes generated data and timing reproducible | Any integer; change it to generate another repeatable sample |
| `benchmark.model` | Model name placed in requests | Any model identifier accepted by the endpoint |
| `benchmark.tokenizer.name` | Tokenizer used to construct and measure prompts | `builtin`, a Hugging Face model name, or a supported local tokenizer |

## Endpoint

| Parameter | What it controls | Options |
|---|---|---|
| `endpoint.url` | Destination used by `aiperf profile` | Any compatible endpoint URL; `preview.invalid` prevents accidental sending |
| `endpoint.type` | Request API shape | Keep `chat` for these online-chat workloads |
| `endpoint.streaming` | Whether responses stream incrementally | `true` or `false` |
| `endpoint.timeout` | Per-request timeout | Seconds, such as `120` |
| `endpoint.connectionReuse` | HTTP connection behavior | `pooled`, `never`, or `sticky-user-sessions` |

`sticky-user-sessions` is useful for multi-turn chat, but actual replica affinity still depends on the deployment or load balancer.

## Dataset and request sizes

| Parameter | What it controls |
|---|---|
| `dataset.type` | `synthetic` generates prompts; `file` reads a dataset or trace |
| `dataset.entries` | Number of unique generated conversations/contexts in the reusable pool |
| `dataset.prompts.isl` | New input tokens generated for a turn |
| `dataset.prompts.osl` | Maximum output tokens requested for a turn |

`isl`, `osl`, `turns`, and `turnDelay` accept these distribution shapes:

| Shape | Syntax | Behavior |
|---|---|---|
| Fixed | `512` or `{value: 512}` | Always the same value |
| Normal | `{mean: 512, stddev: 50}` | Values clustered around the mean |
| Log-normal | `{mean: 512, median: 400}` | Mostly smaller values with a long upper tail |
| Multimodal | `{peaks: [{value: 128, weight: 80}, {value: 2048, weight: 20}]}` | Mixture of distinct request classes |
| Empirical | `{points: [{value: 128, weight: 60}, {value: 512, weight: 40}]}` | Samples exact measured values and frequencies |

Add `min` and `max` to any distribution to impose hard bounds. Weights are relative and do not need to sum to 100.

## Prefix and KV-cache similarity

Choose one prefix model; the two forms are mutually exclusive.

### Shared system and per-user context

```yaml
prefixPrompts:
  sharedSystemLength: 512
  userContextLength: 256
```

| Parameter | What it controls |
|---|---|
| `sharedSystemLength` | Identical starting tokens shared by every session; encourages global prefix-cache reuse |
| `userContextLength` | Context unique to one generated session but reusable by later turns in that session |

### Prefix pool

```yaml
prefixPrompts:
  poolSize: 10
  length: 512
```

| Parameter | What it controls |
|---|---|
| `poolSize` | Number of distinct shared prefixes; a smaller pool creates more reuse |
| `length` | Token length of each shared prefix; longer prefixes make successful hits more valuable |

AIPerf creates cacheable request similarity. Cache capacity, eviction, TTL, block size, and hit latency are properties of the endpoint or simulator.

## Conversations

| Parameter | What it controls | Common form |
|---|---|---|
| `dataset.turns` | Turns per conversation | `{mean: 6, stddev: 2, min: 2, max: 12}` |
| `dataset.turnDelay` | User think time between turns, in milliseconds | `{mean: 10000, stddev: 3000}` |
| `dataset.turnDelayRatio` | Multiplier applied to all turn delays | `1.0`; below `1` speeds them up, above `1` slows them down |

Later turns include earlier conversation history when AIPerf sends the workload, creating per-session prefix reuse and growing input context.

## Traffic phase

With AIPerf `0.11.0`, phase `name` is `warmup` or `profiling`.

| Parameter | What it controls | Options |
|---|---|---|
| `name` | Whether results count toward the benchmark | `warmup` or `profiling` |
| `type` | Scheduling strategy | `constant`, `poisson`, `gamma`, `concurrency`, `user_centric`, or `fixed_schedule` |
| `duration` | Time during which new requests are issued | Seconds or strings such as `30m` and `3h` |
| `requests` | Request-count stop condition | Positive integer |
| `sessions` | Completed-conversation stop condition | Positive integer |
| `rate` | Global average request rate | Requests per second; supports decimals |
| `concurrency` | Maximum simultaneous in-flight requests | Positive integer |
| `gracePeriod` | Time allowed for in-flight requests to finish | Seconds or duration string |

Traffic types:

- `constant`: evenly spaced requests at `rate`.
- `poisson`: independently timed requests around `rate`.
- `gamma`: variable arrivals around `rate`; also set `smoothness`.
- `concurrency`: immediately replaces completed requests to maintain `concurrency`.
- `user_centric`: schedules repeated turns for `users` at a global `rate`.
- `fixed_schedule`: replays exact timestamps from a compatible trace.

Additional timing controls:

| Parameter | Applies to | What it does |
|---|---|---|
| `smoothness` | `gamma` | Below `1` is bursty, `1` is Poisson-like, above `1` is smoother |
| `users` | `user_centric` | Number of simultaneous virtual users; approximate per-user turn gap is `users / rate` seconds |
| `rateRamp` | Rate modes | Gradually increases request rate over the specified duration |
| `concurrencyRamp` | Applicable phases | Gradually increases allowed concurrency |
| `prefillConcurrency` | Applicable phases | Caps requests concurrently processing prompt/prefill tokens |
| `prefillRamp` | Applicable phases | Gradually increases the prefill limit |
| `seamless` | Non-first phases | Starts the next phase without waiting for prior in-flight requests |
| `cancellation.rate` | Applicable phases | Percentage of requests intentionally cancelled |
| `cancellation.delay` | Applicable phases | Seconds after sending before cancellation |

## Recommended validation

Validate a file before running it:

```bash
aiperf config validate workloads/chat-job-1.yaml
```

Generate requests without sending anything:

```bash
uv run workloads generate chat-job-1 --count 100
```
