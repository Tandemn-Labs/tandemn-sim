# Data generation benchmark controller

This repository defines, runs, pauses, resumes, and records synthetic AI
inference benchmarks. The controller is intentionally benchmark-agnostic: its
behavior comes from the selected benchmark's `overview.yaml`, not from
benchmark-specific Python code.

The canonical hierarchy is:

```text
benchmark
└── jobs
    └── requests and responses
```

## Canonical terminology

These terms have specific meanings throughout the repository.

| Term | Definition |
|---|---|
| **Benchmark** | A named collection of jobs scheduled on one shared clock. |
| **Job** | One independently launched online generator or batch process within a benchmark. |
| **Run** | One execution attempt of a benchmark, identified by a run ID. |
| **Online job** | A time-bounded job that generates requests throughout an active window. |
| **Batch job** | A finite set of requests released at a start time and evaluated against a deadline. |
| **Request** | One payload generated or loaded by a job and sent toward an API. |
| **Response** | The result or error paired with a request ID. |
| **Artifact** | Runtime state or request/response data produced by a benchmark run. |

`Benchmark` and `job` are the only hierarchy terms. Traffic shape, request
shape, and dataset shape are properties of a job; they are not additional
levels in the hierarchy.

## Benchmark catalog

The repository contains six canonical benchmark definitions:

| Benchmark ID | Environment | Duration | Human-readable definition | Executable today |
|---|---|---:|---|---|
| `simulation_3h` | Simulation | 3 hours | `benchmarks/simulation_3h/benchmark.md` | Yes |
| `simulation_12h` | Simulation | 12 hours | `benchmarks/simulation_12h/benchmark.md` | Definition only |
| `simulation_24h` | Simulation | 24 hours | `benchmarks/simulation_24h/benchmark.md` | Definition only |
| `real_life_3h` | Real-life replay | 3 hours | `benchmarks/real_life_3h/benchmark.md` | Definition only |
| `real_life_12h` | Real-life replay | 12 hours | `benchmarks/real_life_12h/benchmark.md` | Definition only |
| `real_life_24h` | Real-life replay | 24 hours | `benchmarks/real_life_24h/benchmark.md` | Definition only |

A benchmark becomes executable when its folder contains:

```text
benchmark.md
overview.yaml
jobs/
  <job-config>.yaml
```

`simulation_3h` currently has this complete structure. The other five
benchmarks retain their detailed source definitions but have not yet been
converted into controller overviews and individual job YAMLs.

## Repository structure

```text
data-generation/
├── benchmarks/
│   ├── simulation_3h/
│   │   ├── benchmark.md
│   │   ├── overview.yaml
│   │   ├── JOB_REFERENCE.md
│   │   └── jobs/
│   │       ├── chat-free-core.yaml
│   │       ├── batch-mini-nightly-eval.yaml
│   │       └── ...
│   ├── simulation_12h/
│   │   └── benchmark.md
│   ├── simulation_24h/
│   │   └── benchmark.md
│   ├── real_life_3h/
│   │   └── benchmark.md
│   ├── real_life_12h/
│   │   └── benchmark.md
│   └── real_life_24h/
│       └── benchmark.md
├── src/data_generation/
│   ├── controller.py
│   └── batch_dataset_generator.py
├── tests/
│   ├── smoke_test/
│   │   ├── overview.yaml
│   │   ├── print_sink.py
│   │   └── jobs/
│   ├── batch_prompt_generation.yaml
│   ├── test_controller.py
│   └── test_batch_dataset_generator.py
├── artifacts/
│   └── benchmarks/
├── benchmark_schema.md
├── pyproject.toml
├── uv.lock
└── README.md
```

Generated artifacts, virtual environments, caches, and local process output
are not repository source and are excluded from version control.

## How benchmark files relate

The three files under an executable benchmark have different responsibilities.

### `benchmark.md`: human-readable source

`benchmark.md` explains the intended benchmark:

- benchmark identity and duration;
- available resources;
- job arrivals and active windows;
- online traffic characteristics;
- batch campaign characteristics;
- model, request-shape, and SLO assumptions.

This file is the design source. The controller does not parse it directly.
When a benchmark is made executable, its contents are translated into
`overview.yaml` and the YAML files under `jobs/`.

The shared conceptual contract for these definitions lives in
`benchmark_schema.md`.

### `overview.yaml`: executable benchmark schedule

`overview.yaml` is the controller's entrypoint for one benchmark. It has two
top-level sections:

```yaml
schema_version: "benchmark/v1"

benchmark:
  id: simulation_3h
  source_benchmark: benchmark.md
  poll_interval_seconds: 60
  shutdown_grace_seconds: 60
  hard_stop_at_offset_minutes: 210

jobs:
  <job-id>:
    enabled: true
    kind: online
    driver: aiperf
    config: jobs/<job-config>.yaml
    start_at_offset_minutes: 0
    stop_at_offset_minutes: 180
```

The `benchmark` section defines the shared clock and controller behavior. The
`jobs` mapping is the complete collection of jobs in that benchmark.

Every `config` path is resolved relative to `overview.yaml`. This keeps each
benchmark self-contained and movable as one folder.

Important benchmark fields:

| Field | Purpose |
|---|---|
| `id` | Stable benchmark ID used in logs, state, and artifact paths. |
| `source_benchmark` | Human-readable `benchmark.md` from which the schedule was derived. |
| `poll_interval_seconds` | How frequently the controller checks benchmark time and job processes. |
| `shutdown_grace_seconds` | Time allowed for jobs to stop before `SIGKILL`. |
| `hard_stop_at_offset_minutes` | Final benchmark-relative cutoff inherited by jobs without their own override. |

### `jobs/*.yaml`: one job's generation configuration

Each job listed in `overview.yaml` points to exactly one job configuration.
The overview controls **when** the job runs. The job YAML controls **what data
it generates**.

Job IDs are keys in `overview.yaml`. Job configuration filenames are readable
descriptions such as `chat-free-core.yaml`. They do not have to repeat the job
ID because the overview provides the mapping.

## Job classification

Jobs are classified independently by lifecycle and driver.

### Lifecycle: online or batch

| Kind | Starts | Ends | Primary timing fields |
|---|---|---|---|
| `online` | First controller tick at or after its start offset | Stop offset, followed by process shutdown | `start_at_offset_minutes`, `stop_at_offset_minutes` |
| `batch` | First controller tick at or after its start offset | Natural completion or hard stop | `start_at_offset_minutes`, `deadline_at_offset_minutes`, `hard_stop_at_offset_minutes` |

An online job produces traffic throughout a half-open active window. A batch
job has a finite request count. Reaching a batch deadline records a deadline
miss but does not immediately terminate the job; the hard stop is the final
cutoff.

### Driver: AIPerf or external process

| Driver | Meaning |
|---|---|
| `aiperf` | The controller runs `aiperf profile` with the referenced job YAML. |
| `external_mock_data` | The controller runs the explicit `command` declared by the job. This is used for the local request sink and is the integration point for future batch senders. |

Every launched job is its own child process and process group. The controller
does not run jobs as Python threads. Multiple jobs can run concurrently because
their processes are independent.

### Enabled and disabled jobs

`enabled: true` places a job into the scheduler. `enabled: false` preserves the
job in the benchmark definition and state map without launching it.

The current `simulation_3h` overview defines 26 jobs:

- 18 enabled online AIPerf jobs;
- 8 disabled batch jobs awaiting sender integration.

### Parent jobs

An additive online job may declare `parent_job_id`. The controller validates
that the additive job's active window stays inside its parent job's window.
This field expresses the relationship between a base stream and extra demand;
it does not combine their processes.

## Online job configuration

Online job YAMLs use AIPerf's configuration schema. They define:

- target model and endpoint type;
- streaming behavior and timeout;
- synthetic input and output token distributions;
- prefix reuse and session turns;
- traffic pattern, rate, concurrency, ramp, and duration;
- random seed for reproducibility.

Example relationship:

```text
benchmarks/simulation_3h/overview.yaml
  jobs.job-benchmark_timing_3hr-0-1.config
    -> jobs/chat-free-core.yaml
       -> AIPerf synthetic requests
```

The full-size `simulation_3h` job YAMLs currently use
`http://preview.invalid/...` endpoints because endpoint integration is
deliberately deferred. They accurately define request and traffic shapes but
will not reach a real model server until those endpoints are replaced or
injected.

The smoke job YAMLs use `http://127.0.0.1:8765` and are safe for local
inspection.

## Batch job configuration

Batch job YAMLs use the repository's `batch-dataset/v1` schema. Each file
defines one finite dataset for one batch job:

- dataset ID;
- model and request format;
- total record count;
- shard count and requests per shard;
- input-token distribution;
- output-token distribution;
- deterministic random seed.

For every generated request:

1. The input and output token lengths are sampled from the configured bounded
   distributions.
2. AIPerf's built-in prompt generator creates unique synthetic prompt text at
   the sampled input length.
3. The generator creates an OpenAI-compatible JSON record.
4. Chat records place output length in `max_tokens`; embedding records omit
   generated-output settings.

The batch generator creates request data. It does not currently send that data.
The batch jobs remain disabled in the benchmark overview until a batch sender
is connected through the job's driver and command.

## Controller execution flow

The controller always works top-down:

```text
controller
  -> load benchmarks/<benchmark-id>/overview.yaml
  -> resolve every jobs/<job-config>.yaml
  -> validate the benchmark and job windows
  -> start the benchmark clock with time.time()
  -> poll all job states
  -> start, observe, or stop each job
  -> save benchmark and job state
  -> capture job request/response artifacts
```

On each tick, the controller:

1. Calculates elapsed active benchmark time using `time.time()`.
2. Checks every job in the overview.
3. Starts eligible pending jobs.
4. Detects exited child processes.
5. Marks batch deadline misses.
6. Stops online jobs whose active windows ended.
7. Force-stops jobs that exceed the shutdown grace period.
8. Synchronizes newly available AIPerf request/response records.
9. Writes the benchmark checkpoint and run index.

The overview is loaded when the controller starts. The controller polls its
in-memory benchmark and job objects; it does not reread the YAML on every tick.

Because dispatch happens on polling boundaries, a job scheduled for `T+120`
starts on the first tick whose elapsed benchmark time is at least 120 minutes.

## Job state model

The controller tracks each job independently:

```text
disabled

pending -> running -> stopping -> completed
   |          |                       |
   |          +---------------------> failed
   |
   +-------------------------------> missed

running -> paused -> pending on resume
```

State meanings:

| State | Meaning |
|---|---|
| `pending` | Enabled but not started yet. |
| `running` | Child process is active. |
| `stopping` | Graceful stop requested; waiting for exit. |
| `completed` | Process exited successfully or after an expected scheduled stop. |
| `failed` | Launch failed or the process exited unexpectedly with a nonzero code. |
| `missed` | The job's valid launch window closed before it started. |
| `disabled` | Present in the benchmark but intentionally excluded from execution. |
| `paused` | Active when the controller was interrupted and eligible for resume. |

## Runs, milestones, pause, and resume

Each benchmark execution receives a UTC run ID such as
`20260724T201601Z`. Runtime state is stored separately from benchmark
definitions.

`benchmark_state.json` records:

- benchmark ID, run ID, status, and elapsed active time;
- benchmark and job milestones;
- every job's current state;
- launch count;
- start, pause, resume, and completion timestamps;
- accumulated process runtime;
- process exit code;
- deadline-miss status.

Pressing `Ctrl+C` raises `KeyboardInterrupt` in the controller. The controller
then:

1. sends `SIGINT` to every active job process group;
2. waits for `shutdown_grace_seconds`;
3. sends `SIGKILL` to any remaining process groups;
4. flushes available job records;
5. saves the benchmark as `paused`.

Resume the same run with:

```bash
uv run controller simulation_3h --resume <run-id>
```

Paused wall-clock time does not advance benchmark time. Completed jobs are not
replayed. Paused jobs are requeued and start again if their benchmark windows
remain open. AIPerf restarts the generator process; it does not restore one
partially completed in-memory request or session.

The controller rejects resume when the persisted job set or core job schedule
no longer matches the selected overview.

## Artifact structure

Runtime output mirrors the benchmark-to-job hierarchy:

```text
artifacts/
└── benchmarks/
    └── <benchmark-id>/
        ├── runs.json
        └── <run-id>/
            ├── benchmark_state.json
            └── jobs/
                └── <job-id>/
                    ├── prompts.jsonl
                    ├── responses.jsonl
                    ├── process.log              # launched, captured-output jobs
                    └── <AIPerf support files>   # AIPerf jobs
```

`runs.json` is the lightweight index of all known runs for one benchmark.
`benchmark_state.json` is the durable checkpoint for one run.

Every job directory is created when the benchmark run starts, including
disabled and future jobs. The canonical data files are:

### `prompts.jsonl`

One compact JSON object per sent request:

```json
{
  "request_id": "request-id",
  "sent_at": "2026-07-24T20:16:01.123456+00:00",
  "request": {
    "model": "model-name",
    "messages": []
  }
}
```

### `responses.jsonl`

One compact JSON object per returned response or request error:

```json
{
  "request_id": "request-id",
  "received_at": "2026-07-24T20:16:02.123456+00:00",
  "status_code": 200,
  "response": [],
  "error": null
}
```

The shared `request_id` joins prompts to responses.

For AIPerf jobs, raw export is enabled automatically. The controller reads
complete raw JSONL records, separates requests and responses, and appends only
unseen request IDs. Synchronization happens on controller ticks and once more
during shutdown. This makes pause and resume append-safe.

AIPerf may also write raw exports, metrics, inputs, and other support files in
the job directory. `process.log` contains captured stdout and stderr unless the
job explicitly enables terminal output.

## Running the repository

Install the project and development dependencies:

```bash
uv sync
```

Validate the default executable benchmark without launching jobs:

```bash
uv run controller --validate-only
```

Run the three-hour simulation:

```bash
uv run controller simulation_3h
```

An explicit overview path is also accepted:

```bash
uv run controller benchmarks/simulation_3h/overview.yaml
```

Use another artifact root:

```bash
uv run controller simulation_3h --artifact-root /path/to/artifacts
```

## Local smoke benchmark

Run the compressed three-minute benchmark:

```bash
uv run controller tests/smoke_test/overview.yaml
```

The smoke overview launches:

- a local threaded HTTP sink on `127.0.0.1:8765`;
- low-volume versions of the enabled online jobs;
- AIPerf generators pointed at that local sink.

The sink prints concise request summaries and returns minimal
OpenAI-compatible responses. It does not call a real model.

If port `8765` is already in use, stop the previous controller and sink before
starting another smoke run:

```bash
lsof -nP -iTCP:8765 -sTCP:LISTEN
```

Stop the active smoke benchmark with `Ctrl+C`.

## Generating batch prompts

Generate three requests from the editable batch smoke configuration:

```bash
uv run controller \
  --generate-batch tests/batch_prompt_generation.yaml \
  --records 3
```

Records print to the terminal by default. Save them only when needed:

```bash
uv run controller \
  --generate-batch tests/batch_prompt_generation.yaml \
  --records 3 \
  --output artifacts/batch_prompt_smoke.jsonl
```

`--records` cannot exceed the job YAML's configured `record_count`.

## Adding or converting a benchmark

Use this sequence to make another benchmark executable:

1. Create or retain `benchmarks/<benchmark-id>/benchmark.md`.
2. Add the exact `Benchmark ID` to the Markdown definition.
3. Create `benchmarks/<benchmark-id>/overview.yaml`.
4. Use `schema_version: "benchmark/v1"`.
5. Set `benchmark.id` equal to the folder name.
6. Set `benchmark.source_benchmark` to `benchmark.md`.
7. Translate every benchmark job into one entry under `jobs`.
8. Add one configuration YAML under `jobs/` for every overview entry.
9. Ensure online jobs have valid start and stop offsets.
10. Ensure batch jobs have start, deadline, and hard-stop offsets.
11. Validate parent-job windows.
12. Run the controller with `--validate-only`.
13. Add or adapt a local smoke benchmark before a full-duration run.

No controller code changes are required for a new benchmark that follows the
existing overview contract and supported job drivers.

## Validation and tests

Run all repository checks:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run python -m unittest discover -s tests
uv run pre-commit run --all-files
```

The tests cover:

- loading all 26 jobs in `simulation_3h`;
- online scheduled stopping;
- batch deadline tracking;
- controller pause and resume;
- append-safe prompt and response capture;
- synthetic batch prompt generation and token bounds.

## Current scope

The current architecture intentionally separates concerns:

- benchmark Markdown defines intent;
- overview YAML defines the executable schedule;
- job YAML defines generated data and traffic;
- the controller owns timing and processes;
- artifacts own runtime state, prompts, and responses.

The remaining major integration is the real request destination and batch
sender. The online job shapes, scheduler, state model, local smoke path, batch
prompt generation, and artifact capture are already implemented.
