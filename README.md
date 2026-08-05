# Tandemn benchmark data generation

This repository runs the benchmark-method matrices used by the Tandemn
Efficiency Index evaluation. Benchmark definitions, scheduling methods,
experiment matrices, metric profiles, and run artifacts have separate owners.

The top-level interface always names either one benchmark-method pair or one
experiment:

```bash
uv run benchmark --benchmark simulation_3h --method dynamo_aic
uv run benchmark --experiment main_simulation
```

Use `--placement-only` to generate plans without submitting jobs or starting
data senders. Use `--validate-only` to validate catalog selections without
running them.

## Repository layout

```text
benchmarks/
  catalog.yaml                    # six authoritative benchmark IDs
  <benchmark>/
    benchmark.md                  # human-readable benchmark definition
    benchmark.yaml                # executable manifest and lifecycle settings
    jobs/
      <job>.yaml                  # schedule, placement spec, and workload
    resources.yaml                # benchmark resource map

configs/
  methods.yaml                    # method catalog and lightweight settings
  metrics/
    catalog.yaml
    <profile>.yaml                # ordered metrics for an experiment

experiments/
  catalog.yaml
  <experiment>.yaml               # benchmark x method x metric matrix

src/data_generation/
  cli.py                          # single repository-level entry point
  controller.py                   # clock, lifecycle, submission, and artifacts
  scheduling.py                   # method-neutral scheduling contracts
  benchmarks/registry.py
  experiments/registry.py
  metrics/registry.py
  methods/
    registry.py                   # method discovery and construction
    dynamo_aic.py                 # Dynamo+AIC scheduler
    gpt_only.py                   # GPT-only scheduler via OpenAI

artifacts/
  <experiment>/<benchmark>/<method>/<run>/
```

`configs/methods.yaml` contains the compact method catalog and settings. Each
implemented scheduler is one module under `src/data_generation/methods/`.
The registry constructs it, and the benchmark controller depends only on the
shared scheduling contract.

## Catalogs

`benchmarks/catalog.yaml` declares the six benchmark identities:

| Benchmark | Environment | Duration | Status |
|---|---:|---:|---|
| `simulation_3h` | simulation | 3 hours | executable |
| `simulation_12h` | simulation | 12 hours | definition only |
| `simulation_24h` | simulation | 24 hours | definition only |
| `real_life_3h` | real hardware | 3 hours | definition only |
| `real_life_12h` | real hardware | 12 hours | definition only |
| `real_life_24h` | real hardware | 24 hours | definition only |

Definition-only entries preserve the paper's complete benchmark taxonomy while
making it explicit that their executable traces have not yet been converted.
They cannot be run accidentally.

`configs/methods.yaml` is the authoritative method list. It includes
FCFS, best fixed, Dynamo+AIC, QLM, QoServe, Helix, rolling CP-SAT, contextual
BO, GPT-only, and Koi. Currently `dynamo_aic` and `gpt_only` are
implemented; the remaining entries are marked `planned` and fail clearly if
selected for execution.

`experiments/` contains the paper-aligned comparison matrices:

- `main_simulation`
- `main_real_hardware`
- `mixed_scheduling`
- `heterogeneous_placement`
- `learning_shifts`
- `agentic_ablations`

Each experiment selects benchmark IDs, method IDs, and one metric profile. The
experiment loader cross-checks all three catalogs before a run begins.

## Job model

An executable benchmark has two distinct layers:

1. `benchmark.yaml` owns shared timing, resources, and the jobs directory.
2. Every file under `jobs/` owns one job's schedule, placement spec, and
   driver-specific workload.

Within a job file, `spec` is the canonical input to placement. It contains the
model, precision, token shape, SLO, and optimization goal. `workload` contains
only the driver-specific AIPerf or finite batch configuration. Consequently, a
method never needs to parse AIPerf configuration to understand a job.

The controller resolves both files and converts the result to `SchedulerJob`.
Methods receive that shared job contract plus the live resource map. An
accepted method decision reserves capacity before submission, so the next job
observes the updated map. Terminal jobs release the same capacity.

Method outputs remain placement-specific. `gpt_only` returns only a GPU type
and the `replicas_needed`, `tp`, `pp`, `dp`, and `ep` parallelism values, or a
null placement. Shared code deterministically resolves the pool, region,
market, backend, device total, accelerator metadata, model, targets, plan
action, ladder, and final plan schema from that response, the canonical job,
and the benchmark resource map. Methods never generate the repository's plan
schema themselves.

## Running selections

Install the project and development dependencies:

```bash
uv sync --group dev
```

Validate a direct pair:

```bash
uv run benchmark \
  --benchmark simulation_3h \
  --method gpt_only \
  --validate-only
```

Run GPT-only placement through OpenAI using the configured Luna model:

```bash
# Set OPENAI_API_KEY in the repository's ignored .env file first.
uv run --group openai benchmark \
  --benchmark simulation_3h \
  --method gpt_only \
  --placement-only
```

Run Dynamo+AIC placement only:

```bash
uv run --group dynamo benchmark \
  --benchmark simulation_3h \
  --method dynamo_aic \
  --placement-only
```

Select one cell from an experiment matrix:

```bash
uv run benchmark \
  --experiment agentic_ablations \
  --benchmark simulation_3h \
  --method gpt_only \
  --validate-only
```

Omitting the optional benchmark and method filters selects the entire
experiment matrix. Validation accepts planned catalog entries so the complete
paper matrix can be checked. Execution requires both an executable benchmark
and an implemented method.

The historical `controller` executable remains as an alias to `benchmark`, but
new commands should use `benchmark`.

## Batch data generation

Generate an OpenAI-compatible JSONL file from one canonical batch job with:

```bash
uv run benchmark \
  --generate-batch benchmarks/simulation_3h/jobs/batch-mini-nightly-eval.yaml \
  --output /tmp/batch-mini-nightly-eval.jsonl
```

Use `--records N` for a smaller deterministic sample.

## Artifacts

Every run is isolated by the full selection key:

```text
artifacts/<experiment>/<benchmark>/<method>/<run>/
```

Direct benchmark-method runs use `adhoc` as the experiment ID. Normal runs
write lifecycle state, placement decisions, resource snapshots, process logs,
and per-job runtime artifacts. Placement-only runs checkpoint `placement.json`
after every decision.

## Adding a benchmark or method

To add a benchmark:

1. Add its human definition and `benchmark.yaml` under
   `benchmarks/<benchmark_id>/`.
2. For an executable benchmark, add its canonical files under `jobs/` and its
   `resources.yaml`.
3. Register it in `benchmarks/catalog.yaml`.
4. Add its ID to the relevant experiment manifests.

To add a method:

1. Implement the shared `SchedulingMethod` contract in
   `src/data_generation/methods/<method_id>.py`.
2. Register its display name, status, and settings in `configs/methods.yaml`.
3. Register its loader in `methods/registry.py`.
4. Add it to the appropriate experiment manifests.

No benchmark loading or controller lifecycle code should change when a method
is added.

## Verification

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run python -m unittest discover -s tests -v
```

See [benchmark execution architecture](architecture/benchmark_execution.md)
for the runtime flow and ownership boundaries.
