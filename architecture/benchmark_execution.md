# Benchmark Execution Architecture

This document describes how a benchmark moves from a controller command to
scheduler placement, data generation, and run artifacts.

## Entry points

Run the three-hour simulation with your choice of method. For example, to run Dynamo+AIC:

```bash
uv run --group dynamo controller simulation_3h --method dynamo-aic
```

Generate placements without submitting jobs or sending prompts:

```bash
uv run --group dynamo controller simulation_3h --method dynamo-aic --placement-only
```

The `controller` command is registered in [`pyproject.toml`](../pyproject.toml)
and invokes `data_generation.controller:main`.

## End-to-end flow

```text
CLI command
  -> controller.main()
  -> load benchmark and job definitions
  -> load the selected scheduling method and resource inventory
  -> start the benchmark clock
  -> process each job at its arrival time
       -> create scheduler-neutral job input
       -> schedule and reserve hardware
       -> persist the placement
       -> submit the accepted job
       -> start its data sender
  -> monitor jobs and persist runtime state
  -> stop senders and release hardware
```

## 1. Controller initialization

[`controller.main()`](../src/data_generation/controller.py) performs the
top-level setup:

1. Load the repository's local `.env` file.
2. Resolve `simulation_3h` to
   [`benchmarks/simulation_3h/overview.yaml`](../benchmarks/simulation_3h/overview.yaml).
3. Load and validate the benchmark and all referenced job files.
4. Load the scheduler selected by `--method`.
5. Enter normal execution or placement-only execution.

## 2. Benchmark and job loading

`load_benchmark()` reads `overview.yaml`. For every job, `load_job()` resolves
its referenced YAML under
[`benchmarks/simulation_3h/jobs/`](../benchmarks/simulation_3h/jobs/).

Scheduler input currently comes from two benchmark layers:

| Input | Authoritative source |
|---|---|
| Job ID and lifecycle | `overview.yaml` |
| Model and workload | AIPerf or batch dataset section |
| SLO and submission features | Job `job.features` section |
| Latency or throughput policy | Job `job.optimization_goal` field |

The loader verifies schedule windows, referenced files, supported drivers, and
required per-job metadata.

## 3. Scheduling method initialization

For `--method dynamo-aic`, `load_dynamo_aic_method()` creates a fresh method
instance for the run using:

- [`resources.yaml`](../benchmarks/simulation_3h/resources.yaml): frozen
  hardware pools and initial capacity;
- [`methods/dynamo_aic.yaml`](../benchmarks/simulation_3h/methods/dynamo_aic.yaml):
  AIC backend, database, and optimization defaults.

Shared job, resource, candidate, and decision contracts are defined in
[`scheduling.py`](../src/data_generation/scheduling.py). Dynamo-specific code
is isolated under
[`methods/dynamo_aic/`](../src/data_generation/methods/dynamo_aic/).

## 4. Timed benchmark execution

`run_benchmark()` creates a run directory and begins the benchmark clock. On
each controller poll, it calls `reconcile_job()` for every job.

When the current benchmark offset reaches a pending job's start time,
`reconcile_job()` calls `start_job()`. Scheduling happens before submission or
data generation.

## 5. Per-job placement

`scheduler_job()` converts the controller job into the scheduler-neutral
`SchedulerJob` contract. `DynamoAicMethod.schedule()` then:

1. Captures the current resource snapshot.
2. Evaluates every hardware pool supported by AIC.
3. Passes the job's model, SLO, workload, policy, and each pool's available
   device count to AIC.
4. Rejects configurations that violate the SLO or available capacity.
5. Compares the feasible hardware candidates using the job's policy.
6. Selects a winner and immediately reserves its devices.

Because reservation occurs inside `schedule()`, the next arriving job observes
the updated resource inventory.

## 6. Submission and data generation

If the decision is rejected, the job becomes `rejected`; no submission or
sender is started.

If the decision is accepted, `start_job()`:

1. Submits the job to Tandemn.
2. Receives a Tandemn job ID and endpoint URL.
3. Writes a runtime AIPerf configuration containing that endpoint.
4. Starts AIPerf as an independent child process.

The controller monitors the process on subsequent polls. At the online job's
stop offset, it stops new requests, drains the sender, and releases the
reserved devices when the job becomes terminal.

## 7. Artifacts and traceability

Run data is stored under:

```text
artifacts/benchmarks/<benchmark-id>/<run-id>/
```

Normal benchmark artifacts include:

```text
benchmark_state.json
resource_map.json
jobs/<job-id>/placement.json
jobs/<job-id>/runtime-aiperf.yaml
jobs/<job-id>/process.log
jobs/<job-id>/prompts.jsonl
jobs/<job-id>/responses.jsonl
```

`placement.json` records the AIC request, all evaluated hardware candidates,
predicted performance, selected parallelism, and the resource snapshots before
and after placement. `resource_map.json` records the latest available and
allocated capacity. `benchmark_state.json` records benchmark and job lifecycle
state.

An unplaced decision records both `failure_category` and `reason`. Capacity,
SLO, and load infeasibility are kept distinct from unsupported models,
unavailable models, unsupported hardware, and method errors. Candidate records
retain the complete per-hardware error details.

The benchmark artifact parent also contains `runs.json`, which indexes run IDs
and their latest state.

## Placement-only execution

`run_placement_only()` processes enabled jobs in arrival order using their
benchmark offsets without waiting for wall-clock time. It schedules and
releases resources but never submits jobs or starts data senders.

Online demand uses AIPerf's request rate and token means. Batch demand uses the
dataset record count divided by its deadline, plus its input/output token means.
The simulated batch allocation is released at that deadline because AIC sizes
the configuration to finish the work by then.

In addition to per-job placement files and `resource_map.json`, this mode writes
`placements.json`, containing every decision and the complete resource-event
timeline for the simulated run. Each completed decision is logged to the
terminal with its original `T+` release offset, hardware, device count,
replica count, and parallelism. AIC's internal reports are suppressed;
`placements.json` retains the complete metrics and candidate details. The file
is atomically checkpointed after every decision, so completed placements remain
available if a later AIC evaluation is interrupted.

## Ownership boundaries

| Component | Responsibility |
|---|---|
| Benchmark folder | Workload, schedule, SLO, policy, resources, method settings |
| Controller | Clock, job lifecycle, submission, sender processes, artifacts |
| Shared scheduling contracts | Stable interface between controller and methods |
| Scheduler method | Candidate evaluation, selection, reservation, release |
| AIPerf | Prompt generation and request sending |
| Artifacts | Placement evidence, resource state, runtime state, requests, responses |

New scheduler implementations should remain under
`src/data_generation/methods/<method-name>/` and implement the shared
`SchedulingMethod` contract without placing method-specific logic in benchmark
loading or job execution.
