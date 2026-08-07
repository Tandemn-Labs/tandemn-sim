# Benchmark execution architecture

The repository models a run as one cell in an experiment matrix:

```text
experiment -> benchmark x method -> metric profile
```

This keeps comparison design separate from workload definitions and method
implementations. The benchmark controller executes a resolved cell; it does
not decide which methods belong in a paper table.

## Resolution flow

```text
benchmark CLI
  -> experiment registry (optional matrix expansion)
  -> benchmark registry + method registry + metric registry
  -> load benchmark manifest and split job definitions
  -> construct the selected SchedulingMethod
  -> controller execution or placement-only simulation
  -> artifacts/<experiment>/<benchmark>/<method>/<run>/
```

The primary entry point is `data_generation.cli:main`, exposed as
`uv run benchmark`. The `controller` script is a compatibility alias to the
same entry point.

Direct execution names one benchmark and method:

```bash
uv run benchmark --benchmark simulation_3h --method dynamo_aic
```

Experiment execution expands the declared cross product. Optional flags filter
it to one benchmark, one method, or one pair:

```bash
uv run benchmark \
  --experiment agentic_ablations \
  --benchmark simulation_3h \
  --method gpt_only \
  --placement-only
```

## Configuration ownership

| Layer | Owns | Does not own |
|---|---|---|
| Benchmark catalog | Benchmark identity, environment, duration, readiness | Job details or method settings |
| Benchmark manifest | Clock, resource map, job schedule location | Method implementation |
| Job file | Schedule, placement spec, and driver workload | Method implementation |
| Method catalog/config | Method identity, readiness, runtime settings | Python implementation |
| Method module | Candidate evaluation, selection, reservation, release | Benchmark lifecycle |
| Experiment | Compared benchmarks, methods, and metric profile | Runtime behavior |
| Metric profile | Ordered reported measurements | Metric collection implementation |
| Controller | Clock, process lifecycle, submission, resume, artifacts | Concrete placement algorithm |

Method metadata and settings therefore live in `configs/methods.yaml`, while
each executable scheduler is one module under `src/data_generation/methods/`.
The shared registry constructs concrete methods and is the only method-aware
boundary used by the CLI.

## Benchmark loading

`load_benchmark()` reads `benchmark.yaml`, verifies that the benchmark is
executable, then loads every YAML file from its `jobs` directory. Each job has
one canonical document:

```yaml
schema_version: benchmark-job/v1
job_id: <job_id>
schedule: {}
spec: {}
workload: {}
```

The loader turns those three sections into the controller's `Job`.
`scheduler_job()` then narrows that value to the method-neutral `SchedulerJob`
contract.

## Scheduling boundary

Every method receives the same two inputs for each arriving job:

1. `SchedulerJob`, containing model and workload requirements.
2. Its method-owned live resource inventory, initialized from the selected
   benchmark's `resources.yaml`.

Every method returns `SchedulingDecision`. Accepted decisions reserve capacity
before Tandemn submission or data generation begins. Rejected decisions use
the shared failure categories. When a job becomes terminal, the controller
calls the method's `release(job_id)` operation.

`gpt_only` sends the canonical job and current resource map to the configured
OpenAI model. The model returns only a GPU type and parallelism values, or a
null placement. Shared code deterministically resolves its pool, region,
market, backend, and device count before validating and reserving capacity.
`dynamo_aic` evaluates supported hardware-region candidates through AIC and
greedily reserves the best feasible result.

Neither method produces the cluster-plan schema. The controller deterministically
builds that shared schema from `SchedulingDecision`, the canonical job, and the
benchmark resource map. This supplies job and tenant IDs, model, accelerator,
device totals, targets, mechanism ID, lifecycle action, and ladder fields through
one common path for every method.

## Timed execution

For a normal run, the controller:

1. Creates the fully keyed run directory.
2. Starts or restores the benchmark clock.
3. Reconciles jobs at every poll.
4. Schedules an eligible job before submission.
5. Submits accepted work and starts its workload driver.
6. Synchronizes runtime records and state.
7. Stops or drains terminal workloads and releases their capacity.

Placement-only execution processes the same arrivals and releases without
waiting for wall time, submitting jobs, or starting drivers. It checkpoints
the aggregate plan after each completed decision.

## Artifact identity

Run output is keyed by every dimension that can change results:

```text
artifacts/<experiment>/<benchmark>/<method>/<run>/
```

The `adhoc` experiment ID is used for direct benchmark-method selections. This
prevents two methods running the same benchmark from sharing state or
overwriting placement evidence.

Normal run artifacts include `benchmark_state.json`, `placement.json`, the
latest `resource_map.json`, and per-job runtime files. Placement-only execution
writes its checkpointed `placement.json` in the same selection-specific run
directory.

## Extension rules

A new benchmark changes benchmark manifests and catalogs, not method code. A
new method implements `SchedulingMethod`, adds an optional global config, and
registers one loader; it does not add controller branches. A new comparison
changes an experiment manifest and metric profile, not benchmark or method
implementation.
