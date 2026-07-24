# AIPerf benchmark controller

The controller runs one benchmark. A benchmark is an overview YAML plus the job
YAMLs it references.

## Structure

```text
src/data_generation/controller.py
benchmarks/
  simulation_3h/
    overview.yaml
    jobs/
      chat-free-core.yaml
      ...
```

`controller.py` is the only entrypoint. It starts the benchmark clock, reads the
overview, launches each job at its scheduled offset, stops jobs at their end
offset, and owns every child process.

`overview.yaml` defines the benchmark ID, timing, and complete job list. Each
job points to one YAML under that benchmark's `jobs/` directory. Online job
files are AIPerf configurations. Batch job files remain placeholders until
their dataset senders are supplied.

Adding another benchmark follows the same layout:

```text
benchmarks/simulation_6h/overview.yaml
benchmarks/simulation_6h/jobs/*.yaml
```

No Python changes are required when the new overview follows the
`benchmark/v1` schema.

## Runtime state

Every controller tick writes:

```text
artifacts/<benchmark-id>/runs.json
artifacts/<benchmark-id>/<run-id>/benchmark_state.json
```

`runs.json` is the lightweight index of runs for that benchmark.
`benchmark_state.json` is the durable snapshot of the controller's in-memory
Python state. It contains benchmark
elapsed time, benchmark milestones, and a `jobs` array containing each job's
state, start, pause, resume, completion, launch count, runtime, and exit code.

## Run

Validate the default `simulation_3h` benchmark:

```bash
uv run controller --validate-only
```

Run it:

```bash
uv run controller simulation_3h
```

Run the smoke benchmark:

```bash
uv run controller tests/smoke_test/overview.yaml
```

Stop with `Ctrl+C`. The controller sends `SIGINT` to every active process group,
waits for the benchmark's shutdown grace period, then force-stops anything
remaining. Active jobs are recorded as paused.

Resume using the run ID printed at startup:

```bash
uv run controller simulation_3h --resume <run-id>
```

Paused wall-clock time does not advance the benchmark. Completed jobs are not
replayed, paused jobs restart for their remaining scheduled window, and future
jobs remain pending. AIPerf restarts a paused generator; it does not restore an
individual in-memory request or session midway through generation.

## Checks

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run python -m unittest discover -s tests
uv run pre-commit run --all-files
```
