# AIPerf data-generation controller

The controller starts one AIPerf process for one benchmark job. It does not
schedule jobs, submit jobs, manage run state, or transform AIPerf output.

## Run a job

Install the project:

    uv sync --group dev

Pass the exact job ID declared by a YAML file under the benchmark jobs
directory:

    uv run controller job-benchmark_timing_3hr-0-1

The controller:

1. Finds the unique job with that ID.
2. Requires the job's driver to be AIPerf.
3. Extracts its workload as the AIPerf configuration.
4. Adds the prompt token distributions declared in the job specification.
5. Runs AIPerf with the temporary configuration.
6. Returns AIPerf's exit code.

The job starts immediately. Schedule offsets in the benchmark job are not
interpreted. AIPerf owns its endpoint behavior and generated artifacts. When a
job does not declare an endpoint URL, AIPerf uses its localhost default.

## Source layout

    src/data_generation/
      __init__.py
      controller.py

controller.py is the only functional data-generation module.

## Verify

    uv run ruff format --check src tests
    uv run ruff check src tests
    uv run mypy src
    uv run python -m unittest tests.test_controller -v
