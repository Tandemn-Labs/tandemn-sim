# Tandemn Simulator

This repository evaluates methods for scheduling AI inference workloads onto GPU clusters. It provides a common simulation and benchmark harness for comparing how different schedulers admit jobs, select hardware, choose parallelism configurations, and create placement plans under changing cluster demand.

The goal is to run the same workload against multiple scheduling approaches and measure placement quality, resource efficiency, latency, throughput, deadline completion, and behavior under contention.

## Scheduler methods

The scheduler implementations and adapters live in `cluster_sim/`. Current approaches include:

- Fixed and best-fit placement baselines.
- Maximum-VRAM placement.
- NVIDIA Dynamo AIConfigurator placement.
- GPT-only and agentic placement methods.
- A Koi planner adapter for Store-backed scheduling runs.

Additional scheduling methods are tracked in `configs/methods.yaml`. Each method receives the same benchmark workload and cluster state so results can be compared consistently.

## Benchmarks

Benchmarks live in `benchmarks/` and cover both simulated and real-hardware environments. They define the cluster resources, model mix, online and batch jobs, arrival patterns, SLOs, deadlines, and run duration.

See [`benchmarks/README.md`](benchmarks/README.md) for one-time local Store setup,
fixed hardware snapshots, and isolated baseline-user creation.

The benchmark catalog currently includes:

- 3-hour, 12-hour, and 24-hour simulation workloads.
- 3-hour, 12-hour, and 24-hour real-hardware workload definitions.
- `simulation_12h_homogenous`, an executable 12-hour simulation benchmark for a homogeneous, on-demand 400-GPU H100 cluster.
- `15min_real`, a 15-minute Orca benchmark on 8 RTX PRO 6000 and 8 L4 GPUs.

See `benchmarks/catalog.yaml` for the complete list and execution status of each benchmark. Experiment groups in `experiments/` combine benchmarks, scheduler methods, and metric suites for specific evaluations.

## Repository layout

```text
benchmarks/          Benchmark manifests, jobs, and resource definitions
cluster_sim/         Simulator, scheduler adapters, planners, and telemetry
configs/             Scheduler and metric catalogs
experiments/         Reproducible experiment definitions
src/data_generation/ AIPerf workload launcher
tests/               Data-generation tests
```

## Getting started

Install the project and development dependencies:

```bash
uv sync --group dev
```

Inspect the cluster simulator options:

```bash
uv run python cluster_sim/cli.py --help
```

Run one AIPerf-backed benchmark job by its YAML job ID:

```bash
uv run controller job-benchmark_timing_3hr-0-1
```

The simulator integrates with Tandemn Store and external serving components, so complete benchmark runs require those services to be available.

## Development checks

```bash
uv run ruff format --check src tests cluster_sim
uv run ruff check src tests cluster_sim
uv run mypy src
uv run python -m unittest discover -s tests
```
