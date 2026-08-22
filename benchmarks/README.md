# Benchmark Store Setup

This is the one-time local setup for the shared benchmark Store. It creates
the global model and hardware state once, then creates isolated users for each
planner baseline.

## Checkout Layout

Keep these repositories as siblings:

```text
tandemn/
├── tandemn-sim/
├── tandemn-store/
├── Tandemn-orca/
├── koi/
└── tandemn-LLMServingSim/
```

Run the remaining commands from `tandemn-sim/`.

## Install

Docker must be running. Install the simulator and its sibling Store/Orca
dependencies:

```bash
uv sync --group dev
```

This keeps Store, Orca, AIConfigurator, OpenAI, and OpenEvolve installed after
future `uv sync` runs.

## Hardware Snapshots

The benchmark uses fixed, repository-local hardware snapshots:

```text
benchmarks/catalog-snapshots/
├── aws.json
├── azure.json
└── SHA256SUMS
```

These snapshots are the benchmark hardware source of truth. Do not refresh
them during a benchmark comparison.

To intentionally refresh them from local cloud credentials, first confirm the
AWS and Azure CLIs are authenticated:

```bash
aws sts get-caller-identity
az account show
```

Then regenerate only the regions and SKUs used by the benchmark resource map:

```bash
mkdir -p benchmarks/catalog-snapshots

uv run python -m tandemn_orca.scripts.aws_hardware_catalog \
  --regions us-east-1,ap-northeast-1 \
  --output benchmarks/catalog-snapshots/aws.json

uv run python -m tandemn_orca.scripts.azure_hardware_catalog \
  --regions westus2,westus \
  --output benchmarks/catalog-snapshots/azure.json

(
  cd benchmarks/catalog-snapshots
  sha256sum aws.json azure.json > SHA256SUMS
  sha256sum --check SHA256SUMS
)
```

The generators write JSON files only. Do not pass Azure's `--store` option.
Copy these three files unchanged to another machine, then run
`sha256sum --check SHA256SUMS` from that machine's `catalog-snapshots/`
directory before static seeding.

## Static Seed

This command destroys local Tandemn Store volumes, starts Postgres and MinIO,
applies migrations, and seeds the shared benchmark state:

```bash
uv run python benchmarks/benchmark_static_seed.py \
  --full-reset \
  --benchmark-id simulation_3h
```

It verifies the snapshots before destroying anything, then seeds:

```text
1 canonical benchmark user
16 model catalogs
1 canonical resource map selected by benchmark ID
1 AWS hardware catalog
1 Azure hardware catalog
```

It exits nonzero if the catalog checksum, expected catalog identity, resource
pool coverage, or final Store counts are wrong.

## Baseline Users

Create one isolated user per planner/run pair after static seeding:

```bash
uv run python benchmarks/baseline_user_setup.py \
  --baseline bestfit \
  --run-id 20260820T000000Z
```

Koi additionally imports its fixed causal graph for that user:

```bash
uv run python benchmarks/baseline_user_setup.py \
  --baseline koi \
  --run-id 20260820T000000Z \
  --koi-root ../koi
```

The command prints one JSON identity object to stdout:

```json
{"baseline": "bestfit", "run_id": "20260820T000000Z", "user_id": "usr_..."}
```

Rerunning the same baseline and run ID reuses that user ID, deletes only that
user's runtime rows, and restores its independent 200-GPU simulation resource map. It
does not alter another baseline user, shared model catalogs, or shared hardware
catalogs.

Valid baselines are:

```text
koi
dynamo_aic
bestfit
vram
prism
cp_sat
gpt_only
openevolve
```

ServingSim launch commands and per-user runtime locking are documented with
the launcher once that step is implemented.
