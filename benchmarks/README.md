# Simulation Benchmark Runbook

This runbook creates the shared Store state, one isolated Store user per
planner, and one standalone simulation process per baseline. It is terminal
and session-manager agnostic: run commands one at a time wherever you manage
long-lived processes.

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

## Simulation Benchmarks

The executable simulation benchmark IDs and their resource-map capacities are:

| Benchmark | Duration | Pools | GPUs |
| --- | ---: | ---: | ---: |
| `simulation_3h` | 3 hours | 9 | 200 |
| `simulation_12h` | 12 hours | 9 | 400 |
| `simulation_24h` | 24 hours | 9 | 800 |

The executable real-hardware benchmark added for the 104-GPU fleet is:

| Benchmark | Duration | Pools | GPUs |
| --- | ---: | ---: | ---: |
| `real100gpubenchmark` | 3 hours plus 30-minute drain | 5 | 104 |

All baseline users must receive the same resource map as the simulation they
run. The following values must always match:

```text
benchmark_static_seed.py --benchmark-id X
baseline_user_setup.py  --benchmark-id X
cluster_sim/cli.py      --benchmark X
```

`baseline_user_setup.py` requires `--benchmark-id`; it must never infer a
benchmark from a default.

## Isolated Baseline Databases

To run baselines concurrently, create one PostgreSQL database per baseline.
This prevents global Store job-ID collisions and keeps all jobs, plans, ranks,
metrics, evidence, and resource maps isolated. Docker is reset once; Postgres
and MinIO containers are shared.

```bash
uv run python benchmarks/benchmark_static_seed.py \
  --full-reset \
  --benchmark-id simulation_3h \
  --baselines bestfit contextual_bandit cp_sat dynamo_aic edf evox gpt_only openevolve prism vram
```

The command creates fixed database names such as:

```text
tandemn_3h_bestfit
tandemn_3h_dynamo_aic
tandemn_3h_gpt_only
```

It prints one JSON object mapping each baseline to its isolated Store user ID.
It already performs the equivalent of `baseline_user_setup.py` in each
database, so do not run that script again for these baselines.

## Static Seed

This command destroys local Tandemn Store volumes, starts Postgres and MinIO,
applies migrations, and seeds the shared benchmark state:

```bash
uv run python benchmarks/benchmark_static_seed.py \
  --full-reset \
  --benchmark-id simulation_12h
```

Use `simulation_3h`, `simulation_12h`, or `simulation_24h` consistently for
the entire campaign; this example uses `simulation_12h`.

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

`--full-reset` destroys the local Store volumes. Stop every simulation process
using that Store before running it. Do not change a resource map beneath a
running simulation; stop it and restart with a fresh baseline user instead.

## Baseline Users

Create one isolated user per planner/run pair after static seeding:

```bash
uv run python benchmarks/baseline_user_setup.py \
  --baseline bestfit \
  --run-id 20260820T000000Z \
  --benchmark-id simulation_12h
```

Koi additionally imports its fixed causal graph for that user:

```bash
uv run python benchmarks/baseline_user_setup.py \
  --baseline koi \
  --run-id 20260820T000000Z \
  --benchmark-id simulation_12h \
  --koi-root ../koi
```

The command prints one JSON identity object to stdout:

```json
{"baseline": "bestfit", "run_id": "20260820T000000Z", "user_id": "usr_..."}
```

Rerunning the same baseline and run ID reuses that user ID, deletes only that
user's runtime rows, and restores its independent selected benchmark resource map. It
does not alter another baseline user, shared model catalogs, or shared hardware
catalogs.

Create users one by one for these registered baselines:

```text
koi
dynamo_aic
bestfit
vram
edf
prism
cp_sat
gpt_only
openevolve
contextual_bandit
evox
```

The command prints the Store user ID required by the simulator:

```json
{"baseline": "bestfit", "run_id": "20260820T000000Z", "user_id": "usr_..."}
```

## Verify Capacity

Before launching a baseline, verify its isolated resource map. Replace
`<USER_ID>` with the ID printed by `baseline_user_setup.py`.

```bash
docker exec tandemn-postgres psql -X -U tandemn -d tandemn -P pager=off -c "
WITH pools AS (
  SELECT r.user_id, pool
  FROM resource_maps r
  CROSS JOIN LATERAL jsonb_path_query(
    r.pools_json,
    '$.clouds.**.machine_pools.*'
  ) AS pool
)
SELECT
  u.name,
  count(*) AS pools,
  sum(
    (pool->>'total_instances')::int *
    (pool->>'gpus_per_instance')::int
  ) AS total_gpus
FROM users u
JOIN pools ON pools.user_id = u.user_id
WHERE u.user_id = '<USER_ID>'
GROUP BY u.name;
"
```

The result must have nine pools and the GPU count in the simulation benchmark
table above. A 12-hour or 24-hour baseline reporting 200 GPUs is incorrect and
must not be launched.

## Service Check

The static seed starts Postgres and MinIO. Confirm Store and ServingSim before
starting any simulation:

```bash
docker compose -f ../tandemn-store/docker-compose.yml ps
docker exec tandemn-postgres pg_isready -U tandemn -d tandemn
docker start servingsim_docker
docker exec servingsim_docker python3 -m serving --help >/dev/null
```

## Run One Baseline

Run this command once per prepared baseline. Replace every angle-bracket value
manually; no shell variables or process-manager commands are required.

```bash
uv run python cluster_sim/cli.py \
  --benchmark <simulation_3h|simulation_12h|simulation_24h> \
  --postgres-url postgresql+psycopg://tandemn:tandemn@localhost:55432/<BASELINE_DATABASE> \
  --user-id <USER_ID> \
  --planner <BASELINE> \
  --logical-epoch 2026-08-22T00:00:00+00:00 \
  --run-name <BENCHMARK>-<BASELINE>-<RUN_ID> \
  --llmservingsim-root ../tandemn-LLMServingSim \
  --container servingsim_docker \
  --parallel-simulators <WORKER_COUNT> \
  --rank-timeout-seconds 330 \
  --trace-root traces/<RUN_ID> \
  --trace-run-id <BENCHMARK>-<BASELINE>-<RUN_ID>
```

Use a unique trace root and run ID for every retry. Do not overwrite traces
from an invalid or interrupted run.

### Baseline Requirements

| Baseline | Additional simulator arguments | External requirement |
| --- | --- | --- |
| `koi` | `--koi-root ../koi --koi-arg=--run-id=<RUN_ID> --koi-arg=--log-dir=logs/koi --koi-arg=--trace=no-llm` | OpenAI-compatible Koi credentials |
| `dynamo_aic` | None | Local AIConfigurator data |
| `bestfit` | None | None |
| `vram` | None | None |
| `edf` | None | None |
| `prism` | None | None |
| `cp_sat` | None | OR-Tools installed by `uv sync` |
| `gpt_only` | None | `OPENAI_MODEL` and OpenAI-compatible credentials |
| `openevolve` | None | OpenAI-compatible credentials |
| `contextual_bandit` | None | None |
| `evox` | None | SkyDiscover and OpenAI-compatible credentials |

LLMServingSim uses local precomputed profiles for these benchmarks. A Hugging
Face token is not required for ordinary simulation runs.

### Koi With Azure Foundry

Set the matching Koi Foundry endpoint and key in the parent process, then pass
only the deployed model name through the existing Koi argument bridge:

```bash
export KOI_FOUNDRY_DEEPSEEK_BASE_URL='https://deepseek-models.openai.azure.com/openai/v1'
export KOI_FOUNDRY_DEEPSEEK_API_KEY='...'
export TANDEMN_POSTGRES_URL='postgresql+psycopg://tandemn:tandemn@localhost:55432/<BASELINE_DATABASE>'

uv run python cluster_sim/cli.py \
  --benchmark simulation_3h \
  --postgres-url "$TANDEMN_POSTGRES_URL" \
  --user-id <USER_ID> \
  --planner koi \
  --koi-root ../koi \
  --koi-arg=--model=DeepSeek-V4-Pro \
  --koi-arg=--trace=all
```

Koi is imported into the simulator process, so it receives those environment
variables directly. `TANDEMN_POSTGRES_URL` must match `--postgres-url` because
Koi creates its own Store client.

## Optional Two-Tick Preflight

Before a long run, use a separate prepared baseline user and add this argument
to the simulator command:

```bash
--ticks 2
```

Do not continue that preflight user as the final benchmark. A tick-limited run
finishes its submitted jobs; create a fresh baseline user with the same
benchmark ID for the full run.

## Results

The common trace and LLM usage files are written to:

```text
traces/<RUN_ID>/simulation/<BENCHMARK>/<BASELINE>/<TRACE_RUN_ID>.jsonl
traces/<RUN_ID>/simulation/<BENCHMARK>/<BASELINE>/<TRACE_RUN_ID>.llm-usage.jsonl
```

Koi also writes native diagnostics under:

```text
logs/koi/<RUN_ID>/runner.log
logs/koi/<RUN_ID>/events.jsonl
```

A full run ends with a `run_finished` event whose termination reason is
`SIMULATION_HARD_STOP`. Inspect the trace for `PROFILE_UNAVAILABLE` and
`PROCESS_CRASH`; do not treat a run containing either as a clean comparison.
