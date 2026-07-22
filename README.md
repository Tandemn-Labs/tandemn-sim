# AIPerf workload generator

This project uses AIPerf to generate mock chat requests and logs them as JSON.
It does not send requests to an endpoint yet.

## First-time setup

Run these commands from the `data-generation` folder:

```bash
uv sync
```

`uv sync` creates the virtual environment and installs AIPerf and the project.
You do not need to run `source .venv/bin/activate` when using `uv run`.

## Generate requests

Generate five requests from workload 1:

```bash
uv run workloads generate chat-job-1
```

Generate a specific number:

```bash
uv run workloads generate chat-job-1 --count 100
```

Show complete prompts instead of 200-character previews:

```bash
uv run workloads generate chat-job-1 --count 5 --full-prompt
```

Use another workload by changing the job name:

```bash
uv run workloads generate chat-job-2
uv run workloads generate chat-job-3
uv run workloads generate chat-job-4
uv run workloads generate chat-job-5
```

Stop generation with `Ctrl+C`.

## Help and validation

Show the available command:

```bash
uv run workloads --help
```

Show the options for `generate`:

```bash
uv run workloads generate --help
```

Validate a workload after editing it:

```bash
uv run aiperf config validate workloads/chat-job-1.yaml
```

Run the code-quality checks:

```bash
uv run ruff check src
uv run ruff format --check src
uv run mypy src
uv run pre-commit run --all-files
```

## KV-cache-oriented requests

`workloads/chat-job-1.yaml` includes request-shape settings that encourage KV
cache reuse:

```yaml
connectionReuse: sticky-user-sessions

prefixPrompts:
  sharedSystemLength: 512
  userContextLength: 256

turns: {mean: 6, stddev: 2, min: 2, max: 12}
```

- `sharedSystemLength` gives every session the same 512-token starting prefix.
- `userContextLength` gives each session a 256-token context reused by its turns.
- `turns` creates multi-turn sessions with recurring context.
- `sticky-user-sessions` asks AIPerf to keep a session on the same connection
  when it eventually sends traffic.

These settings create cacheable similarity; they do not enable a KV cache.
Cache capacity, eviction, TTL, and hit behavior belong to the active endpoint or
simulator.

The current command prints request shapes immediately. The three-hour traffic
phase in the YAML is retained for a future sender, but this preview command does
not wait three hours or reproduce its arrival timing.

## Use from Python

The same generator can be called without the CLI:

```python
from data_generation import generate

generate("chat-job-1", count=100, full_prompt=False)
```

Edit the workload behavior in `workloads/chat-job-1.yaml`. The full parameter
reference is in `workloads/00-PARAMETERS.md`.
