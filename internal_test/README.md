# Internal benchmark dashboard

`internal_test` opens an idle local dashboard for the complete
`simulation_3h` benchmark. Nothing starts until **Start 3-hour run** is clicked
in the browser.

```bash
uv run python internal_test/run.py
```

Open <http://127.0.0.1:8765>, review the 26-job schedule, and click the start
button. The controller then runs the original three-hour workload clock at
full speed and preserves the benchmark's configured:

- online start and stop times, traffic patterns, rates, ramps, and concurrency;
- batch start times, complete record counts, deadlines, and hard stops;
- 180-minute offered-work horizon and 30-minute drain window.

All online and batch requests are sent to the local receiver. The receiver
returns minimal OpenAI-compatible responses, so no model server or cloud
account is required.

The dashboard remains available after completion. Press `Ctrl+C` in the
terminal to stop it. Stopping while the run is active asks the controller to
pause and clean up its child processes.

To use another port or executable benchmark:

```bash
uv run python internal_test/run.py --port 9876
uv run python internal_test/run.py \
  --benchmark benchmarks/simulation_3h/overview.yaml
```

Only generated YAML configs and the dashboard plan are stored under
`internal_test/.runtime/`. They contain no generated prompts or responses.
AI-Perf runs with `summary` export in an OS-temporary directory. Its
`inputs.json` path is redirected to the operating system's null device, and the
controller does not create prompt or response JSONL files. The remaining small
summary/log directory is deleted automatically when the controller exits. The
source benchmark is never modified.

Online jobs use a reusable pool of at most 32 generated entries. This preserves
their input/output token distributions and traffic timing without retaining
thousands of large prompts per process. Batch jobs generate into a bounded
queue, send each request, and discard it.

## Dashboard data

Each job has its own local endpoint route and dashboard view. The view includes:

- controller-relative lifecycle state;
- configured traffic shape and rate metadata;
- observed requests per second at one-second resolution;
- batch completion progress;
- direct chart scrubbing over a minute-based three-hour time axis.

The dashboard keeps only one numeric request counter per elapsed second and a
running total for each job. The detailed series is returned only for the job
currently selected in the dashboard. It never stores request bodies, prompt
text, response text, character counts, request-shape metadata, or per-request
artifacts.

The internal test enables the eight batch jobs that are disabled in the source
overview by attaching `internal_test/batch_sender.py` as their local sender.
Their configured datasets are generated and streamed with bounded memory.
