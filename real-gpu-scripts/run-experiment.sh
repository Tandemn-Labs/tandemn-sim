#!/usr/bin/env bash

: "${LOG_DIR:?LOG_DIR is required}"
mkdir -p -- "$LOG_DIR"
pids=()

cleanup() {
    local status=$?
    trap - EXIT INT TERM

    for pid in "${pids[@]:-}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done

    sleep 10

    for pid in "${pids[@]:-}"; do
        kill -KILL -- "-$pid" 2>/dev/null || true
    done

    wait "${pids[@]}" 2>/dev/null || true
    exit "$status"
}

start() {
    local name=$1
    shift
    setsid "$@" >"$LOG_DIR/$name.log" 2>&1 &
    pids+=("$!")
}

trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup EXIT


### Reset chunk-manager database
/home/user/chunk-manager/reset-chunk-manager-db.sh
/home/user/chunk-manager/.venv/bin/python ~/chunk-manager/examples/python/add_job_and_chunks.py \
    --target "$CHUNK_MANAGER_TARGET" \
    01ARZ3NDEKTSV4RRFFQ69G5FAV:4 \
    00000000000000000000000001:50


### Reset tandemn-store and get user id
TANDEMN_POSTGRES_URL="postgresql+psycopg://tandemn:tandemn@localhost:55432/tandemn" 
export OUTPUT="$(
    /home/user/tandemn-sim/.venv/bin/python \
    /home/user/tandemn-sim/benchmarks/benchmark_static_seed.py \
    --full-reset \
    --benchmark-id "$BENCHMARK" \
    --baselines "$BASELINE"
)"
export TANDEMN_USER_ID="$(
   /home/user/tandemn-sim/.venv/bin/python -c 'import json, sys; print(json.load(sys.stdin)["bestfit"])' \
    <<<"$OUTPUT"
)"
export TANDEMN_POSTGRES_URL="postgresql+psycopg://tandemn:tandemn@localhost:55432/tandemn_${BENCHMARK}_${BASELINE}"
export POSTGRES_URL="$TANDEMN_POSTGRES_URL"


### Run orca
start orca \
uv run --directory "/home/user/tandemn-system" tandemn-orca \
  --user-id "$TANDEMN_USER_ID" \
  --namespace dynamo-system \
  --batch-namespace tandemn-system \
  --chunk-manager-target "$TANDEMN_CHUNK_MANAGER_TARGET" \
  --batch-worker-secret "$TANDEMN_BATCH_WORKER_SECRET" \
  --router-config-dir "$TANDEMN_ROUTER_CONFIG_DIR" \
  --router-binary "$TANDEMN_ROUTER_BINARY" \
  --interval-seconds 10


### Run runner2.py
start runner2 \
  uv run --directory "/home/user/tandemn-sim/" python cluster_sim/runner2.py \
  --benchmark "$BENCHMARK" \
  --user-id "$TANDEMN_USER_ID" \
  --planner bestfit \
  --logical-epoch 2026-01-01T00:00:00+00:00


### Run real_scenario.py
start real_scenario \
/home/user/tandemn-sim/.venv/bin/python -m data_generation.real_scenario \
  "$BENCHMARK" \
  --user-id "$TANDEMN_USER_ID" \
  --router-config-dir "$TANDEMN_ROUTER_CONFIG_DIR" \
  --source-cloud gcp \
  --source-region us-central1 \
  --source-zone us-central1-a \
  --ready-timeout 1200


# Stop the remaining workload when either Orca or runner2 exits.
wait -n "${pids[@]}"
