#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${LOG_DIR:?LOG_DIR is required}"
mkdir -p -- "$LOG_DIR"
pids=()
started_pid=
orca_pid=
runner_pid=
real_scenario_pid=
first_exited_pid=
first_exited_status=
delay_orca_shutdown=false
unexpected_exit=false
interrupted=false

declare -A pid_to_name=()

kill_descendants() {
    local ppid=$1
    local signal=$2
    local pids_file
    for pids_file in /proc/[0-9]*/status; do
        local cpid="${pids_file%/status}"
        cpid="${cpid#/proc/}"
        if [[ -r "$pids_file" ]] && grep -qE "^PPid:[[:space:]]*${ppid}$" "$pids_file" 2>/dev/null; then
            kill_descendants "$cpid" "$signal"
        fi
    done
    kill "-$signal" "$ppid" 2>/dev/null || true
}

kill_managed_process() {
    local pid=$1
    local signal=$2

    kill_descendants "$pid" "$signal"
    # The group signal also reaches children reparented after the root exits.
    kill "-$signal" -- "-$pid" 2>/dev/null || true
}

cleanup() {
    local status=$?
    local pid
    trap - EXIT
    # Do not let a second Ctrl-C abort cleanup before the SIGKILL escalation.
    trap '' INT TERM

    printf 'Stopping managed processes...\n' >&2

    if [[ "$delay_orca_shutdown" == true && "$interrupted" != true ]]; then
        for pid in "${pids[@]:-}"; do
            [[ "$pid" == "$orca_pid" || "$pid" == "$real_scenario_pid" ]] && continue
            kill_managed_process "$pid" TERM
        done

        sleep 10

        for pid in "${pids[@]:-}"; do
            [[ "$pid" == "$orca_pid" || "$pid" == "$real_scenario_pid" ]] && continue
            kill_managed_process "$pid" KILL
        done

        sleep 10
        for pid in "$orca_pid" "$real_scenario_pid"; do
            kill_managed_process "$pid" TERM
        done
        sleep 10
        for pid in "$orca_pid" "$real_scenario_pid"; do
            kill_managed_process "$pid" KILL
        done
    else
        for pid in "${pids[@]:-}"; do
            kill_managed_process "$pid" TERM
        done

        sleep 10

        for pid in "${pids[@]:-}"; do
            kill_managed_process "$pid" KILL
        done
    fi

    local failed=false
    for pid in "${pids[@]:-}"; do
        local ec=0
        if [[ "$pid" == "$first_exited_pid" ]]; then
            ec=$first_exited_status
        else
            wait "$pid" 2>/dev/null || ec=$?
        fi
        if [[ "$pid" == "$first_exited_pid" && "$unexpected_exit" == true ]]; then
            local name=${pid_to_name[$pid]:-unknown}
            printf '\n=== Process "%s" (PID %s) exited unexpectedly with code 0 ===\n' "$name" "$pid" >&2
            if [[ -f "$LOG_DIR/$name.log" ]]; then
                printf 'Last log lines from %s/%s.log:\n' "$LOG_DIR" "$name" >&2
                tail -n 50 "$LOG_DIR/$name.log" >&2
            fi
            failed=true
        elif [[ "$ec" != 0 && "$ec" != 143 && "$ec" != 137 && "$ec" != 127 ]]; then
            local name=${pid_to_name[$pid]:-unknown}
            printf '\n=== Process "%s" (PID %s) exited with code %s ===\n' "$name" "$pid" "$ec" >&2
            if [[ -f "$LOG_DIR/$name.log" ]]; then
                printf 'Last log lines from %s/%s.log:\n' "$LOG_DIR" "$name" >&2
                tail -n 50 "$LOG_DIR/$name.log" >&2
            fi
            failed=true
        fi
    done

    if $failed; then
        printf '\ntandemn-sim FAILED\n' >&2
    fi

    exit "$status"
}

start() {
    local name=$1
    shift
    setsid "$@" >"$LOG_DIR/$name.log" 2> >(tee -a "$LOG_DIR/$name.log" >&2) &
    started_pid=$!
    pids+=("$started_pid")
    pid_to_name[$started_pid]=$name
}

trap 'interrupted=true; exit 130' INT
trap 'interrupted=true; exit 143' TERM
trap cleanup EXIT


#################### DO ALL THE CLEANUP ####################

### Reset chunk-manager database
/home/user/chunk-manager/reset-chunk-manager-db.sh
/home/user/chunk-manager/.venv/bin/python ~/chunk-manager/examples/python/add_job_and_chunks.py \
    --target "$CHUNK_MANAGER_TARGET" \
    0000000000ZQQJMH7ZKN9JB75S:4 \
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

### Clear Kubernetes contexts
bash "$SCRIPT_DIR/clear_k8s_contexts.sh"

### Clear logs
rm -rf -- "$LOG_DIR"/* "$LOG_DIR"/.[!.]* "$LOG_DIR"/..?*

### Clear S3 bucket for output
bash "$SCRIPT_DIR/clear-s3-outputs.sh"


#################### START EXPERIMENT ####################

### Run GPU metrics collectors
start gpu_metrics bash "$SCRIPT_DIR/collect-gpu-metrics.sh"


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
orca_pid=$started_pid


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
real_scenario_pid=$started_pid


sleep 10


### Run runner2.py
start runner2 \
  uv run --directory "/home/user/tandemn-sim/" python cluster_sim/runner2.py \
  --benchmark "$BENCHMARK" \
  --user-id "$TANDEMN_USER_ID" \
  --planner bestfit \
  --logical-epoch 2026-01-01T00:00:00+00:00
runner_pid=$started_pid


# The first managed process to exit ends the experiment. Only runner2 gives
# Orca and real_scenario two reconciliation intervals to finish workloads.
wait -n -p first_exited_pid "${pids[@]}"
first_exited_status=$?

if [[ "$first_exited_pid" == "$runner_pid" ]]; then
    delay_orca_shutdown=true
elif [[ "$first_exited_status" == 0 ]]; then
    unexpected_exit=true
    exit 1
fi

exit "$first_exited_status"
