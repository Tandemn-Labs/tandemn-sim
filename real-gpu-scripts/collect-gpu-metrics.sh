#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTEXTS_FILE="$TANDEMN_CLUSTER_CONTEXT"
LOG_DIR=${LOG_DIR:-"$SCRIPT_DIR/logs"}/gpu-metrics-collectors
LOCAL_PORT_BASE=${LOCAL_PORT_BASE:-9090}
PROMETHEUS_NAMESPACE=${PROMETHEUS_NAMESPACE:-monitoring}
PROMETHEUS_SERVICE=${PROMETHEUS_SERVICE:-kube-prometheus-stack-prometheus}
PROMETHEUS_REMOTE_PORT=${PROMETHEUS_REMOTE_PORT:-9090}
DYNAMO_NAMESPACE=${DYNAMO_NAMESPACE:-dynamo-system}
BATCH_NAMESPACE=${BATCH_NAMESPACE:-tandemn-system}
PYTHON_BIN=${PYTHON_BIN:-python3}

: "${TANDEMN_USER_ID:?TANDEMN_USER_ID is required}"
: "${SYS:?SYS is required}"
ROUTER_TOKEN=${ROUTER_TOKEN:-${TANDEMN_ROUTER_TELEMETRY_TOKEN:?ROUTER_TOKEN or TANDEMN_ROUTER_TELEMETRY_TOKEN is required}}

[[ -f "$CONTEXTS_FILE" ]] || {
    printf 'Contexts file not found: %s\n' "$CONTEXTS_FILE" >&2
    exit 1
}
[[ -d "$SYS" ]] || {
    printf 'Tandemn system directory not found: %s\n' "$SYS" >&2
    exit 1
}
[[ $LOCAL_PORT_BASE =~ ^[0-9]+$ ]] || {
    printf 'LOCAL_PORT_BASE must be numeric: %s\n' "$LOCAL_PORT_BASE" >&2
    exit 1
}

for required_command in kubectl setsid uv "$PYTHON_BIN"; do
    command -v "$required_command" >/dev/null || {
        printf 'Required command not found: %s\n' "$required_command" >&2
        exit 1
    }
done

mkdir -p -- "$LOG_DIR"
pids=()
last_pid=

cleanup() {
    local status=$?
    trap - EXIT INT TERM

    if ((${#pids[@]})); then
        for pid in "${pids[@]}"; do
            kill -TERM -- "-$pid" 2>/dev/null || true
        done

        sleep 10

        for pid in "${pids[@]}"; do
            kill -KILL -- "-$pid" 2>/dev/null || true
        done

        wait "${pids[@]}" 2>/dev/null || true
    fi
    exit "$status"
}

start() {
    local name=$1
    shift
    setsid "$@" >"$LOG_DIR/$name.log" 2>&1 &
    last_pid=$!
    pids+=("$last_pid")
}

wait_for_port_forward() {
    local pid=$1
    local port=$2
    local context=$3
    local attempt

    for ((attempt = 0; attempt < 100; attempt++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            printf 'Port-forward for %s exited; see %s/port-forward-%s.log\n' \
                "$context" "$LOG_DIR" "$port" >&2
            return 1
        fi
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            return 0
        fi
        sleep 0.1
    done

    printf 'Timed out waiting for Prometheus port-forward for %s on port %s\n' \
        "$context" "$port" >&2
    return 1
}

trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup EXIT

context_lines="$("$PYTHON_BIN" - "$CONTEXTS_FILE" <<'PY'
import json
import sys

with open(sys.argv[1]) as contexts_file:
    contexts = json.load(contexts_file)

if not isinstance(contexts, dict):
    raise SystemExit("contexts JSON must be an object mapping labels to kube contexts")

for label, context in contexts.items():
    if not isinstance(context, str) or not context:
        raise SystemExit(f"invalid kube context for {label!r}")
    print(context)
PY
)"

[[ -n "$context_lines" ]] || {
    printf 'No kube contexts found in %s\n' "$CONTEXTS_FILE" >&2
    exit 1
}
mapfile -t contexts <<<"$context_lines"

for index in "${!contexts[@]}"; do
    context=${contexts[index]}
    local_port=$((LOCAL_PORT_BASE + index))

    start "port-forward-$local_port" \
        kubectl --context "$context" --namespace "$PROMETHEUS_NAMESPACE" \
        port-forward --address 127.0.0.1 "svc/$PROMETHEUS_SERVICE" \
        "$local_port:$PROMETHEUS_REMOTE_PORT"
    port_forward_pid=$last_pid
    wait_for_port_forward "$port_forward_pid" "$local_port" "$context"

    start "collector-$local_port" \
        uv run --directory "$SYS" tandemn-gpu-metrics-collector \
        --prometheus-url "http://127.0.0.1:$local_port" \
        --kube-context "$context" \
        --namespace "$DYNAMO_NAMESPACE" \
        --batch-namespace "$BATCH_NAMESPACE" \
        --user-id "$TANDEMN_USER_ID" \
        --router-telemetry-token "$ROUTER_TOKEN"

    printf 'Started collector for %s at http://127.0.0.1:%s\n' "$context" "$local_port"
done

printf 'Collectors are running. Press Ctrl-C to stop all collectors and port-forwards.\n'
wait -n "${pids[@]}"
