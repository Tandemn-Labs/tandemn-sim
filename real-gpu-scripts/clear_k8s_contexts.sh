#!/usr/bin/env bash
set -Eeuo pipefail

: "${TANDEMN_CLUSTER_CONTEXTS:?TANDEMN_CLUSTER_CONTEXTS is required}"

CONTEXTS_FILE=$TANDEMN_CLUSTER_CONTEXTS
PYTHON_BIN=${PYTHON_BIN:-python3}
DYNAMO_NAMESPACE=${DYNAMO_NAMESPACE:-dynamo-system}
BATCH_NAMESPACE=${BATCH_NAMESPACE:-tandemn-system}

[[ -f "$CONTEXTS_FILE" ]] || {
    printf 'Contexts file not found: %s\n' "$CONTEXTS_FILE" >&2
    exit 1
}

for required_command in kubectl "$PYTHON_BIN"; do
    command -v "$required_command" >/dev/null || {
        printf 'Required command not found: %s\n' "$required_command" >&2
        exit 1
    }
done

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

for context in "${contexts[@]}"; do
    printf 'Clearing resources from context: %s\n' "$context"

    kubectl --context "$context" --namespace "$DYNAMO_NAMESPACE" \
        delete dgd --all --ignore-not-found
    kubectl --context "$context" --namespace "$BATCH_NAMESPACE" \
        delete job,lws --all --ignore-not-found
done
